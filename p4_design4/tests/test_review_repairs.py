"""Regression coverage for checkpoint interoperability and review findings."""

from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import load_model
from legacy_grid import legacy_axes
from losses import TrainingConfig
from main import make_site_split
from metrics import MineMetrics
from network import ModelConfig, SPAIIceInversionNet
from protocol import validate_matlab_config, validate_record
from protocol import sha256_file
from predict_features import predict_records
from inference import SPAIDeployment
from safety import (
    bearing_capacity,
    capacity_error,
    validation_grid,
    ordinary_kriging,
    thickness_capacity_relative_bias,
    composition_capacity_relative_bias,
)
from event_association import associate_events
from mine_ablations import CASES, build_ablation


class ReviewRepairs(unittest.TestCase):
    def test_generic_prediction_does_not_inherit_held_out_status_from_model(self):
        cfg = ModelConfig(cnn_channels=8, gru_hidden=4, unet_base=4, delay_channels=4)
        model = SPAIDeployment(SPAIIceInversionNet(cfg)).eval()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            file = folder / "record.h5"
            with h5py.File(file, "w") as f:
                f.attrs.update(icemap_schema="icemap.features.v1", site_id="fixture")
                for name in ("phase", "E", "tau"):
                    f[name] = np.ones((2, 32), dtype=np.float32)
                f["scatter_valid"] = np.ones((2, 32))
                f["time_s"] = np.arange(32) * 1e-11
                f["depth_m"] = np.arange(32) * 0.001
            bundle = {
                "model_config": asdict(cfg),
                "evaluation_role": "site_held_out",
                "model_kind": "spai_six_loss",
            }
            report = predict_records(model, bundle, [file], folder / "unverified")
            self.assertFalse(report["held_out_independence_verified"])
            digest = sha256_file(file)
            bundle["data_split_sha256"] = {
                "test": [digest],
                "train": [],
                "validation": [],
            }
            report = predict_records(model, bundle, [file], folder / "heldout")
            self.assertTrue(report["held_out_independence_verified"])
            bundle["data_split_sha256"]["train"] = [digest]
            report = predict_records(model, bundle, [file], folder / "contaminated")
            self.assertFalse(report["held_out_independence_verified"])

    def test_matched_cnns_run_the_six_losses_and_backpropagate(self):
        cfg = ModelConfig(cnn_channels=8, gru_hidden=4, unet_base=4, delay_channels=4)
        from losses import MineLoss

        x = torch.randn(1, 2, 64)
        batch = {
            "phase": x,
            "scatter": x.square(),
            "tof": x.abs(),
            "scatter_valid": torch.ones_like(x, dtype=torch.bool),
            "time_s": torch.arange(64) * 2e-11,
            "depth_m": torch.arange(64) * 0.001,
            "boundary_target": torch.zeros_like(x),
            "layer_count": torch.ones(1, 2, dtype=torch.long),
            "relative_density_target": torch.ones(1, 2, 10),
            "layer_density_gt": torch.ones(1, 2, 10) * 0.9,
            "layer_thickness_m_gt": torch.ones(1, 2, 10) * 0.02,
            "density_profile_gt": torch.ones_like(x) * 0.9,
        }
        for case in CASES:
            with self.subTest(case=case):
                model = build_ablation(cfg, case).train()
                out = model(
                    **{
                        k: batch[k]
                        for k in (
                            "phase",
                            "scatter",
                            "tof",
                            "scatter_valid",
                            "time_s",
                            "depth_m",
                        )
                    }
                )
                loss, _, _ = MineLoss()(out, batch, 79)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertLessEqual(
                    model.ablation_metadata["total_relative_difference"], 0.01
                )
                self.assertTrue(
                    any(
                        p.grad is not None and p.grad.abs().sum() > 0
                        for p in model.parameters()
                    )
                )
                model.eval()
                inputs = {
                    k: batch[k]
                    for k in (
                        "phase",
                        "scatter",
                        "tof",
                        "scatter_valid",
                        "time_s",
                        "depth_m",
                    )
                }
                with tempfile.TemporaryDirectory() as directory:
                    file = Path(directory) / "ablation.pt"
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "model_config": asdict(cfg),
                            "training_config": asdict(TrainingConfig()),
                            "ablation_case": case,
                        },
                        file,
                    )
                    loaded, _ = load_model(file, "cpu")
                    with torch.inference_mode():
                        expected = model(**inputs)
                        actual = loaded(inputs)
                    torch.testing.assert_close(
                        actual["layer_thickness_m"], expected["layer_thickness_m"]
                    )

    def test_observed_event_association_monotone_and_missing_evidence(self):
        estimated = torch.tensor([[[1.0, 3.0, 5.0, 7.0]]], dtype=torch.float64) * 1e-9
        observed = (
            torch.tensor([[[0.9, 2.0, 3.1, 5.1, 8.0]]], dtype=torch.float64) * 1e-9
        )
        count = torch.tensor([[2]])
        result, used = associate_events(
            estimated,
            count,
            observed,
            torch.ones_like(observed, dtype=torch.bool),
            0.3e-9,
        )
        torch.testing.assert_close(result[0, 0, :3], observed[0, 0, [0, 2, 3]])
        self.assertTrue(used.all())
        with self.assertRaisesRegex(ValueError, "insufficient"):
            associate_events(
                estimated,
                count,
                observed[..., :2],
                torch.ones_like(observed[..., :2], dtype=torch.bool),
                0.3e-9,
            )
        with self.assertRaisesRegex(ValueError, "no monotone"):
            associate_events(
                estimated,
                count,
                observed,
                torch.ones_like(observed, dtype=torch.bool),
                0.01e-9,
            )

    def test_pure_spai_save_load_cpu_prediction_roundtrip(self):
        torch.manual_seed(11)
        cfg = ModelConfig(cnn_channels=8, gru_hidden=4, unet_base=4, delay_channels=4)
        original = SPAIIceInversionNet(cfg).eval()
        x = torch.randn(1, 2, 32)
        inputs = {
            "phase": x,
            "scatter": x.square(),
            "tof": x.abs(),
            "scatter_valid": torch.ones_like(x, dtype=torch.bool),
            "time_s": torch.arange(32, dtype=torch.float64) * 1e-11,
            "depth_m": torch.arange(32) * 0.001,
        }
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "spai.pt"
            torch.save(
                {
                    "model": original.state_dict(),
                    "model_config": asdict(cfg),
                    "training_config": asdict(TrainingConfig()),
                },
                file,
            )
            deployed, bundle = load_model(file, "cpu")
            with torch.inference_mode():
                expected = original(**inputs)
                actual = deployed(inputs)
            for key in (
                "layer_thickness_m",
                "layer_density",
                "layer_delays_s",
                "layer_count",
            ):
                torch.testing.assert_close(actual[key], expected[key])
            self.assertTrue(all(torch.is_tensor(v) for v in actual.values()))
            self.assertEqual(bundle["model_kind"], "spai_six_loss")

    def test_matlab_config_is_checked_not_ignored(self):
        cfg = ModelConfig()
        raw = asdict(cfg)
        validate_matlab_config(json.dumps(raw), cfg)
        raw["max_layers"] = 9
        with self.assertRaisesRegex(ValueError, "max_layers"):
            validate_matlab_config(json.dumps(raw), cfg)

    def test_formal_protocol_rejects_missing_calibration(self):
        with tempfile.TemporaryDirectory() as d:
            with h5py.File(Path(d) / "f.h5", "w") as f:
                f.attrs["workflow_mode"] = "calibrated_experiment"
                with self.assertRaisesRegex(ValueError, "matlab_config"):
                    validate_record(f, ModelConfig(), TrainingConfig())

    def test_site_folds_disjoint_and_tested_once(self):
        records = [(Path(f"{i}_{j}.h5"), str(i)) for i in range(10) for j in range(2)]
        held_out = []
        for fold in range(5):
            splits, _ = make_site_split(records, fold, 43, 0.2)
            sites = {k: {str(p).split("_")[0] for p in v} for k, v in splits.items()}
            self.assertFalse(sites["train"] & sites["test"])
            self.assertFalse(sites["validation"] & sites["test"])
            self.assertFalse(sites["train"] & sites["validation"])
            held_out += splits["test"]
        self.assertCountEqual(held_out, [p for p, _ in records])

    def test_legacy_grid_must_be_confirmed(self):
        with tempfile.TemporaryDirectory() as d:
            with h5py.File(Path(d) / "f.h5", "w") as f:
                f["phase"] = np.zeros((16, 2, 1))
                with self.assertRaisesRegex(ValueError, "calibrated axes"):
                    legacy_axes(f)
                time, depth, _ = legacy_axes(f, 0.002)
                np.testing.assert_allclose(np.diff(time), 0.004 / 299792458)
                np.testing.assert_allclose(np.diff(depth), 0.002)

    def test_capacity_reference_examples_and_units(self):
        self.assertAlmostEqual(bearing_capacity([0.10], [0.91])["capacity_kg"], 350.0)
        self.assertAlmostEqual(bearing_capacity([0.11], [0.91])["capacity_kg"], 423.5)
        self.assertAlmostEqual(bearing_capacity([0.10], [0.85])["capacity_kg"], 175.0)
        self.assertAlmostEqual(bearing_capacity([0.10], [0.90])["capacity_kg"], 350.0)
        self.assertAlmostEqual(float(thickness_capacity_relative_bias(0.1)), 0.21)
        self.assertAlmostEqual(float(composition_capacity_relative_bias(1, 0)), 1.0)
        self.assertAlmostEqual(capacity_error([350], [175])["BCE_percent"], 100.0)

    def test_kriging_interpolates_observed_sites(self):
        xy = validation_grid()
        values = np.arange(25) + 175.0
        result = ordinary_kriging(
            xy, values, xy, correlation_length_m=3.0, variance_kg2=100.0
        )
        np.testing.assert_allclose(result["capacity_kg"], values, atol=1e-9)
        np.testing.assert_allclose(result["weights"].sum(-1), 1.0, atol=1e-10)
        np.testing.assert_allclose(result["variance_kg2"], 0, atol=1e-9)

    def test_unmatched_metric_policy_is_explicit(self):
        m = MineMetrics()
        m.update([], [], [0.1], [0.9])
        result = m.summary()
        self.assertEqual(result["LER_percent"], 100.0)
        self.assertEqual(result["AE_h_percent"], 100.0)
        self.assertEqual(result["AE_rho_percent"], 100.0)
        self.assertIn("full_unmatched_mass", result["metrics_version"])
        with self.assertRaises(ValueError):
            m.update([0.1], [-0.1], [0.1], [0.9])


if __name__ == "__main__":
    unittest.main()
