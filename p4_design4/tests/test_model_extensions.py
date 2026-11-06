from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
import unittest
import json
import h5py

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration import CalibrationConfig, ResidualPhysicalCalibration
from inference import load_model
from losses import MineLoss, TrainingConfig
from network import ModelConfig, SPAIIceInversionNet
from model_catalog import build_network, ArchitectureConfig, VARIANTS
from robustness import (
    AugmentationConfig,
    perturb_inputs,
    monte_carlo_dropout,
    PhysicalEnsemble,
)
from scene import Scene, encode_context, CONTEXT_DIM, read_scene_h5
from train_opf import CalibratedSPAI
from training_config import RetrainingConfig


class ModelExtensions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self):
        return ModelConfig(cnn_channels=8, gru_hidden=4, unet_base=4, delay_channels=8)

    def batch(self):
        torch.manual_seed(5)
        B, T, N = 2, 3, 65
        batch = {k: torch.randn(B, T, N) for k in ["phase", "scatter", "tof"]}
        batch.update(
            scatter_valid=torch.ones(B, T, N, dtype=torch.bool),
            time_s=torch.arange(N)[None].expand(B, -1) * (0.002 / 299792458),
            depth_m=torch.arange(N)[None].expand(B, -1) * 0.001,
        )
        return batch

    def test_default_factory_preserves_mine_parameters_and_output(self):
        torch.manual_seed(31)
        original = SPAIIceInversionNet(self.config()).eval()
        torch.manual_seed(31)
        selected = build_network(self.config()).eval()
        for key, value in original.state_dict().items():
            torch.testing.assert_close(
                value, selected.state_dict()[key], rtol=0, atol=0
            )
        b = self.batch()
        with torch.no_grad():
            one = original(
                **{
                    k: b[k]
                    for k in [
                        "phase",
                        "scatter",
                        "tof",
                        "scatter_valid",
                        "time_s",
                        "depth_m",
                    ]
                }
            )
            two = selected(
                **{
                    k: b[k]
                    for k in [
                        "phase",
                        "scatter",
                        "tof",
                        "scatter_valid",
                        "time_s",
                        "depth_m",
                    ]
                }
            )
        torch.testing.assert_close(
            one["layer_thickness_m"], two["layer_thickness_m"], rtol=0, atol=0
        )

    def test_all_variants_have_gradients_and_analytic_thickness_for_odd_length(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = CalibratedSPAI(
                    self.config(),
                    variant,
                    {"transformer_heads": 2, "temporal_hidden": 4},
                )
                result = model(self.batch())
                self.assertEqual(tuple(result["layer_density"].shape), (2, 3, 10))
                expected = (
                    299792458
                    * result["layer_delays_s"]
                    / (2 * (1 + 0.845 * result["layer_density"]))
                )
                torch.testing.assert_close(result["layer_thickness_m"], expected)
                loss = (
                    result["boundary_logits"].square().mean()
                    + result["rho_tilde"].mean()
                    + result["layer_thickness_m"].mean()
                )
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
                self.assertTrue(any(g.abs().sum() > 0 for g in grads))

    def test_method_checkpoint_and_context_are_self_contained(self):
        cfg = self.config()
        r = asdict(ArchitectureConfig(context_dim=CONTEXT_DIM, temporal_hidden=4))
        backbone = CalibratedSPAI(cfg, "multiscale", r)
        dim = (2 * cfg.gru_hidden + cfg.unet_base + cfg.delay_channels) * 16 + 32
        head = ResidualPhysicalCalibration(
            dim, torch.zeros(dim), torch.ones(dim), CalibrationConfig(width=8)
        )
        scene = Scene.from_dict({"uav": {"altitude_agl_m": 2.5}})
        vector = encode_context(scene)
        bundle = {
            "backbone_config": asdict(cfg),
            "backbone_state": backbone.state_dict(),
            "backbone_variant": "multiscale",
            "architecture_config": r,
            "calibration_config": asdict(head.config),
            "calibration_state": head.state_dict(),
            "input_dim": dim,
            "pool_bins": 16,
            "default_context": vector,
        }
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "model.pt"
            torch.save(bundle, p)
            model, _ = load_model(p, "cpu")
        b = self.batch()
        with torch.no_grad():
            out = model(b)
            b["context"] = torch.tensor(vector)[None, None].expand(2, 3, -1)
            explicit = model(b)
        torch.testing.assert_close(
            out["layer_thickness_m"], explicit["layer_thickness_m"]
        )

    def test_missing_context_is_not_a_measured_zero_and_invalid_metadata_rejected(self):
        missing = encode_context(Scene())
        zero = encode_context(Scene.from_dict({"ice": {"snow_depth_m": 0.0}}))
        self.assertNotEqual(missing, zero)
        for bad in [
            {"climate": {"relative_humidity": 65}},
            {"uav": {"altitude_agl_m": -1}},
            {"ice": {"surface_temperature_c": -300}},
            {"unknown": {}},
        ]:
            with self.assertRaises((TypeError, ValueError)):
                Scene.from_dict(bad)

    def test_identity_and_dropout_augmentation_do_not_modify_targets(self):
        b = self.batch()
        b["target"] = torch.ones(2, 3)
        identity = perturb_inputs(b, AugmentationConfig())
        torch.testing.assert_close(identity["phase"], b["phase"], rtol=0, atol=0)
        dropped = perturb_inputs(
            b, AugmentationConfig(modality_dropout_probability=1.0)
        )
        self.assertEqual(float(dropped["tof"].abs().sum()), 0.0)
        self.assertFalse(dropped["scatter_valid"].any())
        torch.testing.assert_close(dropped["target"], b["target"])

    def test_optional_h5_scene_metadata_preserves_missing_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.h5"
            with h5py.File(path, "w") as file:
                file.attrs["scene_metadata_json"] = json.dumps(
                    {"ice": {"snow_depth_m": None}, "uav": {"altitude_agl_m": 2.5}}
                )
            scene = read_scene_h5(path)
        self.assertIsNone(scene.ice.snow_depth_m)
        self.assertEqual(scene.uav.altitude_agl_m, 2.5)

    def test_uncertainty_restores_batchnorm_and_module_modes(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.BatchNorm1d(4)
                self.drop = nn.Dropout(0.5)

            def forward(self, x):
                rho = self.drop(self.norm(x)).abs().reshape(2, 1, 4) + 0.5
                tau = torch.ones_like(rho) * 1e-9
                return {
                    "layer_count": torch.ones(2, 1, dtype=torch.long),
                    "layer_density": rho,
                    "layer_delays_s": tau,
                    "layer_thickness_m": 299792458 * tau / (2 * (1 + 0.845 * rho)),
                }

        model = Toy().train()
        before = model.norm.running_mean.clone()
        result = monte_carlo_dropout(model, torch.randn(2, 4), 8)
        torch.testing.assert_close(model.norm.running_mean, before)
        self.assertTrue(model.training and model.norm.training and model.drop.training)
        self.assertTrue((result["layer_density_marginal_std"] > 0).any())
        self.assertTrue(
            torch.isnan(result["layer_density_conditional_mean"][..., 1:]).all()
        )
        self.assertTrue((result["layer_presence_probability"][..., 1:] == 0).all())

    def test_physical_ensemble_uses_physical_recalculation(self):
        class Fixed(nn.Module):
            def __init__(self, density):
                super().__init__()
                self.density = density

            def forward(self, inputs):
                rho = torch.full((1, 2, 3), self.density)
                return {
                    "layer_density": rho,
                    "layer_delays_s": torch.ones_like(rho) * 1e-9,
                    "count_logits": torch.zeros_like(rho),
                }

        out = PhysicalEnsemble([Fixed(0.8), Fixed(1.0)], [0.25, 0.75])({})
        torch.testing.assert_close(out["layer_density"], torch.full((1, 2, 3), 0.95))
        torch.testing.assert_close(
            out["layer_thickness_m"],
            299792458
            * out["layer_delays_s"]
            / (2 * (1 + 0.845 * out["layer_density"])),
        )

    def test_invalid_architecture_and_training_settings_fail_early(self):
        for config in [
            ArchitectureConfig(tcn_kernel=2),
            ArchitectureConfig(dropout=float("nan")),
        ]:
            with self.assertRaises(ValueError):
                config.validate()
        with self.assertRaises(ValueError):
            RetrainingConfig(backbone_lr=-1).validate()
        with self.assertRaises(ValueError):
            build_network(self.config(), "typo")


if __name__ == "__main__":
    unittest.main()
