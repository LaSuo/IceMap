"""Missing physical truth must never turn a partial evaluation into a pass."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predict_opf import predict_file


class FixedPhysicalModel:
    def __call__(self, inputs):
        assert set(inputs) == {
            "phase",
            "tof",
            "scatter",
            "time_s",
            "depth_m",
            "scatter_valid",
        }
        B, T, _ = inputs["phase"].shape
        h = torch.full((B, T, 1), 0.002)
        rho = torch.full_like(h, 0.9)
        return {
            "layer_count": torch.ones((B, T), dtype=torch.long),
            "layer_thickness_m": h,
            "layer_density": rho,
            "layer_delays_s": 2 * h * (1 + 0.845 * rho) / 299792458,
        }


class SuiteEvaluation(unittest.TestCase):
    def test_empty_truth_keeps_prediction_but_prevents_full_pass(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            data = folder / "test.h5"
            with h5py.File(data, "w") as f:
                f.attrs["confirmed_legacy_grid_step_m"] = 0.001
                for key in ["phase", "tau", "E"]:
                    f[key] = np.ones((8, 2, 1), dtype=np.float32)
                truth = np.zeros((8, 2, 1), dtype=np.float32)
                truth[:2, 0, 0] = 0.9
                f["label3"] = truth
                f["file_names"] = np.array(list(b"record"), dtype=np.uint8)[:, None]
            model_path = folder / "model.pt"
            model_path.write_bytes(b"fixture")
            bundle = {
                "model_kind": "fixture",
                "evaluation_role": "unit test",
                "reference_alignment": None,
            }
            # The scorer is device-independent; avoid requiring CUDA in tests.
            with patch.object(torch.Tensor, "cuda", lambda self: self):
                result = predict_file(
                    FixedPhysicalModel(),
                    bundle,
                    model_path,
                    data,
                    folder / "output",
                    targets={"LER_percent": 0.0},
                )
            self.assertEqual(result["frames"], 2)
            self.assertEqual(result["scored_frames"], 1)
            self.assertEqual(
                result["evaluation_status"], "partial_ground_truth_pending_confirmation"
            )
            self.assertIsNone(result["within_one_percentage_point"])
            self.assertEqual(result["unscorable_empty_gt_frames"][0]["frame"], 1)
            self.assertFalse(result["labels_supplied_to_model"])
            self.assertEqual(
                len(json.loads((folder / "output/predictions.json").read_text())), 2
            )
            bundle["evaluation_policies"] = {
                "test": {"empty_label3": "accepted_ablation_failure"}
            }
            with patch.object(torch.Tensor, "cuda", lambda self: self):
                accepted = predict_file(
                    FixedPhysicalModel(),
                    bundle,
                    model_path,
                    data,
                    folder / "accepted",
                    targets={"LER_percent": 0.0},
                )
            self.assertEqual(
                accepted["evaluation_status"], "partial_gt_ablation_failure_accepted"
            )
            self.assertIsNone(accepted["within_one_percentage_point"])


if __name__ == "__main__":
    unittest.main()
