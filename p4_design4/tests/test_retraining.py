"""Regression checks for rebuilding a training run after all caches are deleted."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retraining_data import ArrayDataset, make_split, supervised_arrays
from train import quality_augmentation
from training_config import RetrainingConfig, historical_post_config, weak_expert_config


class RetrainingTests(unittest.TestCase):
    def source(self):
        S, T, N = 8, 2, 256
        source = {
            k: np.zeros((S, T, N), dtype=np.float32)
            for k in ["phase", "scatter", "tof", "label1", "label3"]
        }
        source["names"] = np.array([f"scene{i//2}_sub{i%2+1}" for i in range(S)])
        for i in range(S):
            for k in ["phase", "scatter", "tof"]:
                source[k][i] = i + 0.1
            source["label3"][i, :, 10:60] = 0.9
            source["label1"][i, :, [20, 55, 108]] = 1
        return source

    def test_physical_label_runs_override_extra_source_boundaries(self):
        arrays = supervised_arrays(self.source())
        self.assertTrue(np.all(arrays["layer_count"] == 1))
        np.testing.assert_allclose(arrays["layer_thickness_m_gt"][..., 0], 0.05)
        self.assertTrue(np.all(arrays["boundary_target"].sum(-1) == 2))
        record = ArrayDataset(arrays, [0])[0]
        np.testing.assert_allclose(
            record["time_s"].diff().numpy(), 0.002 / 299792458, rtol=2e-5
        )

    def test_rebuilt_split_removes_duplicate_without_reading_test_labels(self):
        source = self.source()
        with tempfile.TemporaryDirectory() as name:
            with h5py.File(Path(name) / "test_unlabelled.h5", "w") as file:
                for k, field in [("phase", "phase"), ("scatter", "E"), ("tof", "tau")]:
                    file[field] = source[k][:1].transpose(2, 1, 0)
                file["file_names"] = np.array(list(b"test"), dtype=np.uint8)[:, None]
            split = make_split(source, name)
        self.assertNotIn(0, split["eligible_indices"])
        self.assertEqual(split["exact_duplicates_excluded"], 1)
        ti, vi = split["train_indices"], split["validation_indices"]
        self.assertFalse(set(ti) & set(vi))
        self.assertFalse({i // 2 for i in ti} & {i // 2 for i in vi})

    def test_quality_augmentation_keeps_physical_truth_and_source_unchanged(self):
        arrays = supervised_arrays(self.source())
        record = ArrayDataset(arrays, [0])[0]
        batch = {k: v[None] for k, v in record.items()}
        before = copy.deepcopy(batch)
        for style in range(1, 9):
            out = quality_augmentation(batch, style)
            for k in ["layer_thickness_m_gt", "layer_density_gt", "layer_count"]:
                torch.testing.assert_close(out[k], before[k])
            torch.testing.assert_close(batch["tof"], before["tof"])

    def test_historical_recipe_preserves_recoverable_settings(self):
        recipe = RetrainingConfig()
        self.assertEqual(recipe.stage_epochs, (20, 20, 40))
        self.assertEqual(recipe.expert_epochs, 300)
        self.assertEqual(weak_expert_config().weak_min_range_m, 0.2)
        self.assertTrue(historical_post_config().enforce_observed_range)


if __name__ == "__main__":
    unittest.main()
