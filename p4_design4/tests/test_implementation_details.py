"""Implementation-value and provenance checks, not additional evaluation experiments."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from implementation_settings import (
    TrainingInitials,
    settings_manifest,
    audit_partition_metadata,
)
from describe_implementation import describe


class ImplementationDetails(unittest.TestCase):
    def test_engineering_optimizer_values_are_explicit_and_validated(self):
        cfg = TrainingInitials().validate()
        self.assertEqual(cfg.learning_rate, 1e-3)
        self.assertEqual(cfg.batch_size, 2)
        self.assertEqual(
            (cfg.adam_beta1, cfg.adam_beta2, cfg.adam_epsilon), (0.9, 0.999, 1e-8)
        )
        with self.assertRaises(ValueError):
            TrainingInitials(adam_beta1=1.0).validate()
        self.assertIn("provisional", settings_manifest()["training_initials_source"])

    def test_inspector_reports_the_existing_architecture_and_80_epoch_schedule(self):
        value = describe()
        self.assertEqual(value["trainable_parameters"], 682597)
        self.assertEqual(len(value["staged_loss_weights"]), 80)
        gru = next(m for m in value["modules"] if m["class"] == "GRU")
        self.assertTrue(gru["bidirectional"])
        self.assertEqual(gru["hidden_size"], 64)
        self.assertEqual(value["training_config"]["stage_epochs"], (20, 20, 40))

    def test_partition_audit_does_not_fabricate_structure_identity(self):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i in range(3):
                path = Path(d) / f"{i}.h5"
                with h5py.File(path, "w") as f:
                    f.attrs["site_id"] = str(i)
                paths.append(path)
            split = dict(zip(("train", "validation", "test"), [[p] for p in paths]))
            result = audit_partition_metadata(split)
            self.assertFalse(result["physical_structure_metadata_complete"])
            for path in paths:
                with h5py.File(path, "a") as f:
                    f.attrs["experiment_metadata_json"] = json.dumps(
                        {"structure_id": "same_ice"}
                    )
            with self.assertRaisesRegex(ValueError, "Physical structure"):
                audit_partition_metadata(split)


if __name__ == "__main__":
    unittest.main()
