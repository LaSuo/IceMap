import json
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from options_io import load_mapping
from retraining_data import read_legacy
from inspect_dataset import signal_quality
from train_opf import CalibratedSPAI
from network import ModelConfig
from inference import load_model


class InputValidation(unittest.TestCase):
    def test_json_and_malformed_options(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "options.json"
            path.write_text('{"seed": 431}')
            self.assertEqual(load_mapping(path)["seed"], 431)
            for text in ["[]", '{"seed": NaN}', "{broken", '{"seed":1,"seed":2}']:
                path.write_text(text)
                with self.assertRaises(ValueError):
                    load_mapping(path)

    def test_yaml_safe_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "options.yml"
            path.write_text("seed: 431\n")
            self.assertEqual(load_mapping(path), {"seed": 431})
            path.write_text("!!python/object:object {}")
            with self.assertRaises(ValueError):
                load_mapping(path)

    def test_hdf5_missing_fields_and_memory_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.h5"
            with h5py.File(path, "w") as file:
                file["phase"] = np.zeros((16, 2, 1))
            with self.assertRaisesRegex(ValueError, "missing datasets"):
                read_legacy(path)
            with h5py.File(path, "w") as file:
                for k in ["phase", "E", "tau", "label1", "label3"]:
                    file[k] = np.ones((16, 2, 1))
                file["file_names"] = np.array([b"record"])
            with self.assertRaises(MemoryError):
                read_legacy(path, max_bytes=1)
            self.assertEqual(read_legacy(path)["names"][0], "record")

    def test_network_nonfinite_shape_and_mask_errors(self):
        model = CalibratedSPAI(
            ModelConfig(cnn_channels=4, gru_hidden=2, unet_base=2, delay_channels=4)
        )
        b = {k: torch.zeros(1, 2, 16) for k in ["phase", "scatter", "tof"]}
        b.update(
            time_s=torch.arange(16)[None] * 1e-11,
            depth_m=torch.arange(16)[None] * 0.001,
            scatter_valid=torch.ones(1, 2, 16, dtype=torch.bool),
        )
        b["phase"][0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            model(b)
        b["phase"].zero_()
        b["scatter_valid"] = torch.full((1, 2, 16), 0.5)
        with self.assertRaisesRegex(ValueError, "binary"):
            model(b)

    def test_invalid_checkpoint_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.pt"
            torch.save({"backbone_state": {}}, path)
            with self.assertRaisesRegex(ValueError, "missing fields"):
                load_model(path, "cpu")

    def test_signal_diagnostics_handle_constant_and_periodic_inputs(self):
        self.assertEqual(signal_quality(np.ones((3, 128)))["status"], "constant")
        wave = np.sin(2 * np.pi * np.arange(128) / 16)[None].repeat(3, 0)
        self.assertAlmostEqual(
            signal_quality(wave)["spectral_peak_cycles_per_bin"], 1 / 16
        )


if __name__ == "__main__":
    unittest.main()
