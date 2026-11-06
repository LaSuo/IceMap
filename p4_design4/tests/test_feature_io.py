from pathlib import Path
import sys
import tempfile
import unittest
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_io import read_extra_features


class ExtraFeatureIO(unittest.TestCase):
    def test_selection_order_units_and_missing_support(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.h5"
            with h5py.File(path, "w") as file:
                file["phase"] = np.zeros((3, 20))
                file.attrs["icemap_extra_schema"] = "icemap.extra_features.v1"
                for name in ["entropy", "centroid"]:
                    data = np.ones((3, 20))
                    data[:, 0] = 0
                    file[f"extra_features/{name}"] = data
                    file[f"extra_features/{name}"].attrs["unit"] = (
                        "1" if name == "entropy" else "cycles/bin"
                    )
                    mask = np.ones((3, 20))
                    mask[:, 0] = 0
                    file[f"extra_feature_valid/{name}"] = mask
            out = read_extra_features(path, ["centroid", "entropy"])
            self.assertEqual(out["features"].shape, (3, 20, 2))
            self.assertEqual(out["units"], ["cycles/bin", "1"])
            self.assertFalse(out["valid"][:, 0].any())
            with self.assertRaises(ValueError):
                read_extra_features(path, ["missing"])
            with self.assertRaises(ValueError):
                read_extra_features(path, ["entropy", "entropy"])

    def test_corrupt_values_and_masks_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.h5"
            with h5py.File(path, "w") as file:
                file["phase"] = np.zeros((3, 20))
                file.attrs["icemap_extra_schema"] = "icemap.extra_features.v1"
                file["extra_features/entropy"] = np.ones((3, 20))
                file["extra_features/entropy"].attrs["unit"] = "1"
                file["extra_feature_valid/entropy"] = np.zeros((3, 20))
            with self.assertRaisesRegex(ValueError, "must be zero"):
                read_extra_features(path)

    def test_complex_channels_cannot_be_silently_cast_to_real(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.h5"
            with h5py.File(path, "w") as file:
                file["phase"] = np.zeros((3, 20))
                file.attrs["icemap_extra_schema"] = "icemap.extra_features.v1"
                file["extra_features/entropy"] = (
                    np.ones((3, 20), dtype=np.complex64) * 1j
                )
                file["extra_features/entropy"].attrs["unit"] = "1"
                file["extra_feature_valid/entropy"] = np.ones((3, 20))
            with self.assertRaisesRegex(ValueError, "real numeric"):
                read_extra_features(path)


if __name__ == "__main__":
    unittest.main()
