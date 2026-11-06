import dataclasses
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration import CalibrationConfig, ResidualPhysicalCalibration
from inference import load_model, EnhancedIceInversion
from network import ModelConfig
from raw_shape_features import raw_shape_features
from train_opf import CalibratedSPAI


class VariantFeatures(unittest.TestCase):
    def test_omitted_modalities_cannot_change_inference(self):
        torch.manual_seed(19)
        cfg = ModelConfig(cnn_channels=8, gru_hidden=4, unet_base=4, delay_channels=8)
        backbone = CalibratedSPAI(cfg)
        dim = (8 + 4 + 8) * 16 + 32
        head = ResidualPhysicalCalibration(
            dim, torch.zeros(dim), torch.ones(dim), CalibrationConfig(width=8)
        )
        model = EnhancedIceInversion(
            backbone, head, input_fingerprints=["phase"]
        ).eval()
        batch = {k: torch.randn(1, 3, 64) for k in ["phase", "scatter", "tof"]}
        batch.update(
            time_s=torch.arange(64)[None] * (0.002 / 299792458),
            depth_m=torch.arange(64)[None] * 0.001,
            scatter_valid=torch.ones(1, 3, 64, dtype=torch.bool),
        )
        other = dict(batch)
        other["scatter"] = batch["scatter"] * 100 + 50
        other["tof"] = batch["tof"] * 200 - 80
        with torch.no_grad():
            first = model(batch)
            second = model(other)
        for key in [
            "count_logits",
            "layer_density",
            "layer_delays_s",
            "layer_thickness_m",
        ]:
            torch.testing.assert_close(first[key], second[key])

    def test_gain_normalization_preserves_shape_and_scale_statistic(self):
        torch.manual_seed(9)
        inputs = {k: torch.randn(2, 3, 80) for k in ["phase", "scatter", "tof"]}
        initial = raw_shape_features(inputs, bins=8)
        scaled = raw_shape_features({k: 3 * v for k, v in inputs.items()}, bins=8)
        self.assertEqual(tuple(initial.shape), (2, 3, 96))
        for channel in range(3):
            begin = channel * 32
            torch.testing.assert_close(
                initial[..., begin : begin + 24], scaled[..., begin : begin + 24]
            )
            torch.testing.assert_close(
                scaled[..., begin + 24], 3 * initial[..., begin + 24]
            )

    def test_configuration_loader_can_disable_erf(self):
        backbone = CalibratedSPAI(ModelConfig(use_erf=False))
        head = ResidualPhysicalCalibration(
            3104,
            torch.zeros(3104),
            torch.ones(3104),
            CalibrationConfig(width=8, dropout_probability=0.1),
        )
        bundle = {
            "backbone_config": dataclasses.asdict(backbone.core.config),
            "backbone_state": backbone.state_dict(),
            "calibration_config": dataclasses.asdict(head.config),
            "calibration_state": head.state_dict(),
            "input_dim": 3104,
            "pool_bins": 16,
            "reference_alignment": None,
        }
        with tempfile.TemporaryDirectory() as name:
            p = Path(name) / "model.pt"
            torch.save(bundle, p)
            loaded, _ = load_model(p, device="cpu")
        self.assertFalse(loaded.backbone.core.config.use_erf)
        self.assertIsNone(loaded.backbone.core.skeleton.erf)
        self.assertIsNone(loaded.backbone.core.density_erf)
        self.assertIsNone(loaded.backbone.core.delay_erf)
        self.assertEqual(loaded.calibration.config.dropout_probability, 0.1)


if __name__ == "__main__":
    unittest.main()
