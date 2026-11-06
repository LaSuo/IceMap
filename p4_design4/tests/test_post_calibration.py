import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration import CalibrationConfig
from post_calibration import PostCalibrationConfig, apply_post_calibration
from quality_calibration import QualityExpertConfig
from quality_features import build_quality_features


class PostCalibrationTests(unittest.TestCase):
    def fixture(self):
        physical = CalibrationConfig()
        rho = torch.full((2, 3, 10), 0.87)
        distance = torch.full_like(rho, 0.2)
        tau = 2 * distance / physical.c0_m_s
        logits = torch.randn(2, 3, 10)
        return {
            "layer_density": rho,
            "layer_delays_s": tau,
            "layer_thickness_m": distance / (1 + 0.845 * rho),
            "count_logits": logits,
            "layer_count": logits.argmax(-1) + 1,
        }, physical

    def test_identity_and_weak_input_keep_range(self):
        pred, physical = self.fixture()
        identity = apply_post_calibration(pred, PostCalibrationConfig(), physical)
        for key in pred:
            torch.testing.assert_close(identity[key], pred[key])
        cfg = PostCalibrationConfig(
            range_scale=1.05,
            range_offset_m=0.002,
            range_quality_low=0.8,
            range_quality_high=1.1,
        )
        weak = apply_post_calibration(
            pred, cfg, physical, {"tof": torch.full((2, 3, 80), 0.4)}
        )
        torch.testing.assert_close(weak["layer_delays_s"], pred["layer_delays_s"])
        torch.testing.assert_close(weak["layer_thickness_m"], pred["layer_thickness_m"])

    def test_geometry_remains_analytic(self):
        pred, physical = self.fixture()
        cfg = PostCalibrationConfig(
            range_scale=0.98, range_offset_m=0.001, count_consensus_strength=0.1
        )
        out = apply_post_calibration(pred, cfg, physical)
        expected = (
            physical.c0_m_s
            * out["layer_delays_s"]
            / (2 * (1 + 0.845 * out["layer_density"]))
        )
        torch.testing.assert_close(out["layer_thickness_m"], expected)

    def test_active_layers_cannot_exceed_observed_time_window(self):
        pred, physical = self.fixture()
        cfg = PostCalibrationConfig(enforce_observed_range=True)
        inputs = {
            "time_s": torch.arange(100).float()[None].expand(2, -1)
            * (0.002 / physical.c0_m_s)
        }
        out = apply_post_calibration(pred, cfg, physical, inputs)
        mask = torch.arange(10) < out["layer_count"].unsqueeze(-1)
        total = (out["layer_delays_s"] * physical.c0_m_s / 2 * mask).sum(-1)
        self.assertTrue((total <= 0.099001).all())

    def test_peak_features_use_relative_spacing(self):
        n = torch.arange(256).float()
        pulse = torch.exp(-0.5 * ((n - 70) / 5) ** 2) + 0.8 * torch.exp(
            -0.5 * ((n - 150) / 5) ** 2
        )
        inputs = {
            k: pulse.expand(1, 3, -1).clone() for k in ["phase", "scatter", "tof"]
        }
        cfg = QualityExpertConfig(feature_mode="peak_spectral")
        a = build_quality_features(inputs, None, cfg)
        b = build_quality_features(
            {k: v.roll(10, -1) for k, v in inputs.items()}, None, cfg
        )
        self.assertEqual(tuple(a.shape), (1, 3, 504))
        self.assertTrue(torch.isfinite(a).all())
        peaks = a[..., -96:].reshape(1, 3, 12, 8)
        shifted = b[..., -96:].reshape(1, 3, 12, 8)
        torch.testing.assert_close(peaks[..., 1, 2], shifted[..., 1, 2])
        torch.testing.assert_close(
            peaks[..., 1, 2], torch.full((1, 3), 0.08), atol=0.001, rtol=0
        )


if __name__ == "__main__":
    unittest.main()
