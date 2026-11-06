import unittest
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration import ResidualPhysicalCalibration, CalibrationConfig


class CalibrationInvariants(unittest.TestCase):
    def test_zero_residual_preserves_backbone_and_physics(self):
        head = ResidualPhysicalCalibration(8, torch.zeros(8), torch.ones(8))
        rho = torch.full((3, 10), 0.85)
        distance = torch.full((3, 10), 0.2)
        out = head(torch.randn(3, 8), rho, distance)
        torch.testing.assert_close(out["layer_density"], rho)
        torch.testing.assert_close(out["layer_delays_s"], 2 * distance / 299792458)
        expected = (
            299792458 * out["layer_delays_s"] / (2 * (1 + 0.845 * out["layer_density"]))
        )
        torch.testing.assert_close(out["layer_thickness_m"], expected)
        self.assertTrue((out["layer_thickness_m"] > 0).all())

    def test_added_corrections_remain_bounded(self):
        cfg = CalibrationConfig()
        head = ResidualPhysicalCalibration(8, torch.zeros(8), torch.ones(8), cfg)
        with torch.no_grad():
            head.density_head.bias.fill_(100)
            head.delay_head.bias.fill_(-100)
        out = head(
            torch.randn(2, 8), torch.full((2, 10), 0.85), torch.full((2, 10), 0.2)
        )
        self.assertLessEqual(
            float(out["density_correction"].abs().max()),
            cfg.density_correction_limit + 1e-6,
        )
        self.assertLessEqual(
            float(out["range_correction_m"].abs().max()),
            cfg.range_correction_limit_m + 1e-6,
        )
        self.assertTrue(torch.isfinite(out["layer_thickness_m"]).all())


if __name__ == "__main__":
    unittest.main()
