import unittest
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration import ResidualPhysicalCalibration, CalibrationConfig
from reference_alignment import ReferenceAlignmentConfig, align_prediction


class ReferenceAlignmentInvariants(unittest.TestCase):
    def make_data(self):
        B, T, K = 2, 4, 10
        config = CalibrationConfig()
        head = ResidualPhysicalCalibration(5, torch.zeros(5), torch.ones(5), config)
        density = torch.full((B, T, K), 0.85)
        distance = torch.linspace(0.1, 0.4, B * T * K).reshape(B, T, K)
        with torch.no_grad():
            pred = head(
                torch.randn(B * T, 5), density.flatten(0, 1), distance.flatten(0, 1)
            )
        pred = {k: v.reshape(B, T, *v.shape[1:]) for k, v in pred.items()}
        base = {
            "layer_density": density,
            "layer_delays_s": 2 * distance / config.c0_m_s,
            "boundary_count": torch.full((B, T), 3.0),
        }
        return pred, base, config

    def test_identity_settings_preserve_original_outputs(self):
        pred, base, physical = self.make_data()
        result = align_prediction(pred, base, ReferenceAlignmentConfig(), physical)
        for key in [
            "count_logits",
            "layer_density",
            "layer_delays_s",
            "layer_thickness_m",
        ]:
            torch.testing.assert_close(result[key], pred[key])

    def test_temporal_consensus_never_mixes_records(self):
        pred, base, physical = self.make_data()
        cfg = ReferenceAlignmentConfig(regression_temporal_consensus="median")
        result = align_prediction(pred, base, cfg, physical)
        self.assertFalse(
            torch.allclose(
                result["layer_delays_s"][0],
                result["layer_delays_s"][1],
                rtol=1e-5,
                atol=1e-15,
            )
        )
        for t in range(1, 4):
            torch.testing.assert_close(
                result["layer_delays_s"][:, 0],
                result["layer_delays_s"][:, t],
                rtol=1e-6,
                atol=1e-15,
            )
        perm = torch.tensor([1, 0])
        rev = align_prediction(
            {k: v[perm] for k, v in pred.items()},
            {k: v[perm] for k, v in base.items()},
            cfg,
            physical,
        )
        torch.testing.assert_close(
            result["layer_thickness_m"][perm], rev["layer_thickness_m"]
        )

    def test_prior_fusion_uses_only_model_predictions(self):
        pred, base, physical = self.make_data()
        pred["count_logits"].zero_()
        pred["count_logits"][..., 0] = 2
        base["boundary_count"].fill_(4)
        cfg = ReferenceAlignmentConfig(
            count_prior_weight=1, density_shrinkage=0.25, range_shrinkage=0.93
        )
        out = align_prediction(pred, base, cfg, physical)
        self.assertTrue((out["layer_count"] == 3).all())
        expected = (
            physical.c0_m_s
            * out["layer_delays_s"]
            / (2 * (1 + 0.845 * out["layer_density"]))
        )
        torch.testing.assert_close(out["layer_thickness_m"], expected)
        base["unused_ground_truth"] = torch.randn(20)
        other = align_prediction(pred, base, cfg, physical)
        torch.testing.assert_close(out["layer_thickness_m"], other["layer_thickness_m"])

    def test_invalid_shrinkage_rejected(self):
        pred, base, physical = self.make_data()
        with self.assertRaises(ValueError):
            align_prediction(
                pred, base, ReferenceAlignmentConfig(density_shrinkage=1.1), physical
            )

    def test_disagreeing_prior_is_disabled_without_truth(self):
        pred, base, physical = self.make_data()
        pred["count_logits"].zero_()
        pred["count_logits"][..., 0] = 20
        base["boundary_count"].fill_(8)
        cfg = ReferenceAlignmentConfig(
            count_prior_weight=30, count_prior_disagreement_limit=0.75
        )
        result = align_prediction(pred, base, cfg, physical)
        torch.testing.assert_close(result["count_logits"], pred["count_logits"])
        self.assertTrue((result["layer_count"] == 1).all())

    def test_global_density_offset_keeps_analytic_thickness(self):
        pred, base, physical = self.make_data()
        result = align_prediction(
            pred, base, ReferenceAlignmentConfig(density_offset_g_cm3=-0.03), physical
        )
        torch.testing.assert_close(
            result["layer_density"], pred["layer_density"] - 0.03
        )
        expected = (
            physical.c0_m_s
            * result["layer_delays_s"]
            / (2 * (1 + 0.845 * result["layer_density"]))
        )
        torch.testing.assert_close(result["layer_thickness_m"], expected)


if __name__ == "__main__":
    unittest.main()
