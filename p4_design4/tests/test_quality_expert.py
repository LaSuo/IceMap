import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quality_calibration import (
    QualityExpertConfig,
    QualityPhysicalExpert,
    quality_gates,
    combine_quality_expert,
)


class QualityExpertTests(unittest.TestCase):
    def test_sparse_strong_input_preserves_frozen_prediction(self):
        cfg = QualityExpertConfig(weak_weight=1.0)
        tau = torch.zeros(2, 3, 100)
        tau[..., :30] = 1.2
        inputs = {"tof": tau}
        self.assertTrue((quality_gates(inputs, cfg)["combined"] == 0).all())
        rho = torch.full((2, 3, 10), 0.87)
        delay = torch.full_like(rho, 1e-9)
        logits = torch.randn(2, 3, 10)
        baseline = {
            "count_logits": logits,
            "layer_count": logits.argmax(-1) + 1,
            "layer_density": rho,
            "layer_delays_s": delay,
            "layer_thickness_m": cfg.c0_m_s * delay / (2 * (1 + 0.845 * rho)),
        }
        expert = {
            "count_logits": logits * 5,
            "layer_density": rho + 0.05,
            "layer_delays_s": delay * 2,
        }
        result = combine_quality_expert(baseline, expert, inputs, cfg)
        for key in [
            "count_logits",
            "layer_count",
            "layer_density",
            "layer_delays_s",
            "layer_thickness_m",
        ]:
            torch.testing.assert_close(result[key], baseline[key])

    def test_expert_active_profile_fits_observation_window(self):
        cfg = QualityExpertConfig(width=8)
        expert = QualityPhysicalExpert(5, torch.zeros(5), torch.ones(5), cfg).eval()
        with torch.no_grad():
            expert.count_head.weight.zero_()
            expert.count_head.bias.zero_()
            expert.count_head.bias[-1] = 10
            result = expert(torch.randn(4, 5))
        distance = result["layer_delays_s"] * cfg.c0_m_s / 2
        self.assertTrue((distance.sum(-1) <= cfg.max_range_m + 1e-6).all())
        expected = (
            cfg.c0_m_s
            * result["layer_delays_s"]
            / (2 * (1 + 0.845 * result["layer_density"]))
        )
        torch.testing.assert_close(result["layer_thickness_m"], expected)

    def test_weak_refinement_can_protect_short_single_layers(self):
        cfg = QualityExpertConfig(
            dense_weight=0,
            weak_weight=1,
            weak_count_weight=0,
            weak_single_layer_only=True,
            weak_min_range_m=0.2,
        )
        rho = torch.full((1, 2, 10), 0.87)
        distance = torch.full_like(rho, 0.15)
        distance[:, 1] = 0.4
        logits = torch.zeros(1, 2, 10)
        logits[..., 0] = 10
        baseline = {
            "layer_density": rho,
            "layer_delays_s": 2 * distance / cfg.c0_m_s,
            "count_logits": logits,
            "layer_count": torch.ones(1, 2, dtype=torch.long),
        }
        expert = {
            "layer_density": rho + 0.05,
            "layer_delays_s": baseline["layer_delays_s"] * 1.2,
            "count_logits": logits,
        }
        signal = torch.zeros(1, 2, 100)
        signal[..., :20] = 0.1
        out = combine_quality_expert(baseline, expert, {"tof": signal}, cfg)
        torch.testing.assert_close(
            out["layer_delays_s"][:, 0], baseline["layer_delays_s"][:, 0]
        )
        self.assertTrue((out["quality_expert_gate"][:, 0] == 0).all())
        self.assertTrue((out["quality_expert_gate"][:, 1] > 0).all())


if __name__ == "__main__":
    unittest.main()
