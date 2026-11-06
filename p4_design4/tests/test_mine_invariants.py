"""Synthetic invariants for physical consistency and metric definitions.
Run: python -B -m unittest discover -s p4_design4/tests -v
"""

import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from network import (
    ModelConfig,
    SPAIIceInversionNet,
    EntropyReliabilityRefinement,
    soft_layer_assignment,
)
from losses import TrainingConfig, stage_weights, MineLoss
from metrics import MineMetrics


class MineInvariants(unittest.TestCase):
    def test_assignment_equation(self):
        q = torch.tensor([1.0, 2.0, 3.5])
        actual = soft_layer_assignment(q, 10, 0.23)
        expected = torch.exp(-(q[:, None] - torch.arange(1, 11)).abs() / 0.23)
        expected = expected / expected.sum(-1, keepdim=True)
        torch.testing.assert_close(actual, expected)

    def test_constant_erf_preserves_features(self):
        module = EntropyReliabilityRefinement(4, 30, 2)
        x = torch.tensor([1.0, 2.0, 3.0, 4.0]).expand(2, 4, 64, 4).clone()
        result, info = module(x)
        torch.testing.assert_close(result, module.norm(x), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            info["reliability"],
            torch.ones_like(info["reliability"]),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_stage_table_and_zero_to_target_warmup(self):
        cfg = TrainingConfig()
        self.assertEqual(sum(cfg.stage_epochs), 80)
        stage, w = stage_weights(19, cfg)
        self.assertEqual(stage, 1)
        self.assertEqual(w["count"], 0.5)
        self.assertEqual(w["pre_rho"], 0)
        stage, w = stage_weights(20, cfg)
        self.assertEqual(stage, 2)
        self.assertEqual(w["pre_rho"], 0)
        self.assertEqual(stage_weights(24, cfg)[1]["pre_rho"], 0.5)
        self.assertEqual(stage_weights(40, cfg)[1]["h"], 0)
        self.assertEqual(
            stage_weights(44, cfg)[1],
            dict(boundary=1.0, count=0.5, pre_rho=0.5, rho=1.0, h=2.0, profile=1.0),
        )

    def test_analytic_thickness_and_delay_gradients(self):
        torch.manual_seed(23)
        cfg = ModelConfig(cnn_channels=8, gru_hidden=4, unet_base=4, delay_channels=8)
        model = SPAIIceInversionNet(cfg).train()
        x = torch.randn(1, 3, 64)
        out = model(
            x,
            x * 0.7,
            x * 0.5,
            time_s=torch.arange(64) * 2e-11,
            depth_m=torch.linspace(0, 0.2, 48),
        )
        expected = (
            cfg.c0_m_s
            * out["layer_delays_s"]
            / (2 * (1 + 0.845 * out["layer_density"]))
        )
        torch.testing.assert_close(out["layer_thickness_m"], expected)
        self.assertTrue((out["boundary_times_s"].diff(dim=-1) >= -1e-14).all())
        (out["layer_density"].sum() + out["layer_thickness_m"].sum()).backward()
        self.assertGreater(
            sum(
                float(p.grad.abs().sum())
                for p in model.tof_encoder.parameters()
                if p.grad is not None
            ),
            0,
        )
        self.assertGreater(
            sum(
                float(p.grad.abs().sum())
                for p in model.skeleton.parameters()
                if p.grad is not None
            ),
            0,
        )

    def test_physical_metadata_is_required(self):
        model = SPAIIceInversionNet(
            ModelConfig(cnn_channels=8, gru_hidden=4, unet_base=4, delay_channels=8)
        )
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 2, 64), torch.zeros(1, 2, 64), torch.zeros(1, 2, 64))

    def test_metric_identity_and_missing_layer(self):
        m = MineMetrics()
        m.update([0.1, 0.2], [0.81, 0.9], [0.1, 0.2], [0.81, 0.9])
        self.assertEqual(m.summary()["LER_percent"], 0)
        self.assertEqual(m.summary()["LE_h_percent"], 0)
        self.assertEqual(m.summary()["AE_rho_percent"], 0)
        missing = MineMetrics()
        missing.update([], [], [0.1, 0.2], [0.81, 0.9])
        self.assertEqual(missing.summary()["LER_percent"], 100)
        self.assertEqual(missing.summary()["AE_h_percent"], 100)
        self.assertIsNone(missing.summary()["LE_h_percent"])


if __name__ == "__main__":
    unittest.main()
