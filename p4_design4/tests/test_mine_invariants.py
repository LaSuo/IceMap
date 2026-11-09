"""Synthetic invariants for physical consistency and metric definitions.
Run: python -B -m unittest discover -s p4_design4/tests -v
"""

import sys
from dataclasses import replace
from itertools import combinations
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
from event_association import decode_interface_roles, ordered_interface_bins


class MineInvariants(unittest.TestCase):
    def test_identity_blocks_and_disabled_calibration_preserve_initial_readout(self):
        config = ModelConfig(
            max_layers=3,
            cnn_channels=8,
            gru_hidden=4,
            unet_base=4,
            delay_channels=8,
            timing_condition_on_count=True,
        )
        original = SPAIIceInversionNet(config).eval()
        extended = SPAIIceInversionNet(
            replace(
                config,
                phase_residual_blocks=1,
                delay_residual_blocks=1,
                identity_extra_blocks=True,
                compact_calibration=True,
                calibration_strength=0,
            )
        ).eval()
        loaded = extended.load_state_dict(original.state_dict(), strict=False)
        self.assertFalse(loaded.unexpected_keys)
        values = torch.rand(1, 3, 64)
        inputs = dict(
            tof=values, phase=values, scatter=values, time_s=torch.arange(64) * 2e-11
        )
        with torch.no_grad():
            baseline, adapted = original(**inputs), extended(**inputs)
        for key in (
            "layer_count",
            "boundary_times_s",
            "layer_density",
            "layer_thickness_m",
        ):
            torch.testing.assert_close(adapted[key], baseline[key])

    def test_compact_calibration_trains_count_and_keeps_analytic_thickness(self):
        config = ModelConfig(
            max_layers=3,
            cnn_channels=8,
            gru_hidden=4,
            unet_base=4,
            delay_channels=8,
            phase_residual_blocks=1,
            delay_residual_blocks=1,
            compact_calibration=True,
            phase_contrast=True,
            input_stride=4,
            minimum_layers=1,
            timing_min_separation_bins=2,
        )
        model = SPAIIceInversionNet(config)
        x = torch.rand(1, 3, 64)
        inputs = dict(
            tof=x, phase=x * 0.7, scatter=x * 0.5, time_s=torch.arange(64) * 2e-11
        )
        out = model(**inputs)
        self.assertEqual(out["calibrated_count_logits"].shape, (1, 3, 4))
        target = torch.full((3,), 2, dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(
            out["calibrated_count_logits"][0], target
        )
        (loss + out["layer_thickness_m"].sum()).backward()
        self.assertGreater(
            model.interface_calibration.count[-1].weight.grad.abs().sum().item(), 0
        )
        self.assertGreater(
            model.interface_calibration.timing.weight.grad.abs().sum().item(), 0
        )
        model.eval()
        with torch.no_grad():
            out = model(**inputs)
        self.assertTrue((out["layer_count"] >= 1).all())
        self.assertTrue((out["boundary_times_s"].diff(dim=-1) >= 0).all())
        expected = (
            config.c0_m_s
            * out["layer_delays_s"]
            / (2 * (1 + config.dielectric_density_coefficient * out["layer_density"]))
        )
        torch.testing.assert_close(out["layer_thickness_m"], expected)

    def test_ordered_timing_cannot_reuse_a_strong_interface(self):
        generator = torch.Generator().manual_seed(77)
        scores = torch.randn(2, 1, 9, 3, generator=generator)
        scores[:, :, 4, :] += 8
        counts = torch.tensor([[1], [2]])
        chosen = ordered_interface_bins(scores, counts, min_gap=2)
        for b, count in enumerate(counts.flatten().tolist()):

            def value(path):
                return sum(
                    float(scores[b, 0, index, rank]) for rank, index in enumerate(path)
                )

            paths = (
                p
                for p in combinations(range(9), count + 1)
                if all(right - left >= 2 for left, right in zip(p, p[1:]))
            )
            self.assertEqual(
                tuple(chosen[b, 0, : count + 1].tolist()), max(paths, key=value)
            )
            self.assertTrue((chosen[b, 0, count:] == chosen[b, 0, count]).all())
        allowed = torch.zeros(2, 1, 9, dtype=torch.bool)
        allowed[0, 0, [0, 8]] = True
        allowed[1, 0, 4] = True
        masked = ordered_interface_bins(scores, counts, min_gap=2, allowed=allowed)
        self.assertEqual(masked[0, 0].tolist(), [0, 8, 8])
        torch.testing.assert_close(masked[1], chosen[1])

    def test_joint_role_decoder_matches_exhaustive_layer_and_timing_search(self):
        rng = torch.Generator().manual_seed(21)
        logits = torch.randn(2, 3, 9, 3, generator=rng)
        prior = torch.randn(2, 3, 6, generator=rng)
        for minimum in (0, 1, 2):
            count, chosen = decode_interface_roles(logits, prior, 3, minimum, 0.6)
            for b in range(2):
                for t in range(3):

                    def path_score(path):
                        layers = len(path) - 1
                        score = prior[b, t, layers] + logits[b, t, path[0], 0]
                        if layers:
                            score += logits[b, t, path[-1], 2]
                            score += sum(logits[b, t, i, 1] for i in path[1:-1])
                            score -= 0.6 * (layers - 1)
                        return float(score)

                    paths = (
                        path
                        for layers in range(minimum, 3)
                        for path in combinations(range(9), layers + 1)
                        if all(y - x >= 3 for x, y in zip(path, path[1:]))
                    )
                    expected = max(paths, key=path_score)
                    layers = len(expected) - 1
                    self.assertEqual(count[b, t].item(), layers)
                    self.assertEqual(tuple(chosen[b, t, : layers + 1]), expected)
                    self.assertTrue((chosen[b, t, layers:] == expected[-1]).all())

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
