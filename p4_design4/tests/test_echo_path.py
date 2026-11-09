"""Echo path regressions for a false phase lobe and a mismatched grid."""

from dataclasses import asdict, replace
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echo_path import EchoPathConfig
from network import ModelConfig, SPAIIceInversionNet


class EchoPathTest(unittest.TestCase):
    def test_shared_phase_and_tof_support_and_grid_fallback(self):
        cfg = EchoPathConfig(
            native_step_s=7e-12,
            surface_bin=80,
            surface_scale_bins=12,
            minimum_gap_bins=40,
            maximum_single_span_bins=240,
            maximum_double_span_bins=330,
            inner_fraction=0.5,
            minimum_tof_support=0.01,
            minimum_relative_phase=0.3,
        ).validate()
        model = SPAIIceInversionNet(
            ModelConfig(
                cnn_channels=8,
                gru_hidden=4,
                unet_base=4,
                delay_channels=8,
                max_layers=3,
                input_stride=4,
                echo_path=asdict(cfg),
            )
        ).eval()
        position = torch.arange(400, dtype=torch.float32)

        def echo(center, width):
            return torch.exp(-0.5 * ((position - center) / width).square())

        phase = 0.4 + 0.34 * (echo(80, 5) + echo(180, 5) + echo(280, 5))
        tof = echo(80, 9) + 0.6 * echo(280, 9)
        double = tof + 0.5 * echo(180, 9)
        inputs = {
            "phase": phase.expand(3, 3, -1),
            "scatter": torch.ones(3, 3, 400),
            "tof": torch.stack((tof, double, double))[:, None, :].expand(-1, 3, -1),
            "time_s": torch.stack((position, position, 2 * position))
            * cfg.native_step_s,
        }
        with torch.inference_mode():
            predicted = model(**inputs)
            model.config = replace(model.config, echo_path=None)
            baseline = model(**inputs)
        self.assertTrue(predicted["echo_path_used"][:2].all())
        self.assertFalse(predicted["echo_path_used"][2].any())
        self.assertTrue((predicted["layer_count"][0] == 1).all())
        self.assertTrue((predicted["layer_count"][1] == 2).all())
        self.assertTrue((predicted["boundary_times_s"].diff(dim=-1)[:2] >= 0).all())
        for key in (
            "layer_count",
            "boundary_times_s",
            "layer_density",
            "layer_thickness_m",
        ):
            torch.testing.assert_close(predicted[key][2], baseline[key][2])
        net = model.config
        expected = (
            net.c0_m_s
            * predicted["layer_delays_s"]
            / (
                2
                * (1 + net.dielectric_density_coefficient * predicted["layer_density"])
            )
        )
        torch.testing.assert_close(predicted["layer_thickness_m"], expected)
