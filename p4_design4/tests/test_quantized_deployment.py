"""Round-trip checks for compact weights, packets and asynchronous deployment."""

import asyncio
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import h5py
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from async_inference import AsyncSPAI, DEFAULT_OUTPUTS
from export_quantized_features import export_archive, iter_feature_archive
from inference import load_model
from network import ModelConfig, SPAIIceInversionNet
from quantized_features import FINGERPRINTS, pack_feature_window
from quantized_weights import Int8Conv1d, compact_weight_bundle


def inputs_for_test():
    rng = torch.Generator().manual_seed(7)
    phase, scatter, tof = (torch.rand(1, 3, 64, generator=rng) for _ in range(3))
    tof[..., 13] = 0.0001
    tof[..., 14] = 1.0
    return {
        "phase": phase * 2 - 1,
        "scatter": scatter,
        "tof": tof,
        "scatter_valid": scatter > 0.2,
        "time_s": torch.arange(64, dtype=torch.float64) * 2e-11,
        "depth_m": torch.arange(64, dtype=torch.float64) * 0.001,
    }


class QuantizedDeployment(unittest.TestCase):
    def test_companding_preserves_weak_echoes_and_exact_axes(self):
        inputs = inputs_for_test()
        linear = pack_feature_window(inputs, "linear8")
        compact = pack_feature_window(inputs, "companded8")
        decoded = compact.unpack()
        self.assertLess(
            compact.payload_bytes, pack_feature_window(inputs, "fp32").payload_bytes
        )
        self.assertEqual(compact.fields["phase"].values.dtype, torch.int8)
        self.assertEqual(compact.fields["tof"].values.dtype, torch.uint8)
        self.assertTrue((linear.unpack()["tof"][..., 13] == 0).all())
        self.assertTrue((decoded["tof"][..., 13] > 0).all())
        for name in ("time_s", "depth_m", "scatter_valid"):
            torch.testing.assert_close(decoded[name], inputs[name], rtol=0, atol=0)

    def test_archive_preserves_packed_values_and_record_order(self):
        inputs = inputs_for_test()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with h5py.File(root / "source.h5", "w") as original:
                for source, name in zip(("phase", "E", "tau"), FINGERPRINTS):
                    samples = torch.cat((inputs[name], inputs[name] * 0.2), dim=0)
                    original[source] = samples.permute(2, 1, 0).numpy()
                original["time_s"] = inputs["time_s"].numpy()
                original["depth_m"] = inputs["depth_m"].numpy()
            report = export_archive(
                root / "source.h5", root / "compact.h5", "companded8", None
            )
            self.assertEqual(report["observations"], 2)
            packets = list(iter_feature_archive(root / "compact.h5"))
            for index, (actual, packet) in enumerate(packets):
                self.assertEqual(actual, index)
                expected = {
                    key: inputs[key] * (0.2 if index else 1.0) for key in FINGERPRINTS
                }
                expected.update(
                    scatter_valid=expected["scatter"] > 0,
                    time_s=inputs["time_s"],
                    depth_m=inputs["depth_m"],
                )
                packed = pack_feature_window(expected, "companded8")
                for key, tensor in packed.unpack().items():
                    torch.testing.assert_close(
                        packet.unpack()[key], tensor, rtol=0, atol=0
                    )

    def test_packed_model_reloads_and_async_results_match_sync(self):
        cfg = ModelConfig(
            max_layers=3,
            cnn_channels=8,
            gru_hidden=4,
            unet_base=4,
            delay_channels=8,
            compact_calibration=True,
            minimum_layers=1,
        )
        core = SPAIIceInversionNet(cfg).eval()
        metadata = {"model_config": asdict(cfg), "training_config": {}}
        bundle = compact_weight_bundle(core, metadata, "all")
        bundle["feature_profile"] = "companded8"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            torch.save(bundle, path)
            loaded, _ = load_model(path, device="cpu")
            self.assertTrue(
                any(isinstance(module, Int8Conv1d) for module in loaded.modules())
            )
            packet = pack_feature_window(inputs_for_test())
            with torch.inference_mode():
                expected = loaded(packet.unpack())

            async def run():
                endpoint = AsyncSPAI.from_checkpoint(path, max_in_flight=1)
                async with endpoint:
                    first = await endpoint.submit(packet, observation_id="first")
                    second = await endpoint.submit(packet, observation_id="second")
                    return await asyncio.gather(first, second)

            results = asyncio.run(run())
        self.assertEqual(
            [result.observation_id for result in results], ["first", "second"]
        )
        for result in results:
            for key in DEFAULT_OUTPUTS:
                torch.testing.assert_close(result.outputs[key], expected[key])
