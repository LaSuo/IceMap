"""Bounded observation-level interface between fingerprint extraction and SPAI."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch

from inference import load_model
from quantized_features import FEATURE_PROFILES, FeaturePacket, pack_feature_window


DEFAULT_OUTPUTS = (
    "layer_count",
    "layer_thickness_m",
    "layer_density",
    "boundary_times_s",
    "layer_delays_s",
)


@dataclass(frozen=True)
class ObservationResult:
    observation_id: object
    outputs: dict


class AsyncSPAI:
    """Run one inference worker with bounded pending and active observations.

    Submit a complete Design3 window and await the returned future whenever its
    result is needed. The next window can be prepared while SPAI is running.
    """

    def __init__(
        self,
        model,
        *,
        device="cpu",
        max_in_flight=2,
        feature_profile="companded8",
        output_keys=DEFAULT_OUTPUTS,
    ):
        if type(max_in_flight) is not int or max_in_flight < 1:
            raise ValueError("max_in_flight must be a positive integer.")
        if feature_profile not in FEATURE_PROFILES:
            raise ValueError("Unsupported feature profile.")
        self.model = model.eval()
        self.device = torch.device(device)
        self.max_in_flight = max_in_flight
        self.feature_profile = feature_profile
        self.output_keys = tuple(output_keys)
        self._jobs = set()
        self._loop = None
        self._executor = None
        self._closed = False

    @classmethod
    def from_checkpoint(cls, checkpoint, *, device="cpu", **kwargs):
        model, bundle = load_model(checkpoint, device=device)
        kwargs.setdefault(
            "feature_profile", bundle.get("feature_profile", "companded8")
        )
        return cls(model, device=device, **kwargs)

    async def __aenter__(self):
        if self._loop is not None:
            raise RuntimeError("An AsyncSPAI endpoint is single-use.")
        self._loop = asyncio.get_running_loop()
        self._slots = asyncio.BoundedSemaphore(self.max_in_flight)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="spai")
        return self

    def _predict(self, packet, observation_id):
        with torch.inference_mode():
            output = self.model(packet.unpack(self.device))
            return ObservationResult(
                observation_id, {key: output[key].cpu() for key in self.output_keys}
            )

    async def submit(self, inputs, *, observation_id=None):
        """Wait for capacity, enqueue inference, and return a result future."""
        if asyncio.get_running_loop() is not self._loop or self._closed:
            raise RuntimeError("Use the endpoint inside its async context.")
        await self._slots.acquire()
        try:
            if self._closed:
                raise RuntimeError("The endpoint has closed.")
            packet = (
                inputs
                if isinstance(inputs, FeaturePacket)
                else pack_feature_window(inputs, self.feature_profile)
            )
            job = self._loop.run_in_executor(
                self._executor, self._predict, packet, observation_id
            )
        except BaseException:
            self._slots.release()
            raise
        self._jobs.add(job)
        job.add_done_callback(self._finished)
        return asyncio.shield(job)

    def _finished(self, job):
        self._jobs.discard(job)
        self._slots.release()
        if not job.cancelled():
            job.exception()

    async def predict(self, inputs, *, observation_id=None):
        """Submit one window and await its associated output."""
        job = await self.submit(inputs, observation_id=observation_id)
        return await job

    async def __aexit__(self, exc_type, exc, traceback):
        self._closed = True
        try:
            if self._jobs:
                await asyncio.shield(
                    asyncio.gather(*self._jobs, return_exceptions=True)
                )
        finally:
            self._executor.shutdown(wait=False)
