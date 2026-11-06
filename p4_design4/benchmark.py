"""Measure actual feature-to-inversion latency; no raw-pipeline/Pi speed claim."""

import argparse
import io
import json
import platform
import time
import torch
from inference import load_model
from main import H5IceDataset, collate_records, move_batch
from network import ModelConfig, SPAIIceInversionNet
from protocol import implementation_contract, sha256_file
from mine_ablations import build_ablation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--sensing-hz", type=float, default=5.0)
    p.add_argument("--allow-debug-data", action="store_true")
    a = p.parse_args()
    if a.iterations < 1 or a.warmup < 0 or a.threads < 1 or a.sensing_hz <= 0:
        p.error("Invalid benchmark budget/rate.")
    from pathlib import Path

    if Path(a.output).exists():
        p.error("Refusing to overwrite an existing benchmark.")
    torch.set_num_threads(a.threads)
    model, bundle = load_model(a.model, a.device)
    if "model_config" not in bundle:
        p.error(
            "Benchmark the formal SPAI checkpoint; enhanced development bundles use a different pipeline."
        )
    cfg = ModelConfig(**bundle["model_config"])
    architecture = build_ablation(cfg, bundle.get("ablation_case", "full"))
    architecture_parameter_count = sum(p.numel() for p in architecture.parameters())
    del architecture
    batch = move_batch(
        collate_records(
            [
                H5IceDataset(
                    [a.data],
                    cfg.max_layers,
                    with_labels=False,
                    model_config=cfg,
                    allow_debug_data=a.allow_debug_data,
                )[0]
            ]
        ),
        a.device,
    )
    inputs = {
        k: batch[k]
        for k in ("phase", "scatter", "tof", "scatter_valid", "time_s", "depth_m")
    }
    if a.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    durations = []
    with torch.inference_mode():
        for index in range(a.warmup + a.iterations):
            if a.device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(inputs)
            if a.device.startswith("cuda"):
                torch.cuda.synchronize()
            duration = time.perf_counter() - start
            if index >= a.warmup:
                durations.append(duration)
    times = torch.tensor(durations, dtype=torch.float64) * 1000
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    contract = implementation_contract(cfg, bundle.get("ablation_case", "full"))
    clock = batch.get("acquisition_time_s")
    lookahead = None
    future = contract["temporal_future_frames"]
    if clock is not None and clock.shape[-1] > future and future > 0:
        waits = (clock[..., future:] - clock[..., :-future]) * 1000
        lookahead = {
            "min_ms": float(waits.min()),
            "max_ms": float(waits.max()),
            "mean_ms": float(waits.mean()),
        }
    report = {
        "scope": "prepared features -> SPAI; excludes loading, D1-D3 and asynchronous scheduling",
        "hardware": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "device": a.device,
        "threads": a.threads,
        "iterations": a.iterations,
        "warmup": a.warmup,
        "input_shape": list(batch["phase"].shape),
        "parameter_count": architecture_parameter_count,
        "unpacked_floating_parameter_bytes": sum(
            x.numel() * x.element_size() for x in model.parameters()
        ),
        "logical_float32_parameter_bytes": 4 * architecture_parameter_count,
        "quantization": bundle.get("quantization"),
        "ablation_case": bundle.get("ablation_case", "full"),
        "serialized_state_bytes": buffer.tell(),
        "model_file_bytes": Path(a.model).stat().st_size,
        "model_sha256": sha256_file(a.model),
        "mean_ms": float(times.mean()),
        "p50_ms": float(times.quantile(0.5)),
        "p95_ms": float(times.quantile(0.95)),
        "latency_unit": "entire supplied B,T,N window",
        "amortized_ms_per_frame": float(times.mean())
        / (batch["phase"].shape[0] * batch["phase"].shape[1]),
        "max_ms": float(times.max()),
        "minimum_erf_lookahead_ms": 1000
        * contract["temporal_future_frames"]
        / a.sensing_hz,
        "observed_acquisition_erf_lookahead": lookahead,
        "latency_note": "centered ERF needs future frames; D1/D3 lookahead and buffering must additionally be measured",
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated() if a.device.startswith("cuda") else None
        ),
        "full_pipeline_peak_rss_bytes": None,
        "full_pipeline_5hz_verified": False,
    }
    Path(a.output).write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
