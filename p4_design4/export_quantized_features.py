"""Stream a legacy fingerprint H5 into an observation-indexed transport archive."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from legacy_grid import legacy_axes
from quantized_features import (
    FEATURE_PROFILES,
    FINGERPRINTS,
    FeaturePacket,
    QuantizedField,
    pack_feature_window,
)


ARCHIVE_FORMAT = "icemap.feature_packet.v1"
LEGACY_NAMES = {"phase": "phase", "scatter": "E", "tof": "tau"}


def export_archive(source, destination, profile, step_m):
    total = 0
    with h5py.File(source, "r") as original:
        raw = {key: original.get(name) for key, name in LEGACY_NAMES.items()}
        if any(
            not isinstance(item, h5py.Dataset)
            or item.ndim != 3
            or item.dtype.kind not in "iuf"
            for item in raw.values()
        ):
            raise ValueError("Expected numeric legacy phase, E and tau arrays [N,T,S].")
        bins, frames, records = raw["phase"].shape
        if (
            min(frames, records) < 1
            or bins < 8
            or any(item.shape != (bins, frames, records) for item in raw.values())
        ):
            raise ValueError("Legacy fingerprints must share a nonempty shape.")
        time, depth, provenance = legacy_axes(original, step_m)
        with h5py.File(destination, "x") as archive:
            archive.attrs.update(
                format=ARCHIVE_FORMAT,
                profile=profile,
                complete=False,
                axis_provenance=provenance,
            )
            archive["time_s"] = time
            archive["depth_m"] = depth
            shape, chunks = (records, frames, bins), (1, frames, bins)
            valid = archive.create_dataset(
                "scatter_valid", shape, dtype=np.bool_, chunks=chunks, compression="lzf"
            )
            for index in range(records):
                inputs = {
                    key: torch.from_numpy(
                        np.asarray(item[:, :, index].T, dtype=np.float32)[None]
                    )
                    for key, item in raw.items()
                }
                inputs.update(
                    scatter_valid=inputs["scatter"] > 0,
                    time_s=torch.from_numpy(time),
                    depth_m=torch.from_numpy(depth),
                )
                packet = pack_feature_window(inputs, profile)
                valid[index] = packet.context["scatter_valid"][0].numpy()
                for name, field in packet.fields.items():
                    values = field.values[0].numpy()
                    if index == 0:
                        group = archive.create_group(name)
                        group.attrs["square"] = field.square
                        group.create_dataset(
                            "values",
                            shape,
                            dtype=values.dtype,
                            chunks=chunks,
                            compression="lzf",
                        )
                        if field.scale is not None:
                            group.create_dataset(
                                "scale", (records, frames, 1), dtype=np.float32
                            )
                    group = archive[name]
                    group["values"][index] = values
                    if field.scale is not None:
                        group["scale"][index] = field.scale[0].numpy()
                total += packet.payload_bytes
            archive.attrs["complete"] = True
    return {
        "profile": profile,
        "observations": records,
        "frames": records * frames,
        "serialized_archive_bytes": Path(destination).stat().st_size,
        "total_observation_payload_bytes_with_repeated_axes": total,
    }


def iter_feature_archive(path):
    """Yield (observation_index, FeaturePacket) without expanding stored INT8 tensors."""
    with h5py.File(path, "r") as archive:
        profile = archive.attrs.get("profile")
        if (
            archive.attrs.get("format") != ARCHIVE_FORMAT
            or profile not in FEATURE_PROFILES
            or not archive.attrs.get("complete", False)
        ):
            raise ValueError("Unsupported or incomplete feature packet archive.")
        axes = {
            name: torch.from_numpy(archive[name][...]) for name in ("time_s", "depth_m")
        }
        for index in range(archive["scatter_valid"].shape[0]):
            valid = torch.from_numpy(archive["scatter_valid"][index : index + 1])
            fields = {}
            for name in FINGERPRINTS:
                group = archive[name]
                scale = (
                    torch.from_numpy(group["scale"][index : index + 1])
                    if "scale" in group
                    else None
                )
                fields[name] = QuantizedField(
                    torch.from_numpy(group["values"][index : index + 1]),
                    scale,
                    bool(group.attrs["square"]),
                )
            yield index, FeaturePacket(fields, dict(axes, scatter_valid=valid), profile)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=FEATURE_PROFILES, default="companded8")
    parser.add_argument("--legacy-grid-mm", type=float)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    step = None if args.legacy_grid_mm is None else args.legacy_grid_mm / 1000
    report = export_archive(args.input, args.output, args.profile, step)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
