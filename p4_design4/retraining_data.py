"""Rebuild supervised records directly from the supplied legacy HDF5 files."""

import hashlib
from pathlib import Path
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from legacy_data import physical_layers, recording_group
from legacy_grid import legacy_axes


def read_legacy(
    path, with_labels=True, max_bytes=None, grid_step_m=None, require_axes=False
):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {path}")
    if max_bytes is not None and (
        not isinstance(max_bytes, (int, float)) or max_bytes <= 0
    ):
        raise ValueError("max_bytes must be positive or None.")
    mapping = [("phase", "phase"), ("scatter", "E"), ("tof", "tau")] + (
        [(k, k) for k in ["label1", "label3"]] if with_labels else []
    )
    try:
        with h5py.File(path, "r") as file:
            missing = {name for _, name in mapping} | {"file_names"}
            missing -= set(file)
            if missing:
                raise ValueError(f"{path.name}: missing datasets {sorted(missing)}")
            shapes = {}
            for _, name in mapping:
                dataset = file[name]
                if dataset.ndim != 3 or dataset.dtype.kind not in {"i", "u", "f"}:
                    raise ValueError(
                        f"{path.name}/{name}: expected real numeric [N,T,S]."
                    )
                shapes[name] = dataset.shape
            if len(set(shapes.values())) != 1:
                raise ValueError(f"{path.name}: feature/label shape mismatch: {shapes}")
            N, T, S = shapes["phase"]
            if N < 8 or T < 2 or S < 1:
                raise ValueError(f"{path.name}: need N>=8,T>=2,S>=1, got {N,T,S}")
            estimated = sum(np.prod(file[name].shape) * 4 for _, name in mapping)
            if max_bytes is not None and estimated > max_bytes:
                raise MemoryError(
                    f"{path.name}: decoded arrays need {estimated/2**30:.2f} GiB, exceeding the explicit limit."
                )
            result = {
                key: file[name][...].transpose(2, 1, 0).astype(np.float32)
                for key, name in mapping
            }
            names = file["file_names"]
            if (
                names.ndim == 2
                and names.shape[1] == S
                and names.dtype.kind in {"u", "i"}
            ):
                if names.dtype.itemsize not in {1, 2}:
                    raise ValueError("Character codes must be uint8 or uint16.")
                encoding = (
                    "utf-8"
                    if names.dtype.itemsize == 1
                    else ("utf-16-be" if names.dtype.byteorder == ">" else "utf-16-le")
                )
                values = [
                    names[:, i].tobytes().decode(encoding).rstrip("\x00").strip()
                    for i in range(S)
                ]
            elif (
                names.ndim == 1
                and len(names) == S
                and names.dtype.kind in {"O", "S", "U"}
            ):
                values = [
                    x.decode("utf-8") if isinstance(x, bytes) else str(x)
                    for x in names[...]
                ]
            else:
                raise ValueError(
                    f"{path.name}: file_names must be [characters,S] codes or S text strings."
                )
            if any(not name for name in values):
                raise ValueError(
                    "Empty record names cannot define recording-disjoint splits."
                )
            result["names"] = np.array(values)
            axes = (
                legacy_axes(file, grid_step_m)
                if require_axes or grid_step_m is not None or "time_s" in file
                else None
            )
    except (OSError, UnicodeError) as error:
        raise ValueError(f"Cannot decode HDF5 {path}: {error}") from error
    if len({v.shape for k, v in result.items() if k != "names"}) != 1:
        raise ValueError("All feature/label arrays must share [record,frame,bin].")
    if len(result["names"]) != result["phase"].shape[0]:
        raise ValueError("File name count differs from record count.")
    if any(not np.isfinite(v).all() for k, v in result.items() if k != "names"):
        raise ValueError("Nonfinite source data.")
    if axes is not None:
        result["time_s"], result["depth_m"] = axes[:2]
    return result


def signature(data, index):
    return hashlib.sha256(
        b"".join(data[k][index].tobytes() for k in ["phase", "scatter", "tof"])
    ).hexdigest()


def supervised_arrays(source):
    S, T, N = source["phase"].shape
    out = {k: source[k] for k in ["phase", "scatter", "tof"]}
    # Programmatic fixture compatibility uses the explicitly historical grid;
    # CLI training always supplies or validates calibrated axes first.
    times = source.get("time_s", np.arange(N, dtype=np.float64) * (0.002 / 299792458))
    depths = source.get("depth_m", np.arange(N, dtype=np.float64) * 0.001)
    dt = float(np.diff(times).mean())
    dz = float(np.diff(depths).mean())
    out["time_s"] = np.broadcast_to(times, (S, N)).copy()
    out["depth_m"] = np.broadcast_to(depths, (S, N)).copy()
    out.update(
        boundary_target=np.zeros((S, T, N), np.float32),
        layer_count=np.zeros((S, T), np.int64),
        layer_density_gt=np.zeros((S, T, 10), np.float32),
        layer_thickness_m_gt=np.zeros((S, T, 10), np.float32),
        density_profile_gt=np.zeros((S, T, N), np.float32),
    )
    for b in range(S):
        for t in range(T):
            segments = physical_layers(source["label3"][b, t])
            if not 1 <= len(segments) <= 10:
                raise ValueError(
                    f"Training record {b}, frame {t}: invalid physical layer count."
                )
            marked = np.flatnonzero(source["label1"][b, t] > 0.5)
            if len(marked) < len(segments) + 1:
                raise ValueError("Missing source surface/bottom boundary labels.")
            h = np.array([(z - a) * dz for a, z, _ in segments])
            rho = np.array([r for _, _, r in segments])
            bins = np.rint(
                marked[0]
                + np.r_[0, np.cumsum(2 * h * (1 + 0.845 * rho) / (299792458 * dt))]
            ).astype(int)
            if bins[-1] >= N or len(np.unique(bins)) != len(segments) + 1:
                raise ValueError(
                    "Physical interfaces cannot be represented on the input grid."
                )
            L = len(segments)
            out["boundary_target"][b, t, bins] = 1
            out["layer_count"][b, t] = L
            out["layer_thickness_m_gt"][b, t, :L] = h
            out["layer_density_gt"][b, t, :L] = rho
            start = segments[0][0]
            out["density_profile_gt"][b, t, : N - start] = source["label3"][
                b, t, start:
            ]
    # This is an explicitly declared proxy: the reference specification does not numerically
    # specify how to obtain relative-density supervision from these files.
    out["relative_density_target"] = out["layer_density_gt"].copy()
    out["scatter_valid"] = out["scatter"] > 0
    return out


def make_split(
    source, data_directory, seed=431, validation_fraction=0.15, max_bytes=None
):
    test_hashes = set()
    for file in sorted(Path(data_directory).glob("test*.h5")):
        test = read_legacy(file, with_labels=False, max_bytes=max_bytes)
        test_hashes.update(signature(test, i) for i in range(len(test["names"])))
    eligible = [
        i
        for i in range(len(source["names"]))
        if signature(source, i) not in test_hashes
    ]
    groups = {}
    for i in eligible:
        count = len(physical_layers(source["label3"][i, 0]))
        groups.setdefault((count, recording_group(source["names"][i])), []).append(i)
    rng = random.Random(seed)
    validation_groups = set()
    for count in sorted({key[0] for key in groups}):
        choices = sorted(key for key in groups if key[0] == count)
        rng.shuffle(choices)
        if len(choices) > 1:
            n = max(1, min(len(choices) - 1, round(validation_fraction * len(choices))))
            validation_groups.update(choices[:n])
    validation = [
        i for key, items in groups.items() if key in validation_groups for i in items
    ]
    train = [
        i
        for key, items in groups.items()
        if key not in validation_groups
        for i in items
    ]
    train_groups = {recording_group(source["names"][i]) for i in train}
    val_groups = {recording_group(source["names"][i]) for i in validation}
    if not train or not validation or train_groups & val_groups:
        raise ValueError(
            "A nonempty, recording-disjoint training/validation split is required."
        )
    return {
        "train_indices": train,
        "validation_indices": validation,
        "eligible_indices": eligible,
        "exact_duplicates_excluded": len(source["names"]) - len(eligible),
        "test_labels_used": False,
        "independent_site_ids_available": False,
    }


class ArrayDataset(Dataset):
    def __init__(self, arrays, indices):
        self.arrays, self.indices = arrays, list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        out = {
            k: torch.from_numpy(np.array(v[i], copy=True))
            for k, v in self.arrays.items()
        }
        N = out["phase"].shape[-1]
        if "time_s" not in out:
            out["time_s"] = torch.arange(N, dtype=torch.float64) * (0.002 / 299792458)
        if "depth_m" not in out:
            out["depth_m"] = torch.arange(N, dtype=torch.float64) * 0.001
        return out


def masked_inputs(batch, fingerprints):
    out = dict(batch)
    for k in ["phase", "scatter", "tof"]:
        if k not in fingerprints:
            out[k] = torch.zeros_like(batch[k])
    if "scatter" not in fingerprints:
        out["scatter_valid"] = torch.zeros_like(batch["scatter_valid"])
    return out
