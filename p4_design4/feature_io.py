"""Explicit access to optional exported channels; main network inputs unchanged."""

from pathlib import Path
import h5py
import numpy as np


def read_delay_events(path):
    """Read observed D3 events, distinct from learned SPAI interface times."""
    with h5py.File(path, "r") as f:
        schema = f.attrs.get("icemap_event_schema", "")
        if isinstance(schema, bytes):
            schema = schema.decode("utf8")
        if schema != "icemap.events.v1":
            raise ValueError("No supported explicit D3 event schema.")
        events = np.asarray(f["tau_events_s"])
        bins = np.asarray(f["tau_event_bins"])
        mask = np.asarray(f["tau_event_valid"])
        count = np.asarray(f["tau_event_count"]).reshape(-1)
        intervals = np.asarray(f["tau_intervals_s"])
        T, N = f["phase"].shape
        if (
            events.ndim != 2
            or events.shape[0] != T
            or bins.shape != events.shape
            or mask.shape != events.shape
            or count.shape != (T,)
        ):
            raise ValueError("Malformed event shapes.")
        if not np.isin(mask, [0, 1]).all() or not np.equal(count, mask.sum(-1)).all():
            raise ValueError("Event count/mask disagreement.")
        if (
            not np.isfinite(events).all()
            or not np.isfinite(bins).all()
            or not np.isfinite(intervals).all()
        ):
            raise ValueError("Nonfinite events.")
        if intervals.shape != (T, max(1, events.shape[1] - 1)):
            raise ValueError("Malformed inter-event interval shape.")
        mask = mask.astype(bool)
        if (
            (events[~mask] != 0).any()
            or (bins[~mask] != 0).any()
            or (bins[mask] < 0).any()
            or (bins[mask] > N - 1).any()
        ):
            raise ValueError("Invalid event padding/bin coordinates.")
        for t, n in enumerate(count.astype(int)):
            if (
                not np.array_equal(mask[t], np.arange(events.shape[1]) < n)
                or (np.diff(events[t, :n]) <= 0).any()
                or not np.allclose(
                    np.diff(events[t, :n]),
                    intervals[t, : max(n - 1, 0)],
                    rtol=1e-6,
                    atol=1e-15,
                )
            ):
                raise ValueError("Events must be ordered with consistent intervals.")
            if (intervals[t, max(n - 1, 0) :] != 0).any():
                raise ValueError("Unused interval padding must be zero.")
            if "time_s" in f and not np.allclose(
                np.interp(
                    bins[t, :n], np.arange(N), np.asarray(f["time_s"]).reshape(-1)
                ),
                events[t, :n],
                rtol=1e-7,
                atol=1e-15,
            ):
                raise ValueError("Event time/bin calibration disagreement.")
    return {
        "time_s": events,
        "bins": bins,
        "valid": mask,
        "count": count.astype(int),
        "intervals_s": intervals,
    }


def read_extra_features(path, names=None):
    """Return [frames,bins,channels], masks, units and ordered channel names.

    Returned descriptors do not get silently appended to phase/E/tau. A model
    that consumes them must declare/train that additional input explicitly.
    """
    path = Path(path)
    with h5py.File(path, "r") as file:
        schema = file.attrs.get("icemap_extra_schema", "")
        if isinstance(schema, bytes):
            schema = schema.decode("utf-8")
        if not isinstance(schema, str) or schema != "icemap.extra_features.v1":
            raise ValueError("H5 has no supported extended feature schema.")
        if "phase" not in file or file["phase"].ndim != 2:
            raise ValueError("Expected phase [frames,bins] in feature record.")
        if "extra_features" not in file or "extra_feature_valid" not in file:
            raise ValueError("Extended feature groups are missing.")
        keys = sorted(file["extra_features"]) if names is None else list(names)
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("Choose a nonempty set of unique feature names.")
        values = []
        masks = []
        units = []
        for name in keys:
            if (
                not isinstance(name, str)
                or "/" in name
                or name not in file["extra_features"]
                or name not in file["extra_feature_valid"]
            ):
                raise ValueError(f"Unknown extended channel: {name!r}")
            dataset = file["extra_features"][name]
            mask_dataset = file["extra_feature_valid"][name]
            if dataset.dtype.kind not in {
                "i",
                "u",
                "f",
            } or mask_dataset.dtype.kind not in {"b", "i", "u", "f"}:
                raise ValueError(
                    f"Extended channel and mask must be real numeric: {name}"
                )
            value = np.asarray(dataset, dtype=np.float32)
            raw_mask = np.asarray(mask_dataset)
            if (
                value.shape != file["phase"].shape
                or raw_mask.shape != value.shape
                or not np.isfinite(value).all()
                or not np.isin(raw_mask, [0, 1]).all()
            ):
                raise ValueError(f"Malformed extended channel: {name}")
            mask = raw_mask.astype(bool)
            if (value[~mask] != 0).any():
                raise ValueError(f"Invalid samples must be zero: {name}")
            unit = dataset.attrs.get("unit")
            if isinstance(unit, bytes):
                unit = unit.decode("utf-8")
            if not isinstance(unit, str) or not unit:
                raise ValueError(f"Unit missing for {name}")
            values.append(value)
            masks.append(mask)
            units.append(unit)
    return {
        "features": np.stack(values, -1),
        "valid": np.stack(masks, -1),
        "names": keys,
        "units": units,
    }
