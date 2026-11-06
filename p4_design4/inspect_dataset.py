"""Label-free fingerprint diagnostics; spectral units are cycles per bin."""

import argparse
import json

import h5py
import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis


def signal_quality(array):
    array = np.asarray(array)
    if (
        array.ndim != 2
        or min(array.shape) < 1
        or array.dtype.kind not in {"i", "u", "f"}
    ):
        raise ValueError("Diagnostic input requires real numeric [frames,bins].")
    finite = np.isfinite(array)
    report = {"shape": list(array.shape), "finite_fraction": float(finite.mean())}
    if not finite.all():
        report.update(status="nonfinite", spectral_peak_cycles_per_bin=None)
        return report
    value = array.astype(np.float64)
    energy = float(np.mean(value**2))
    standard = float(value.std())
    report.update(
        status="constant" if standard <= np.finfo(float).eps else "ok",
        nonzero_fraction=float(np.mean(value != 0)),
        rms=float(np.sqrt(energy)),
        crest_factor=float(np.max(np.abs(value)) / np.sqrt(energy)) if energy else None,
        excess_kurtosis=(
            float(kurtosis(value.ravel(), fisher=True, bias=False))
            if value.size >= 4 and standard > np.finfo(float).eps
            else None
        ),
    )
    if standard > np.finfo(float).eps and value.shape[-1] >= 8:
        frequency, power = welch(
            value,
            fs=1.0,
            nperseg=min(256, value.shape[-1]),
            axis=-1,
            detrend="constant",
        )
        report["spectral_peak_cycles_per_bin"] = float(
            frequency[np.argmax(power.mean(0))]
        )
    else:
        report["spectral_peak_cycles_per_bin"] = None
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--record", type=int, default=0)
    args = p.parse_args()
    result = {}
    with h5py.File(args.data, "r") as file:
        for name in ["phase", "E", "tau"]:
            if name not in file or file[name].ndim != 3:
                raise ValueError(f"{name} must be [bins,frames,records].")
            if not 0 <= args.record < file[name].shape[-1]:
                raise IndexError("Record index outside dataset.")
            result[name] = signal_quality(file[name][:, :, args.record].T)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
