"""Match raw OPF subwindows to legacy HDF5 rows and extract original labels."""

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import re

import h5py
import numpy as np

from protocol import sha256_file


LABELS = ("label1", "label2", "label2_len", "label2_mask", "label3")
FINGERPRINTS = ("phase", "E", "tau")
RAW_PATTERN = re.compile(r"^(e\d+r\d+_t\d+b\d+)_sub(\d+)(?:_|$)", re.I)
LEGACY_PATTERN = re.compile(r"^(\d{6})_(e\d+r\d+_t\d+b\d+)_.*?_sub(\d+)(?:_|$)", re.I)


def legacy_names(file):
    values = file["file_names"][...]
    if values.ndim == 2 and values.dtype == np.uint8:
        return [bytes(row).decode("utf-8").strip("\x00 ") for row in values.T]
    if values.ndim == 1 and values.dtype.kind in "OSU":
        return [
            (value.decode("utf-8") if isinstance(value, bytes) else str(value)).strip(
                "\x00 "
            )
            for value in values
        ]
    raise ValueError("Unsupported legacy HDF5 filename table.")


def row_index(file, date):
    result = defaultdict(list)
    names = legacy_names(file)
    if any(file[field].shape[-1] != len(names) for field in LABELS + FINGERPRINTS):
        raise ValueError("A legacy record axis does not match the filename table.")
    for index, name in enumerate(names):
        match = LEGACY_PATTERN.match(Path(name).stem)
        if match and match[1] == date:
            result[(match[2].lower(), int(match[3]))].append((index, name))
    return result, len(names)


def identical_rows(first, i, second, j, fields):
    return all(
        np.array_equal(first[name][..., i], second[name][..., j]) for name in fields
    )


def build_manifest(raw_dir, train, test, date):
    raw = defaultdict(list)
    unparsed = []
    for path in sorted(raw_dir.glob("*.h5")):
        match = RAW_PATTERN.match(path.stem)
        if match:
            raw[(match[1].lower(), int(match[2]))].append(path)
        else:
            unparsed.append(path.name)
    if not raw:
        raise ValueError("No recognizable raw OPF recordings.")
    train_index, train_count = row_index(train, date)
    test_index, test_count = row_index(test, date)
    rows = []
    extracted = []
    for key in sorted(raw.keys() | train_index.keys() | test_index.keys()):
        raw_files = raw.get(key, [])
        train_rows, test_rows = train_index.get(key, []), test_index.get(key, [])
        row = {
            "key": f"{key[0]}_sub{key[1]}",
            "raw_filename": "|".join(path.name for path in raw_files),
            "train_row_zero_based": train_rows[0][0] if len(train_rows) == 1 else -1,
            "test_row_zero_based": test_rows[0][0] if len(test_rows) == 1 else -1,
            "train_filename": "|".join(name for _, name in train_rows),
            "test_filename": "|".join(name for _, name in test_rows),
            "duplicate_features_identical": "",
            "duplicate_labels_identical": "",
            "label_archive_row_zero_based": -1,
            "raw_nominal_total_mm": "",
            "legacy_median_total_mm": "",
        }
        if not raw_files:
            row["status"] = "legacy_only"
        elif max(len(raw_files), len(train_rows), len(test_rows)) > 1:
            row["status"] = "ambiguous_identifier"
        elif not (train_rows or test_rows):
            row["status"] = "raw_only"
        else:
            row["status"] = "unique_identifier"
            if train_rows and test_rows:
                a, b = train_rows[0][0], test_rows[0][0]
                row["duplicate_features_identical"] = identical_rows(
                    train, a, test, b, FINGERPRINTS
                )
                row["duplicate_labels_identical"] = identical_rows(
                    train, a, test, b, LABELS
                )
                if not (
                    row["duplicate_features_identical"]
                    and row["duplicate_labels_identical"]
                ):
                    row["status"] = "legacy_duplicate_conflict"
            if row["status"] == "unique_identifier":
                src, src_rows = (test, test_rows) if test_rows else (train, train_rows)
                index = src_rows[0][0]
                layer_profile = src["label3"][..., index]
                nominal = [
                    float(s)
                    for s in re.findall(r"_h_(\d+(?:\.\d+)?)_rho_", raw_files[0].stem)
                ]
                row["raw_nominal_total_mm"] = sum(nominal) if nominal else ""
                row["legacy_median_total_mm"] = float(
                    np.median(np.count_nonzero(layer_profile > 0, axis=0))
                )
                row["label_archive_row_zero_based"] = len(extracted)
                extracted.append((row, src, index))
        rows.append(row)
    return rows, extracted, unparsed, (train_count, test_count)


def export_labels(path, items):
    if not items:
        raise ValueError("There are no unambiguous legacy labels to extract.")
    text = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "x") as output:
        output.attrs["format"] = "icemap.opf.legacy_labels.v1"
        output.attrs["scope"] = (
            "identifier-matched historical labels; independent of recovered features"
        )
        for name in LABELS:
            values = np.stack(
                [src[name][..., index] for _, src, index in items], axis=-1
            )
            output.create_dataset(name, data=values, compression="lzf")
        for name in ("key", "raw_filename", "train_filename", "test_filename"):
            output.create_dataset(
                name, data=[row[name] for row, _, _ in items], dtype=text
            )
        for name in ("train_row_zero_based", "test_row_zero_based"):
            output.create_dataset(name, data=[row[name] for row, _, _ in items])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--legacy-dir", type=Path, required=True)
    parser.add_argument("--legacy-date", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if re.fullmatch(r"\d{6}", args.legacy_date) is None:
        parser.error("--legacy-date must be the six-digit legacy acquisition prefix.")
    train_path = args.legacy_dir / "train.h5"
    test_path = args.legacy_dir / "test_opf.h5"
    with h5py.File(train_path) as train, h5py.File(test_path) as test:
        rows, items, unparsed, sizes = build_manifest(
            args.raw_dir, train, test, args.legacy_date
        )
        args.output.mkdir(parents=True, exist_ok=False)
        with (args.output / "matches.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        export_labels(args.output / "labels_matched.h5", items)
    summary = {
        "legacy_date": args.legacy_date,
        "legacy_train_total": sizes[0],
        "legacy_test_total": sizes[1],
        "unique_raw_labels_extracted": len(items),
        "identifier_status_counts": {
            state: sum(row["status"] == state for row in rows)
            for state in sorted({row["status"] for row in rows})
        },
        "train_test_identical_duplicates": sum(
            row["duplicate_features_identical"] is True
            and row["duplicate_labels_identical"] is True
            for row in rows
        ),
        "raw_nominal_vs_legacy_total_disagreements_mm": [
            row["key"]
            for row in rows
            if row["raw_nominal_total_mm"] != ""
            and abs(row["raw_nominal_total_mm"] - row["legacy_median_total_mm"]) > 1
        ],
        "unparsed_raw_filenames": unparsed,
        "source_sha256": {
            path.name: sha256_file(path) for path in (train_path, test_path)
        },
    }
    (args.output / "match_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
