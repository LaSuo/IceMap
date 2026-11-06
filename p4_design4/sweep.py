"""Plan development variant/ERF/seed runs; not the site-held-out main protocol."""

import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys

from model_catalog import VARIANTS, ALIASES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--variants", nargs="+", choices=(*VARIANTS, *ALIASES), default=list(VARIANTS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[431])
    parser.add_argument("--erf", nargs="+", choices=["on", "off"], default=["on"])
    parser.add_argument("--profile", choices=["fresh", "historical"], default="fresh")
    parser.add_argument("--legacy-grid-mm", type=float)
    parser.add_argument("--allow-relative-density-proxy", action="store_true")
    parser.add_argument("--allow-test-selected-history", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--recipe-json", type=Path)
    parser.add_argument("--architecture-config", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--progress", action="store_true", help="Optional tqdm run-level progress"
    )
    args = parser.parse_args()
    args.variants = [ALIASES.get(v, v) for v in args.variants]
    choices = list(itertools.product(args.variants, args.seeds, args.erf))
    if len(choices) != len(set(choices)):
        parser.error("Duplicate sweep choices would overwrite the same run.")
    iterator = choices
    if args.progress:
        from tqdm import tqdm

        iterator = tqdm(choices, desc="model runs", unit="run")
    for variant, seed, erf in iterator:
        folder = args.output_root / f"{variant}_seed{seed}_erf{erf}"
        command = [
            sys.executable,
            "-B",
            str(Path(__file__).with_name("train.py")),
            "--data-dir",
            str(args.data_dir),
            "--output-dir",
            str(folder),
            "--network-variant",
            variant,
            "--seed",
            str(seed),
            "--erf",
            erf,
            "--profile",
            args.profile,
            "--device",
            args.device,
        ]
        for name in ["recipe_json", "architecture_config"]:
            if getattr(args, name) is not None:
                command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
        if args.smoke:
            command += ["--smoke"]
        if args.legacy_grid_mm is not None:
            command += ["--legacy-grid-mm", str(args.legacy_grid_mm)]
        for key in ("allow_relative_density_proxy", "allow_test_selected_history"):
            if getattr(args, key):
                command += ["--" + key.replace("_", "-")]
        print(
            json.dumps({"command": command, "execution_enabled": args.execute}),
            flush=True,
        )
        if args.execute:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
