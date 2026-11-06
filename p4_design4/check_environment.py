"""Inspect dependency versions and CUDA without installing or altering them."""

import argparse
from importlib import metadata
import json
from pathlib import Path
import sys

from packaging.requirements import Requirement


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cuda", action="store_true")
    args = p.parse_args()
    result = []
    passed = sys.version_info[:2] == (3, 10)
    for line in Path(__file__).with_name("requirements.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        requirement = Requirement(line)
        try:
            version = metadata.version(requirement.name)
            valid = version in requirement.specifier
        except metadata.PackageNotFoundError:
            version = None
            valid = False
        result.append(
            {
                "package": requirement.name,
                "installed": version,
                "required": str(requirement.specifier),
                "pass": valid,
            }
        )
        passed &= valid
    import torch

    cuda = torch.cuda.is_available()
    print(
        json.dumps(
            {
                "python": sys.version.split()[0],
                "dependencies": result,
                "cuda_available": cuda,
                "torch_cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if cuda else None,
                "pass": passed and (cuda or not args.cuda),
            },
            indent=2,
        )
    )
    if not passed or (args.cuda and not cuda):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
