"""Print validated option templates, e.g. --kind scene > scene.json."""

import argparse
from dataclasses import asdict
import json

from model_catalog import ArchitectureConfig
from robustness import AugmentationConfig
from scene import Scene
from training_config import RetrainingConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        choices=["scene", "architecture", "recipe", "augmentation"],
        required=True,
    )
    args = parser.parse_args()
    templates = {
        "scene": Scene().to_dict(),
        "architecture": asdict(ArchitectureConfig()),
        "recipe": asdict(RetrainingConfig()),
        "augmentation": asdict(AugmentationConfig()),
    }
    print(json.dumps(templates[args.kind], indent=2))


if __name__ == "__main__":
    main()
