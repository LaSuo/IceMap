"""Strict JSON/YAML option loading; never execute configuration as code."""

import json
from pathlib import Path


def load_mapping(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError(
            "Configuration exceeds 4 MiB; a dataset may have been supplied by mistake."
        )
    text = path.read_text(encoding="utf-8-sig")
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as error:
                raise RuntimeError(
                    "YAML options require PyYAML; install requirements.txt."
                ) from error
            try:
                value = yaml.safe_load(text)
            except yaml.YAMLError as error:
                raise ValueError("Invalid or unsafe YAML syntax.") from error
        elif path.suffix.lower() == ".json":

            def invalid(value):
                raise ValueError("Nonstandard JSON constant " + value)

            def unique(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError("Duplicate JSON key " + key)
                    result[key] = value
                return result

            value = json.loads(text, parse_constant=invalid, object_pairs_hook=unique)
        else:
            raise ValueError("Expected .json, .yaml or .yml configuration.")
    except (ValueError, TypeError) as error:
        raise ValueError(f"Cannot read configuration {path}: {error}") from error
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ValueError("Configuration must be a mapping with string keys.")
    return value
