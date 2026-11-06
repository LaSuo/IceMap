"""Anonymous runtime and complete code-tree fingerprints for reproducibility."""

import hashlib
import json
import platform
import sys
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
import torch
from protocol import sha256_file


def capture_environment():
    packages = {}
    for name in ("torch", "numpy", "h5py", "scipy", "PyYAML", "packaging", "tqdm"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_names": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ],
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def code_fingerprints(root):
    root = Path(root)
    files = {
        p.relative_to(root).as_posix(): sha256_file(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and (
            p.suffix in (".py", ".m", ".yml")
            or p.name.startswith("requirements")
            and p.suffix == ".txt"
        )
    }
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"code_sha256": digest, "files": files}
