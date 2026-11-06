from pathlib import Path
import ast
import json
import re
import runpy

root = Path(__file__).resolve().parents[1]
spec = runpy.run_path(str(root / "audit/mine_defaults.py"))["MINE_VALUES"]


def defaults(file, cls):
    tree = ast.parse((root / file).read_text(encoding="utf-8-sig"))
    node = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == cls)
    return {
        x.target.id: ast.literal_eval(x.value)
        for x in node.body
        if isinstance(x, ast.AnnAssign)
    }


m = defaults("p4_design4/network.py", "ModelConfig")
t = defaults("p4_design4/losses.py", "TrainingConfig")
checks = {}
for key in ["max_layers", "assignment_temperature", "dielectric_density_coefficient"]:
    checks["python." + key] = m[key] == spec[key]
checks["python.erf_windows"] = [list(x) for x in m["erf_windows"]] == spec[
    "erf_windows_depth_frames"
]
checks["python.erf_beta"] = list(m["erf_beta"]) == spec["erf_beta"]
checks["python.stage_epochs"] = list(t["stage_epochs"]) == spec["stage_epochs"]
checks["python.warmup_epochs"] = t["warmup_epochs"] == spec["warmup_epochs"]
for name, spec_name in [
    ("lambda_b", "boundary"),
    ("lambda_count", "count"),
    ("lambda_pre_rho", "relative_density"),
    ("lambda_rho", "density"),
    ("lambda_h", "thickness"),
    ("lambda_profile", "profile"),
]:
    checks["python." + name] = t[name] == spec["loss_weights"][spec_name]
s = (root / "common/icemap_config.m").read_text(encoding="utf-8-sig")
for key in [
    "frequency_low_hz",
    "frequency_high_hz",
    "bandwidth_low_hz",
    "bandwidth_high_hz",
    "surface_sigma_initial_bins",
    "surface_sigma_final_bins",
    "surface_sigma_anneal_frames",
    "scattering_gamma",
    "assignment_temperature",
    "max_layers",
    "dielectric_density_coefficient",
]:
    value = float(re.search(r"cfg\." + key + r"\s*=\s*([\d.eE+-]+)\s*;", s).group(1))
    checks["matlab." + key] = value == spec[key]
checks["matlab.loss_weights"] = "cfg.loss_weights = [1 0.5 0.5 1 2 1];" in s
checks["matlab.erf_windows"] = "cfg.erf_windows = [30 2; 60 3; 90 4];" in s
checks["matlab.erf_beta"] = "cfg.erf_beta = [1 1 1];" in s
checks["matlab.stage_epochs"] = "cfg.stage_epochs = [20 20 40];" in s
checks["matlab.warmup"] = "cfg.loss_warmup_epochs = 5;" in s
report = {
    "scope": "static default-value comparison with the visually checked reference transcription",
    "checks": checks,
    "passed": all(checks.values()),
    "runtime_validation": "not performed",
}
assert all(checks.values()), report
print("Static mine parameter checks:", len(checks), "PASS")
