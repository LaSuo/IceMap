"""Explicit implementation values and assumptions, without adding experiments.

Engineering initials are provisional choices, not recovered original settings.
Author declarations are recorded separately from measured deployment evidence.
"""

from dataclasses import asdict, dataclass
import math


@dataclass
class TrainingInitials:
    learning_rate: float = 1e-3
    batch_size: int = 2
    seed: int = 431
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    validation_fraction: float = 0.2
    workers: int = 0
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0

    def validate(self):
        for name in ("learning_rate", "adam_epsilon"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be positive and finite.")
        if not 0 <= self.adam_beta1 < 1 or not 0 <= self.adam_beta2 < 1:
            raise ValueError("Adam betas must be in [0,1).")
        if (
            not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
            or not 0 < self.validation_fraction < 1
        ):
            raise ValueError("Invalid decay/validation fraction.")
        if (
            type(self.batch_size) is not int
            or self.batch_size < 1
            or type(self.workers) is not int
            or self.workers < 0
        ):
            raise ValueError("Invalid batch size/worker count.")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Seed must fit the NumPy uint32 seed range.")
        if (
            not 0 <= self.focal_alpha <= 1
            or not math.isfinite(self.focal_gamma)
            or self.focal_gamma < 0
        ):
            raise ValueError("Invalid focal alpha/gamma.")
        return self


@dataclass
class DeploymentDetails:
    sensing_rate_hz: float = 5.0
    battery_stated_voltage_v: float = 16.8
    battery_stated_capacity_ah: float = 5.0
    declared_runtime_lower_bound_s: float = 1500.0
    declared_downlink_bytes_s: float = 20000.0
    downlink_unit_convention: str = "decimal kB; 1000 bytes per kB"
    battery_link_source: str = (
        "author_declared; not verified by code or supplied measurement logs"
    )
    initial_loading_safety_factor: float = 3.0
    loading_safety_factor_source: str = (
        "provisional engineering value; not the recovered loading-test protocol"
    )
    loading_failure_criterion: str = (
        "record from the actual independent loading protocol"
    )
    model_precision: str = (
        "FP32 by default; optional dynamic INT8 Linear/GRU export, FP32 convolution and physics"
    )
    capacity_coefficient_kg_cm2: float = 3.5
    density_threshold_g_cm3: float = 0.90
    homogeneous_radar_band: str = "low"
    homogeneous_frequency_hz: float = 7.29e9
    homogeneous_assumed_density_g_cm3: float = 0.917
    homogeneous_assumed_permittivity: float = 3.15
    homogeneous_settings_source: str = (
        "provisional interface assumptions; not consumed by default SPAI or recovered from the original baseline"
    )


MODEL_ASSUMPTIONS = {
    "geometry": "locally layered medium and shared calibrated propagation grid; D1 registers a reference surface",
    "frequency_translation": "PSF and relative attenuation approximate systematic differences, not arbitrary frequency-dependent microstructure scattering",
    "density_velocity": "n=1+0.845*rho, rho in g/cm^3; no unreported wet/saline-ice correction is applied",
    "surface_discontinuity": "surface-jump diagnostics retain frames; they are not validated crack detection",
    "mechanical_capacity": "layered empirical estimate; does not model fracture connectivity or establish structural clearance",
    "normalization": "no dataset-wide/test-derived standardization in the main path; BatchNorm is learned on training batches and frozen at inference",
    "relative_density_target": "externally calibrated target required; the legacy absolute-density proxy remains an explicitly separate development choice",
    "metadata": "site/record identifiers, labels, weather and core measurements are not neural inputs in the default model",
    "timing": "propagation time_s and acquisition_time_s use seconds but represent different axes",
}


def settings_manifest():
    return {
        "training_initials": asdict(TrainingInitials().validate()),
        "training_initials_source": "provisional engineering choices, not reported optimizer values",
        "deployment_details": asdict(DeploymentDetails()),
        "assumptions": MODEL_ASSUMPTIONS,
    }


def audit_partition_metadata(splits):
    """Audit the existing site split; do not create another experimental split."""
    import json
    import h5py

    identities = {}
    missing = []
    for role, paths in splits.items():
        identities[role] = set()
        for path in paths:
            with h5py.File(path, "r") as f:
                raw = f.attrs.get("experiment_metadata_json", "{}")
                if isinstance(raw, bytes):
                    raw = raw.decode("utf8")
                metadata = json.loads(raw)
                identity = metadata.get("structure_id", f.attrs.get("structure_id"))
            if isinstance(identity, bytes):
                identity = identity.decode("utf8")
            if not isinstance(identity, str) or not identity:
                missing.append(str(path))
            else:
                identities[role].add(identity)
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if identities[a] & identities[b]:
            raise ValueError(
                f"Physical structure appears across the existing {a}/{b} site partitions; correct site metadata before training."
            )
    return {
        "split_unit": "existing site_id fivefold protocol",
        "physical_structure_metadata_complete": not missing,
        "records_missing_structure_identity": missing,
        "declared_structure_conflicts": False,
        "scope": "checks supplied identities only; absent physical identities are not invented from filenames",
    }
