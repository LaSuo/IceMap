"""Recoverable training recipe; historical development choices are explicit.

The deleted reference backbone followed several warm starts whose complete
optimizer/RNG lineage was not retained. This recipe can train the same model
family from scratch; identical weights or historical metrics are not assured.
The historical posterior/gates below were selected using test-labelled
development data and must not be described as independent test validation.
"""

from dataclasses import dataclass
import math

from calibration import CalibrationConfig
from post_calibration import PostCalibrationConfig
from quality_calibration import QualityExpertConfig
from reference_alignment import ReferenceAlignmentConfig


@dataclass
class RetrainingConfig:
    seed: int = 431
    expert_seed: int = 617
    validation_fraction: float = 0.15
    stage_epochs: tuple = (20, 20, 40)
    backbone_batch: int = 16
    backbone_lr: float = 0.0003
    backbone_weight_decay: float = 0.0001
    aux_bce: float = 1.0
    aux_location: float = 1.0
    aux_time: float = 1.5
    aux_thickness: float = 3.0
    aux_density: float = 1.0
    sigma: int = 3
    head_epochs: int = 500
    head_lr: float = 0.0005
    head_batch: int = 256
    head_weight_decay: float = 0.001
    expert_epochs: int = 300
    expert_lr: float = 0.0005
    expert_batch: int = 1024
    expert_weight_decay: float = 0.001
    augmentation_styles: int = 9
    cache_batch: int = 4
    pool_bins: int = 16
    raw_shape_bins: int = 64

    def validate(self):
        self.stage_epochs = tuple(self.stage_epochs)
        if len(self.stage_epochs) != 3 or any(
            not isinstance(x, int) or x < 1 for x in self.stage_epochs
        ):
            raise ValueError("Three positive stage epoch budgets are required.")
        for k in ["seed", "expert_seed"]:
            if not isinstance(getattr(self, k), int) or getattr(self, k) < 0:
                raise ValueError("Invalid " + k)
        for k in [
            "backbone_batch",
            "head_epochs",
            "head_batch",
            "expert_epochs",
            "expert_batch",
            "augmentation_styles",
            "cache_batch",
            "pool_bins",
            "raw_shape_bins",
            "sigma",
        ]:
            if not isinstance(getattr(self, k), int) or getattr(self, k) < 1:
                raise ValueError("Invalid " + k)
        if self.augmentation_styles > 9 or not 0 < self.validation_fraction < 1:
            raise ValueError("Invalid styles or validation fraction.")
        for k in ["backbone_lr", "head_lr", "expert_lr"]:
            if not math.isfinite(getattr(self, k)) or getattr(self, k) <= 0:
                raise ValueError("Invalid " + k)
        for k in [
            "backbone_weight_decay",
            "head_weight_decay",
            "expert_weight_decay",
            "aux_bce",
            "aux_location",
            "aux_time",
            "aux_thickness",
            "aux_density",
        ]:
            if not math.isfinite(getattr(self, k)) or getattr(self, k) < 0:
                raise ValueError("Invalid " + k)
        return self


def calibration_config():
    return CalibrationConfig(
        width=256, dropout_probability=0.0, range_correction_limit_m=1.6
    )


def historical_reference_config():
    return ReferenceAlignmentConfig(
        count_prior_weight=21.8808652054,
        count_prior_sigma=0.5,
        density_shrinkage=1.0,
        range_shrinkage=1.0,
        regression_temporal_consensus="median",
        count_prior_disagreement_limit=0.75,
        density_offset_g_cm3=-0.03,
    )


def historical_post_config():
    return PostCalibrationConfig(
        range_scale=1.0,
        range_offset_m=-0.0005,
        count_consensus_strength=0.13,
        range_quality_low=0.8,
        range_quality_high=1.1,
        enforce_observed_range=True,
    )


def dense_expert_config(max_range_m=2.199):
    return QualityExpertConfig(
        width=256,
        feature_mode="peak_spectral",
        dropout_probability=0.1,
        max_range_m=max_range_m,
        dense_weight=1.0,
        weak_weight=0.0,
        density_weight=0.5,
        range_weight=1.0,
        dense_count_weight=1.0,
        weak_count_weight=0.0,
    )


def weak_expert_config(max_range_m=2.199):
    return QualityExpertConfig(
        width=256,
        feature_mode="backbone",
        dropout_probability=0.0,
        max_range_m=max_range_m,
        dense_weight=0.0,
        weak_weight=1.0,
        weak_start=1.1,
        weak_full=0.2,
        density_weight=1.0,
        range_weight=1.0,
        weak_count_weight=0.0,
        weak_single_layer_only=True,
        weak_exclude_dense=True,
        weak_min_range_m=0.2,
    )
