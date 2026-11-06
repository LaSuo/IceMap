"""Validated optional scene metadata. Missing observations remain None.

Metadata alone never changes dielectric physics or labels. Context encoding is
used only by an explicitly configured research network. Scalar run metadata can
be supplied by JSON; callers may also provide per-record/frame context tensors.
"""

from dataclasses import asdict, dataclass, field, fields
import json
import math


@dataclass
class IceSurface:
    surface_rms_roughness_m: float | None = None
    correlation_length_m: float | None = None
    snow_depth_m: float | None = None
    snow_density_kg_m3: float | None = None
    surface_temperature_c: float | None = None
    water_salinity_psu: float | None = None
    liquid_water_fraction: float | None = None
    surface_slope_rad: float | None = None


@dataclass
class UAV:
    altitude_agl_m: float | None = None
    speed_m_s: float | None = None
    roll_rad: float | None = None
    pitch_rad: float | None = None
    yaw_rad: float | None = None
    vertical_uncertainty_m: float | None = None
    horizontal_uncertainty_m: float | None = None
    frame_interval_s: float | None = None
    antenna_lever_arm_m: tuple | None = None


@dataclass
class Climate:
    air_temperature_c: float | None = None
    relative_humidity: float | None = None
    wind_speed_m_s: float | None = None
    pressure_pa: float | None = None
    precipitation_mm_h: float | None = None
    solar_irradiance_w_m2: float | None = None


@dataclass
class Radar:
    low_frequency_hz: float | None = None
    high_frequency_hz: float | None = None
    low_bandwidth_hz: float | None = None
    high_bandwidth_hz: float | None = None
    clock_offset_s: float | None = None
    noise_power_low: float | None = None
    noise_power_high: float | None = None


@dataclass
class Scene:
    ice: IceSurface = field(default_factory=IceSurface)
    uav: UAV = field(default_factory=UAV)
    climate: Climate = field(default_factory=Climate)
    radar: Radar = field(default_factory=Radar)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - {
            "ice",
            "uav",
            "climate",
            "radar",
        }:
            raise ValueError("Unknown scene group or invalid mapping.")
        result = cls(
            **{
                k: kind(**value.get(k, {}))
                for k, kind in [
                    ("ice", IceSurface),
                    ("uav", UAV),
                    ("climate", Climate),
                    ("radar", Radar),
                ]
            }
        )
        result.validate()
        return result

    def validate(self):
        signed = {
            "roll_rad",
            "pitch_rad",
            "yaw_rad",
            "surface_slope_rad",
            "clock_offset_s",
        }
        fractions = {"relative_humidity", "liquid_water_fraction"}
        for group in fields(self):
            item = getattr(self, group.name)
            for f in fields(item):
                value = getattr(item, f.name)
                if value is None:
                    continue
                if f.name == "antenna_lever_arm_m":
                    if (
                        not isinstance(value, (list, tuple))
                        or len(value) != 3
                        or any(not math.isfinite(float(x)) for x in value)
                    ):
                        raise ValueError(
                            "antenna_lever_arm_m must contain three finite body-frame coordinates."
                        )
                    continue
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError("Expected finite scalar: " + f.name)
                if f.name in fractions and not 0 <= value <= 1:
                    raise ValueError(f.name + " must be a fraction in [0,1].")
                if f.name.endswith("temperature_c") and value < -273.15:
                    raise ValueError("Temperature below absolute zero.")
                if (
                    f.name not in signed | fractions
                    and not f.name.endswith("temperature_c")
                    and value < 0
                ):
                    raise ValueError("Negative " + f.name)
        return self

    def to_dict(self):
        self.validate()
        return asdict(self)


# Fixed engineering scales, not a learned ice law; append an observed-mask so
# an absent value differs from a genuine zero. No target thickness/density.
CONTEXT_FIELDS = (
    ("ice", "surface_rms_roughness_m", 0.01),
    ("ice", "snow_depth_m", 0.1),
    ("ice", "surface_temperature_c", 20.0),
    ("ice", "water_salinity_psu", 35.0),
    ("ice", "liquid_water_fraction", 1.0),
    ("ice", "surface_slope_rad", 1.0),
    ("uav", "altitude_agl_m", 5.0),
    ("uav", "speed_m_s", 5.0),
    ("uav", "roll_rad", 1.0),
    ("uav", "pitch_rad", 1.0),
    ("uav", "vertical_uncertainty_m", 0.1),
    ("uav", "frame_interval_s", 0.1),
    ("climate", "air_temperature_c", 20.0),
    ("climate", "relative_humidity", 1.0),
    ("climate", "wind_speed_m_s", 10.0),
    ("climate", "pressure_pa", 100000.0),
)
CONTEXT_DIM = 2 * len(CONTEXT_FIELDS)


def encode_context(scene):
    scene.validate()
    values, observed = [], []
    for group, name, scale in CONTEXT_FIELDS:
        value = getattr(getattr(scene, group), name)
        values.append(0.0 if value is None else max(-10.0, min(10.0, value / scale)))
        observed.append(float(value is not None))
    return values + observed


def read_scene_h5(path):
    """Optional metadata reader; never treats record/site IDs as context."""
    import h5py

    with h5py.File(path, "r") as file:
        value = file.attrs.get("scene_metadata_json")
    if value is None:
        return Scene()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return Scene.from_dict(json.loads(value))
