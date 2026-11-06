"""Fixed reference constants used by source-level consistency checks."""

MINE_VALUES = {
    "dielectric_density_coefficient": 0.845,
    "frequency_low_hz": 7.29e9,
    "frequency_high_hz": 60e9,
    "bandwidth_low_hz": 1.4e9,
    "bandwidth_high_hz": 5.5e9,
    "surface_sigma_initial_bins": 160,
    "surface_sigma_final_bins": 80,
    "surface_sigma_anneal_frames": 50,
    "scattering_gamma": 1.5,
    "assignment_temperature": 0.23,
    "max_layers": 10,
    "erf_windows_depth_frames": [[30, 2], [60, 3], [90, 4]],
    "erf_beta": [1, 1, 1],
    "stage_epochs": [20, 20, 40],
    "warmup_epochs": 5,
    "loss_weights": {
        "boundary": 1.0,
        "count": 0.5,
        "relative_density": 0.5,
        "density": 1.0,
        "thickness": 2.0,
        "profile": 1.0,
    },
}
