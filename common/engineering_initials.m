function values = engineering_initials()
    % Approximate starting calibration, not recovered experimental measurements.
    % Assumptions: approximately 1 mm shared range grid, about 1.5 m UAV height,
    % propagation window reaching beyond 13 ns, and roughly unit-scale amplitudes.
    % The algorithm does not silently normalize signals to enforce these assumptions.
    values.surface_range_m = [0.5 3.0];
    values.shallow_end_s = 13e-9; % radar-relative two-way time, not an ice-only depth
    values.psf_sigma_grid_bins = [8 16 24 32 48 64];
    values.lambda_attenuation = 1e-2;
    values.lambda_deconvolution = 1e-3;
    values.spatial_sigma_m = 0.5;
    values.spatial_epsilon_m = 0.01;
    values.boundary_phase_threshold_rad = 0.4;
    values.boundary_dilation_bins = 2;
    values.tof_min_peak_distance_bins = 10;
    values.tof_min_prominence = 0.05; % absolute input-amplitude units; calibrate for receiver gain
    values.cross_frequency_max_relative_residual = 0.5;
    values.surface_discontinuity_m = 0.05;
end
