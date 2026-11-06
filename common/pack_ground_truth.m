function [gt, boundary_target] = pack_ground_truth(core, time_s, surface_time_s, ...
                                                depth_m, relative_density_target, cfg)
    % Convert independently measured layer properties to the training contract.
    % surface_time_s is a measured registration anchor, in the current echo grid.
    % Relative-density targets must be supplied with a documented scattering
    % calibration; absolute core densities are never silently substituted.
    h = core.layer_thickness_m(:).';
    rho = core.layer_density_g_cm3(:).';
    L = numel(h);
    T = numel(surface_time_s);
    K = cfg.max_layers;
    assert(L >= 1 && L <= K && numel(rho) == L && all(h > 0) && all(rho > 0));
    time_s = time_s(:).';
    depth_m = depth_m(:).';
    surface_time_s = surface_time_s(:);
    assert(all(diff(time_s) > 0) && all(diff(depth_m) > 0) && depth_m(1) >= 0);
    assert(isequal(size(relative_density_target), [T L]), ...
           'Supply relative-density targets for every frame and true layer.');
    assert(all(isfinite(relative_density_target(:))) && all(relative_density_target(:) >= 0));
    two_way_delay = 2 * h .* (1 + cfg.dielectric_density_coefficient * rho) / cfg.c0_m_s;
    boundary_target = zeros(T, numel(time_s));
    boundary_times_s_gt = zeros(T, K + 1);
    for t = 1:T
        if isfield(core, 'boundary_times_s')
            assert(isequal(size(core.boundary_times_s), [T L + 1]), 'Measured boundary times must be [T,L+1].');
            timestamps = core.boundary_times_s(t, :);
            assert(all(isfinite(timestamps)) && all(diff(timestamps) > 0), 'Invalid measured interface times.');
        else
            timestamps = surface_time_s(t) + [0 cumsum(two_way_delay)];
        end
        assert(timestamps(1) >= time_s(1) && timestamps(end) <= time_s(end), ...
               'A physical boundary lies outside the calibrated echo window.');
        bins = zeros(1, L + 1);
        for j = 1:L + 1
            [~, bins(j)] = min(abs(time_s - timestamps(j)));
        end
        assert(numel(unique(bins)) == L + 1, 'Shared grid is too coarse to label distinct physical boundaries.');
        boundary_target(t, bins) = 1;
        boundary_times_s_gt(t, 1:L + 1) = timestamps;
    end
    gt.layer_count = L * ones(T, 1);
    gt.relative_density_target = zeros(T, K);
    gt.relative_density_target(:, 1:L) = relative_density_target;
    gt.layer_density_gt = zeros(T, K);
    gt.layer_density_gt(:, 1:L) = repmat(rho, T, 1);
    gt.layer_thickness_m_gt = zeros(T, K);
    gt.layer_thickness_m_gt(:, 1:L) = repmat(h, T, 1);
    gt.depth_m = depth_m;
    z = [0 cumsum(h)];
    profile = zeros(1, numel(depth_m));
    for l = 1:L
        profile(depth_m >= z(l) & depth_m < z(l + 1)) = rho(l);
    end
    gt.density_profile_gt = repmat(profile, T, 1);
    gt.boundary_times_s_gt = boundary_times_s_gt;
    for name = {'boundary_measurements_m', 'specimen_density_g_cm3', 'specimen_mass_g', 'specimen_volume_cm3', ...
              'mass_resolution_g', 'volume_relative_uncertainty', 'boundary_measurement_resolution_m', 'density_uncertainty_g_cm3'}
        if isfield(core, name{1})
            gt.measurements.(name{1}) = core.(name{1});
        end
    end
    if isfield(core, 'boundary_times_s')
        gt.boundary_target_method = 'independently_measured_propagation_times';
    else
        gt.boundary_target_method = 'dielectric_projection_from_measured_core';
    end
end
