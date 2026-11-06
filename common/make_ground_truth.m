function truth = make_ground_truth(boundary_measurements_m, mass_g, volume_cm3, ...
                                 persistent_transition, rho_above, rho_below)
    % Independent physical measurements and uncertainty-aware aggregation.
    % Boundary positions are each measured 3 times at 1 mm resolution.
    % Mass/volume rows are layers, 3 representative specimens per layer.
    assert(size(boundary_measurements_m, 2) == 3, 'Three boundary measurements required.');
    assert(size(mass_g, 2) == 3 && isequal(size(mass_g), size(volume_cm3)), ...
           'R=3 mass/volume specimens per layer are required.');
    assert(all(isfinite(boundary_measurements_m(:))) && all(isfinite(mass_g(:))) && all(isfinite(volume_cm3(:))));
    assert(all(volume_cm3(:) > 0) && all(mass_g(:) > 0));
    z_gt = mean(boundary_measurements_m, 2);
    h_gt = diff(z_gt);
    L = numel(h_gt);
    assert(size(mass_g, 1) == L && all(h_gt > 0));
    assert(numel(persistent_transition) == max(L - 1, 0) && numel(rho_above) == max(L - 1, 0) && numel(rho_below) == max(L - 1, 0));
    accepted = persistent_transition(:) & abs(rho_above(:) - rho_below(:)) > 0.01;
    assert(all(accepted), 'An internal boundary fails persistent-structure/density-change criteria.');
    rho_specimen = mass_g ./ volume_cm3;
    truth.boundaries_m = z_gt;
    truth.boundary_measurements_m = boundary_measurements_m;
    truth.specimen_mass_g = mass_g;
    truth.specimen_volume_cm3 = volume_cm3;
    truth.layer_thickness_m = h_gt;
    truth.layer_density_g_cm3 = mean(rho_specimen, 2);
    truth.density_std_g_cm3 = std(rho_specimen, 0, 2);
    truth.specimen_density_g_cm3 = rho_specimen;
    truth.num_layers = L;
    truth.R = 3;
    truth.boundary_measurement_resolution_m = 0.001;
    truth.density_uncertainty_g_cm3 = 0.01;
    truth.mass_resolution_g = 0.05;
    truth.volume_relative_uncertainty = 0.008;
    truth.density_sampling_interval_m = 0.05;
    truth.boundary_density_neighborhood_m = 0.02;
    % Relative-density targets are NOT generated here: their scattering-to-target
    % construction is absent from the reference specification and must be provided explicitly.
end
