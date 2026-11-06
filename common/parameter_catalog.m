function rows = parameter_catalog(cfg)
    % Human/machine-readable parameter values, units, provenance and assumptions.
    % Changed reference values are recorded explicitly as user overrides.
    if nargin < 1
        cfg = icemap_config();
    end
    rows = struct('name', {}, 'value', {}, 'unit', {}, 'stage', {}, 'source', {}, 'note', {});
    mine = { 'c0_m_s', 299792458, 'm/s', 'shared'; ...
           'dielectric_density_coefficient', .845, 'cm^3/g', 'D4'; ...
           'frequency_low_hz', 7.29e9, 'Hz', 'D2'; 'frequency_high_hz', 60e9, 'Hz', 'D2'; ...
           'bandwidth_low_hz', 1.4e9, 'Hz', 'D2'; 'bandwidth_high_hz', 5.5e9, 'Hz', 'D2'; ...
           'surface_sigma_initial_bins', 160, 'bins', 'D1'; 'surface_sigma_final_bins', 80, 'bins', 'D1'; ...
           'surface_sigma_anneal_frames', 50, 'frames', 'D1'; 'scattering_gamma', 1.5, '1', 'D1'; ...
           'assignment_temperature', .23, '1', 'D4'; 'max_layers', 10, 'layers', 'D4'; ...
           'erf_windows', [30 2; 60 3; 90 4], 'depth bins x frames', 'D4'; ...
           'erf_beta', [1 1 1], '1', 'D4'; 'stage_epochs', [20 20 40], 'epochs', 'D4'; ...
           'loss_warmup_epochs', 5, 'epochs', 'D4'; 'loss_weights', [1 .5 .5 1 2 1], '1', 'D4'};
    for i = 1:size(mine, 1)
        key = mine{i, 1};
        source = 'mine_reported';
        if ~isequal(cfg.(key), mine{i, 2})
            source = 'user_override_of_reported_value';
        end
        rows(end + 1) = entry(key, cfg.(key), mine{i, 3}, mine{i, 4}, source, ''); %#ok<AGROW>
    end
    initials = engineering_initials();
    names = fieldnames(initials);
    units = {'m', 's', 'bins', 'input-amplitude squared', '1', 'm', 'm', 'rad', 'bins', 'bins', 'input-amplitude', '1', 'm'};
    stages = {'D1', 'D2', 'D2', 'D2', 'D2', 'D1', 'D1', 'D3', 'D3', 'D3', 'D3', 'D2', 'D1'};
    for i = 1:numel(names)
        key = names{i};
        source = cfg.calibration_status;
        note = 'Approximate initialization; verify against acquisition grid, receiver scaling and independent calibration.';
        if strcmp(source, 'measured')
            note = 'Measurement-calibrated value declared by caller; preserve calibration_provenance.';
        end
        rows(end + 1) = entry(key, cfg.(key), units{i}, stages{i}, source, note); %#ok<AGROW>
    end
    choices = {'tracking_band', 'surface_min_peak_distance_bins', 'surface_candidate_prominence', 'spatial_neighborhood_frames', ...
             'min_cos_incidence', 'psf_family', 'deep_evidence_policy', 'psf_support_sigmas', 'max_operator_nonzeros', ...
             'attenuation_floor', 'attenuation_max_iter', 'solver_tolerance', 'phase_confidence_min', 'phase_trend_bins', ...
             'coherence_depth_bins', 'coherence_frame_radius', 'coherence_floor', 'tof_smooth_bins', 'tof_centroid_halfwidth_bins', 'tof_skeleton_sigma_bins'};
    choice_units = {'enum', 'bins', 'input-amplitude', 'frames', '1', 'enum', 'enum', 'standard deviations', 'matrix nonzeros', '1', ...
                  'iterations', 'relative KKT residual', '1', 'bins', 'bins', 'frames', '1', 'bins', 'bins', 'bins'};
    for i = 1:numel(choices)
        key = choices{i};
        rows(end + 1) = entry(key, cfg.(key), choice_units{i}, 'engineering', 'implementation_choice', 'Not a numerically specified original experiment parameter.'); %#ok<AGROW>
    end
end

function row = entry(name, value, unit, stage, source, note)
    row = struct('name', name, 'value', value, 'unit', unit, 'stage', stage, 'source', source, 'note', note);
end
