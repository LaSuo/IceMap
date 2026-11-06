function validate_processing_config(cfg)
    if ~isstruct(cfg) || ~isscalar(cfg)
        error('IceMap:ConfigType', 'Processing configuration must be one scalar struct.');
    end
    if isfield(cfg, 'calibration_status') && ~any(strcmp(cfg.calibration_status, {'engineering_initial', 'measured'}))
        error('IceMap:CalibrationStatus', 'calibration_status must be engineering_initial or measured.');
    end
    positive = {'c0_m_s', 'frequency_low_hz', 'frequency_high_hz', 'bandwidth_low_hz', 'bandwidth_high_hz', ...
              'surface_sigma_initial_bins', 'surface_sigma_final_bins', 'assignment_temperature', 'spatial_sigma_m', ...
              'spatial_epsilon_m', 'min_cos_incidence', 'psf_support_sigmas', 'lambda_attenuation', ...
              'lambda_deconvolution', 'attenuation_floor', 'solver_tolerance', 'boundary_phase_threshold_rad', ...
              'coherence_floor', 'tof_skeleton_sigma_bins'};
    nonnegative = {'dielectric_density_coefficient', 'scattering_gamma', 'surface_candidate_prominence', ...
                 'phase_confidence_min', 'tof_min_prominence'};
    integers = {'max_operator_nonzeros', 'max_layers', 'surface_sigma_anneal_frames', 'surface_min_peak_distance_bins', 'spatial_neighborhood_frames', ...
              'attenuation_max_iter', 'phase_trend_bins', 'coherence_depth_bins', 'tof_smooth_bins', 'tof_min_peak_distance_bins'};
    zero_integers = {'boundary_dilation_bins', 'coherence_frame_radius', 'tof_centroid_halfwidth_bins'};
    names = [positive nonnegative integers zero_integers];
    for i = 1:numel(names)
        k = names{i};
        if ~isfield(cfg, k)
            error('IceMap:ConfigField', 'Missing processing field: %s.', k);
        end
        v = cfg.(k);
        if isempty(v)
            continue
        end
        if ~isnumeric(v) || ~isreal(v) || ~isscalar(v) || ~isfinite(v)
            error('IceMap:ConfigValue', '%s requires a finite real scalar.', k);
        end
        if any(strcmp(k, positive)) && v <= 0
            error('IceMap:ConfigRange', '%s must be >0.', k);
        end
        if any(strcmp(k, nonnegative)) && v < 0
            error('IceMap:ConfigRange', '%s must be >=0.', k);
        end
        if any(strcmp(k, integers)) && (v < 1 || v ~= round(v))
            error('IceMap:ConfigInteger', '%s must be a positive integer.', k);
        end
        if any(strcmp(k, zero_integers)) && (v < 0 || v ~= round(v))
            error('IceMap:ConfigInteger', '%s must be a nonnegative integer.', k);
        end
    end
    if cfg.min_cos_incidence > 1 || cfg.coherence_floor > 1 || cfg.phase_confidence_min > 1
        error('IceMap:ConfigFraction', 'Cosine/coherence thresholds cannot exceed one.');
    end
    if ~isempty(cfg.surface_range_m)
        v = cfg.surface_range_m;
        if ~isnumeric(v) || ~isreal(v) || numel(v) ~= 2 || any(~isfinite(v)) || diff(v) <= 0
            error('IceMap:SurfaceRange', 'surface_range_m needs two finite increasing endpoints.');
        end
    end
    if ~isempty(cfg.psf_sigma_grid_bins)
        v = cfg.psf_sigma_grid_bins;
        if ~isnumeric(v) || ~isreal(v) || ~isvector(v) || any(~isfinite(v)) || any(v <= 0)
            error('IceMap:PSFGrid', 'PSF widths must be a finite positive vector.');
        end
    end
    if ~isempty(cfg.shallow_end_s) && (~isnumeric(cfg.shallow_end_s) || ~isreal(cfg.shallow_end_s) || ~isscalar(cfg.shallow_end_s) || ~isfinite(cfg.shallow_end_s))
        error('IceMap:ShallowSplit', 'shallow_end_s must be one calibrated finite time.');
    end
    if ~any(strcmp(cfg.deep_evidence_policy, {'error', 'unity'}))
        error('IceMap:EvidencePolicy', 'Deep evidence policy must be error or unity.');
    end
    if ~isfield(cfg, 'fingerprint_input_mode') || ~any(strcmp(cfg.fingerprint_input_mode, {'dual_band', 'fused'}))
        error('IceMap:FingerprintInput', 'Expected dual_band or fused fingerprint input.');
    end
    for name = {'cross_frequency_max_relative_residual', 'surface_discontinuity_m'}
        if isfield(cfg, name{1}) && ~isempty(cfg.(name{1}))
            value = cfg.(name{1});
            if ~isnumeric(value) || ~isreal(value) || ~isscalar(value) || ~isfinite(value) || value <= 0
                error('IceMap:DiagnosticThreshold', '%s must be a positive calibrated scalar or empty.', name{1});
            end
        end
    end
end
