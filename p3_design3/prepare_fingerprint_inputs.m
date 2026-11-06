function s = prepare_fingerprint_inputs(low, high, time_s, cfg)
    [low, high, time_s] = validate_echoes(low, high, time_s);
    validate_processing_config(cfg);
    require_config(cfg, {'boundary_phase_threshold_rad', 'boundary_dilation_bins', 'tof_min_peak_distance_bins', 'tof_min_prominence'});
    [T, R, C, N] = size(high);
    if T < 2
        error('IceMap:FingerprintFrames', 'At least two frames are required for temporal scattering reliability.');
    end
    if N < 8
        error('IceMap:FingerprintBins', 'Alternative feature methods require at least eight bins.');
    end
    p = cfg.methods.d3;
    names = fieldnames(p);
    for i = 1:numel(names)
        k = names{i};
        v = p.(k);
        if strcmp(k, 'scale_bins')
            if ~isnumeric(v) || ~isreal(v) || ~isvector(v) || isempty(v) || any(~isfinite(v)) || any(v <= 0) || any(diff(v) <= 0)
                error('IceMap:FeatureScales', 'scale_bins must be a strictly increasing positive finite vector.');
            end
        elseif ~(isnumeric(v) || islogical(v)) || ~isreal(v) || ~isscalar(v) || ~isfinite(v)
            error('IceMap:FeatureOption', 'Feature option %s must be a finite real scalar.', k);
        end
    end
    positive = {'phase_huber_delta_rad', 'phase_tolerance', 'amplitude_floor_fraction', 'absolute_noise_floor', ...
              'spectral_power_floor', 'maximum_relative_phase_step'};
    for i = 1:numel(positive)
        if p.(positive{i}) <= 0
            error('IceMap:FeatureOption', '%s must be positive.', positive{i});
        end
    end
    integers = {'phase_max_iter', 'noise_training_bins', 'minimum_training_bins', 'minimum_scale_votes', ...
              'spectral_window_bins', 'spectral_nfft', 'analytic_smooth_bins', 'maximum_window_elements'};
    for i = 1:numel(integers)
        v = p.(integers{i});
        if v < 1 || v ~= round(v)
            error('IceMap:FeatureOption', '%s must be a positive integer.', integers{i});
        end
    end
    if p.noise_guard_bins < 0 || p.noise_guard_bins ~= round(p.noise_guard_bins) || p.noise_mad_multiplier < 0 || p.scale_peak_fraction <= 0 || p.scale_peak_fraction > 1 || p.minimum_scale_votes > numel(p.scale_bins) || p.spectral_entropy_weight < 0
        error('IceMap:FeatureOption', 'Invalid guard, noise multiplier, scale vote, or entropy parameter.');
    end
    if p.amplitude_floor_fraction > 1 || p.maximum_relative_phase_step > pi || p.minimum_training_bins > 2 * p.noise_training_bins
        error('IceMap:FeatureOption', 'Amplitude fraction <=1, phase-step <=pi and achievable training-cell count required.');
    end
    if R * C * N > p.maximum_window_elements
        error('IceMap:FeatureBudget', 'Per-frame channel work plane exceeds the feature budget.');
    end
    if mod(p.spectral_window_bins, 2) ~= 1 || p.spectral_window_bins < 3 || p.spectral_nfft < p.spectral_window_bins
        error('IceMap:FeatureFFT', 'Odd spectral window >=3 and nfft >= window required.');
    end
    for k = {'strict_phase_evidence', 'allow_no_events', 'include_endpoint_events'}
        if ~any(p.(k{1}) == [0 1])
            error('IceMap:FeatureOption', '%s must be Boolean.', k{1});
        end
    end
    analytic_low = low;
    analytic_high = high;
    if isreal(low)
        analytic_low = hilbert(low.').';
    end
    if isreal(high)
        for t = 1:T
            X = reshape(high(t, :, :, :), R * C, N);
            analytic_high(t, :, :, :) = reshape(hilbert(X.').', 1, R, C, N);
        end
    end
    s = struct('low', low, 'high', high, 'analytic_low', analytic_low, 'analytic_high', analytic_high, ...
             'low_envelope', abs(analytic_low), 'high_amplitude', reshape(sqrt(mean(mean(abs(analytic_high).^2, 2), 3)), T, N), ...
             'time_s', time_s, 'T', T, 'R', R, 'C', C, 'N', N, 'cfg', cfg, 'options', p, ...
             'analytic_low_constructed', isreal(low), 'analytic_high_constructed', isreal(high));
end
