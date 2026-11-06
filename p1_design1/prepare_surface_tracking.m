function s = prepare_surface_tracking(low, high, time_s, ins, cfg)
    [low, high, time_s] = validate_echoes(low, high, time_s);
    validate_processing_config(cfg);
    require_config(cfg, {'surface_range_m', 'spatial_sigma_m', 'spatial_epsilon_m'});
    p = cfg.methods.d1;
    T = size(low, 1);
    N = size(low, 2);
    validate_method_options(p);
    ins = validate_ins(ins, T);
    names = fieldnames(p);
    for i = 1:numel(names)
        value = p.(names{i});
        assert((isnumeric(value) || islogical(value)) && isscalar(value) && isreal(value) && isfinite(value), 'IceMap:ResearchOption', 'Invalid D1 option: %s', names{i});
    end
    assert(p.max_iter == round(p.max_iter) && p.correlation_max_lag_bins == round(p.correlation_max_lag_bins));
    assert(numel(cfg.surface_range_m) == 2 && all(isfinite(cfg.surface_range_m)) && diff(cfg.surface_range_m) > 0);
    assert(cfg.min_cos_incidence > 0 && cfg.min_cos_incidence <= 1 && isfinite(cfg.scattering_gamma) && cfg.scattering_gamma >= 0);
    if p.use_altitude_prior
        assert(strcmp(cfg.design1_version, 'v1'), 'Altitude prior is supported only by D1 v2.');
    end
    assert(p.candidate_count >= 1 && p.candidate_count == round(p.candidate_count));
    assert(p.kalman_process_variance_bins2 > 0 && p.kalman_measurement_variance_bins2 > 0 && p.initial_variance_bins2 > 0);
    assert(p.transition_sigma_bins > 0 && p.huber_delta_bins > 0 && p.trend_lambda >= 0 && p.max_iter >= 1 && p.tolerance > 0);
    assert(p.innovation_gate_sigma > 0 && p.max_jump_bins >= 0 && p.correlation_max_lag_bins >= 0);
    assert(p.template_update >= 0 && p.template_update <= 1 && cfg.spatial_sigma_m > 0 && cfg.spatial_epsilon_m > 0);
    assert(p.correlation_min_score >= 0 && p.correlation_min_score <= 1);
    for k = {'z_delta_m', 'x_m', 'y_m'}
        assert(isfield(ins, k{1}) && numel(ins.(k{1})) == T && all(isfinite(ins.(k{1})(:))), 'Invalid INS.');
        ins.(k{1}) = ins.(k{1})(:);
    end
    range = cfg.c0_m_s * time_s / 2;
    dr = mean(diff(range));
    search = find(range >= cfg.surface_range_m(1) & range <= cfg.surface_range_m(2));
    assert(numel(search) >= 3, 'Surface search interval requires at least three bins.');
    if strcmp(cfg.tracking_band, 'low')
        env = abs(low);
    elseif strcmp(cfg.tracking_band, 'high')
        env = reshape(sqrt(mean(mean(abs(high).^2, 2), 3)), T, N);
    else
        error('Research tracking_band must be low or high.');
    end
    valid = sum(env(:, search).^2, 2) > 0;
    if ~valid(1)
        error('IceMap:SurfaceInitialization', 'The first frame has no surface evidence; initialization cannot be fabricated.');
    end
    if ~all(valid) && ~p.allow_missing_surface_frames
        error('IceMap:SurfaceEvidence', 'Missing surface evidence; explicit gap handling is disabled.');
    end
    if p.max_missing_surface_frames < 1 || p.max_missing_surface_frames ~= round(p.max_missing_surface_frames)
        error('IceMap:GapOptions', 'Positive integer missing-frame limit required.');
    end
    loc = cell(T, 1);
    emission = cell(T, 1);
    observed = zeros(T, 1);
    confidence = zeros(T, 1);
    gap = 0;
    for t = 1:T
        if ~valid(t)
            gap = gap + 1;
            if gap > p.max_missing_surface_frames
                error('IceMap:SurfaceGap', 'Missing-surface run exceeds %d frames.', p.max_missing_surface_frames);
            end
            loc{t} = min(max(loc{t - 1} + ins.z_delta_m(t) / dr, search(1)), search(end));
            emission{t} = ones(size(loc{t})) * log(numel(loc{t}));
            observed(t) = loc{t}(1);
            confidence(t) = 0;
            continue
        else
            gap = 0;
        end
        a = env(t, search);
        [peaks, idx] = findpeaks(a, 'MinPeakDistance', cfg.surface_min_peak_distance_bins, 'MinPeakProminence', cfg.surface_candidate_prominence);
        if isempty(idx)
            [peaks, idx] = max(a);
        end
        [~, order] = sort(peaks, 'descend');
        order = order(1:min(numel(order), p.candidate_count));
        loc{t} = reshape(search(idx(order)), [], 1);
        power = reshape(peaks(order).^2, [], 1);
        emission{t} = -log(max(power / sum(power), realmin));
        observed(t) = loc{t}(1);
        confidence(t) = exp(-emission{t}(1));
    end

    s = struct('low', low, 'high', high, 'time_s', time_s, 'ins', ins, 'cfg', cfg, 'options', p, ...
             'T', T, 'N', N, 'envelope', env, 'search', search, 'dr', dr, 'candidates', {loc}, ...
             'emission', {emission}, 'observed', observed, 'confidence', confidence, 'evidence_valid', valid);
end
