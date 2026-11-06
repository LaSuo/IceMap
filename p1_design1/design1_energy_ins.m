function out = design1_energy_ins(low, high, time_s, ins, cfg)
    % Energy likelihood, INS prior, shared registration and slope correction.
    % INS z_delta_m(k) is downward-positive radar-to-surface displacement
    % predicted between frames k-1 and k. x_m,y_m are horizontal positions.
    [low, high, time_s] = validate_echoes(low, high, time_s);
    validate_processing_config(cfg);
    require_config(cfg, {'surface_range_m', 'spatial_sigma_m', 'spatial_epsilon_m'});
    assert(cfg.spatial_sigma_m > 0 && cfg.spatial_epsilon_m > 0);
    assert(numel(cfg.surface_range_m) == 2 && diff(cfg.surface_range_m) > 0);
    T = size(low, 1);
    N = size(low, 2);
    ins = validate_ins(ins, T);
    for name = {'z_delta_m', 'x_m', 'y_m'}
        assert(isfield(ins, name{1}) && numel(ins.(name{1})) == T, ['Missing INS ' name{1}]);
        assert(all(isfinite(ins.(name{1})(:))), 'Nonfinite INS.');
    end
    range_m = cfg.c0_m_s * time_s / 2;
    dr = mean(diff(range_m));
    search = find(range_m >= cfg.surface_range_m(1) & range_m <= cfg.surface_range_m(2));
    assert(~isempty(search), 'Surface search interval lies outside time_s.');
    if strcmp(cfg.tracking_band, 'low')
        envelope = abs(low);
    elseif strcmp(cfg.tracking_band, 'high')
        envelope = reshape(sqrt(mean(mean(abs(high).^2, 2), 3)), T, N);
    else
        error('tracking_band must be low or high.');
    end
    track = zeros(T, 1);
    sigma = zeros(T, 1);
    confidence = zeros(T, 1);
    for k = 1:T
        % Bin 1 is initialization: 160 bins; bin 50 and later: 80 bins.
        fraction = min((k - 1) / max(cfg.surface_sigma_anneal_frames - 1, 1), 1);
        sigma(k) = cfg.surface_sigma_initial_bins + fraction * ...
            (cfg.surface_sigma_final_bins - cfg.surface_sigma_initial_bins);
        a = envelope(k, search);
        if ~any(a > 0)
            error('IceMap:SurfaceEvidence', 'No surface evidence in frame %d.', k);
        end
        [pks, loc] = findpeaks(a, 'MinPeakDistance', cfg.surface_min_peak_distance_bins, ...
                            'MinPeakProminence', cfg.surface_candidate_prominence);
        if isempty(loc)
            [pks, loc] = max(a);
        end
        candidates = search(loc);
        power = pks(:).^2;
        eta = power / max(sum(power), realmin);
        if k == 1
            kappa = ones(size(eta));
        else
            prediction = track(k - 1) + ins.z_delta_m(k) / dr;
            kappa = exp(-0.5 * ((candidates(:) - prediction) / sigma(k)).^2);
        end
        [~, best] = max(eta .* kappa);
        track(k) = candidates(best);
        confidence(k) = eta(best) * kappa(best);
    end
    % One shared reference preserves alignment between both radar bands.
    shift_bins = track - track(1);
    registered_low = shift_depth(low, shift_bins);
    registered_high = shift_depth(high, shift_bins);
    % The reference zeta sign, not an undocumented world-elevation convention.
    zeta = zeros(T, 1);
    for k = 2:T
        zeta(k) = zeta(k - 1) + (track(k) - track(k - 1)) * dr - ins.z_delta_m(k);
    end
    theta = zeros(T, 1);
    for k = 1:T
        neighbors = max(1, k - cfg.spatial_neighborhood_frames + 1):k;
        o = hypot(ins.x_m(k) - ins.x_m(neighbors), ins.y_m(k) - ins.y_m(neighbors));
        o = o(:);
        dz = zeta(k) - zeta(neighbors);
        dz = dz(:);
        w = exp(-o.^2 / (2 * cfg.spatial_sigma_m^2));
        theta(k) = atan(sum(w .* dz ./ max(o, cfg.spatial_epsilon_m)) / sum(w));
    end
    % E' = E/cos(theta)^gamma; complex amplitude gets sqrt(power gain).
    cos_incidence = max(cos(theta), cfg.min_cos_incidence);
    power_gain = cos_incidence.^(-cfg.scattering_gamma);
    out.data_l = registered_low .* sqrt(power_gain);
    out.data_h = registered_high .* reshape(sqrt(power_gain), T, 1, 1, 1);
    out.time_s = time_s;
    out.surface_track_bins = track;
    out.surface_sigma_bins = sigma;
    out.surface_confidence = confidence;
    out.shift_bins = shift_bins;
    out.delay_correction_s = 2 * shift_bins * dr / cfg.c0_m_s;
    out.surface_zeta_m = zeta;
    out.surface_jump_m = [0; abs(diff(zeta))];
    out.discontinuity_evaluated = ~isempty(cfg.surface_discontinuity_m);
    out.surface_discontinuity = false(T, 1);
    if out.discontinuity_evaluated
        out.surface_discontinuity = out.surface_jump_m > cfg.surface_discontinuity_m;
    end
    out.crack_detection_validated = false; % jump diagnostics are not fracture identification
    out.surface_inclination_rad = theta;
    out.power_gain = power_gain;
    out.geometry_low = registered_low;
    out.geometry_high = registered_high;
    out.params = cfg;
end

function out = shift_depth(data, shift)
    sz = size(data);
    T = sz(1);
    N = sz(end);
    out = zeros(sz, 'like', data);
    for k = 1:T
        idx = repmat({':'}, 1, ndims(data));
        idx{1} = k;
        flat = reshape(data(idx{:}), [], N);
        shifted = interp1(1:N, flat.', (1:N) + shift(k), 'linear', 0).';
        out(idx{:}) = reshape(shifted, [1 sz(2:end)]);
    end
end
