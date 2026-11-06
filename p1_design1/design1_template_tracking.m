function out = design1_template_tracking(low, high, time_s, ins, cfg)
    % Complete method entry; shared code handles validated I/O and registration.
    s = prepare_surface_tracking(low, high, time_s, ins, cfg);
    p = s.options;
    T = s.T;
    N = s.N;
    dr = s.dr;
    ins = s.ins;
    env = s.envelope;
    search = s.search;
    loc = s.candidates;
    emission = s.emission;
    observed = s.observed;
    confidence = s.confidence;
    track = observed;
    diagnostic = struct();
    covariance = [];
    reference = env(1, search);
    reference = reference - mean(reference);
    track(1) = observed(1);
    lags = -floor(p.correlation_max_lag_bins):floor(p.correlation_max_lag_bins);
    diagnostic.correlation_rejected = false(T, 1);
    for t = 2:T
        scores = zeros(size(lags));
        for j = 1:numel(lags)
            a = interp1(1:N, env(t, :), search + lags(j), 'linear', 0);
            a = a - mean(a);
            scores(j) = real(reference * a') / max(norm(reference) * norm(a), eps);
        end
        [best, j] = max(scores);
        if best <= p.correlation_min_score
            track(t) = min(max(track(t - 1) + ins.z_delta_m(t) / dr, search(1)), search(end));
            confidence(t) = 0;
            diagnostic.correlation_rejected(t) = true;
            continue
        end
        track(t) = min(max(track(1) + lags(j), search(1)), search(end));
        confidence(t) = max(0, best);
        aligned = interp1(1:N, env(t, :), search + lags(j), 'linear', 0);
        aligned = aligned - mean(aligned);
        reference = (1 - p.template_update) * reference + p.template_update * aligned;
    end
    out = register_surface_track(s, track, confidence, diagnostic, covariance, false);
end
