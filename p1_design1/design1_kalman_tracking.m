function out = design1_kalman_tracking(low, high, time_s, ins, cfg)
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
    mu = observed(1);
    P = p.initial_variance_bins2;
    if p.use_altitude_prior
        require_config(cfg.scene.uav, {'altitude_agl_m'});
        mu = (cfg.scene.uav.altitude_agl_m - range(1)) / dr + 1;
    end
    for t = 1:T
        if t > 1
            mu = mu + ins.z_delta_m(t) / dr;
            P = P + p.kalman_process_variance_bins2;
        end
        if ~s.evidence_valid(t)
            track(t) = min(max(mu, search(1)), search(end));
            covariance(t) = P;
            confidence(t) = 0;
            continue
        end
        R = p.kalman_measurement_variance_bins2 ./ max(exp(-emission{t}), .01);
        cost = emission{t} + .5 * (loc{t} - mu).^2 ./ (P + R);
        [~, j] = min(cost);
        innovation = loc{t}(j) - mu;
        if abs(innovation) <= p.innovation_gate_sigma * sqrt(P + R(j))
            K = P / (P + R(j));
            mu = mu + K * innovation;
            P = (1 - K)^2 * P + K^2 * R(j);
            confidence(t) = exp(-emission{t}(j));
        else
            confidence(t) = 0;
        end
        mu = min(max(mu, search(1)), search(end));
        track(t) = mu;
        covariance(t) = P;
    end
    out = register_surface_track(s, track, confidence, diagnostic, covariance, false);
end
