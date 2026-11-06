function out = design1_rts_smoothing(low, high, time_s, ins, cfg)
    % INS-controlled residual-velocity filter, probabilistic peak association,
    % missed detections, and fixed-interval covariance smoothing.
    s = prepare_surface_tracking(low, high, time_s, ins, cfg);
    p = s.options;
    T = s.T;
    if p.rts_process_acceleration_bins2 <= 0 || p.rts_initial_velocity_variance <= 0 || p.rts_clutter_mass <= 0 || p.rts_detection_probability <= 0 || p.rts_detection_probability >= 1
        error('IceMap:RTSOptions', 'Positive variances/clutter and detection probability in (0,1) required.');
    end
    H = [1 0];
    I = eye(2);
    transitions = repmat(eye(2), 1, 1, T);
    if isfield(s.ins, 'timestamp_s')
        steps = [1; diff(s.ins.timestamp_s)];
        time_basis = 'seconds';
    else
        steps = ones(T, 1);
        time_basis = 'frames';
    end
    filtered = zeros(2, T);
    predicted = zeros(2, T);
    Pf = zeros(2, 2, T);
    Pp = zeros(2, 2, T);
    x = [s.observed(1); 0];
    P = diag([p.initial_variance_bins2 p.rts_initial_velocity_variance]);
    confidence = zeros(T, 1);
    associations = cell(T, 1);
    for t = 1:T
        dt = steps(t);
        F = [1 dt; 0 1];
        transitions(:, :, t) = F;
        Q = p.rts_process_acceleration_bins2 * [dt^4 / 4 dt^3 / 2; dt^3 / 2 dt^2];
        if t > 1
            x = F * x + [s.ins.z_delta_m(t) / s.dr; 0];
            P = F * P * F' + Q;
        end
        predicted(:, t) = x;
        Pp(:, :, t) = P;
        R = p.kalman_measurement_variance_bins2;
        S = H * P * H' + R;
        innovations = s.candidates{t} - H * x;
        gated = abs(innovations) <= p.innovation_gate_sigma * sqrt(S);
        likelihood = exp(-.5 * innovations.^2 / S - s.emission{t}) / sqrt(2 * pi * S);
        likelihood(~gated) = 0;
        if ~s.evidence_valid(t)
            likelihood(:) = 0;
        end
        mass = p.rts_detection_probability * likelihood;
        missed = (1 - p.rts_detection_probability) * p.rts_clutter_mass;
        beta = mass / (missed + sum(mass));
        beta0 = missed / (missed + sum(mass));
        if any(beta > 0)
            innovation = sum(beta .* innovations);
            spread = sum(beta .* innovations.^2) - innovation^2;
            K = P * H' / S;
            Pc = (I - K * H) * P * (I - K * H)' + K * R * K';
            x = x + K * innovation;
            P = beta0 * P + (1 - beta0) * Pc + K * max(spread, 0) * K';
        end
        P = (P + P') / 2;
        if any(~isfinite(x)) || any(~isfinite(P(:))) || min(eig(P)) < -1e-8
            error('IceMap:FilterCovariance', 'Invalid covariance in frame %d.', t);
        end
        filtered(:, t) = x;
        Pf(:, :, t) = P;
        confidence(t) = 1 - beta0;
        associations{t} = struct('candidate_bins', s.candidates{t}, 'probabilities', beta, 'missed_probability', beta0);
    end
    smooth = filtered;
    Ps = Pf;
    for t = T - 1:-1:1
        denominator = Pp(:, :, t + 1);
        if rcond(denominator) < 1e-12
            error('IceMap:SmoothCovariance', 'Singular predicted covariance at frame %d.', t + 1);
        end
        A = (Pf(:, :, t) * transitions(:, :, t + 1)') / denominator;
        smooth(:, t) = filtered(:, t) + A * (smooth(:, t + 1) - predicted(:, t + 1));
        Ps(:, :, t) = Pf(:, :, t) + A * (Ps(:, :, t + 1) - denominator) * A';
        Ps(:, :, t) = (Ps(:, :, t) + Ps(:, :, t)') / 2;
        if any(~isfinite(Ps(:, :, t)), 'all') || min(eig(Ps(:, :, t))) < -1e-8
            error('IceMap:SmoothCovariance', 'Invalid smoothed covariance at frame %d.', t);
        end
    end
    track = min(max(smooth(1, :).', s.search(1)), s.search(end));
    covariance = max(reshape(Ps(1, 1, :), T, 1), 0);
    diagnostics = struct('filtered_states', filtered, 'smoothed_states', smooth, 'smoothed_covariance', Ps, 'associations', {associations});
    diagnostics.time_basis = time_basis;
    out = register_surface_track(s, track, confidence, diagnostics, covariance, true);
end
