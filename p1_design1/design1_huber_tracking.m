function out = design1_huber_tracking(low, high, time_s, ins, cfg)
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
    motion = [0; cumsum(ins.z_delta_m(2:end) / dr)];
    y = observed - motion;
    D = diff(speye(T), 2, 1);
    if T < 3
        D = sparse(0, T);
    end
    x = y;
    converged = false;
    for iteration = 1:p.max_iter
        residual = x - y;
        w = max(confidence, .01) .* min(1, p.huber_delta_bins ./ max(abs(residual), eps));
        next = (spdiags(w, 0, T, T) + p.trend_lambda * (D' * D)) \ (w .* y);
        if norm(next - x, Inf) <= p.tolerance * (1 + norm(x, Inf))
            converged = true;
            x = next;
            break
        end
        x = next;
    end
    track = min(max(x + motion, search(1)), search(end));
    diagnostic.iterations = iteration;
    diagnostic.converged = converged;
    if p.require_convergence && ~converged
        error('IceMap:TrendConvergence', 'D1 Huber IRLS did not converge.');
    end
    out = register_surface_track(s, track, confidence, diagnostic, covariance, true);
end
