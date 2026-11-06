function out = design1_viterbi_tracking(low, high, time_s, ins, cfg)
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
    costs = emission{1};
    back = cell(T, 1);
    for t = 2:T
        residual = loc{t}.' - loc{t - 1} - ins.z_delta_m(t) / dr;
        transition = .5 * (residual / p.transition_sigma_bins).^2;
        if p.max_jump_bins > 0
            transition(abs(residual) > p.max_jump_bins) = Inf;
        end
        [best, parent] = min(costs + transition, [], 1);
        costs = best.' + emission{t};
        back{t} = parent;
        assert(any(isfinite(costs)), 'IceMap:TrackPath', 'No feasible Viterbi path.');
    end
    [diagnostic.path_cost, j] = min(costs);
    for t = T:-1:1
        track(t) = loc{t}(j);
        confidence(t) = exp(-emission{t}(j));
        if t > 1
            j = back{t}(j);
        end
    end
    out = register_surface_track(s, track, confidence, diagnostic, covariance, true);
end
