function [phase, coherence, valid, info] = robust_phase_fingerprint(high, cfg, p)
    % Directional Huber IRLS on channel-normalized phase increments.
    [T, R, C, N] = size(high);
    phase = zeros(T, N);
    coherence = phase;
    valid = false(T, N);
    iterations = zeros(T, 1);
    converged = false(T, 1);
    for t = 1:T
        X = reshape(high(t, :, :, :), R * C, N);
        products = X(:, 2:end) .* conj(X(:, 1:end - 1));
        amplitude = abs(products);
        floorvalue = max(max(amplitude, [], 2) * p.amplitude_floor_fraction, realmin);
        observed = amplitude > floorvalue;
        unit = products ./ max(amplitude, floorvalue);
        unit(~observed) = 0;
        mu = angle(sum(unit, 1));
        weights = double(observed);
        for iteration = 1:p.phase_max_iter
            residual = angle(unit .* exp(-1i * mu));
            weights = double(observed) .* min(1, p.phase_huber_delta_rad ./ max(abs(residual), eps));
            next = angle(sum(weights .* unit, 1));
            change = abs(angle(exp(1i * (next - mu))));
            mu = next;
            if max(change) <= p.phase_tolerance
                converged(t) = true;
                break
            end
        end
        concentration = abs(sum(weights .* unit, 1)) ./ max(sum(weights, 1), eps);
        reliable = concentration >= cfg.phase_confidence_min & any(observed, 1);
        if nnz(reliable) < 2
            if p.strict_phase_evidence
                error('IceMap:PhaseEvidence', 'Alternative phase estimation lacks two valid bins in frame %d.', t);
            end
            trend = zeros(1, N - 1);
        else
            index = find(reliable);
            trend = interp1(index, unwrap(mu(index)), 1:N - 1, 'linear', 'extrap');
            trend = movmedian(trend, cfg.phase_trend_bins);
        end
        gradient = angle(exp(1i * (mu - trend)));
        gradient(~reliable) = 0;
        phase(t, :) = [gradient gradient(end)];
        coherence(t, :) = [concentration concentration(end)];
        valid(t, :) = [reliable reliable(end)];
        iterations(t) = iteration;
    end
    info = struct('iterations', iterations, 'converged', converged, 'method', 'directional_huber');
end
