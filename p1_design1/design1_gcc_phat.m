function out = design1_gcc_phat(low, high, time_s, ins, cfg)
    % Linear-lag spectral-whitened envelope registration and sub-bin peak fitting.
    s = prepare_surface_tracking(low, high, time_s, ins, cfg);
    p = s.options;
    if p.phat_exponent < 0 || p.phat_exponent > 1 || p.spectral_floor <= 0 || p.phat_peak_ratio <= 1
        error('IceMap:PHATOptions', 'Exponent in [0,1], floor >0 and peak ratio >1 required.');
    end
    T = s.T;
    W = numel(s.search);
    nfft = 2^nextpow2(2 * W - 1);
    track = s.observed;
    confidence = s.confidence;
    accepted = false(T, 1);
    accepted(1) = true;
    estimated_lag = zeros(T, 1);
    peak_ratio = nan(T, 1);
    previous = 1;
    for t = 2:T
        reference = s.envelope(previous, s.search);
        current = s.envelope(t, s.search);
        reference = reference - mean(reference);
        current = current - mean(current);
        predicted = track(previous) + sum(s.ins.z_delta_m(previous + 1:t)) / s.dr;
        if norm(reference) <= eps || norm(current) <= eps
            track(t) = min(max(predicted, s.search(1)), s.search(end));
            confidence(t) = 0;
            continue
        end
        window = .5 - .5 * cos(2 * pi * (0:W - 1) / max(W - 1, 1));
        cross = fft(current .* window, nfft) .* conj(fft(reference .* window, nfft));
        magnitude = abs(cross);
        threshold = max(p.spectral_floor * max(magnitude), realmin);
        transfer = cross ./ max(magnitude, threshold).^p.phat_exponent;
        transfer(magnitude < threshold) = 0;
        correlation = real(ifft(transfer));
        maxlag = min(W - 1, p.correlation_max_lag_bins);
        lags = -maxlag:maxlag;
        values = correlation(mod(lags, nfft) + 1);
        allowed = abs(lags - (predicted - track(previous))) <= p.innovation_gate_sigma * sqrt(p.kalman_measurement_variance_bins2);
        values(~allowed) = -Inf;
        if ~any(isfinite(values))
            track(t) = min(max(predicted, s.search(1)), s.search(end));
            confidence(t) = 0;
            continue
        end
        [peak, j] = max(values);
        other = values;
        other(max(1, j - 1):min(numel(values), j + 1)) = -Inf;
        competitor = max(other);
        if ~isfinite(competitor)
            competitor = 0;
        end
        peak_ratio(t) = max(peak, 0) / max(competitor, eps);
        if peak <= 0 || peak_ratio(t) < p.phat_peak_ratio
            track(t) = min(max(predicted, s.search(1)), s.search(end));
            confidence(t) = 0;
            continue
        end
        fractional = 0;
        if j > 1 && j < numel(values) && all(isfinite(values(j - 1:j + 1)))
            curvature = values(j - 1) - 2 * peak + values(j + 1);
            if curvature < 0
                fractional = max(-.5, min(.5, .5 * (values(j - 1) - values(j + 1)) / curvature));
            end
        end
        lag = lags(j) + fractional;
        track(t) = min(max(track(previous) + lag, s.search(1)), s.search(end));
        estimated_lag(t) = lag;
        accepted(t) = true;
        confidence(t) = 1 - 1 / peak_ratio(t);
        previous = t;
    end
    diagnostics = struct('accepted', accepted, 'relative_lag_bins', estimated_lag, 'peak_ratio', peak_ratio, 'linear_fft_length', nfft);
    out = register_surface_track(s, track, confidence, diagnostics, [], false);
end
