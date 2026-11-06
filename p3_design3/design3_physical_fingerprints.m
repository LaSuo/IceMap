function out = design3_physical_fingerprints(low, high, time_s, cfg)
    % Calibrated signal-processing method.
    % calibrated boundary times and layer-wise two-way propagation delays.
    [low, high, time_s] = validate_echoes(low, high, time_s);
    validate_processing_config(cfg);
    require_config(cfg, {'boundary_phase_threshold_rad', 'boundary_dilation_bins', ...
                        'tof_min_peak_distance_bins', 'tof_min_prominence'});
    assert(cfg.boundary_phase_threshold_rad > 0 && cfg.boundary_dilation_bins >= 0);
    assert(cfg.coherence_floor > 0 && cfg.coherence_floor <= 1);
    [T, ~, ~, N] = size(high);
    assert(T >= 2, 'Density fingerprints require at least two adjacent frames.');
    phase = zeros(T, N);
    phase_coherence = zeros(T, N);
    amplitude = reshape(sqrt(mean(mean(abs(high).^2, 2), 3)), T, N);
    for t = 1:T
        X = reshape(high(t, :, :, :), [], N);
        products = X(:, 2:end) .* conj(X(:, 1:end - 1));
        gradient = angle(sum(products, 1));
        coherence = abs(sum(products, 1)) ./ (sum(abs(products), 1) + eps);
        % Circular interpolation of the high-confidence slowly varying trend.
        reliable = coherence >= cfg.phase_confidence_min;
        if nnz(reliable) < 2
            error('IceMap:PhaseEvidence', 'Insufficient high-confidence phase samples in frame %d.', t);
        end
        idx = find(reliable);
        trend_angle = unwrap(gradient(idx));
        trend = interp1(idx, trend_angle, 1:(N - 1), 'linear', 'extrap');
        trend = movmean(trend, cfg.phase_trend_bins);
        residual = angle(exp(1i * (gradient - trend)));
        phase(t, :) = [residual residual(end)];
        phase_coherence(t, :) = [coherence coherence(end)];
    end
    boundary = abs(phase) >= cfg.boundary_phase_threshold_rad;
    dilation = ones(1, 2 * cfg.boundary_dilation_bins + 1);
    mask = conv2(double(boundary), dilation, 'same') > 0;
    % Coherence over channel, local depth and unmasked adjacent frames.
    coherence = zeros(T, N);
    count = zeros(T, N);
    for t = 1:T
        X = reshape(high(t, :, :, :), [], N);
        for j = max(1, t - cfg.coherence_frame_radius):min(T, t + cfg.coherence_frame_radius)
            if j == t
                continue
            end
            Y = reshape(high(j, :, :, :), [], N);
            valid = ~mask(t, :) & ~mask(j, :);
            cross = sum(X .* conj(Y), 1) .* valid;
            pX = sum(abs(X).^2, 1) .* valid;
            pY = sum(abs(Y).^2, 1) .* valid;
            z = movsum(cross, cfg.coherence_depth_bins);
            powerX = movsum(pX, cfg.coherence_depth_bins);
            powerY = movsum(pY, cfg.coherence_depth_bins);
            observed = movsum(double(valid), cfg.coherence_depth_bins) > 0 & powerX > 0 & powerY > 0;
            coh = min(1, max(0, abs(z).^2 ./ max(powerX .* powerY, realmin)));
            coherence(t, :) = coherence(t, :) + coh .* observed;
            count(t, :) = count(t, :) + observed;
        end
    end
    coherence = coherence ./ max(count, 1);
    valid_scatter = ~mask & count > 0;
    E = amplitude ./ max(coherence, cfg.coherence_floor);
    E(~valid_scatter) = 0; % masked specular samples cannot be density evidence
    % Delay fingerprints: dense event map plus explicit timestamps/intervals.
    tau = zeros(T, N);
    event_times = cell(T, 1);
    intervals = cell(T, 1);
    event_bins = cell(T, 1);
    for t = 1:T
        if isreal(low(t, :))
            envelope = abs(hilbert(low(t, :)));
        else
            envelope = abs(low(t, :));
        end
        envelope = movmean(envelope, cfg.tof_smooth_bins);
        [heights, loc] = findpeaks(envelope, 'MinPeakDistance', cfg.tof_min_peak_distance_bins, ...
                                'MinPeakProminence', cfg.tof_min_prominence);
        refined = zeros(size(loc));
        for j = 1:numel(loc)
            win = max(1, loc(j) - cfg.tof_centroid_halfwidth_bins):min(N, loc(j) + cfg.tof_centroid_halfwidth_bins);
            weights = envelope(win).^2;
            refined(j) = sum(win .* weights) / max(sum(weights), realmin);
            tau(t, :) = tau(t, :) + heights(j) * exp(-0.5 * (((1:N) - refined(j)) / cfg.tof_skeleton_sigma_bins).^2);
        end
        event_bins{t} = refined;
        event_times{t} = interp1(1:N, time_s, refined, 'linear');
        intervals{t} = diff(event_times{t});
    end
    out.phase = phase;
    out.E = E;
    out.tau = tau;
    out.time_s = time_s;
    out.phase_coherence = phase_coherence;
    out.boundary_mask = mask;
    out.scatter_valid = valid_scatter;
    out.scattering_coherence = coherence;
    out.tau_event_bins = event_bins;
    out.tau_events_s = event_times;
    out.tau_intervals_s = intervals;
    out.params = cfg;
end
