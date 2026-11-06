function [tau, bins, times, intervals, diagnostic] = build_event_fingerprint(score, envelope, time_s, cfg, threshold, threshold_valid)
    [T, N] = size(score);
    p = cfg.methods.d3;
    if ~isequal(size(score), size(envelope), size(threshold), size(threshold_valid)) || ~isreal(score) || ~isreal(envelope) || ~isreal(threshold) || ...
        any(~isfinite(score(:))) || any(score(:) < 0) || any(~isfinite(envelope(:))) || any(envelope(:) < 0) || ...
        any(~isfinite(threshold(:))) || any(threshold(:) < 0) || any(~ismember(threshold_valid(:), [0 1]))
        error('IceMap:EventScore', 'Event score/envelope/threshold shape or value invalid.');
    end
    if ~isnumeric(time_s) || ~isreal(time_s) || ~isvector(time_s) || numel(time_s) ~= N || any(~isfinite(time_s)) || any(diff(time_s) <= 0)
        error('IceMap:EventTime', 'Event time axis must be a finite increasing N-vector.');
    end
    tau = zeros(T, N);
    bins = cell(T, 1);
    times = bins;
    intervals = bins;
    count = zeros(T, 1);
    ratios = cell(T, 1);
    for t = 1:T
        [height, loc] = findpeaks(score(t, :), 'MinPeakDistance', cfg.tof_min_peak_distance_bins, 'MinPeakProminence', cfg.tof_min_prominence);
        if p.include_endpoint_events
            if score(t, 1) > score(t, 2)
                loc = [1 loc];
                height = [score(t, 1) height];
            end
            if score(t, N) > score(t, N - 1)
                loc = [loc N];
                height = [height score(t, N)];
            end
        end
        accepted = threshold_valid(t, loc) & height > threshold(t, loc) & envelope(t, loc) > 0;
        loc = loc(accepted);
        height = height(accepted);
        [~, order] = sort(height, 'descend');
        chosen = [];
        for j = order
            if isempty(chosen) || all(abs(loc(j) - chosen) >= cfg.tof_min_peak_distance_bins)
                chosen(end + 1) = loc(j);
            end
        end
        loc = sort(chosen);
        refined = zeros(size(loc));
        quality = zeros(size(loc));
        for j = 1:numel(loc)
            radius = min(cfg.tof_centroid_halfwidth_bins, floor(cfg.tof_min_peak_distance_bins / 2));
            window = max(1, loc(j) - radius):min(N, loc(j) + radius);
            weight = envelope(t, window).^2;
            if sum(weight) <= realmin
                error('IceMap:EventEvidence', 'Selected event has no envelope support.');
            end
            refined(j) = sum(window .* weight) / sum(weight);
            amplitude = interp1(1:N, envelope(t, :), refined(j), 'linear');
            tau(t, :) = tau(t, :) + amplitude * exp(-.5 * (((1:N) - refined(j)) / cfg.tof_skeleton_sigma_bins).^2);
            quality(j) = score(t, loc(j)) / max(threshold(t, loc(j)), p.absolute_noise_floor);
        end
        if any(diff(refined) <= 0)
            error('IceMap:EventOrder', 'Sub-bin refinement produced unordered events.');
        end
        if isempty(refined) && ~p.allow_no_events
            error('IceMap:NoEvents', 'No supported events in frame %d.', t);
        end
        bins{t} = refined;
        times{t} = interp1(1:N, time_s, refined, 'linear');
        intervals{t} = diff(times{t});
        count(t) = numel(refined);
        ratios{t} = quality;
    end
    diagnostic = struct('event_count', count, 'score_to_threshold', {ratios});
end
