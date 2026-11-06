function [coherence, valid] = scattering_reliability(high, mask, cfg)
    % Bounded local temporal coherence; no invalid neighbor counts as evidence.
    [T, ~, ~, N] = size(high);
    total = zeros(T, N);
    count = zeros(T, N);
    for t = 1:T
        X = reshape(high(t, :, :, :), [], N);
        for j = max(1, t - cfg.coherence_frame_radius):min(T, t + cfg.coherence_frame_radius)
            if j == t
                continue
            end
            Y = reshape(high(j, :, :, :), [], N);
            observed = ~mask(t, :) & ~mask(j, :);
            cross = movsum(sum(X .* conj(Y), 1) .* observed, cfg.coherence_depth_bins);
            px = movsum(sum(abs(X).^2, 1) .* observed, cfg.coherence_depth_bins);
            py = movsum(sum(abs(Y).^2, 1) .* observed, cfg.coherence_depth_bins);
            has = px > 0 & py > 0 & movsum(double(observed), cfg.coherence_depth_bins) > 0;
            value = min(1, max(0, abs(cross).^2 ./ max(px .* py, realmin)));
            total(t, :) = total(t, :) + value .* has;
            count(t, :) = count(t, :) + has;
        end
    end
    coherence = total ./ max(count, 1);
    valid = ~mask & count > 0;
end
