function [threshold, valid, background, noise] = local_mad_threshold(score, p)
    % Guarded training cells; robust engineering threshold, not a calibrated Pfa.
    [T, N] = size(score);
    threshold = zeros(T, N);
    background = threshold;
    noise = threshold;
    valid = false(T, N);
    for t = 1:T
        for n = 1:N
            left = max(1, n - p.noise_guard_bins - p.noise_training_bins):n - p.noise_guard_bins - 1;
            right = n + p.noise_guard_bins + 1:min(N, n + p.noise_guard_bins + p.noise_training_bins);
            indices = [left right];
            indices = indices(indices >= 1 & indices <= N);
            if numel(indices) < p.minimum_training_bins
                continue
            end
            values = score(t, indices);
            center = median(values);
            sigma = 1.4826 * median(abs(values - center));
            background(t, n) = center;
            noise(t, n) = sigma;
            threshold(t, n) = center + p.noise_mad_multiplier * sigma + p.absolute_noise_floor;
            valid(t, n) = true;
        end
    end
end
