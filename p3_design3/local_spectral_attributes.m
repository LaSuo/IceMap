function [attributes, valid] = local_spectral_attributes(envelope, p)
    % Local one-sided amplitude-envelope PSD; frequency is cycles per bin.
    [T, N] = size(envelope);
    W = p.spectral_window_bins;
    radius = (W - 1) / 2;
    if N * max(W, p.spectral_nfft) > p.maximum_window_elements
        error('IceMap:FeatureBudget', 'Local spectral window/FFT matrix exceeds budget.');
    end
    period = 2 * (N - 1);
    index = mod(((1:N).' + (-radius:radius)) - 1, period);
    index = 1 + min(index, period - index);
    window = .5 - .5 * cos(2 * pi * (0:W - 1) / max(W - 1, 1));
    window_power = sum(window.^2);
    frequency = (0:floor(p.spectral_nfft / 2)) / p.spectral_nfft;
    K = numel(frequency);
    names = {'entropy', 'flatness', 'centroid', 'bandwidth', 'power'};
    attributes = struct();
    for k = 1:numel(names)
        attributes.(names{k}) = zeros(T, N);
    end
    valid = false(T, N);
    for t = 1:T
        samples = reshape(envelope(t, index(:)), N, W);
        center = sum(samples .* window, 2) / sum(window);
        samples = (samples - center) .* window;
        transform = fft(samples, p.spectral_nfft, 2);
        power = abs(transform(:, 1:K)).^2 / (window_power * p.spectral_nfft);
        if mod(p.spectral_nfft, 2) == 0
            power(:, 2:end - 1) = 2 * power(:, 2:end - 1);
        else
            power(:, 2:end) = 2 * power(:, 2:end);
        end
        total = sum(power, 2);
        observed = total > p.spectral_power_floor;
        probability = power ./ max(total, p.spectral_power_floor);
        entropy = -sum(probability .* log(max(probability, realmin)), 2) / log(K);
        centroid = sum(probability .* frequency, 2);
        bandwidth = sqrt(max(0, sum(probability .* (frequency - centroid).^2, 2)));
        floorvalue = max(total / K * 1e-12, realmin);
        flatness = exp(mean(log(max(power, floorvalue)), 2)) ./ max(mean(power, 2), realmin);
        attributes.entropy(t, :) = min(1, max(0, entropy)).';
        attributes.flatness(t, :) = min(1, max(0, flatness)).';
        attributes.centroid(t, :) = centroid.';
        attributes.bandwidth(t, :) = bandwidth.';
        attributes.power(t, :) = total.';
        valid(t, :) = observed.';
    end
    for k = 1:numel(names)
        attributes.(names{k})(~valid) = 0;
    end
end
