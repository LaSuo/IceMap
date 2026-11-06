function out = depth_reflect_filter(data, kernel)
    % Reflect padding, including radii larger than the original support.
    N = size(data, 2);
    radius = (numel(kernel) - 1) / 2;
    if N < 2 || radius ~= round(radius) || any(~isfinite(kernel)) || ~isreal(kernel)
        error('IceMap:FeatureFilter', 'Finite odd real filter and at least two input bins required.');
    end
    period = 2 * (N - 1);
    indices = mod((1 - radius:N + radius) - 1, period);
    indices = 1 + min(indices, period - indices);
    out = conv2(data(:, indices), reshape(kernel, 1, []), 'valid');
end
