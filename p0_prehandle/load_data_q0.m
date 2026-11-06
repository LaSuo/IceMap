function [radarl_data, radarh_data, label] = load_data_q0(path_name, file_name, options)
    % Raw adapter: timestamps in seconds and explicit range calibration.
    if nargin < 3
        options = import_initials();
    end
    required = {'low_sample_rate_hz', 'sync_tolerance_s', 'high_range_bin_m'};
    for k = 1:numel(required)
        if ~isfield(options, required{k})
            error('IceMap:ImportCalibration', 'Missing options.%s.', required{k});
        end
    end
    if ~isfield(options, 'data_root')
        options.data_root = 'data_q0';
    end
    if ~isfield(options, 'high_clock_offset_s')
        options.high_clock_offset_s = 0;
    end
    if ~isfield(options, 'output_range_step_m')
        options.output_range_step_m = .001;
    end
    for key = {'high_range_bin_m', 'output_range_step_m'}
        v = options.(key{1});
        if ~isscalar(v) || ~isfinite(v) || v <= 0
            error('IceMap:ImportCalibration', '%s must be positive in meters.', key{1});
        end
    end
    base = fullfile(options.data_root, path_name);
    [lc, lt, L, lb] = load_radarl(fullfile(base, 'data', 'radarl', string(file_name) + ".txt"), 1, options.low_sample_rate_hz);
    [hc, ht, H, hb] = load_radarh(fullfile(base, 'data', 'radarh', string(file_name) + ".txt"));
    [li, hi, sync] = synchronize_radar_frames(lb + lt, hb + ht, options.sync_tolerance_s, options.high_clock_offset_s);
    L = reshape(L(li, 1, :), numel(li), []);
    H = H(hi, :, :, :);
    ln = size(L, 2);
    laxis = lc.AreaStart + (0:ln - 1) * lc.DisResolution;
    lout = laxis(1):options.output_range_step_m:laxis(end);
    if numel(lout) < 2
        error('IceMap:ImportGrid', 'Low-band range grid too short.');
    end
    L = interp1(laxis, L.', lout, 'linear').';
    N = hc.SamplesPerChirp;
    half = floor(N / 2);
    if half < 2
        error('IceMap:ImportGrid', 'Need at least four ADC samples per chirp.');
    end
    window = .5 - .5 * cos(2 * pi * (0:N - 1) / max(1, N - 1));
    H = fft(H .* reshape(window, 1, 1, 1, N), N, 4);
    H = H(:, :, :, 1:half);
    haxis = (0:half - 1) * options.high_range_bin_m;
    hout = 0:options.output_range_step_m:haxis(end);
    if numel(hout) < 2
        error('IceMap:ImportGrid', 'High-band range grid too short.');
    end
    shape = [size(H, 1), size(H, 2), size(H, 3), size(H, 4)];
    flat = reshape(H, [], half);
    flat = interp1(haxis, flat.', hout, 'linear').';
    H = reshape(flat, numel(hi), shape(2), shape(3), numel(hout));
    radarl_data = struct('dist_axis', lout, 'sync_data', L, 'sync', sync, 'config', lc, 'import_options', options);
    radarh_data = struct('dist_axis', hout, 'sync_data', H, 'sync', sync, 'config', hc, 'import_options', options);
    label_file = fullfile(base, 'label', string(file_name) + ".mat");
    if isfile(label_file)
        label = load(label_file);
    else
        label = struct();
    end
    % Frame-dependent labels remain in their source coordinates; caller registers
    % them explicitly using sync.low_indices/high_indices. No synthetic padding.
end
