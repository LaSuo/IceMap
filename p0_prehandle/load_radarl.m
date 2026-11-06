function [cfg, timestamps, radar_matrix, baseTime] = load_radarl(filepath, isBaseband, sample_rate_hz)
    % Read complete observed frames only; nominal FPS/runtime never create data.
    if nargin < 3
        sample_rate_hz = [];
    end
    if ~isscalar(isBaseband) || ~ismember(isBaseband, [0 1])
        error('IceMap:Baseband', 'isBaseband must be 0 or 1.');
    end
    fid = fopen(filepath, 'r');
    if fid < 0
        error('IceMap:RawOpen', 'Cannot open %s.', filepath);
    end
    closer = onCleanup(@()fclose(fid));
    cfg = read_radar_header(fid, 10);
    values = {};
    absolute = [];
    pending = [];
    width = [];
    while true
        line = fgetl(fid);
        if ~ischar(line)
            break
        end
        if isempty(strtrim(line))
            continue
        end
        x = parse_radar_numbers(line);
        if isempty(pending)
            if numel(x) ~= 1
                error('IceMap:Timestamp', 'Expected one timestamp before each low-band frame.');
            end
            pending = x;
        else
            if isempty(width)
                width = numel(x);
            end
            if numel(x) ~= width
                error('IceMap:FrameLength', 'Low-band frame %d: inconsistent bin count.', numel(values) + 1);
            end
            if ~isempty(absolute) && pending <= absolute(end)
                error('IceMap:Timestamp', 'Timestamps must be strictly increasing in seconds.');
            end
            values{end + 1, 1} = x;
            absolute(end + 1) = pending;
            pending = []; %#ok<AGROW>
        end
    end
    if ~isempty(pending)
        error('IceMap:IncompleteFrame', 'EOF after low-band timestamp without samples.');
    end
    if isempty(values)
        error('IceMap:EmptyRecording', 'No complete low-band frame.');
    end
    matrix = vertcat(values{:});
    baseTime = absolute(1);
    timestamps = absolute - baseTime;
    if isBaseband
        if mod(width, 2)
            error('IceMap:FrameLength', 'Baseband frame needs equal real/imaginary halves.');
        end
        if isempty(sample_rate_hz) && isfield(cfg, 'SampleRate')
            sample_rate_hz = cfg.SampleRate;
        end
        if isfield(cfg, 'SampleRate') && ~isempty(sample_rate_hz) && abs(cfg.SampleRate - sample_rate_hz) > 1e-6 * max(abs([cfg.SampleRate sample_rate_hz]))
            error('IceMap:SampleRateConflict', 'Supplied/initial ADC sample rate conflicts with the measured file header.');
        end
        if isempty(sample_rate_hz) || ~isscalar(sample_rate_hz) || ~isfinite(sample_rate_hz) || sample_rate_hz <= 0
            error('IceMap:SampleRate', 'Supply calibrated low-band ADC sample_rate_hz or header SampleRate in Hz.');
        end
        matrix = matrix(:, 1:width / 2) + 1i * matrix(:, width / 2 + 1:end);
        cfg.SampleRate = sample_rate_hz;
        cfg.DisResolution = 299792458 / (2 * sample_rate_hz);
        if ~isfield(cfg, 'AreaStart')
            cfg.AreaStart = 0;
        end
        cfg.AreaEnd = cfg.AreaStart + cfg.DisResolution * size(matrix, 2);
    end
    radar_matrix = reshape(matrix, size(matrix, 1), 1, size(matrix, 2));
    cfg.actual_frames = size(matrix, 1);
    cfg.timestamps_unit = 's';
    cfg.synthetic_frames = 0;
end
