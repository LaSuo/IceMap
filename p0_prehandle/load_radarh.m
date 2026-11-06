function [cfg, timestamps, radar_matrix, baseTime] = load_radarh(filepath)
    % Timestamp, then one antenna-id/data pair per receiver. No silent zero frames.
    fid = fopen(filepath, 'r');
    if fid < 0
        error('IceMap:RawOpen', 'Cannot open %s.', filepath);
    end
    closer = onCleanup(@()fclose(fid));
    cfg = read_radar_header(fid, 18);
    for key = {'RxNum', 'ChirpPerFrame', 'SamplesPerChirp'}
        k = key{1};
        if ~isfield(cfg, k) || ~isfinite(cfg.(k)) || cfg.(k) < 1 || cfg.(k) ~= round(cfg.(k))
            error('IceMap:RawHeader', 'Missing/invalid %s.', k);
        end
    end
    % This recording format encodes three enabled receivers by the value 7.
    cfg.RxNumRecorded = cfg.RxNum;
    if cfg.RxNum == 7
        cfg.RxNum = 3;
    end
    if cfg.RxNum > 3
        error('IceMap:ReceiverEncoding', 'Unsupported receiver count/mask; decode the acquisition format explicitly.');
    end
    R = cfg.RxNum;
    C = cfg.ChirpPerFrame;
    N = cfg.SamplesPerChirp;
    frames = {};
    absolute = [];
    state = 0;
    seen = false(1, R);
    ant = 0;
    stamp = 0;
    while true
        line = fgetl(fid);
        if ~ischar(line)
            break
        end
        if isempty(strtrim(line))
            continue
        end
        x = parse_radar_numbers(line);
        if state == 0
            if numel(x) ~= 1
                error('IceMap:Timestamp', 'Expected scalar timestamp.');
            end
            stamp = x;
            if ~isempty(absolute) && stamp <= absolute(end)
                error('IceMap:Timestamp', 'High-band timestamps must strictly increase in seconds.');
            end
            current = zeros(1, R, C, N);
            seen(:) = false;
            state = 1;
        elseif state == 1
            if numel(x) ~= 1 || x ~= round(x) || x < 0 || x >= R
                error('IceMap:AntennaId', 'Expected zero-based receiver id in 0..%d.', R - 1);
            end
            ant = x + 1;
            if seen(ant)
                error('IceMap:DuplicateReceiver', 'Receiver %d occurs twice within a frame.', ant - 1);
            end
            state = 2;
        elseif state == 2
            if numel(x) ~= C * N
                error('IceMap:FrameLength', 'High-band frame %d receiver %d: expected %d values, got %d.', numel(frames) + 1, ant - 1, C * N, numel(x));
            end
            current(1, ant, :, :) = reshape(x, N, C).';
            seen(ant) = true;
            if all(seen)
                frames{end + 1, 1} = current;
                absolute(end + 1) = stamp;
                state = 0; %#ok<AGROW>
            else
                state = 1;
            end
        end
    end
    if state ~= 0
        error('IceMap:IncompleteFrame', 'EOF in partial high-band frame; no zero padding is permitted.');
    end
    if isempty(frames)
        error('IceMap:EmptyRecording', 'No complete high-band frames.');
    end
    radar_matrix = cat(1, frames{:});
    baseTime = absolute(1);
    timestamps = absolute - baseTime;
    cfg.actual_frames = numel(frames);
    cfg.timestamps_unit = 's';
    cfg.synthetic_frames = 0;
end
