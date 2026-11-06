function [il, ih, metadata] = synchronize_radar_frames(tl, th, tolerance_s, high_clock_offset_s)
    % Monotone nearest-neighbor association; each observed frame used at most once.
    if nargin < 4
        high_clock_offset_s = 0;
    end
    if ~isscalar(tolerance_s) || ~isfinite(tolerance_s) || tolerance_s < 0 || ~isscalar(high_clock_offset_s) || ~isfinite(high_clock_offset_s)
        error('IceMap:SyncConfig', 'Supply nonnegative finite tolerance and finite measured clock offset, in seconds.');
    end
    tl = tl(:);
    th = th(:) + high_clock_offset_s;
    if isempty(tl) || isempty(th) || any(~isfinite([tl; th])) || any(diff(tl) <= 0) || any(diff(th) <= 0)
        error('IceMap:Timestamp', 'Finite strictly increasing acquisition timestamps required.');
    end
    if max(tl(1), th(1)) > min(tl(end), th(end))
        error('IceMap:NoTimeOverlap', 'No temporal overlap; automatic clock rewriting is forbidden.');
    end
    il = [];
    ih = [];
    i = 1;
    j = 1;
    while i <= numel(tl) && j <= numel(th)
        delta = tl(i) - th(j);
        if abs(delta) <= tolerance_s
            if j < numel(th) && abs(tl(i) - th(j + 1)) < abs(delta)
                j = j + 1;
                continue
            end
            if i < numel(tl) && abs(tl(i + 1) - th(j)) < abs(delta)
                i = i + 1;
                continue
            end
            il(end + 1) = i;
            ih(end + 1) = j;
            i = i + 1;
            j = j + 1; %#ok<AGROW>
        elseif delta < 0
            i = i + 1;
        else
            j = j + 1;
        end
    end
    if isempty(il)
        error('IceMap:NoMatchedFrames', 'No observed pairs satisfy synchronization tolerance.');
    end
    metadata = struct('low_indices', il, 'high_indices', ih, 'low_time_s', tl(il), ...
                    'high_time_s', th(ih), 'time_error_s', tl(il) - th(ih), 'tolerance_s', tolerance_s, ...
                    'high_clock_offset_s', high_clock_offset_s, 'dropped_low', numel(tl) - numel(il), ...
                    'dropped_high', numel(th) - numel(ih), 'synthetic_frames', 0);
end
