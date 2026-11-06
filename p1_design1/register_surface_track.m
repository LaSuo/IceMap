function out = register_surface_track(s, track, confidence, diagnostic, covariance, offline)
    low = s.low;
    high = s.high;
    time_s = s.time_s;
    ins = s.ins;
    cfg = s.cfg;
    p = s.options;
    T = s.T;
    N = s.N;
    dr = s.dr;
    if numel(track) ~= T || any(~isfinite(track)) || numel(confidence) ~= T || any(~isfinite(confidence))
        error('IceMap:TrackOutput', 'Tracker returned invalid length or nonfinite states.');
    end
    track = track(:);
    confidence = confidence(:);
    confidence(~s.evidence_valid) = 0;
    shift = track - track(1);
    registeredL = shift_depth(low, shift);
    registeredH = shift_depth(high, shift);
    zeta = [0; cumsum(diff(track) * dr - ins.z_delta_m(2:end))];
    theta = zeros(T, 1);
    for t = 1:T
        j = max(1, t - cfg.spatial_neighborhood_frames + 1):t;
        distance = hypot(ins.x_m(t) - ins.x_m(j), ins.y_m(t) - ins.y_m(j));
        distance = distance(:);
        dz = zeta(t) - zeta(j);
        w = exp(-distance.^2 / (2 * cfg.spatial_sigma_m^2));
        theta(t) = atan(sum(w .* dz(:) ./ max(distance, cfg.spatial_epsilon_m)) / sum(w));
    end
    gain = max(cos(theta), cfg.min_cos_incidence).^(-cfg.scattering_gamma);
    if p.attitude_correction
        for name = {'roll_rad', 'pitch_rad'}
            assert(isfield(ins, name{1}) && numel(ins.(name{1})) == T && all(isfinite(ins.(name{1})(:))), 'IceMap:Attitude', 'Missing or invalid per-frame attitude.');
            assert(all(abs(ins.(name{1})(:)) < pi / 2), 'Attitude correction assumes a downward antenna.');
        end
        gain = gain .* max(cos(ins.roll_rad(:)) .* cos(ins.pitch_rad(:)), cfg.min_cos_incidence).^(-cfg.scattering_gamma);
    end
    out = struct('data_l', registeredL .* sqrt(gain), 'data_h', registeredH .* reshape(sqrt(gain), T, 1, 1, 1), ...
               'geometry_low', registeredL, 'geometry_high', registeredH, 'time_s', time_s, ...
               'surface_track_bins', track, 'surface_sigma_bins', [], 'surface_confidence', confidence, ...
               'shift_bins', shift, 'delay_correction_s', 2 * shift * dr / cfg.c0_m_s, 'surface_zeta_m', zeta, ...
               'surface_inclination_rad', theta, 'power_gain', gain, 'params', cfg, 'diagnostics', diagnostic);
    out.offline = offline;
    out.surface_evidence_valid = s.evidence_valid;
    out.track_std_bins = [];
    if ~isempty(covariance)
        out.track_std_bins = sqrt(covariance);
    end
end

function out = shift_depth(data, shift)
    sz = size(data);
    N = sz(end);
    out = zeros(sz, 'like', data);
    for t = 1:sz(1)
        idx = repmat({':'}, 1, ndims(data));
        idx{1} = t;
        flat = reshape(data(idx{:}), [], N);
        a = interp1(1:N, flat.', (1:N) + shift(t), 'linear', 0).';
        out(idx{:}) = reshape(a, [1 sz(2:end)]);
    end
end
