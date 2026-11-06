function out = design3_robust_phasor(low, high, time_s, cfg)
    % Robust channel phase consensus, median scattering and guarded MAD events.
    s = prepare_fingerprint_inputs(low, high, time_s, cfg);
    p = s.options;
    [phase, concentration, phase_valid, phase_info] = robust_phase_fingerprint(s.analytic_high, cfg, p);
    mask = conv2(double(abs(phase) >= cfg.boundary_phase_threshold_rad & phase_valid), ones(1, 2 * cfg.boundary_dilation_bins + 1), 'same') > 0;
    [reliability, valid] = scattering_reliability(s.analytic_high, mask, cfg);
    amplitude = zeros(s.T, s.N);
    dispersion = amplitude;
    for t = 1:s.T
        values = abs(reshape(s.analytic_high(t, :, :, :), [], s.N));
        center = median(values, 1);
        amplitude(t, :) = center;
        dispersion(t, :) = 1.4826 * median(abs(values - center), 1) ./ max(center, p.absolute_noise_floor);
    end
    E = amplitude .* sqrt(reliability);
    valid = valid & amplitude > p.absolute_noise_floor;
    E(~valid) = 0;
    score = movmean(s.low_envelope, cfg.tof_smooth_bins, 2);
    [threshold, threshold_valid, background, noise] = local_mad_threshold(score, p);
    [tau, bins, times, intervals, event_info] = build_event_fingerprint(score, s.low_envelope, s.time_s, cfg, threshold, threshold_valid);
    extras = struct('channel_amplitude_dispersion', dispersion, 'phase_disagreement', 1 - concentration, ...
                  'echo_background', background, 'echo_noise_mad', noise);
    validity = struct('channel_amplitude_dispersion', amplitude > p.absolute_noise_floor, 'phase_disagreement', phase_valid, ...
                    'echo_background', threshold_valid, 'echo_noise_mad', threshold_valid);
    units = struct('channel_amplitude_dispersion', '1', 'phase_disagreement', '1', 'echo_background', 'low_amplitude', 'echo_noise_mad', 'low_amplitude');
    info = struct('phase_estimator', phase_info, 'events', event_info, 'threshold', threshold, 'threshold_valid', threshold_valid, 'scattering_convention', 'median_amplitude_times_sqrt_coherence');
    out = assemble_fingerprint_output(s, phase, E, tau, concentration, mask, valid, reliability, bins, times, intervals, extras, validity, units, info);
end
