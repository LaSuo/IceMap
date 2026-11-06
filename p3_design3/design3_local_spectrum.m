function out = design3_local_spectrum(low, high, time_s, cfg)
    % Windowed spectral morphology of high/low envelopes and entropy gating.
    s = prepare_fingerprint_inputs(low, high, time_s, cfg);
    p = s.options;
    [phase, concentration, phase_valid, phase_info] = robust_phase_fingerprint(s.analytic_high, cfg, p);
    mask = conv2(double(abs(phase) >= cfg.boundary_phase_threshold_rad & phase_valid), ones(1, 2 * cfg.boundary_dilation_bins + 1), 'same') > 0;
    [reliability, valid] = scattering_reliability(s.analytic_high, mask, cfg);
    [high_spectrum, high_valid] = local_spectral_attributes(s.high_amplitude, p);
    [low_spectrum, low_valid] = local_spectral_attributes(s.low_envelope, p);
    focus = max(0, 1 - high_spectrum.entropy).^p.spectral_entropy_weight;
    E = s.high_amplitude .* focus .* sqrt(reliability);
    valid = valid & high_valid;
    E(~valid) = 0;
    score = s.low_envelope .* max(0, 1 - low_spectrum.entropy).^p.spectral_entropy_weight;
    score(~low_valid) = 0;
    [threshold, threshold_valid] = local_mad_threshold(score, p);
    threshold_valid = threshold_valid & low_valid;
    [tau, bins, times, intervals, event_info] = build_event_fingerprint(score, s.low_envelope, s.time_s, cfg, threshold, threshold_valid);
    ratio = log((high_spectrum.power + p.spectral_power_floor) ./ (low_spectrum.power + p.spectral_power_floor));
    extras = struct('spectral_entropy', high_spectrum.entropy, 'spectral_flatness', high_spectrum.flatness, ...
                  'spectral_centroid', high_spectrum.centroid, 'spectral_bandwidth', high_spectrum.bandwidth, 'cross_band_log_energy_ratio', ratio);
    validity = struct('spectral_entropy', high_valid, 'spectral_flatness', high_valid, 'spectral_centroid', high_valid, ...
                    'spectral_bandwidth', high_valid, 'cross_band_log_energy_ratio', high_valid & low_valid);
    units = struct('spectral_entropy', '1', 'spectral_flatness', '1', 'spectral_centroid', 'cycles/bin', ...
                 'spectral_bandwidth', 'cycles/bin', 'cross_band_log_energy_ratio', '1');
    info = struct('phase_estimator', phase_info, 'events', event_info, 'window_bins', p.spectral_window_bins, ...
                'nfft', p.spectral_nfft, 'frequency_unit', 'cycles/bin', 'padding', 'reflect', 'scattering_convention', 'spectral_focus_times_amplitude_times_sqrt_coherence');
    out = assemble_fingerprint_output(s, phase, E, tau, concentration, mask, valid, reliability, bins, times, intervals, extras, validity, units, info);
end
