function out = design3_analytic_attributes(low, high, time_s, cfg)
    % Analytic envelope/phase-step and complex nonlinear local energy attributes.
    s = prepare_fingerprint_inputs(low, high, time_s, cfg);
    p = s.options;
    [phase, concentration, phase_valid, phase_info] = robust_phase_fingerprint(s.analytic_high, cfg, p);
    mask = conv2(double(abs(phase) >= cfg.boundary_phase_threshold_rad & phase_valid), ones(1, 2 * cfg.boundary_dilation_bins + 1), 'same') > 0;
    [reliability, valid] = scattering_reliability(s.analytic_high, mask, cfg);
    amplitude = movmedian(s.high_amplitude, p.analytic_smooth_bins, 2);
    E = amplitude .* sqrt(reliability);
    valid = valid & amplitude > p.absolute_noise_floor;
    E(~valid) = 0;
    z = s.analytic_low;
    envelope = abs(z);
    floorvalue = max(max(envelope, [], 2) * p.amplitude_floor_fraction, p.absolute_noise_floor);
    product = z(:, 2:end) .* conj(z(:, 1:end - 1));
    step = angle(product);
    step_valid = abs(z(:, 2:end)) > floorvalue & abs(z(:, 1:end - 1)) > floorvalue & abs(step) <= p.maximum_relative_phase_step;
    phase_step = [step step(:, end)];
    instantaneous = phase_step / (2 * pi);
    instantaneous_valid = [step_valid step_valid(:, end)];
    energy = zeros(s.T, s.N);
    energy(:, 2:end - 1) = max(0, abs(z(:, 2:end - 1)).^2 - real(z(:, 1:end - 2) .* conj(z(:, 3:end))));
    energy_valid = false(s.T, s.N);
    energy_valid(:, 2:end - 1) = envelope(:, 1:end - 2) > floorvalue & envelope(:, 2:end - 1) > floorvalue & envelope(:, 3:end) > floorvalue;
    log_envelope = log(max(envelope, floorvalue));
    slope = [diff(log_envelope, 1, 2) zeros(s.T, 1)];
    slope_valid = [step_valid false(s.T, 1)];
    score = movmean(sqrt(energy), p.analytic_smooth_bins, 2);
    [threshold, threshold_valid] = local_mad_threshold(score, p);
    [tau, bins, times, intervals, event_info] = build_event_fingerprint(score, envelope, s.time_s, cfg, threshold, threshold_valid);
    extras = struct('analytic_envelope', envelope, 'analytic_phase_step', phase_step, 'analytic_instantaneous_frequency', instantaneous, ...
                  'analytic_nonlinear_energy', energy, 'log_envelope_slope', slope);
    validity = struct('analytic_envelope', envelope > floorvalue, 'analytic_phase_step', instantaneous_valid, ...
                    'analytic_instantaneous_frequency', instantaneous_valid, 'analytic_nonlinear_energy', energy_valid, 'log_envelope_slope', slope_valid);
    units = struct('analytic_envelope', 'low_amplitude', 'analytic_phase_step', 'rad/bin', 'analytic_instantaneous_frequency', 'cycles/bin', ...
                 'analytic_nonlinear_energy', 'low_amplitude^2', 'log_envelope_slope', '1/bin');
    info = struct('phase_estimator', phase_info, 'events', event_info, 'analytic_low_constructed', s.analytic_low_constructed, ...
                'analytic_high_constructed', s.analytic_high_constructed, 'phase_step_is_wrapped', true, ...
                'scattering_convention', 'spatial_median_amplitude_times_sqrt_coherence');
    out = assemble_fingerprint_output(s, phase, E, tau, concentration, mask, valid, reliability, bins, times, intervals, extras, validity, units, info);
end
