function out = design3_scale_space(low, high, time_s, cfg)
    % Scale-normalized Gaussian/Laplacian fingerprints with persistence voting.
    s = prepare_fingerprint_inputs(low, high, time_s, cfg);
    p = s.options;
    scales = p.scale_bins;
    S = numel(scales);
    if s.T * s.N * S > p.maximum_window_elements
        error('IceMap:FeatureBudget', 'Scale response stack exceeds the feature budget.');
    end
    if max(scales) > s.N / 2
        error('IceMap:FeatureScales', 'Gaussian scales must not exceed half the observed window.');
    end
    phases = zeros(s.T, s.N, S);
    confidences = phases;
    energies = phases;
    responses = phases;
    phase_valid = false(size(phases));
    phase_diagnostics = cell(S, 1);
    for k = 1:S
        sigma = scales(k);
        radius = min(s.N - 1, ceil(4 * sigma));
        axis = -radius:radius;
        if s.R * s.C * (s.N + 2 * radius) > p.maximum_window_elements
            error('IceMap:FeatureBudget', 'Scale-space padded channel window exceeds budget.');
        end
        gaussian = exp(-.5 * (axis / sigma).^2);
        gaussian = gaussian / sum(gaussian);
        laplacian = (axis.^2 / sigma^4 - 1 / sigma^2) .* gaussian;
        laplacian = laplacian - mean(laplacian);
        filtered = complex(zeros(size(s.analytic_high)));
        for t = 1:s.T
            X = reshape(s.analytic_high(t, :, :, :), [], s.N);
            Z = depth_reflect_filter(X, gaussian);
            filtered(t, :, :, :) = reshape(Z, 1, s.R, s.C, s.N);
            energies(t, :, k) = mean(abs(X - Z).^2, 1);
        end
        [phases(:, :, k), confidences(:, :, k), phase_valid(:, :, k), phase_diagnostics{k}] = robust_phase_fingerprint(filtered, cfg, p);
        responses(:, :, k) = max(0, -sigma^2 * depth_reflect_filter(s.low_envelope, laplacian));
    end
    weights = confidences .* phase_valid;
    phasor = sum(weights .* exp(1i * phases), 3);
    phase = angle(phasor);
    concentration = abs(phasor) ./ max(sum(weights, 3), eps);
    observed_phase = sum(weights, 3) > 0;
    phase(~observed_phase) = 0;
    mask = conv2(double(abs(phase) >= cfg.boundary_phase_threshold_rad & observed_phase), ones(1, 2 * cfg.boundary_dilation_bins + 1), 'same') > 0;
    [reliability, valid] = scattering_reliability(s.analytic_high, mask, cfg);
    energy = mean(energies, 3);
    E = sqrt(energy) .* sqrt(reliability);
    valid = valid & energy > p.absolute_noise_floor^2;
    E(~valid) = 0;
    peaks = max(responses, [], 2);
    votes = responses > max(p.absolute_noise_floor, p.scale_peak_fraction * peaks);
    persistence = sum(votes, 3) / S;
    [strongest, winner] = max(responses, [], 3);
    score = strongest .* persistence;
    score(sum(votes, 3) < p.minimum_scale_votes) = 0;
    [threshold, threshold_valid] = local_mad_threshold(score, p);
    [tau, bins, times, intervals, event_info] = build_event_fingerprint(score, s.low_envelope, s.time_s, cfg, threshold, threshold_valid);
    dominant = reshape(scales(winner(:)), s.T, s.N);
    extras = struct('multiscale_persistence', persistence, 'multiscale_band_energy', energy, 'dominant_scale_bins', dominant);
    validity = struct('multiscale_persistence', any(responses > p.absolute_noise_floor, 3), ...
                    'multiscale_band_energy', energy > p.absolute_noise_floor^2, 'dominant_scale_bins', strongest > p.absolute_noise_floor);
    units = struct('multiscale_persistence', '1', 'multiscale_band_energy', 'high_amplitude^2', 'dominant_scale_bins', 'bin');
    info = struct('scales_bins', scales, 'phase_estimator', {phase_diagnostics}, 'events', event_info, 'threshold', threshold, ...
                'scattering_convention', 'scale_band_rms_times_sqrt_coherence', 'padding', 'reflect');
    out = assemble_fingerprint_output(s, phase, E, tau, concentration, mask, valid, reliability, bins, times, intervals, extras, validity, units, info);
end
