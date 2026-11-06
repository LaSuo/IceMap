function out = design2_snr_fusion(low, high, time_s, cfg)
    [out, p] = prepare_collaboration(low, high, time_s, cfg);
    T = size(low, 1);
    Ns = out.shallow_samples;
    N = size(low, 2);
    start = tic;
    out.branch_diagnostics = cell(T, 1);
    for t = 1:T
        y = low(t, 1:Ns).';
        h = out.kernel{t};
        offsets = -(numel(h) - 1) / 2:(numel(h) - 1) / 2;
        [H, D] = shallow_operator(h, N, Ns);
        L = out.data_l(t, :);
        Z = reshape(mean(mean(out.data_h(t, :, :, :), 2), 3), 1, N);
        % Difference-based local noise estimate: engineering, not calibrated SNR.
        noiseL = max(movmean(abs([diff(L) 0]).^2, p.snr_window_bins) / 2, p.noise_floor);
        noiseH = max(movmean(abs([diff(Z) 0]).^2, p.snr_window_bins) / 2, p.noise_floor);
        snrL = max(movmean(abs(L).^2, p.snr_window_bins) - noiseL, 0) ./ noiseL;
        snrH = max(movmean(abs(Z).^2, p.snr_window_bins) - noiseH, 0) ./ noiseH;
        penetration = exp(-max(0, (1:N) - Ns) / p.penetration_decay_bins);
        weight = snrH .* penetration ./ max(snrL + snrH .* penetration, p.noise_floor);
        out.data_l(t, :) = weight .* Z + (1 - weight) .* L;
        out.branch_diagnostics{t} = struct('high_weight', weight, 'snr_low_proxy', snrL, 'snr_high_proxy', snrH);

    end
    out.fused = reshape(mean(mean(out.data_h, 2), 3), T, N);
    out.fused(:, Ns + 1:end) = out.data_l(:, Ns + 1:end);
    out.fused = out.data_l;
    out.tpf = out.tpf + toc(start) / T;
end
