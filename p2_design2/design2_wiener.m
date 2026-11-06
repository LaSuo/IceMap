function out = design2_wiener(low, high, time_s, cfg)
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
        nfft = 2^nextpow2(Ns + numel(h) - 1);
        kernel = zeros(nfft, 1);
        kernel(mod(offsets, nfft) + 1) = h;
        transfer = fft(kernel);
        padded = zeros(nfft, 1);
        padded(1:Ns) = y;
        restored = ifft(conj(transfer) .* fft(padded) ./ (abs(transfer).^2 + p.wiener_nsr));
        x = restored(1:Ns);
        out.data_l(t, 1:Ns) = x.';
    end
    out.fused = reshape(mean(mean(out.data_h, 2), 3), T, N);
    out.fused(:, Ns + 1:end) = out.data_l(:, Ns + 1:end);
    out.tpf = out.tpf + toc(start) / T;
end
