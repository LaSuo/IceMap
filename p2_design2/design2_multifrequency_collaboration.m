function out = design2_multifrequency_collaboration(low, high, time_s, cfg)
    % Calibrated signal-processing method.
    % linear (zero-padded) regularized deconvolution. No shape redistribution.
    [low, high, time_s] = validate_echoes(low, high, time_s);
    validate_processing_config(cfg);
    require_config(cfg, {'shallow_end_s', 'psf_sigma_grid_bins', ...
                        'lambda_attenuation', 'lambda_deconvolution'});
    assert(cfg.lambda_attenuation > 0 && cfg.lambda_deconvolution > 0);
    assert(all(isfinite(cfg.psf_sigma_grid_bins)) && all(cfg.psf_sigma_grid_bins > 0));
    assert(strcmp(cfg.psf_family, 'gaussian'), 'Only the explicitly documented Gaussian PSF family is implemented.');
    [T, R, C, N] = size(high);
    Ns = find(time_s <= cfg.shallow_end_s, 1, 'last');
    assert(~isempty(Ns) && Ns >= 3 && Ns < N, 'shallow_end_s must split the calibrated grid into nonempty regions.');
    D = (Ns + 1):N;
    Nd = numel(D);
    G = diff(speye(Nd), 1, 1);
    out.data_l = complex(zeros(T, N));
    out.data_h = complex(zeros(T, R, C, N));
    out.rL_bins = zeros(T, 1);
    out.attenuation_relative = ones(T, N);
    out.attenuation_kkt_residual = zeros(T, 1);
    out.kernel = cell(T, 1);
    out.attenuation_identifiable = true(T, 1);
    out.relative_fit_residual = zeros(T, 2);
    out.fit_residual_valid = false(T, 2);
    out.psf_grid_edge = false(T, 1);
    out.cross_frequency_mismatch = false(T, 1);
    start = tic;
    for t = 1:T
        yL = low(t, :).';
        yH = reshape(mean(mean(high(t, :, :, :), 2), 3), N, 1);
        if ~any(abs(yH(1:Ns)) > 0) || ~any(abs(yL(1:Ns)) > 0)
            error('IceMap:ShallowEvidence', 'Both bands need shallow evidence to identify the PSF in frame %d.', t);
        end
        best = Inf;
        for r = cfg.psf_sigma_grid_bins(:).'
            hw = min(N - 1, ceil(cfg.psf_support_sigmas * r));
            offsets = -hw:hw;
            h = exp(-0.5 * (offsets / r).^2);
            h = h / sum(h);
            if N * numel(h) > cfg.max_operator_nonzeros
                error('IceMap:OperatorBudget', 'PSF operator exceeds configured nonzero budget; reduce support/window or explicitly raise the budget.');
            end
            operator = spdiags(repmat(h, N, 1), offsets, N, N);
            residual = yL(1:Ns) - operator(1:Ns, 1:Ns) * yH(1:Ns);
            cost = real(residual' * residual); % specified formulation, no extra r^2 term
            if cost < best
                best = cost;
                RL = operator;
                best_r = r;
                best_h = h;
            end
        end
        out.rL_bins(t) = best_r;
        out.kernel{t} = best_h;
        out.psf_grid_edge(t) = best_r == min(cfg.psf_sigma_grid_bins) || best_r == max(cfg.psf_sigma_grid_bins);
        out.relative_fit_residual(t, 1) = sqrt(best / max(sum(abs(yL(1:Ns)).^2), realmin));
        out.fit_residual_valid(t, 1) = true;
        matchedH = RL(D, D) * yH(D);
        % Real nonnegative physical attenuation. specified formulation, complex observations:
        % min_a ||matchedH - diag(yL)*a||^2 + lambda ||G*a||^2, a>=0.
        if ~any(abs(yL(D)) > 0)
            if strcmp(cfg.deep_evidence_policy, 'error')
                error('IceMap:DeepEvidence', 'No low-frequency deep evidence in frame %d; relative attenuation is unidentifiable.', t);
            elseif strcmp(cfg.deep_evidence_policy, 'unity')
                a = ones(Nd, 1);
                residual_kkt = 0;
                out.attenuation_identifiable(t) = false;
            else
                error('IceMap:EvidencePolicy', 'Expected deep_evidence_policy error or unity.');
            end
        else
            Q = spdiags(abs(yL(D)).^2, 0, Nd, Nd) + cfg.lambda_attenuation * (G' * G);
            rhs = real(conj(yL(D)) .* matchedH);
            if nnz(Q) == 0
                a = ones(Nd, 1);
                residual_kkt = 0;
            else
                [a, residual_kkt] = nonnegative_quadratic(Q, rhs, cfg);
            end
        end
        out.attenuation_relative(t, D) = a;
        if any(abs(matchedH) > 0) && out.attenuation_identifiable(t)
            out.relative_fit_residual(t, 2) = norm(matchedH - a .* yL(D)) / max(norm(matchedH), realmin);
            out.fit_residual_valid(t, 2) = true;
        end
        if ~isempty(cfg.cross_frequency_max_relative_residual)
            out.cross_frequency_mismatch(t) = any(out.relative_fit_residual(t, out.fit_residual_valid(t, :)) > cfg.cross_frequency_max_relative_residual);
        end
        out.attenuation_kkt_residual(t) = residual_kkt;
        if residual_kkt > 10 * cfg.solver_tolerance
            error('IceMap:Convergence', 'specified formulation did not converge; increase attenuation_max_iter or inspect calibration.');
        end
        % Regularized linear inverse, same shallow sub-operator as specified formulation.
        RS = RL(1:Ns, 1:Ns);
        sharp = (RS' * RS + cfg.lambda_deconvolution * speye(Ns)) \ (RS' * yL(1:Ns));
        correctedL = yL;
        correctedL(1:Ns) = sharp;
        out.data_l(t, :) = correctedL.';
        gain = ones(N, 1);
        gain(D) = 1 ./ max(a, cfg.attenuation_floor);
        out.data_h(t, :, :, :) = high(t, :, :, :) .* reshape(gain, 1, 1, 1, N);
    end
    out.tpf = toc(start) / T;
    out.time_s = time_s;
    out.shallow_samples = Ns;
    out.params = cfg;
    out.fused = reshape(mean(mean(out.data_h, 2), 3), T, N);
    out.fused(:, D) = out.data_l(:, D);
    out.mismatch_threshold_calibrated = ~isempty(cfg.cross_frequency_max_relative_residual);
    out.translation_assumption = 'PSF and relative attenuation approximate systematic distortion; unmodeled frequency-dependent scattering remains in residuals';
end

function [a, kkt] = nonnegative_quadratic(Q, rhs, cfg)
    % Red/black exact coordinate minimization for the same tridiagonal QP.
    % Entries of one color are uncoupled; vectorization changes no objective.
    n = numel(rhs);
    d = full(diag(Q));
    d = max(d, realmin);
    lower = full(diag(Q, -1));
    upper = full(diag(Q, 1));
    a = max(real(Q \ rhs), 0);
    scale = max(1, norm(rhs, Inf));
    odd = (1:2:n).';
    even = (2:2:n).';
    for it = 0:cfg.attenuation_max_iter
        g = Q * a - rhs;
        violation = g;
        violation(a <= 0) = min(g(a <= 0), 0);
        kkt = norm(violation, Inf) / scale;
        if kkt <= cfg.solver_tolerance
            return
        end
        if it == cfg.attenuation_max_iter
            break
        end
        for color = 1:2
            if color == 1
                indices = odd;
            else
                indices = even;
            end
            neighbor = zeros(numel(indices), 1);
            left = indices > 1;
            right = indices < n;
            neighbor(left) = neighbor(left) + lower(indices(left) - 1) .* a(indices(left) - 1);
            neighbor(right) = neighbor(right) + upper(indices(right)) .* a(indices(right) + 1);
            a(indices) = max(0, (rhs(indices) - neighbor) ./ d(indices));
        end
    end
end
