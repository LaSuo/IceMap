function out = design2_lowrank_denoising(low, high, time_s, cfg)
    % Joint complex low-rank + sparse corruption decomposition before PSF fitting.
    % Removes sparse outliers, so isolated true reflectors can also be suppressed.
    [low, high, time_s] = validate_echoes(low, high, time_s);
    p = cfg.methods.d2;
    validate_method_options(p);
    [T, R, C, N] = size(high);
    if p.rpca_max_elements < 1 || p.rpca_max_elements ~= round(p.rpca_max_elements) || T * (1 + R * C) * N > p.rpca_max_elements
        error('IceMap:LowRankBudget', 'Joint matrix exceeds the explicit element budget.');
    end
    if T < 2
        error('IceMap:LowRankEvidence', 'At least two frames required for a low-rank prior.');
    end
    if p.rpca_max_iter < 1 || p.rpca_max_iter ~= round(p.rpca_max_iter) || p.rpca_tolerance <= 0 || p.rpca_lambda < 0 || p.rpca_penalty_growth <= 1 || p.rpca_penalty_cap <= 0
        error('IceMap:RPCAOptions', 'Invalid decomposition regularization or solver configuration.');
    end
    matrix = [low; reshape(high, T * R * C, N)];
    scale = norm(matrix, 'fro');
    if scale <= eps
        error('IceMap:LowRankEvidence', 'No nonzero evidence to decompose.');
    end
    lambda = p.rpca_lambda;
    if lambda == 0
        lambda = 1 / sqrt(max(size(matrix)));
    end
    spectral = norm(matrix, 2);
    dual = matrix / max(spectral, max(abs(matrix(:))) / lambda);
    penalty = 1.25 / max(spectral, eps);
    maximum = penalty * p.rpca_penalty_cap;
    L = zeros(size(matrix), 'like', matrix);
    sparsepart = L;
    converged = false;
    history = zeros(p.rpca_max_iter, 1);
    for iteration = 1:p.rpca_max_iter
        [U, S, V] = svd(matrix - sparsepart + dual / penalty, 'econ');
        singular = diag(S);
        keep = singular > 1 / penalty;
        if any(keep)
            L = U(:, keep) * diag(singular(keep) - 1 / penalty) * V(:, keep)';
        else
            L(:) = 0;
        end
        z = matrix - L + dual / penalty;
        sparsepart = max(0, 1 - lambda ./ max(penalty * abs(z), realmin)) .* z;
        residual = matrix - L - sparsepart;
        relative = norm(residual, 'fro') / scale;
        history(iteration) = relative;
        if relative <= p.rpca_tolerance
            converged = true;
            break
        end
        dual = dual + penalty * residual;
        penalty = min(maximum, penalty * p.rpca_penalty_growth);
    end
    if cfg.methods.d2.require_convergence && ~converged
        error('IceMap:RPCAConvergence', 'Low-rank decomposition did not converge.');
    end
    cleanlow = L(1:T, :);
    cleanhigh = reshape(L(T + 1:end, :), T, R, C, N);
    if ~any(abs(cleanlow(:)) > 0)
        error('IceMap:LowRankEvidence', 'Low-rank prior removed all low-frequency evidence.');
    end
    out = design2_multifrequency_collaboration(cleanlow, cleanhigh, time_s, cfg);
    out.lowrank_diagnostics = struct('converged', converged, 'iterations', iteration, 'relative_residual', relative, ...
                                   'lambda', lambda, 'effective_rank', sum(keep), 'residual_history', history(1:iteration), ...
                                   'sparse_energy_fraction', norm(sparsepart, 'fro')^2 / scale^2);
end
