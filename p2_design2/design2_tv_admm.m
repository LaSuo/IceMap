function out = design2_tv_admm(low, high, time_s, cfg)
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
        [x, info] = tv_admm(H, y, D, p, cfg.lambda_deconvolution);
        out.branch_diagnostics{t} = info;
        if p.require_convergence && ~info.converged
            error('IceMap:TVConvergence', 'TV-ADMM did not converge in frame %d.', t);
        end
        out.data_l(t, 1:Ns) = x.';
    end
    out.fused = reshape(mean(mean(out.data_h, 2), 3), T, N);
    out.fused(:, Ns + 1:end) = out.data_l(:, Ns + 1:end);
    out.tpf = out.tpf + toc(start) / T;
end

function [x, info] = tv_admm(H, y, D, p, ridge)
    % min .5||H*x-y||^2 + .5*ridge||x||^2 + tv_weight*sum(abs(D*x)).
    rho = p.admm_rho;
    n = numel(y);
    m = size(D, 1);
    A = decomposition(H' * H + ridge * speye(n) + rho * (D' * D), 'chol');
    x = y;
    z = D * x;
    u = zeros(m, 1, 'like', y);
    converged = false;
    for iteration = 1:p.admm_max_iter
        x = A \ (H' * y + rho * D' * (z - u));
        dx = D * x;
        previous = z;
        v = dx + u;
        z = max(0, 1 - p.tv_weight ./ max(rho * abs(v), realmin)) .* v;
        u = u + dx - z;
        primal = norm(dx - z);
        dual = rho * norm(D' * (z - previous));
        ep = sqrt(m) * p.absolute_tolerance + p.relative_tolerance * max(norm(dx), norm(z));
        ed = sqrt(n) * p.absolute_tolerance + p.relative_tolerance * norm(rho * D' * u);
        if primal <= ep && dual <= ed
            converged = true;
            break
        end
    end
    info = struct('converged', converged, 'iterations', iteration, 'primal_residual', primal, 'dual_residual', dual, ...
                'objective', .5 * norm(H * x - y)^2 + .5 * ridge * norm(x)^2 + p.tv_weight * sum(abs(D * x)));
end
