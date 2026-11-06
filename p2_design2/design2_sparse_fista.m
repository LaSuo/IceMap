function out = design2_sparse_fista(low, high, time_s, cfg)
    % Complex sparse finite-window inverse, line search, monotone restart, and
    % proximal-gradient stopping: .5||Hx-y||^2 + ridge/2||x||^2 + lambda||x||_1.
    [out, p] = prepare_collaboration(low, high, time_s, cfg);
    N = size(low, 2);
    n = out.shallow_samples;
    if p.sparse_weight < 0 || p.fista_max_iter < 1 || p.fista_max_iter ~= round(p.fista_max_iter) || p.fista_tolerance <= 0 || p.fista_backtracking <= 1 || p.fista_max_backtracks < 1
        error('IceMap:FISTAOptions', 'Invalid sparse weight, iteration budget, tolerance or line search.');
    end
    out.branch_diagnostics = cell(size(low, 1), 1);
    start = tic;
    for t = 1:size(low, 1)
        [H, ~] = shallow_operator(out.kernel{t}, N, n);
        y = low(t, 1:n).';
        ridge = cfg.lambda_deconvolution;
        x = y;
        z = x;
        momentum = 1;
        L = max(norm(H' * H, 1) + ridge, eps);
        objective = @(v).5 * norm(H * v - y)^2 + .5 * ridge * norm(v)^2 + p.sparse_weight * sum(abs(v));
        history = zeros(p.fista_max_iter, 1);
        converged = false;
        previous = objective(x);
        for iteration = 1:p.fista_max_iter
            gradient = H' * (H * z - y) + ridge * z;
            base = .5 * norm(H * z - y)^2 + .5 * ridge * norm(z)^2;
            accepted = false;
            for bt = 1:p.fista_max_backtracks
                v = z - gradient / L;
                next = max(0, 1 - p.sparse_weight ./ max(L * abs(v), realmin)) .* v;
                delta = next - z;
                smooth = .5 * norm(H * next - y)^2 + .5 * ridge * norm(next)^2;
                bound = base + real(gradient' * delta) + .5 * L * norm(delta)^2;
                if smooth <= bound + 1e-12 * max(1, abs(base))
                    accepted = true;
                    break
                else
                    L = L * p.fista_backtracking;
                end
            end
            if ~accepted
                error('IceMap:FISTALineSearch', 'No valid majorizer in frame %d.', t);
            end
            value = objective(next);
            if value > previous + 1e-12 * max(1, abs(previous))
                z = x;
                momentum = 1;
                history(iteration) = previous;
                continue
            end
            gradnext = H' * (H * next - y) + ridge * next;
            v = next - gradnext / L;
            proximal = max(0, 1 - p.sparse_weight ./ max(L * abs(v), realmin)) .* v;
            residual = L * norm(next - proximal) / max(1, norm(next));
            history(iteration) = value;
            old = x;
            x = next;
            previous = value;
            if residual <= p.fista_tolerance
                converged = true;
                break
            end
            nextmomentum = (1 + sqrt(1 + 4 * momentum^2)) / 2;
            z = x + (momentum - 1) / nextmomentum * (x - old);
            momentum = nextmomentum;
        end
        if p.require_convergence && ~converged
            error('IceMap:FISTAConvergence', 'Sparse solver did not converge in frame %d.', t);
        end
        out.data_l(t, 1:n) = x.';
        out.branch_diagnostics{t} = struct('converged', converged, 'iterations', iteration, 'gradient_mapping', residual, 'objective_history', history(1:iteration));
    end
    out.fused(:, n + 1:end) = out.data_l(:, n + 1:end);
    out.tpf = out.tpf + toc(start) / size(low, 1);
end
