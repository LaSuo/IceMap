function out = design2_alternating_alignment(low, high, time_s, cfg)
    % V8: demonstrate a relaxed alternating update schedule around the v3 fits.
    % Each outer iteration updates the PSF width, then the relative attenuation.
    % The shallow PSF target is the same grid argmin as v3 and is independent
    % of attenuation. There is no feedback from attenuation to the PSF fit.
    % This is a decoupled fixed-point demonstration, not joint alternating
    % minimization of a coupled physical likelihood. All traces use actual
    % iterates and measured residuals; no synthetic convergence curve is added.
    %
    % r(k+1) = (1-w_r)*r(k) + w_r*r_target
    % a(k+1) = (1-w_a)*a(k) + w_a*argmin_{a>=0} F_deep(a;r(k+1))
    % With both relaxation factors set to one, the result recovers v3 up to
    % the QP solver tolerance. Finite relaxed runs may differ from v3.
    % Hyperparameters live in cfg.methods.d2_alternating. Reported constants,
    % partition boundaries and the v3 default path remain unchanged.
    [low, high, time_s] = validate_echoes(low, high, time_s);
    validate_processing_config(cfg);
    require_config(cfg, {'shallow_end_s', 'psf_sigma_grid_bins', ...
                         'lambda_attenuation', 'lambda_deconvolution'});
    if ~strcmp(cfg.psf_family, 'gaussian')
        error('IceMap:AlternatingPSFFamily', 'V8 requires the Gaussian PSF family used by v3.');
    elseif ~isfield(cfg.methods, 'd2_alternating')
        error('IceMap:AlternatingOptions', 'Missing cfg.methods.d2_alternating.');
    end
    p = cfg.methods.d2_alternating;
    validate_alternating_options(p, cfg);
    [T, R, C, N] = size(high);
    Ns = find(time_s <= cfg.shallow_end_s, 1, 'last');
    if isempty(Ns) || Ns < 3 || Ns >= N
        error('IceMap:AlternatingPartition', 'shallow_end_s must split the grid into nonempty shallow and deep blocks.');
    end
    S = 1:Ns;
    D = Ns + 1:N;
    Nd = numel(D);
    G = diff(speye(Nd), 1, 1);
    history_columns = 17 + double(p.store_attenuation_history) * Nd;
    if T * (p.max_iter + 1) * history_columns > p.max_history_elements
        error('IceMap:AlternatingHistoryBudget', ...
              'Iterate traces exceed max_history_elements; disable full attenuation storage, reduce the input/iterations, or raise the explicit budget.');
    end
    % Fit the shallow target once and retain only per-candidate costs.
    % Candidate ordering and tie handling match the v3 grid search.
    out = initialize_output(T, R, C, N);
    started = tic;
    for t = 1:T
        yL = low(t, :).';
        yH = reshape(mean(mean(high(t, :, :, :), 2), 3), N, 1);
        if ~any(abs(yL(S)) > 0) || ~any(abs(yH(S)) > 0)
            error('IceMap:ShallowEvidence', 'Both bands need shallow PSF evidence in frame %d.', t);
        end
        [target_r, target_cost, grid_cost] = shallow_psf_target(yL(S), yH(S), N, Ns, cfg);
        r = p.initial_psf_bins;
        if r == 0
            r = max(cfg.psf_sigma_grid_bins);
        end
        a = p.initial_attenuation * ones(Nd, 1);
        identifiable = any(abs(yL(D)) > 0);
        if ~identifiable && strcmp(cfg.deep_evidence_policy, 'error')
            error('IceMap:DeepEvidence', 'No low-frequency deep evidence in frame %d; relative attenuation is unidentifiable.', t);
        end
        if identifiable
            Q = spdiags(abs(yL(D)).^2, 0, Nd, Nd) + cfg.lambda_attenuation * (G' * G);
        else
            Q = sparse(Nd, Nd);
        end
        [operator, kernel] = gaussian_operator(r, N, cfg);
        matched = operator(D, D) * yH(D);
        costs = measured_costs(operator(S, S), yL(S), yH(S), matched, yL(D), a, G, cfg);
        trace = initialize_trace(p, Nd);
        trace.psf_width_bins(1) = r;
        trace.objectives(1, :) = costs;
        trace.psf_target_relative_gap(1) = abs(r - target_r) / max(1, abs(target_r));
        trace.kkt_residual(1) = attenuation_kkt(Q, real(conj(yL(D)) .* matched), a, identifiable);
        trace.attenuation_statistics(1, :) = [min(a) mean(a) max(a)];
        if p.store_attenuation_history
            trace.attenuation(1, :) = a.';
        end
        converged = false;
        for iteration = 1:p.max_iter
            previous_r = r;
            previous_a = a;
            previous_costs = costs;
            % Block 1: relax toward the attenuation-independent shallow fit.
            r = previous_r + p.psf_relaxation * (target_r - previous_r);
            if abs(r - target_r) <= p.parameter_tolerance * max(1, abs(target_r))
                % End the interpolation at the actual v3 grid kernel, avoiding
                % an extra support bin caused solely by roundoff near ceil().
                r = target_r;
            end
            [operator, kernel] = gaussian_operator(r, N, cfg);
            matched = operator(D, D) * yH(D);
            rhs = real(conj(yL(D)) .* matched);
            % Block 2: solve the v3 nonnegative QP at this actual PSF iterate.
            if identifiable
                [candidate_a, inner_kkt, inner_iterations] = solve_attenuation(Q, rhs, previous_a, cfg);
            else
                candidate_a = ones(Nd, 1);
                inner_kkt = 0;
                inner_iterations = 0;
            end
            a = previous_a + p.attenuation_relaxation * (candidate_a - previous_a);
            a = max(a, 0);
            % A relaxed positive entry only approaches an active zero bound.
            % Complete the numerical handoff when the conditional solution is
            % exactly zero and this entry is within the parameter tolerance.
            active_zero = candidate_a == 0 & a <= p.parameter_tolerance * max(1, abs(previous_a));
            a(active_zero) = 0;
            kkt = attenuation_kkt(Q, rhs, a, identifiable);
            costs = measured_costs(operator(S, S), yL(S), yH(S), matched, yL(D), a, G, cfg);
            changes = [abs(r - previous_r) / max(1, abs(previous_r)), ...
                       norm(a - previous_a, Inf) / max(1, norm(previous_a, Inf))];
            objective_change = max(abs(costs - previous_costs) ./ max(1, abs(previous_costs)));
            target_gap = abs(r - target_r) / max(1, abs(target_r));
            trace.psf_width_bins(iteration + 1) = r;
            trace.objectives(iteration + 1, :) = costs;
            trace.relative_parameter_change(iteration + 1, :) = changes;
            trace.relative_objective_change(iteration + 1) = objective_change;
            trace.psf_target_relative_gap(iteration + 1) = target_gap;
            trace.kkt_residual(iteration + 1) = kkt;
            trace.inner_kkt_residual(iteration + 1) = inner_kkt;
            trace.inner_iterations(iteration + 1) = inner_iterations;
            trace.inner_solve_performed(iteration + 1) = identifiable;
            trace.active_zero_count(iteration + 1) = nnz(active_zero & previous_a > 0);
            trace.attenuation_statistics(iteration + 1, :) = [min(a) mean(a) max(a)];
            if p.store_attenuation_history
                trace.attenuation(iteration + 1, :) = a.';
            end
            if iteration >= p.min_iter && max(changes) <= p.parameter_tolerance && ...
                    target_gap <= p.parameter_tolerance && objective_change <= p.objective_tolerance && ...
                    kkt <= p.kkt_tolerance
                converged = true;
                break
            end
        end
        if ~converged && p.require_convergence
            error('IceMap:AlternatingConvergence', ...
                  'V8 frame %d did not converge in %d outer iterations (parameter change %.3g, PSF gap %.3g, objective change %.3g, KKT %.3g).', ...
                  t, iteration, max(changes), target_gap, objective_change, kkt);
        end
        trace = finish_trace(trace, iteration + 1);
        trace.psf_target_bins = target_r;
        trace.psf_grid_cost = grid_cost;
        trace.psf_target_shallow_cost = target_cost;
        trace.objective_columns = {'shallow_psf_residual_squared', 'deep_residual_squared_plus_smoothness'};
        trace.parameter_change_columns = {'psf_width_relative_change', 'attenuation_relative_change'};
        trace.attenuation_statistic_columns = {'minimum', 'mean', 'maximum'};
        trace.converged = converged;
        trace.attenuation_identifiable = identifiable;
        trace.objectives_are_joint_descent_certificate = false;
        % Apply compensation once to the original input, avoiding repeated gains.
        RS = operator(S, S);
        sharp = (RS' * RS + cfg.lambda_deconvolution * speye(Ns)) \ (RS' * yL(S));
        out.data_l(t, :) = yL.';
        out.data_l(t, S) = sharp.';
        gain = ones(N, 1);
        gain(D) = 1 ./ max(a, cfg.attenuation_floor);
        out.data_h(t, :, :, :) = high(t, :, :, :) .* reshape(gain, 1, 1, 1, N);
        out.rL_bins(t) = r;
        out.kernel{t} = kernel;
        out.psf_target_bins(t) = target_r;
        out.attenuation_relative(t, D) = a;
        out.attenuation_kkt_residual(t) = kkt;
        out.attenuation_identifiable(t) = identifiable;
        out.relative_fit_residual(t, 1) = sqrt(costs(1) / max(sum(abs(yL(S)).^2), realmin));
        out.fit_residual_valid(t, 1) = true;
        if identifiable && any(abs(matched) > 0)
            out.relative_fit_residual(t, 2) = norm(matched - a .* yL(D)) / max(norm(matched), realmin);
            out.fit_residual_valid(t, 2) = true;
        end
        out.psf_grid_edge(t) = target_r == min(cfg.psf_sigma_grid_bins) || target_r == max(cfg.psf_sigma_grid_bins);
        if ~isempty(cfg.cross_frequency_max_relative_residual)
            out.cross_frequency_mismatch(t) = any(out.relative_fit_residual(t, out.fit_residual_valid(t, :)) > cfg.cross_frequency_max_relative_residual);
        end
        out.alternating_iterations(t) = iteration;
        out.alternating_converged(t) = converged;
        out.alternating_history{t} = trace;
    end
    out.tpf = toc(started) / T;
    out.time_s = time_s;
    out.shallow_samples = Ns;
    out.params = cfg;
    out.fused = reshape(mean(mean(out.data_h, 2), 3), T, N);
    out.fused(:, D) = out.data_l(:, D);
    out.mismatch_threshold_calibrated = ~isempty(cfg.cross_frequency_max_relative_residual);
    out.translation_assumption = 'Same Gaussian PSF, local attenuation commutation and partition assumptions as v3';
    out.optimization_contract = 'relaxed alternating schedule toward the decoupled v3 fits; not coupled joint alternating minimization';
    out.psf_fit_independent_of_attenuation = true;
    out.is_joint_alternating_optimization = false;
end

function validate_alternating_options(p, cfg)
    defaults = method_defaults();
    if ~isstruct(p) || ~isscalar(p) || ~isequal(sort(fieldnames(p)), sort(fieldnames(defaults.d2_alternating)))
        error('IceMap:AlternatingOptions', 'V8 options must match the declared d2_alternating configuration.');
    end
    validate_method_options(p);
    for name = {'max_iter', 'min_iter', 'max_history_elements'}
        value = p.(name{1});
        if value < 1 || value ~= fix(value)
            error('IceMap:AlternatingOptions', '%s must be a positive integer.', name{1});
        end
    end
    if p.min_iter > p.max_iter
        error('IceMap:AlternatingOptions', 'min_iter cannot exceed max_iter.');
    end
    for name = {'psf_relaxation', 'attenuation_relaxation'}
        if p.(name{1}) <= 0 || p.(name{1}) > 1
            error('IceMap:AlternatingOptions', '%s must be in (0,1].', name{1});
        end
    end
    for name = {'parameter_tolerance', 'objective_tolerance', 'kkt_tolerance'}
        if p.(name{1}) <= 0
            error('IceMap:AlternatingOptions', '%s must be positive.', name{1});
        end
    end
    for name = {'require_convergence', 'store_attenuation_history'}
        if ~any(p.(name{1}) == [0 1])
            error('IceMap:AlternatingOptions', '%s must be Boolean.', name{1});
        end
    end
    if p.initial_attenuation < 0 || p.initial_psf_bins < 0
        error('IceMap:AlternatingOptions', 'Initial attenuation and PSF sentinel must be nonnegative.');
    elseif p.initial_psf_bins > 0 && ...
            (p.initial_psf_bins < min(cfg.psf_sigma_grid_bins) || p.initial_psf_bins > max(cfg.psf_sigma_grid_bins))
        error('IceMap:AlternatingOptions', 'initial_psf_bins must lie inside the PSF grid extent; zero selects its maximum.');
    end
end

function out = initialize_output(T, R, C, N)
    out = struct('data_l', complex(zeros(T, N)), 'data_h', complex(zeros(T, R, C, N)), ...
                 'rL_bins', zeros(T, 1), 'psf_target_bins', zeros(T, 1), ...
                 'attenuation_relative', ones(T, N), 'attenuation_kkt_residual', zeros(T, 1), ...
                 'attenuation_identifiable', false(T, 1), 'relative_fit_residual', zeros(T, 2), ...
                 'fit_residual_valid', false(T, 2), 'psf_grid_edge', false(T, 1), ...
                 'cross_frequency_mismatch', false(T, 1), 'alternating_iterations', zeros(T, 1), ...
                 'alternating_converged', false(T, 1));
    out.kernel = cell(T, 1);
    out.alternating_history = cell(T, 1);
end

function [best_r, best_cost, grid_cost] = shallow_psf_target(yL, yH, N, Ns, cfg)
    grid = cfg.psf_sigma_grid_bins(:).';
    grid_cost = zeros(numel(grid), 1);
    best_cost = Inf;
    best_r = grid(1);
    for i = 1:numel(grid)
        operator = gaussian_operator(grid(i), N, cfg);
        residual = yL - operator(1:Ns, 1:Ns) * yH;
        grid_cost(i) = real(residual' * residual);
        if ~isfinite(grid_cost(i))
            error('IceMap:AlternatingFinite', 'PSF residual overflow; inspect input amplitude scaling.');
        elseif grid_cost(i) < best_cost
            best_cost = grid_cost(i);
            best_r = grid(i);
        end
    end
end

function [operator, kernel] = gaussian_operator(r, N, cfg)
    half_width = min(N - 1, ceil(cfg.psf_support_sigmas * r));
    offsets = -half_width:half_width;
    kernel = exp(-0.5 * (offsets / r).^2);
    kernel = kernel / sum(kernel);
    if N * numel(kernel) > cfg.max_operator_nonzeros
        error('IceMap:OperatorBudget', 'V8 PSF operator exceeds max_operator_nonzeros.');
    end
    operator = spdiags(repmat(kernel, N, 1), offsets, N, N);
end

function costs = measured_costs(RS, shallow_low, shallow_high, matched, deep_low, a, G, cfg)
    shallow = shallow_low - RS * shallow_high;
    deep = matched - a .* deep_low;
    costs = [real(shallow' * shallow), ...
             real(deep' * deep) + cfg.lambda_attenuation * sum(abs(G * a).^2)];
    if any(~isfinite(costs))
        error('IceMap:AlternatingFinite', 'Nonfinite iterate objective; inspect echoes, calibration and initial values.');
    end
end

function residual = attenuation_kkt(Q, rhs, a, identifiable)
    if ~identifiable
        residual = 0; % sentinel only; the separate identifiable flag is false
        return
    end
    gradient = Q * a - rhs;
    violation = gradient;
    violation(a <= 0) = min(gradient(a <= 0), 0);
    residual = norm(violation, Inf) / max(1, norm(rhs, Inf));
    if ~isfinite(residual)
        error('IceMap:AlternatingFinite', 'Nonfinite attenuation optimality residual.');
    end
end

function [a, residual, iteration] = solve_attenuation(Q, rhs, previous, cfg)
    % The same nonnegative tridiagonal QP as v3, with an optional warm start.
    a = max(real(Q \ rhs), 0);
    objective = @(x) real(0.5 * x' * Q * x - rhs' * x);
    if objective(previous) < objective(a)
        a = previous;
    end
    diagonal = max(full(diag(Q)), realmin);
    lower = full(diag(Q, -1));
    upper = full(diag(Q, 1));
    n = numel(rhs);
    colors = {(1:2:n).', (2:2:n).'};
    for iteration = 0:cfg.attenuation_max_iter
        residual = attenuation_kkt(Q, rhs, a, true);
        if residual <= cfg.solver_tolerance
            return
        elseif iteration == cfg.attenuation_max_iter
            break
        end
        for color = 1:2
            indices = colors{color};
            neighbor = zeros(numel(indices), 1);
            left = indices > 1;
            right = indices < n;
            neighbor(left) = neighbor(left) + lower(indices(left) - 1) .* a(indices(left) - 1);
            neighbor(right) = neighbor(right) + upper(indices(right)) .* a(indices(right) + 1);
            a(indices) = max(0, (rhs(indices) - neighbor) ./ diagonal(indices));
        end
    end
    error('IceMap:AlternatingInnerConvergence', 'V8 inner attenuation QP failed after %d iterations (KKT %.3g).', iteration, residual);
end

function trace = initialize_trace(p, Nd)
    rows = p.max_iter + 1;
    trace = struct('iteration', (0:p.max_iter).', 'psf_width_bins', zeros(rows, 1), ...
                   'objectives', zeros(rows, 2), 'relative_parameter_change', zeros(rows, 2), ...
                   'relative_objective_change', zeros(rows, 1), 'psf_target_relative_gap', zeros(rows, 1), ...
                   'kkt_residual', zeros(rows, 1), 'inner_kkt_residual', zeros(rows, 1), ...
                   'inner_iterations', zeros(rows, 1), 'inner_solve_performed', false(rows, 1), ...
                   'active_zero_count', zeros(rows, 1), 'attenuation_statistics', zeros(rows, 3));
    if p.store_attenuation_history
        trace.attenuation = zeros(rows, Nd);
    end
end

function trace = finish_trace(trace, rows)
    names = fieldnames(trace);
    for i = 1:numel(names)
        trace.(names{i}) = trace.(names{i})(1:rows, :);
    end
end
