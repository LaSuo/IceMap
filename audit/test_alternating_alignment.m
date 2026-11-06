function tests = test_alternating_alignment
    tests = functiontests(localfunctions);
end

function setupOnce(tc)
    root = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(root, 'common'), fullfile(root, 'p2_design2'));
end

function [low, high, time_s, cfg] = alignment_case()
    T = 2;
    N = 96;
    x = 1:N;
    signal = (0.3 + exp(-0.5 * ((x - 24) / 3).^2) + ...
              0.4 * exp(-0.5 * ((x - 72) / 5).^2)) .* exp(1i * (0.2 + 0.005 * x));
    kernel = exp(-0.5 * ((-5:5) / 1.2).^2);
    kernel = kernel / sum(kernel);
    low = repmat(conv(signal, kernel, 'same'), T, 1);
    high = repmat(reshape(signal, 1, 1, 1, N), T, 2, 2, 1);
    time_s = (0:N - 1) * (0.002 / 299792458);
    cfg = icemap_config(struct('design2_version', 'v8', 'shallow_end_s', time_s(48), ...
                               'psf_sigma_grid_bins', [0.6 1.2 2.4], ...
                               'lambda_attenuation', 0.1, 'lambda_deconvolution', 0.01, ...
                               'solver_tolerance', 1e-9));
end

function testDefaultRemainsV3AndD1DoesNotGainV8(tc)
    cfg = icemap_config();
    verifyEqual(tc, cfg.design2_version, 'v3');
    catalog = algorithm_catalog();
    verifyEqual(tc, catalog.d2{8}, 'design2_alternating_alignment');
    verifyError(tc, @()icemap_config(struct('design1_version', 'v8')), 'IceMap:Version');
end

function testUnitRelaxationRecoversV3(tc)
    [low, high, time_s, cfg] = alignment_case();
    cfg.methods.d2_alternating.psf_relaxation = 1;
    cfg.methods.d2_alternating.attenuation_relaxation = 1;
    baseline = design2_multifrequency_collaboration(low, high, time_s, cfg);
    branch = design2_select(low, high, time_s, cfg);
    verifyEqual(tc, branch.algorithm_version, 'v8');
    verifyEqual(tc, branch.rL_bins, baseline.rL_bins);
    verifyEqual(tc, branch.data_l, baseline.data_l, 'AbsTol', 1e-12);
    verifyEqual(tc, branch.data_h, baseline.data_h, 'AbsTol', 2e-6);
    verifyTrue(tc, all(branch.alternating_converged));
end

function testRelaxedTraceMatchesActualBlockResiduals(tc)
    [low, high, time_s, cfg] = alignment_case();
    cfg.methods.d2_alternating.store_attenuation_history = true;
    out = design2_select(low, high, time_s, cfg);
    verifyTrue(tc, out.psf_fit_independent_of_attenuation);
    verifyFalse(tc, out.is_joint_alternating_optimization);
    verifyTrue(tc, all(out.alternating_converged));
    verifyTrue(tc, all(out.alternating_iterations > 2));
    trace = out.alternating_history{1};
    verifyLessThan(tc, trace.psf_target_bins, trace.psf_width_bins(1));
    verifyFalse(tc, trace.inner_solve_performed(1));
    verifyTrue(tc, all(trace.inner_solve_performed(2:end)));
    verifyEqual(tc, numel(trace.iteration), out.alternating_iterations(1) + 1);
    verifyEqual(tc, trace.attenuation(end, :), out.attenuation_relative(1, out.shallow_samples + 1:end));
    S = 1:out.shallow_samples;
    D = out.shallow_samples + 1:numel(time_s);
    high_mean = reshape(mean(mean(high(1, :, :, :), 2), 3), [], 1);
    % Independent convolution evaluation checks that histories were measured
    % on each stored iterate, including the zero-filled partition boundaries.
    for row = 1:numel(trace.iteration)
        r = trace.psf_width_bins(row);
        half = min(numel(time_s) - 1, ceil(cfg.psf_support_sigmas * r));
        kernel = exp(-0.5 * (((-half:half).' / r).^2));
        kernel = kernel / sum(kernel);
        shallow_prediction = conv(high_mean(S), kernel, 'same');
        deep_prediction = conv(high_mean(D), kernel, 'same');
        a = trace.attenuation(row, :).';
        actual = [sum(abs(low(1, S).' - shallow_prediction).^2), ...
                  sum(abs(deep_prediction - a .* low(1, D).').^2) + cfg.lambda_attenuation * sum(diff(a).^2)];
        verifyEqual(tc, trace.objectives(row, :), actual, 'AbsTol', 1e-10);
    end
end

function testBudgetStopReturnsActualUnconvergedState(tc)
    [low, high, time_s, cfg] = alignment_case();
    cfg.methods.d2_alternating.max_iter = 1;
    cfg.methods.d2_alternating.min_iter = 1;
    cfg.methods.d2_alternating.require_convergence = false;
    out = design2_select(low, high, time_s, cfg);
    verifyFalse(tc, any(out.alternating_converged));
    verifyEqual(tc, out.alternating_iterations, ones(size(low, 1), 1));
    verifyNotEqual(tc, out.rL_bins(1), out.psf_target_bins(1));
    verifyEqual(tc, out.rL_bins(1), out.alternating_history{1}.psf_width_bins(end));
    cfg.methods.d2_alternating.require_convergence = true;
    verifyError(tc, @()design2_select(low, high, time_s, cfg), 'IceMap:AlternatingConvergence');
end

function testInvalidOptionsAndTraceBudgetsFailExplicitly(tc)
    [low, high, time_s, cfg] = alignment_case();
    bad = cfg;
    bad.methods.d2_alternating.psf_relaxation = 0;
    verifyError(tc, @()design2_select(low, high, time_s, bad), 'IceMap:AlternatingOptions');
    bad = cfg;
    bad.methods.d2_alternating.initial_psf_bins = 10;
    verifyError(tc, @()design2_select(low, high, time_s, bad), 'IceMap:AlternatingOptions');
    bad = cfg;
    bad.methods.d2_alternating.min_iter = 1000;
    verifyError(tc, @()design2_select(low, high, time_s, bad), 'IceMap:AlternatingOptions');
    bad = cfg;
    bad.methods.d2_alternating.store_attenuation_history = 2;
    verifyError(tc, @()design2_select(low, high, time_s, bad), 'IceMap:AlternatingOptions');
    cfg.methods.d2_alternating.max_history_elements = 10;
    verifyError(tc, @()design2_select(low, high, time_s, cfg), 'IceMap:AlternatingHistoryBudget');
end

function testUnityFallbackIsMarkedUnidentifiable(tc)
    [low, high, time_s, cfg] = alignment_case();
    Ns = find(time_s <= cfg.shallow_end_s, 1, 'last');
    low(:, Ns + 1:end) = 0;
    cfg.deep_evidence_policy = 'unity';
    out = design2_select(low, high, time_s, cfg);
    verifyFalse(tc, any(out.attenuation_identifiable));
    verifyEqual(tc, out.attenuation_relative, ones(size(low)));
    verifyFalse(tc, any(out.alternating_history{1}.inner_solve_performed));
    verifyFalse(tc, out.alternating_history{1}.attenuation_identifiable);
end

function testAttenuationInitializationDoesNotPretendToChangePSFFit(tc)
    [low, high, time_s, cfg] = alignment_case();
    cfg.methods.d2_alternating.initial_attenuation = 0.2;
    first = design2_select(low, high, time_s, cfg);
    cfg.methods.d2_alternating.initial_attenuation = 2;
    second = design2_select(low, high, time_s, cfg);
    verifyEqual(tc, first.psf_target_bins, second.psf_target_bins);
    verifyEqual(tc, first.rL_bins, second.rL_bins);
    verifyEqual(tc, first.data_l, second.data_l, 'AbsTol', 1e-12);
end

function testActiveZeroBoundsDoNotPreventOuterConvergence(tc)
    [low, high, time_s, cfg] = alignment_case();
    Ns = find(time_s <= cfg.shallow_end_s, 1, 'last');
    low(:, Ns + 1:end) = 1;
    high(:, :, :, Ns + 1:end) = -1;
    out = design2_select(low, high, time_s, cfg);
    verifyTrue(tc, all(out.alternating_converged));
    verifyEqual(tc, out.attenuation_relative(:, Ns + 1:end), zeros(size(low, 1), numel(time_s) - Ns));
    verifyLessThanOrEqual(tc, max(out.attenuation_kkt_residual), cfg.methods.d2_alternating.kkt_tolerance);
    verifyGreaterThan(tc, sum(out.alternating_history{1}.active_zero_count), 0);
end
