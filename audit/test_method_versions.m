function tests = test_method_versions
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    root = fileparts(fileparts(mfilename('fullpath')));
    addpath(root, fullfile(root, 'common'), fullfile(root, 'p1_design1'), fullfile(root, 'p2_design2'), fullfile(root, 'p3_design3'));
    testCase.TestData.root = root;
end

function [low, high, time, ins, cfg, track] = tracking_case()
    T = 8;
    N = 128;
    time = (0:N - 1) * (.002 / 299792458);
    track = 40 + (0:T - 1)';
    low = complex(zeros(T, N));
    high = complex(zeros(T, 2, 2, N));
    for t = 1:T
        low(t, track(t)) = 1;
        low(t, 90) = .2;
        high(t, :, :, track(t)) = 1i;
    end
    ins = struct('z_delta_m', [0; diff(track) * .001], 'x_m', (0:T - 1)' * .02, 'y_m', zeros(T, 1));
    cfg = icemap_config(struct('surface_range_m', [.02 .07], 'spatial_sigma_m', .5, 'spatial_epsilon_m', .01));
end

function testMineTrackingDispatcherIsExact(testCase)
    [low, high, time, ins, cfg] = tracking_case();
    a = design1_energy_ins(low, high, time, ins, cfg);
    b = design1_select(low, high, time, ins, cfg);
    verifyEqual(testCase, a.data_l, b.data_l);
    verifyEqual(testCase, a.data_h, b.data_h);
    verifyEqual(testCase, a.shift_bins, b.shift_bins);
end

function testAllTrackingBranchesCleanMotion(testCase)
    [low, high, time, ins, cfg, track] = tracking_case();
    for version = {'v1', 'v3', 'v4', 'v5', 'v6', 'v7'}
        cfg.design1_version = version{1};
        out = design1_select(low, high, time, ins, cfg);
        verifyLessThan(testCase, max(abs(out.surface_track_bins - track)), 1.1);
        verifyTrue(testCase, all(isfinite(out.data_l(:))) && all(isfinite(out.data_h(:))));
        verifyEqual(testCase, size(out.data_h), size(high));
    end
end

function testOfflineTrackingRejectsTransientDistractor(testCase)
    [low, high, time, ins, cfg, track] = tracking_case();
    low(4, 65) = 5;
    cfg.design1_version = 'v3';
    cfg.methods.d1.transition_sigma_bins = 2;
    out = design1_select(low, high, time, ins, cfg);
    verifyEqual(testCase, out.surface_track_bins, track);
end

function testCorrelationFallsBackToInsWithoutDiscriminativeEnvelope(testCase)
    [low, high, time, ins, cfg] = tracking_case();
    low(:) = 1;
    cfg.design1_version = 'v5';
    out = design1_select(low, high, time, ins, cfg);
    verifyTrue(testCase, all(out.diagnostics.correlation_rejected(2:end)));
    verifyEqual(testCase, diff(out.surface_track_bins), ins.z_delta_m(2:end) / .001, 'AbsTol', 1e-10);
end

function [low, high, time, cfg] = fusion_case()
    T = 3;
    N = 96;
    x = 1:N;
    time = (0:N - 1) * (.002 / 299792458);
    z = (.2 + exp(-.5 * ((x - 25) / 3).^2) + .4 * exp(-.5 * ((x - 70) / 5).^2)) .* exp(1i * (.2 + .005 * x));
    h = exp(-.5 * ((-4:4) / 1.2).^2);
    h = h / sum(h);
    y = conv(z, h, 'same');
    low = repmat(y, T, 1);
    high = repmat(reshape(z, 1, 1, 1, N), T, 2, 2, 1);
    cfg = icemap_config(struct('shallow_end_s', time(48), 'psf_sigma_grid_bins', [.8 1.2], ...
                             'lambda_attenuation', .1, 'lambda_deconvolution', .01, 'solver_tolerance', 1e-6));
end

function testMineFusionDispatcherIsExact(testCase)
    [low, high, time, cfg] = fusion_case();
    a = design2_multifrequency_collaboration(low, high, time, cfg);
    b = design2_select(low, high, time, cfg);
    verifyEqual(testCase, a.data_l, b.data_l);
    verifyEqual(testCase, a.data_h, b.data_h);
end

function testGradientZeroRecoversOriginalRidgeAndWienerIdentityLimit(testCase)
    [low, high, time, cfg] = fusion_case();
    original = design2_multifrequency_collaboration(low, high, time, cfg);
    cfg.design2_version = 'v1';
    cfg.methods.d2.ridge_gradient_weight = 0;
    out = design2_select(low, high, time, cfg);
    verifyEqual(testCase, out.data_l, original.data_l, 'AbsTol', 1e-12);
    cfg.design2_version = 'v2';
    cfg.psf_sigma_grid_bins = .001;
    out = design2_select(low, high, time, cfg);
    n = out.shallow_samples;
    verifyEqual(testCase, out.data_l(:, 1:n), low(:, 1:n) / (1 + cfg.methods.d2.wiener_nsr), 'AbsTol', 1e-12);
end

function testAllFusionBranchesFiniteComplexAndConverged(testCase)
    [low, high, time, cfg] = fusion_case();
    for version = {'v1', 'v2', 'v4', 'v5', 'v6', 'v7'}
        cfg.design2_version = version{1};
        out = design2_select(low, high, time, cfg);
        verifyEqual(testCase, size(out.data_l), size(low));
        verifyEqual(testCase, size(out.data_h), size(high));
        verifyTrue(testCase, all(isfinite(out.data_l(:))) && all(isfinite(out.data_h(:))));
        verifyGreaterThan(testCase, norm(imag(out.data_l(:))), 0);
        if strcmp(version{1}, 'v4')
            verifyTrue(testCase, out.branch_diagnostics{1}.converged);
        end
        if strcmp(version{1}, 'v5')
            verifyTrue(testCase, all(out.branch_diagnostics{1}.high_weight >= 0 & out.branch_diagnostics{1}.high_weight <= 1));
        end
    end
end

function testNestedValidationAndAttitudeRequirements(testCase)
    cfg = icemap_config(struct('methods', struct('d1', struct('trend_lambda', 10))));
    verifyEqual(testCase, cfg.methods.d1.trend_lambda, 10);
    verifyEqual(testCase, cfg.methods.d1.template_update, .05);
    encoded = scene_to_json(cfg.scene);
    verifyTrue(testCase, contains(encoded, 'null'));
    decoded = jsondecode(encoded);
    verifyTrue(testCase, isempty(decoded.ice.snow_depth_m));
    verifyError(testCase, @()icemap_config(struct('scene', struct('climate', struct('relative_humidity', 30)))), 'IceMap:SceneRange');
    [low, high, time, ins, cfg] = tracking_case();
    cfg.design1_version = 'v1';
    cfg.methods.d1.attitude_correction = true;
    verifyError(testCase, @()design1_select(low, high, time, ins, cfg), 'IceMap:Attitude');
end

function testMainFilePipelineStillRunsAndExportsBothVersionIds(testCase)
    T = 4;
    N = 128;
    time_s = (0:N - 1) * (.002 / 299792458);
    radarl = complex(zeros(T, N));
    radarl(:, 40) = 1i;
    radarl(:, 80) = .8i;
    z = exp(1i * [zeros(1, 59) 1.5 * ones(1, N - 59)]);
    radarh = repmat(reshape(z, 1, 1, 1, N), T, 2, 2, 1);
    ins = struct('z_delta_m', zeros(T, 1), 'x_m', (0:T - 1)' * .02, 'y_m', zeros(T, 1));
    options = struct('surface_range_m', [.02 .055], 'spatial_sigma_m', .5, 'spatial_epsilon_m', .01, ...
                   'shallow_end_s', time_s(60), 'psf_sigma_grid_bins', 1, 'lambda_attenuation', 1, ...
                   'lambda_deconvolution', .1, 'solver_tolerance', 1e-5, 'attenuation_max_iter', 20000, ...
                   'boundary_phase_threshold_rad', .4, 'boundary_dilation_bins', 2, 'tof_min_peak_distance_bins', 10, ...
                   'tof_min_prominence', .2, 'tof_smooth_bins', 1, 'phase_trend_bins', 21);
    folder = tempname;
    mkdir(folder);
    cleanup = onCleanup(@()clean_test_folder(folder));
    input = fullfile(folder, 'input.mat');
    output = fullfile(folder, 'features.mat');
    save(input, 'radarl', 'radarh', 'time_s', 'ins');
    features = run_icemap(input, output, options);
    verifyEqual(testCase, size(features.phase), [T N]);
    verifyEqual(testCase, features.design1_metadata.algorithm_version, 'v2');
    verifyEqual(testCase, features.design2_metadata.algorithm_version, 'v3');
    verifyTrue(testCase, isfile(output));
    generated = run_icemap_variants(input, folder, {'v1'}, {'v1'}, options);
    verifyTrue(testCase, isfile(generated{1}));
    verifyError(testCase, @()run_icemap_variants(input, folder, {'v1'}, {'v1'}, options), 'IceMap:OutputExists');
end

function clean_test_folder(folder)
    % Remove only files created by this test; never recursively delete a path.
    files = dir(fullfile(folder, '*.mat'));
    for i = 1:numel(files)
        delete(fullfile(files(i).folder, files(i).name));
    end
    rmdir(folder);
end
