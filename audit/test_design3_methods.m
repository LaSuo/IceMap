function tests = test_design3_methods
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    root = fileparts(fileparts(mfilename('fullpath')));
    addpath(genpath(root));
end

function [low, high, time, cfg] = feature_case()
    T = 4;
    N = 192;
    x = 1:N;
    time = (0:N - 1) * (.002 / 299792458);
    envelope = exp(-.5 * ((x - 60) / 3).^2) + .8 * exp(-.5 * ((x - 132) / 4).^2);
    low = repmat(envelope .* exp(1i * .2 * x), T, 1);
    amplitude = .2 + envelope;
    phase = .02 * sin(x / 10) + (x >= 100) * 1.2;
    high = complex(zeros(T, 2, 2, N));
    for t = 1:T
        for r = 1:2
            for c = 1:2
                high(t, r, c, :) = reshape(amplitude .* exp(1i * (phase + .05 * t + .1 * r + .03 * c)), 1, 1, 1, N);
            end
        end
    end
    cfg = icemap_config(struct('boundary_phase_threshold_rad', .4, 'boundary_dilation_bins', 1, ...
                             'tof_min_peak_distance_bins', 20, 'tof_min_prominence', .02, 'tof_smooth_bins', 1, ...
                             'tof_centroid_halfwidth_bins', 2, 'phase_trend_bins', 21));
end

function testDefaultIsExactOriginalMethod(testCase)
    [low, high, time, cfg] = feature_case();
    a = design3_physical_fingerprints(low, high, time, cfg);
    b = design3_select(low, high, time, cfg);
    verifyEqual(testCase, cfg.design3_version, 'v1');
    for name = {'phase', 'E', 'tau', 'scatter_valid'}
        verifyEqual(testCase, a.(name{1}), b.(name{1}));
    end
    verifyEqual(testCase, b.algorithm_method, 'design3_physical_fingerprints');
end

function testAllAlternativeContractsAndAdditionalChannels(testCase)
    [low, high, time, cfg] = feature_case();
    for version = {'v2', 'v3', 'v4', 'v5'}
        cfg.design3_version = version{1};
        out = design3_select(low, high, time, cfg);
        verifyEqual(testCase, size(out.phase), size(low));
        verifyTrue(testCase, all(isfinite(out.E(:))) && all(isfinite(out.tau(:))));
        verifyTrue(testCase, all(out.E(~out.scatter_valid) == 0));
        names = fieldnames(out.extra_features);
        verifyGreaterThanOrEqual(testCase, numel(names), 3);
        for i = 1:numel(names)
            name = names{i};
            value = out.extra_features.(name);
            valid = out.extra_feature_valid.(name);
            verifyEqual(testCase, size(value), size(low));
            verifyTrue(testCase, all(isfinite(value(:))) && isreal(value));
            verifyTrue(testCase, all(value(~valid) == 0));
        end
        for t = 1:size(low, 1)
            verifyTrue(testCase, all(diff(out.tau_event_bins{t}) > 0));
            verifyTrue(testCase, all(out.tau_events_s{t} >= time(1) & out.tau_events_s{t} <= time(end)));
        end
    end
end

function testRobustMethodFindsKnownEchoEvents(testCase)
    [low, high, time, cfg] = feature_case();
    cfg.design3_version = 'v2';
    out = design3_select(low, high, time, cfg);
    for t = 1:size(low, 1)
        verifyLessThan(testCase, min(abs(out.tau_event_bins{t} - 60)), 2);
        verifyLessThan(testCase, min(abs(out.tau_event_bins{t} - 132)), 2);
    end
end

function testRobustPhaseResistsAnAmplitudeOutlierChannel(testCase)
    [low, high, time, cfg] = feature_case();
    cfg.design3_version = 'v2';
    baseline = design3_select(low, high, time, cfg);
    rng(12);
    high(:, 1, 1, :) = 1000 * exp(1i * randn(size(high(:, 1, 1, :))));
    out = design3_select(low, high, time, cfg);
    error = abs(angle(exp(1i * (out.phase - baseline.phase))));
    verifyLessThan(testCase, median(error(:)), .2);
end

function testKnownAnalyticToneAttributes(testCase)
    [low, high, time, cfg] = feature_case();
    x = 0:size(low, 2) - 1;
    omega = .3;
    low = repmat(exp(1i * omega * x), size(low, 1), 1);
    cfg.design3_version = 'v5';
    out = design3_select(low, high, time, cfg);
    verifyEqual(testCase, out.extra_features.analytic_instantaneous_frequency(:, 3:end - 3), ones(size(low, 1), size(low, 2) - 5) * omega / (2 * pi), 'AbsTol', 1e-12);
    verifyEqual(testCase, out.extra_features.analytic_nonlinear_energy(:, 3:end - 3), ones(size(low, 1), size(low, 2) - 5) * (1 - cos(2 * omega)), 'AbsTol', 1e-12);
end

function testRealEchoesUseAnalyticConversionWithoutShapeLoss(testCase)
    [low, high, time, cfg] = feature_case();
    cfg.design3_version = 'v5';
    out = design3_select(real(low), real(high), time, cfg);
    verifyTrue(testCase, out.diagnostics.analytic_low_constructed);
    verifyTrue(testCase, out.diagnostics.analytic_high_constructed);
    verifyEqual(testCase, size(out.phase), size(low));
    verifyTrue(testCase, all(isfinite(out.E(:))));
end

function testKnownSpectralCentroidAndConstantMissingEvidence(testCase)
    [~, ~, ~, cfg] = feature_case();
    p = cfg.methods.d3;
    x = 0:191;
    wave = repmat(1 + .5 * cos(2 * pi * x / 8), 3, 1);
    [spectrum, valid] = local_spectral_attributes(wave, p);
    verifyTrue(testCase, all(valid(:)));
    verifyLessThan(testCase, abs(median(spectrum.centroid(:, 33:end - 33), 'all') - .125), .02);
    verifyTrue(testCase, all(spectrum.entropy(:) >= 0 & spectrum.entropy(:) <= 1));
    [constant, valid] = local_spectral_attributes(ones(3, 192), p);
    verifyFalse(testCase, any(valid(:)));
    verifyEqual(testCase, constant.entropy, zeros(3, 192));
end

function testEndpointEventsAndEmptyEventPolicy(testCase)
    [~, ~, time, cfg] = feature_case();
    score = zeros(2, numel(time));
    score(:, 1) = 1;
    [~, bins, times] = build_event_fingerprint(score, score, time, cfg, zeros(size(score)), true(size(score)));
    verifyEqual(testCase, bins{1}, 1);
    verifyEqual(testCase, times{1}, time(1));
    cfg.methods.d3.allow_no_events = false;
    verifyError(testCase, @()build_event_fingerprint(zeros(size(score)), score, time, cfg, zeros(size(score)), true(size(score))), 'IceMap:NoEvents');
end

function testEvidenceAndParameterFailurePolicies(testCase)
    [low, high, time, cfg] = feature_case();
    cfg.design3_version = 'v2';
    high(:) = 0;
    verifyError(testCase, @()design3_select(low, high, time, cfg), 'IceMap:PhaseEvidence');
    cfg.methods.d3.strict_phase_evidence = false;
    out = design3_select(low, high, time, cfg);
    verifyFalse(testCase, any(out.scatter_valid(:)));
    cfg.methods.d3.scale_bins = [0 1];
    verifyError(testCase, @()design3_select(low, high, time, cfg), 'IceMap:FeatureScales');
    [low, high, time, cfg] = feature_case();
    cfg.design3_version = 'v4';
    cfg.methods.d3.spectral_nfft = 16;
    verifyError(testCase, @()design3_select(low, high, time, cfg), 'IceMap:FeatureFFT');
    [low, high, time, cfg] = feature_case();
    cfg.design3_version = 'v4';
    cfg.methods.d3.maximum_window_elements = 10;
    verifyError(testCase, @()design3_select(low, high, time, cfg), 'IceMap:FeatureBudget');
end

function testExtendedFeaturesAreExportedWithMasksAndUnits(testCase)
    [low, high, time, cfg] = feature_case();
    cfg.design3_version = 'v4';
    out = design3_select(low, high, time, cfg);
    T = size(low, 1);
    N = size(low, 2);
    K = 10;
    out.boundary_target = zeros(T, N);
    out.boundary_target(:, [21 109]) = 1;
    density = zeros(T, K);
    density(:, 1) = .9;
    thickness = zeros(T, K);
    thickness(:, 1) = .05;
    out.ground_truth = struct('layer_count', ones(T, 1), 'relative_density_target', density, 'layer_density_gt', density, ...
                            'layer_thickness_m_gt', thickness, 'density_profile_gt', zeros(T, N), 'depth_m', (0:N - 1) * .001);
    folder = tempname;
    mkdir(folder);
    cleanup = onCleanup(@()clean_fixture(folder));
    matfile = fullfile(folder, 'features.mat');
    h5file = fullfile(folder, 'features.h5');
    save(matfile, '-struct', 'out');
    export_features_h5(matfile, h5file, 'synthetic_site', 'synthetic_record', 'test_only_proxy');
    verifyEqual(testCase, h5readatt(h5file, '/', 'icemap_extra_schema'), 'icemap.extra_features.v1');
    verifyEqual(testCase, h5readatt(h5file, '/extra_features/spectral_centroid', 'unit'), 'cycles/bin');
    verifyEqual(testCase, h5read(h5file, '/extra_features/spectral_entropy'), out.extra_features.spectral_entropy.');
    versions = jsondecode(h5readatt(h5file, '/', 'algorithm_versions_json'));
    verifyEqual(testCase, versions.design3, 'v4');
end

function clean_fixture(folder)
    files = dir(folder);
    for i = 1:numel(files)
        if ~files(i).isdir
            delete(fullfile(folder, files(i).name));
        end
    end
    rmdir(folder);
end
