function tests = test_input_safety
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    root = fileparts(fileparts(mfilename('fullpath')));
    addpath(genpath(root));
end

function testMalformedEchoes(testCase)
    low = ones(2, 32);
    high = ones(2, 1, 1, 32);
    time = (0:31) * 1e-11;
    verifyError(testCase, @()validate_echoes('wrong', high, time), 'IceMap:EchoType');
    verifyError(testCase, @()validate_echoes([], high, time), 'IceMap:EmptyEcho');
    verifyError(testCase, @()validate_echoes(low, ones(2, 2, 32), time), 'IceMap:EchoShape');
    bad = low;
    bad(1) = NaN;
    verifyError(testCase, @()validate_echoes(bad, high, time), 'IceMap:EchoFinite');
    badtime = time;
    badtime(2) = badtime(1);
    verifyError(testCase, @()validate_echoes(low, high, badtime), 'IceMap:TimeOrder');
    badtime = time;
    badtime(2) = time(2) * 1.1;
    verifyError(testCase, @()validate_echoes(low, high, badtime), 'IceMap:TimeUniform');
end

function testINSShapeAndClock(testCase)
    ins = struct('x_m', [0 1], 'y_m', [0 0], 'z_delta_m', [0 0]);
    out = validate_ins(ins, 2);
    verifyEqual(testCase, size(out.x_m), [2 1]);
    ins.timestamp_s = [1 1];
    verifyError(testCase, @()validate_ins(ins, 2), 'IceMap:INSClock');
    ins = rmfield(ins, 'timestamp_s');
    ins.z_delta_m = 1;
    verifyError(testCase, @()validate_ins(ins, 2), 'IceMap:INSShape');
end

function testMainVersionLabels(testCase)
    cfg = icemap_config();
    verifyEqual(testCase, cfg.design1_version, 'v2');
    verifyEqual(testCase, cfg.design2_version, 'v3');
    catalog = algorithm_catalog();
    verifyEqual(testCase, catalog.d1{2}, 'design1_energy_ins');
    verifyEqual(testCase, catalog.d2{3}, 'design2_multifrequency_collaboration');
end

function testExplicitSurfaceGapPolicy(testCase)
    T = 5;
    N = 96;
    time = (0:N - 1) * (.002 / 299792458);
    track = 30 + (0:T - 1)';
    low = zeros(T, N);
    high = zeros(T, 1, 1, N);
    for t = 1:T
        low(t, track(t)) = 1;
        high(t, 1, 1, track(t)) = 1;
    end
    low(3, :) = 0;
    ins = struct('x_m', (0:T - 1)' * .1, 'y_m', zeros(T, 1), 'z_delta_m', [0; ones(T - 1, 1) * .001]);
    cfg = icemap_config(struct('surface_range_m', [.01 .06], 'spatial_sigma_m', .5, 'spatial_epsilon_m', .01, 'design1_version', 'v1'));
    verifyError(testCase, @()design1_select(low, high, time, ins, cfg), 'IceMap:SurfaceEvidence');
    cfg.methods.d1.allow_missing_surface_frames = true;
    out = design1_select(low, high, time, ins, cfg);
    verifyFalse(testCase, out.surface_evidence_valid(3));
    verifyEqual(testCase, out.surface_confidence(3), 0);
    verifyLessThan(testCase, max(abs(out.surface_track_bins - track)), 1.1);
end

function testInvalidConfigurationFailsBeforeNumericalWork(testCase)
    verifyError(testCase, @()icemap_config(struct('spatial_neighborhood_frames', 0)), 'IceMap:ConfigInteger');
    verifyError(testCase, @()icemap_config(struct('lambda_deconvolution', Inf)), 'IceMap:ConfigValue');
    verifyError(testCase, @()icemap_config(struct('coherence_floor', 2)), 'IceMap:ConfigFraction');
    verifyError(testCase, @()icemap_config(struct('deep_evidence_policy', 'guess')), 'IceMap:EvidencePolicy');
end

function testExplicitUnityDeepPolicyMarksUnidentifiableFrames(testCase)
    T = 2;
    N = 64;
    time = (0:N - 1) * (.002 / 299792458);
    low = zeros(T, N);
    low(:, 20) = 1;
    high = ones(T, 1, 1, N);
    cfg = icemap_config(struct('shallow_end_s', time(32), 'psf_sigma_grid_bins', 1, 'lambda_attenuation', .1, 'lambda_deconvolution', .1));
    verifyError(testCase, @()design2_select(low, high, time, cfg), 'IceMap:DeepEvidence');
    cfg.deep_evidence_policy = 'unity';
    out = design2_select(low, high, time, cfg);
    verifyFalse(testCase, any(out.attenuation_identifiable));
    verifyEqual(testCase, out.attenuation_relative, ones(T, N));
end
