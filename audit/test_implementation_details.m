function tests = test_implementation_details
    tests = functiontests(localfunctions);
end

function setupOnce(tc)
    root = fileparts(fileparts(mfilename('fullpath')));
    addpath(genpath(root));
    tc.TestData.folder = tempname;
    mkdir(tc.TestData.folder);
end

function teardownOnce(tc)
    files = dir(tc.TestData.folder);
    for i = 1:numel(files)
        if ~files(i).isdir
            delete(fullfile(tc.TestData.folder, files(i).name));
        end
    end
    rmdir(tc.TestData.folder);
end

function testDiscontinuityIsDiagnosticAndFramesRemain(tc)
    cfg = icemap_config();
    cfg.surface_range_m = [.01 .15];
    cfg.spatial_sigma_m = .1;
    cfg.spatial_epsilon_m = .001;
    cfg.surface_discontinuity_m = .01;
    T = 3;
    N = 128;
    time = (0:N - 1) * 1e-11;
    low = complex(zeros(T, N));
    high = complex(zeros(T, 1, 1, N));
    track = [40 40 70];
    for t = 1:T
        low(t, track(t)) = 1;
        high(t, 1, 1, track(t)) = 1;
    end
    ins = struct('z_delta_m', zeros(T, 1), 'x_m', (0:T - 1)' * .1, 'y_m', zeros(T, 1));
    out = design1_energy_ins(low, high, time, ins, cfg);
    verifyEqual(tc, out.surface_discontinuity, [false; false; true]);
    verifyEqual(tc, size(out.data_l, 1), T);
    verifyFalse(tc, out.crack_detection_validated);
end

function testInitialCalibrationCatalogHasValuesUnitsAndSource(tc)
    cfg = icemap_config();
    rows = parameter_catalog(cfg);
    verifyEqual(tc, cfg.calibration_status, 'engineering_initial');
    verifyEqual(tc, cfg.shallow_end_s, 13e-9);
    verifyEqual(tc, cfg.psf_sigma_grid_bins, [8 16 24 32 48 64]);
    idx = find(strcmp({rows.name}, 'lambda_deconvolution'));
    verifyEqual(tc, rows(idx).unit, '1');
    verifyEqual(tc, rows(idx).source, 'engineering_initial');
    idx = find(strcmp({rows.name}, 'scattering_gamma'));
    verifyEqual(tc, rows(idx).source, 'mine_reported');
    cfg.scattering_gamma = 1.4;
    rows = parameter_catalog(cfg);
    idx = find(strcmp({rows.name}, 'scattering_gamma'));
    verifyEqual(tc, rows(idx).source, 'user_override_of_reported_value');
end

function testCrossFrequencyMismatchIsMeasuredNotSilentlyDiscarded(tc)
    cfg = icemap_config();
    N = 64;
    T = 2;
    time = (0:N - 1) * 1e-11;
    cfg.shallow_end_s = time(32);
    cfg.psf_sigma_grid_bins = [1 2];
    cfg.lambda_attenuation = .1;
    cfg.lambda_deconvolution = .01;
    cfg.deep_evidence_policy = 'unity';
    cfg.cross_frequency_max_relative_residual = 1e-4;
    low = complex(zeros(T, N));
    low(:, 16) = 1;
    high = complex(zeros(T, 1, 1, N));
    high(:, 1, 1, 20) = 1;
    out = design2_multifrequency_collaboration(low, high, time, cfg);
    verifyTrue(tc, all(out.cross_frequency_mismatch));
    verifyTrue(tc, all(out.fit_residual_valid(:, 1)));
    verifyFalse(tc, any(out.fit_residual_valid(:, 2)));
    verifyEqual(tc, size(out.data_l, 1), T);
end

function testMeasuredTargetsAndMetadataExport(tc)
    cfg = icemap_config();
    T = 2;
    N = 128;
    K = cfg.max_layers;
    time = (0:N - 1) * 1e-11;
    core = make_ground_truth([0 0 0; .05 .05 .05], [9 9 9], [10 10 10], [], [], []);
    tau = 2 * .05 * (1 + .845 * .9) / cfg.c0_m_s;
    core.boundary_times_s = repmat([.1e-9 .1e-9 + tau], T, 1);
    [G, b] = pack_ground_truth(core, time, .1e-9 * ones(T, 1), (0:N - 1) * .001, .9 * ones(T, 1), cfg);
    verifyEqual(tc, G.boundary_target_method, 'independently_measured_propagation_times');
    verifySize(tc, G.boundary_times_s_gt, [T K + 1]);
    S = struct('phase', zeros(T, N), 'E', ones(T, N), 'tau', zeros(T, N), 'time_s', time, 'scatter_valid', true(T, N), ...
             'ground_truth', G, 'boundary_target', b, 'params', cfg, ...
             'workflow_mode', 'pipeline_debug');
    S.experiment_metadata = struct('structure_id', 'fixture_ice', 'site_id', 'fixture_site', 'session_id', 'fixture_session', ...
                                 'environment_id', 'fixture_environment', 'water_body_id', 'fixture_body', 'domain', 'laboratory', 'setting', 'laboratory', ...
                                 'radar_config_id', 'fixture_dual', 'surface_condition_id', 'uncovered');
    S.calibration_provenance = struct('calibration_id', 'fixture', 'selection_role', 'external', 'external_reference_id', 'synthetic_fixture');
    source = fullfile(tc.TestData.folder, 'export.mat');
    target = fullfile(tc.TestData.folder, 'export.h5');
    save(source, '-struct', 'S');
    export_features_h5(source, target, 'fixture_site', 'fixture_record', 'synthetic_fixture');
    verifyEqual(tc, h5read(target, '/boundary_times_s_gt'), G.boundary_times_s_gt.');
    verifyEqual(tc, h5read(target, '/measurements/boundary_measurements_m'), core.boundary_measurements_m.');
    meta = jsondecode(h5readatt(target, '/', 'experiment_metadata_json'));
    verifyEqual(tc, meta.structure_id, 'fixture_ice');
end
