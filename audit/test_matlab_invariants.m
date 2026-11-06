function tests = test_matlab_invariants
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    root = fileparts(fileparts(mfilename('fullpath')));
    addpath(root, fullfile(root, 'common'), fullfile(root, 'p1_design1'), fullfile(root, 'p2_design2'), fullfile(root, 'p3_design3'));
    testCase.TestData.root = root;
end

function testSurfaceTrackingAndAnnealing(testCase)
    T = 60;
    N = 240;
    time = (0:N - 1) * 1e-11;
    cfg = icemap_config();
    cfg.surface_range_m = [.05 .25];
    cfg.spatial_sigma_m = .5;
    cfg.spatial_epsilon_m = .01;
    low = complex(zeros(T, N));
    high = complex(zeros(T, 2, 2, N));
    track = 80 + mod((1:T).', 3);
    dr = cfg.c0_m_s * mean(diff(time)) / 2;
    for t = 1:T
        low(t, track(t)) = 1;
        high(t, :, :, track(t)) = 1;
    end
    ins.z_delta_m = [0; diff(track) * dr];
    ins.x_m = (0:T - 1)' * .02;
    ins.y_m = zeros(T, 1);
    out = design1_energy_ins(low, high, time, ins, cfg);
    verifyEqual(testCase, out.surface_track_bins, track);
    verifyEqual(testCase, out.surface_sigma_bins(1), 160);
    verifyEqual(testCase, out.surface_sigma_bins(50:end), 80 * ones(T - 49, 1));
    verifyEqual(testCase, out.surface_zeta_m, zeros(T, 1), 'AbsTol', 1e-12);
end

function testMaskAndPhysicalDelay(testCase)
    T = 4;
    N = 128;
    time = (0:N - 1) * 1e-11;
    cfg = icemap_config();
    cfg.boundary_phase_threshold_rad = .4;
    cfg.boundary_dilation_bins = 2;
    cfg.tof_min_peak_distance_bins = 10;
    cfg.tof_min_prominence = .2;
    cfg.tof_smooth_bins = 1;
    cfg.phase_trend_bins = 21;
    low = complex(zeros(T, N));
    low(:, 40) = 1i;
    low(:, 80) = .8i;
    x = exp(1i * [zeros(1, 59) 1.5 * ones(1, N - 59)]);
    high = repmat(reshape(x, 1, 1, 1, N), T, 2, 2, 1);
    out = design3_physical_fingerprints(low, high, time, cfg);
    verifyTrue(testCase, any(out.boundary_mask(:)));
    verifyEqual(testCase, out.E(out.boundary_mask), zeros(nnz(out.boundary_mask), 1));
    for t = 1:T
        verifyEqual(testCase, out.tau_intervals_s{t}, 40e-11, 'AbsTol', 1e-14);
    end
end
