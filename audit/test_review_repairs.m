function tests = test_review_repairs
    tests = functiontests(localfunctions);
end

function setupOnce(tc)
    root = fileparts(fileparts(mfilename('fullpath')));
    addpath(genpath(root));
    tc.TestData.directory = tempname;
    mkdir(tc.TestData.directory);
end

function teardownOnce(tc)
    % Remove only individual files created by these fixtures (no recursive delete).
    folder = tc.TestData.directory;
    files = dir(folder);
    for k = 1:numel(files)
        if ~files(k).isdir
            delete(fullfile(folder, files(k).name));
        end
    end
    rmdir(folder);
end

function testUnlabelledEventExportAndFailureAtomicity(tc)
    folder = tc.TestData.directory;
    input = fullfile(folder, 'events.mat');
    output = fullfile(folder, 'events.h5');
    s = struct('phase', zeros(2, 16), 'E', ones(2, 16), 'tau', zeros(2, 16), ...
             'scatter_valid', true(2, 16), 'time_s', (0:15) * 1e-11, 'depth_m', (0:15) * .001);
    s.tau_event_bins = {[2 7]; 2};
    s.tau_events_s = {[1 6] * 1e-11; 1e-11};
    s.tau_intervals_s = {5e-11; []};
    save(input, '-struct', 's');
    export_features_h5(input, output, 'synthetic', 'event_fixture');
    verifyEqual(tc, h5readatt(output, '/', 'has_labels'), 0);
    verifyEqual(tc, h5read(output, '/tau_event_bins'), [1 1; 6 0]);
    verifyEqual(tc, h5read(output, '/tau_event_count'), [2; 1]);
    bad = fullfile(folder, 'partial.h5');
    s.tau_event_bins{1} = [2 7 9];
    save(input, '-struct', 's');
    verifyError(tc, @()export_features_h5(input, bad, 'synthetic', 'bad_fixture'), 'IceMap:EventExport');
    verifyFalse(tc, isfile(bad));
end

function testLowActualFramesAndComplex(tc)
    p = fullfile(tc.TestData.directory, 'low.txt');
    write_low(p, {'10', '[1 2 3 4]', '10.21', '[5 6 7 8]'});
    [cfg, t, x, base] = load_radarl(p, 1, 2.5e9);
    verifyEqual(tc, size(x), [2 1 2]);
    verifyEqual(tc, reshape(x(1, 1, :), 1, []), [1 + 3i 2 + 4i]);
    verifyEqual(tc, t, [0 .21], 'AbsTol', 1e-12);
    verifyEqual(tc, base, 10);
    verifyEqual(tc, cfg.actual_frames, 2);
end

function testIncompleteAndMalformedLow(tc)
    p = fullfile(tc.TestData.directory, 'badlow.txt');
    write_low(p, {'10', '1 2 3 4', '11'});
    verifyError(tc, @()load_radarl(p, 1, 2.5e9), 'IceMap:IncompleteFrame');
    verifyError(tc, @()parse_radar_numbers('system(''bad'')'), 'IceMap:NumericSyntax');
    verifyError(tc, @()parse_radar_numbers('1.2.3'), 'IceMap:NumericSyntax');
    verifyError(tc, @()parse_radar_numbers('NaN'), 'IceMap:NumericSyntax');
end

function testHighFrameLengthAndReceiverOrder(tc)
    p = fullfile(tc.TestData.directory, 'high.txt');
    write_high(p, {'20', '1', '5 6 7 8', '0', '1 2 3 4'});
    [cfg, t, x, base] = load_radarh(p);
    verifyEqual(tc, [size(x, 1) size(x, 2) size(x, 3) size(x, 4)], [1 2 1 4]);
    verifyEqual(tc, reshape(x(1, 1, 1, :), 1, []), 1:4);
    verifyEqual(tc, cfg.actual_frames, 1);
    verifyEqual(tc, t, 0);
    verifyEqual(tc, base, 20);
    write_high(p, {'20', '0', '1 2 3'});
    verifyError(tc, @()load_radarh(p), 'IceMap:FrameLength');
    write_high(p, {'20', '0', '1 2 3 4', '0', '1 2 3 4'});
    verifyError(tc, @()load_radarh(p), 'IceMap:DuplicateReceiver');
end

function testSyncNoFabrication(tc)
    [a, b, m] = synchronize_radar_frames([0 .2 .4 .6], [.01 .41 .61], .02);
    verifyEqual(tc, a, [1 3 4]); % Dropped acquisition is not filled by a duplicate.
    verifyEqual(tc, b, [1 2 3]);
    verifyLessThanOrEqual(tc, max(abs(m.time_error_s)), .02);
    verifyEqual(tc, m.synthetic_frames, 0);
    verifyEqual(tc, numel(unique(a)), numel(a));
    verifyError(tc, @()synchronize_radar_frames([0 .2], [1 1.2], .1), 'IceMap:NoTimeOverlap');
end

function testPSFFitUsesOnlyShallowBlock(tc)
    cfg = icemap_config();
    N = 64;
    T = 2;
    time = (0:N - 1) * 1e-11;
    cfg.shallow_end_s = time(32);
    cfg.psf_sigma_grid_bins = [1 2 3];
    cfg.lambda_attenuation = .1;
    cfg.lambda_deconvolution = .01;
    cfg.deep_evidence_policy = 'unity';
    cfg.solver_tolerance = 1e-7;
    low = complex(zeros(T, N));
    low(:, 30) = 1;
    high = complex(zeros(T, 1, 1, N));
    high(:, 1, 1, 30) = 1;
    a = design2_multifrequency_collaboration(low, high, time, cfg);
    high(:, 1, 1, 33) = 1e6;
    b = design2_multifrequency_collaboration(low, high, time, cfg);
    verifyEqual(tc, a.rL_bins, b.rL_bins);
    verifyEqual(tc, a.data_l(:, 1:32), b.data_l(:, 1:32));
end

function write_low(p, lines)
    fid = fopen(p, 'w');
    closer = onCleanup(@()fclose(fid));
    fprintf(fid, 'FPS: 5\nruntime: 100\nAreaStart: 0\n');
    for j = 4:10
        fprintf(fid, 'header%d\n', j);
    end
    for j = 1:numel(lines)
        fprintf(fid, '%s\n', lines{j});
    end
end

function write_high(p, lines)
    fid = fopen(p, 'w');
    closer = onCleanup(@()fclose(fid));
    fprintf(fid, 'RxNum: 2\nChirpPerFrame: 1\nSamplesPerChirp: 4\nFPS: 5\nruntime: 100\n');
    for j = 6:18
        fprintf(fid, 'header%d\n', j);
    end
    for j = 1:numel(lines)
        fprintf(fid, '%s\n', lines{j});
    end
end
