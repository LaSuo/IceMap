function tests = test_h5_adapter
    tests = functiontests(localfunctions);
end

function setupOnce(tc)
    root = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(root, 'p0_prehandle'));
    tc.TestData.folder = tempname;
    mkdir(tc.TestData.folder);
end

function teardownOnce(tc)
    files = dir(tc.TestData.folder);
    for i = 1:numel(files)
        if ~files(i).isdir
            delete(fullfile(files(i).folder, files(i).name));
        end
    end
    rmdir(tc.TestData.folder);
end

function testSeparateNativeGridsProduceCompatibleEchoes(tc)
    filename = fullfile(tc.TestData.folder, 'valid.h5');
    make_fixture(filename, 3, 3, false);
    original = h5read(filename, '/X4M03_data_real');
    record = import_radar_h5(filename, struct('RangeBins', 128, 'MaxFrames', 2));
    verifySize(tc, record.radarl, [2 128]);
    verifySize(tc, record.radarh, [2 2 3 128]);
    verifyEqual(tc, record.workflow_mode, 'pipeline_debug');
    verifyTrue(tc, record.calibration_provenance.uses_ideal_values);
    verifyFalse(tc, record.sync_metadata.measured);
    verifyEqual(tc, sqrt(mean(abs(record.radarl(:)).^2)), 1, 'AbsTol', 1e-12);
    verifyEqual(tc, diff(record.time_s), repmat(8 / (127 * 299792458), 1, 127), 'AbsTol', 1e-22);
    verifyEqual(tc, h5read(filename, '/X4M03_data_real'), original);
end

function testFrameMismatchCannotBeSilentlySynchronized(tc)
    filename = fullfile(tc.TestData.folder, 'different_frames.h5');
    make_fixture(filename, 3, 2, false);
    verifyError(tc, @()import_radar_h5(filename), 'IceMap:H5Frames');
end

function testCorruptedEchoIsNotReplacedWithIdealValues(tc)
    filename = fullfile(tc.TestData.folder, 'nonfinite.h5');
    make_fixture(filename, 3, 3, true);
    verifyError(tc, @()import_radar_h5(filename), 'IceMap:H5Finite');
end

function testMissingBandHasSpecificError(tc)
    filename = fullfile(tc.TestData.folder, 'missing_band.h5');
    h5create(filename, '/unrelated_data', [3 20]);
    verifyError(tc, @()import_radar_h5(filename), 'IceMap:H5Dataset');
end

function testExtrapolationAndOversizedReadsAreRejected(tc)
    filename = fullfile(tc.TestData.folder, 'budgets.h5');
    make_fixture(filename, 3, 3, false);
    verifyError(tc, @()import_radar_h5(filename, struct('OutputRangeM', [0 5])), 'IceMap:H5RangeOverlap');
    verifyError(tc, @()import_radar_h5(filename, struct('MaxReadMiB', 0.001)), 'IceMap:H5MemoryBudget');
end

function testLastSingleFrameCannotEnterDensityExtraction(tc)
    filename = fullfile(tc.TestData.folder, 'last_frame.h5');
    make_fixture(filename, 3, 3, false);
    verifyError(tc, @()import_radar_h5(filename, struct('FirstFrame', 3)), 'IceMap:H5TooFewFrames');
end

function make_fixture(filename, low_frames, high_frames, corrupt)
    low = repmat(exp(-((linspace(0, 4, 64) - 1.5) / 0.3).^2), low_frames, 1);
    high = repmat(reshape(exp(-((linspace(0, 4, 80) - 1.5) / 0.2).^2), 1, 1, 1, 80), high_frames, 2, 3, 1);
    if corrupt
        low(1, 1) = NaN;
    end
    arrays = {low, zeros(size(low)), high, zeros(size(high))};
    names = {'X4M03_data_real', 'X4M03_data_imag', 'BGT60TR13C_data_real', 'BGT60TR13C_data_imag'};
    for i = 1:numel(names)
        h5create(filename, ['/' names{i}], size(arrays{i}), 'Datatype', 'double');
        h5write(filename, ['/' names{i}], arrays{i});
    end
end
