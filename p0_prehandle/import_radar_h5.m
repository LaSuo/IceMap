function record = import_radar_h5(input_h5, options)
    % Adapt split-complex HDF5 echoes using explicit demonstration assumptions.
    % Source layout as read by MATLAB:
    %   X4M03_data_real/imag: [frames, low_samples]
    %   BGT60TR13C_data_real/imag: [frames, receivers, chirps, high_samples]
    % The sample axes are interpreted as complex range profiles, not raw ADC.
    % No FFT, measured synchronization, or measured INS is implied here.
    % options.LowRangeM/HighRangeM assign ideal endpoints to the native grids.
    % The returned record is always pipeline_debug, never calibrated data.
    if nargin < 2
        options = struct();
    end
    defaults = struct('FirstFrame', 1, 'MaxFrames', 4, 'MaxReceivers', Inf, ...
                      'MaxChirps', 8, 'RangeBins', 512, 'LowRangeM', [0 4], ...
                      'HighRangeM', [0 4], 'OutputRangeM', [0 4], ...
                      'FrameIntervalS', 0.1, 'SpeedMS', 0.5, ...
                      'AltitudeM', 1.5, 'DepthExtentM', 0.5, 'DepthBins', 256, ...
                      'AmplitudeMode', 'unit_rms', 'MaxReadMiB', 256);
    options = adapter_options(defaults, options);
    input_h5 = char(string(input_h5));
    if ~isfile(input_h5)
        error('IceMap:H5MissingFile', 'Raw HDF5 file does not exist: %s', input_h5);
    end
    low_shape = checked_pair(input_h5, 'X4M03', 2);
    high_shape = checked_pair(input_h5, 'BGT60TR13C', 4);
    if low_shape(1) ~= high_shape(1)
        error('IceMap:H5Frames', 'Band frame counts differ; ideal index synchronization requires equal counts.');
    elseif options.FirstFrame > low_shape(1)
        error('IceMap:H5FrameSelection', 'FirstFrame is outside the recording.');
    end
    frames = min(options.MaxFrames, low_shape(1) - options.FirstFrame + 1);
    receivers = min(options.MaxReceivers, high_shape(2));
    chirps = min(options.MaxChirps, high_shape(3));
    if frames < 2
        error('IceMap:H5TooFewFrames', 'Design3 needs at least two adjacent frames.');
    end
    if any([low_shape(end), high_shape(end)] < 8)
        error('IceMap:H5Samples', 'Each radar band needs at least eight native samples.');
    end
    % Conservative allowance for split-complex reads and double work arrays.
    read_mib = 64 * frames * (low_shape(end) + receivers * chirps * high_shape(end)) / 2^20;
    output_mib = 64 * frames * options.RangeBins * (1 + receivers * chirps) / 2^20;
    if read_mib + output_mib > options.MaxReadMiB
        error('IceMap:H5MemoryBudget', ...
              'Estimated adapter work %.1f MiB exceeds MaxReadMiB %.1f; reduce frames/chirps/receivers/bins or raise the explicit budget.', ...
              read_mib + output_mib, options.MaxReadMiB);
    end
    low = read_pair(input_h5, 'X4M03', [options.FirstFrame 1], [frames low_shape(2)]);
    high = read_pair(input_h5, 'BGT60TR13C', [options.FirstFrame 1 1 1], ...
                     [frames receivers chirps high_shape(4)]);
    low_grid = linspace(options.LowRangeM(1), options.LowRangeM(2), low_shape(2));
    high_grid = linspace(options.HighRangeM(1), options.HighRangeM(2), high_shape(4));
    range_m = linspace(options.OutputRangeM(1), options.OutputRangeM(2), options.RangeBins);
    low = interp1(low_grid, low.', range_m, 'linear').';
    flat = reshape(high, [], high_shape(4));
    flat = interp1(high_grid, flat.', range_m, 'linear').';
    high = reshape(flat, frames, receivers, chirps, options.RangeBins);
    [low, low_gain] = scale_amplitude(low, options.AmplitudeMode);
    [high, high_gain] = scale_amplitude(high, options.AmplitudeMode);
    clock = ((options.FirstFrame - 1):(options.FirstFrame + frames - 2)).' * options.FrameIntervalS;
    ins = struct('z_delta_m', zeros(frames, 1), ...
                 'x_m', (0:frames - 1).' * options.FrameIntervalS * options.SpeedMS, ...
                 'y_m', zeros(frames, 1), 'roll_rad', zeros(frames, 1), ...
                 'pitch_rad', zeros(frames, 1), 'yaw_rad', zeros(frames, 1), ...
                 'timestamp_s', clock);
    record = struct('radarl', low, 'radarh', high, ...
                    'time_s', 2 * range_m / 299792458, 'ins', ins, ...
                    'depth_m', linspace(0, options.DepthExtentM, options.DepthBins), ...
                    'acquisition_time_s', clock, 'workflow_mode', 'pipeline_debug', ...
                    'site_id', 'demo_unknown_site');
    record.sync_metadata = struct('method', 'assumed_equal_frame_indices', ...
                                  'measured', false, 'clock_offset_s', 0, ...
                                  'frame_interval_s', options.FrameIntervalS);
    record.sensor_registration = struct('measured', false, ...
                                        'rotation_low_to_high', eye(3), ...
                                        'translation_low_to_high_m', [0 0 0]);
    record.calibration_provenance = struct('status', 'engineering_initial', ...
                                           'uses_ideal_values', true, ...
                                           'source_file', input_h5, ...
                                           'source_low_shape_matlab', low_shape, ...
                                           'source_high_shape_matlab', high_shape, ...
                                           'selected_frames', [options.FirstFrame options.FirstFrame + frames - 1], ...
                                           'selected_receivers', receivers, 'selected_chirps', chirps, ...
                                           'amplitude_divisor_low', low_gain, ...
                                           'amplitude_divisor_high', high_gain, ...
                                           'adapter_options', options);
    record.experiment_metadata = struct('purpose', 'end_to_end_demonstration', ...
                                        'site_id', record.site_id, ...
                                        'measured_site_identity_available', false, ...
                                        'ideal_altitude_m', options.AltitudeM, ...
                                        'altitude_role', 'scene_metadata_only; the observed echo selects the surface', ...
                                        'assumptions', {{'native samples treated as range profiles', ...
                                                         'uniform assumed range endpoints', ...
                                                         'equal-index dual-band synchronization', ...
                                                         'constant-speed level trajectory with zero vertical motion', ...
                                                         'zero attitude and zero sensor lever arm', ...
                                                         'not an accuracy benchmark'}});
end

function options = adapter_options(defaults, supplied)
    if ~isstruct(supplied) || ~isscalar(supplied)
        error('IceMap:H5Options', 'Import options must be one scalar struct.');
    end
    options = defaults;
    names = fieldnames(supplied);
    for i = 1:numel(names)
        if ~isfield(defaults, names{i})
            error('IceMap:H5Option', 'Unknown import option: %s.', names{i});
        end
        options.(names{i}) = supplied.(names{i});
    end
    for item = {{'FirstFrame', 1}, {'MaxFrames', 2}, {'MaxReceivers', 1}, ...
                {'MaxChirps', 1}, {'RangeBins', 128}, {'DepthBins', 8}}
        name = item{1}{1};
        minimum = item{1}{2};
        value = options.(name);
        allow_inf = any(strcmp(name, {'MaxFrames', 'MaxReceivers', 'MaxChirps'}));
        if ~isnumeric(value) || ~isreal(value) || ~isscalar(value) || isnan(value) || ...
                value < minimum || (~isfinite(value) && ~allow_inf) || ...
                (isfinite(value) && value ~= fix(value))
            error('IceMap:H5Option', '%s requires an integer >=%d%s.', name, minimum, infinity_hint(allow_inf));
        end
    end
    for name = {'FrameIntervalS', 'AltitudeM', 'DepthExtentM', 'MaxReadMiB'}
        value = options.(name{1});
        if ~isnumeric(value) || ~isreal(value) || ~isscalar(value) || ~isfinite(value) || value <= 0
            error('IceMap:H5Option', '%s must be a positive finite scalar.', name{1});
        end
    end
    if ~isnumeric(options.SpeedMS) || ~isreal(options.SpeedMS) || ~isscalar(options.SpeedMS) || ...
            ~isfinite(options.SpeedMS) || options.SpeedMS < 0
        error('IceMap:H5Option', 'SpeedMS must be finite and nonnegative.');
    end
    for name = {'LowRangeM', 'HighRangeM', 'OutputRangeM'}
        value = options.(name{1});
        if ~isnumeric(value) || ~isreal(value) || ~isvector(value) || numel(value) ~= 2 || ...
                any(~isfinite(value)) || value(1) < 0 || value(2) <= value(1)
            error('IceMap:H5Range', '%s must contain increasing nonnegative endpoints in meters.', name{1});
        end
        options.(name{1}) = double(value(:).');
    end
    lo = max(options.LowRangeM(1), options.HighRangeM(1));
    hi = min(options.LowRangeM(2), options.HighRangeM(2));
    if options.OutputRangeM(1) < lo || options.OutputRangeM(2) > hi
        error('IceMap:H5RangeOverlap', 'OutputRangeM must lie inside both native range intervals; extrapolation is disabled.');
    end
    options.AmplitudeMode = char(string(options.AmplitudeMode));
    if ~any(strcmp(options.AmplitudeMode, {'unit_rms', 'preserve'}))
        error('IceMap:H5Amplitude', 'AmplitudeMode must be unit_rms or preserve.');
    end
end

function hint = infinity_hint(enabled)
    if enabled
        hint = ' or positive Inf';
    else
        hint = '';
    end
end

function shape = checked_pair(filename, prefix, rank)
    try
        real_info = h5info(filename, ['/' prefix '_data_real']);
        imag_info = h5info(filename, ['/' prefix '_data_imag']);
    catch ME
        error('IceMap:H5Dataset', 'Cannot read the %s real/imag pair: %s', prefix, ME.message);
    end
    shape = real_info.Dataspace.Size;
    if numel(shape) ~= rank || ~isequal(shape, imag_info.Dataspace.Size) || any(shape < 1)
        error('IceMap:H5Shape', '%s needs matching nonempty real/imag arrays of rank %d.', prefix, rank);
    end
    for info = {real_info, imag_info}
        if ~any(strcmp(info{1}.Datatype.Class, {'H5T_FLOAT', 'H5T_INTEGER'}))
            error('IceMap:H5Type', '%s real/imag datasets must be numeric.', prefix);
        end
    end
end

function value = read_pair(filename, prefix, start, count)
    real_part = h5read(filename, ['/' prefix '_data_real'], start, count);
    imag_part = h5read(filename, ['/' prefix '_data_imag'], start, count);
    value = complex(double(real_part), double(imag_part));
    if any(~isfinite(value(:)))
        error('IceMap:H5Finite', '%s contains NaN/Inf; invalid echoes are not replaced with ideal data.', prefix);
    end
    value = reshape(value, count);
end

function [value, divisor] = scale_amplitude(value, mode)
    peak = max(abs(value(:)));
    if peak == 0
        error('IceMap:H5ZeroSignal', 'Selected radar data contain no signal.');
    elseif ~isfinite(peak)
        error('IceMap:H5Amplitude', 'Selected radar amplitude is not finite.');
    end
    divisor = 1;
    if strcmp(mode, 'unit_rms')
        % Scaling by the peak first avoids overflow when computing energy.
        divisor = peak * sqrt(mean(abs(value(:) / peak).^2));
        value = value / divisor;
    end
end
