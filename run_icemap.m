function [features, echo_views] = run_icemap(input_mat, output_mat, overrides)
    % MATLAB stages only. Dataset-specific import and label creation are separate.
    % Required MAT fields: radarl [T,N], radarh [T,R,C,N], time_s [1,N],
    % ins.z_delta_m/ins.x_m/ins.y_m. Ground truth is optional for inference.
    % Optional second output contains amplitude maps for the demo figure.
    if nargin < 3
        overrides = struct();
    end
    root = fileparts(mfilename('fullpath'));
    addpath(fullfile(root, 'common'), fullfile(root, 'p1_design1'), ...
            fullfile(root, 'p2_design2'), fullfile(root, 'p3_design3'));
    cfg = icemap_config(overrides);
    S = load(input_mat);
    stage_timer = tic;
    d1 = design1_select(S.radarl, S.radarh, S.time_s, S.ins, cfg);
    duration1 = toc(stage_timer);
    stage_timer = tic;
    d2 = design2_select(d1.data_l, d1.data_h, d1.time_s, cfg);
    duration2 = toc(stage_timer);
    stage_timer = tic;
    if strcmp(cfg.fingerprint_input_mode, 'dual_band')
        features = design3_select(d2.data_l, d2.data_h, d2.time_s, cfg);
    elseif strcmp(cfg.fingerprint_input_mode, 'fused')
        % A scalar fused stream cannot retain receiver/chirp diversity. Replication
        % only supplies the expected tensor layout; it creates no new observations.
        shape = size(d2.data_h);
        fused_high = repmat(reshape(d2.fused, shape(1), 1, 1, shape(4)), 1, shape(2), shape(3), 1);
        features = design3_select(d2.fused, fused_high, d2.time_s, cfg);
    else
        error('IceMap:FingerprintInput', 'Unsupported fingerprint stream mode.');
    end
    features.fingerprint_input_mode = cfg.fingerprint_input_mode;
    features.parameter_catalog = parameter_catalog(cfg);
    duration3 = toc(stage_timer);
    features.execution_timing = struct('D1_seconds', duration1, 'D2_seconds', duration2, ...
                                     'D3_seconds', duration3, 'observations', size(S.radarl, 1), 'scope', 'batch array processing, excluding acquisition/save/network/asynchronous scheduling');
    features.execution_environment = struct('matlab_version', version, 'matlab_release', version('-release'), 'architecture', computer, 'toolboxes', ver);
    for name = {'experiment_metadata', 'calibration_provenance'}
        if isfield(S, name{1})
            features.(name{1}) = S.(name{1});
        end
    end
    for name = {'acquisition_time_s', 'sync_metadata', 'sensor_registration', 'depth_m'}
        if isfield(S, name{1})
            features.(name{1}) = S.(name{1});
        end
    end
    features.design1_metadata = rmfield(d1, {'data_l', 'data_h', 'geometry_low', 'geometry_high'});
    features.design2_metadata = rmfield(d2, {'data_l', 'data_h', 'fused'});
    features.design3_metadata = struct('algorithm_version', features.algorithm_version, 'algorithm_method', features.algorithm_method);
    if isfield(S, 'ground_truth')
        features.ground_truth = S.ground_truth;
    end
    if isfield(S, 'site_id')
        features.site_id = S.site_id;
    end
    if isfield(S, 'workflow_mode')
        features.workflow_mode = S.workflow_mode;
    else
        features.workflow_mode = 'unverified';
    end
    % ToF-domain boundary targets must undergo the same registration as echoes.
    if isfield(S, 'boundary_target')
        b = S.boundary_target;
        assert(isequal(size(b), size(S.radarl)));
        for t = 1:size(b, 1)
            b(t, :) = interp1(1:numel(S.time_s), double(b(t, :)), ...
                           (1:numel(S.time_s)) + d1.shift_bins(t), 'nearest', 0);
        end
        features.boundary_target = b;
        if isfield(features, 'ground_truth') && isfield(features.ground_truth, 'boundary_times_s_gt')
            for t = 1:size(b, 1)
                valid = 1:(features.ground_truth.layer_count(t) + 1);
                features.ground_truth.boundary_times_s_gt(t, valid) = features.ground_truth.boundary_times_s_gt(t, valid) - d1.delay_correction_s(t);
            end
        end
    end
    if nargout > 1
        shape = size(S.radarl);
        echo_views = struct('raw_low', abs(S.radarl), ...
                            'raw_high', high_band_rms(S.radarh, shape), ...
                            'aligned_low', abs(d2.data_l), ...
                            'aligned_high', high_band_rms(d2.data_h, shape), ...
                            'fused', abs(d2.fused));
    end
    save(output_mat, '-struct', 'features', '-v7.3');
end

function amplitude = high_band_rms(echo, shape)
    amplitude = reshape(sqrt(mean(mean(abs(echo).^2, 2), 3)), shape);
end
