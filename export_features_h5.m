function export_features_h5(feature_mat, output_h5, site_id, record_id, relative_target_method)
    % Export the C-order schema consumed by p4_design4/main.py.
    % Formal targets require independent measurements. Explicit pipeline_debug
    % records may carry nominal demonstration targets with recorded provenance.
    % No density/thickness is inferred from radar amplitudes or filenames here.
    assert(~isfile(output_h5), 'Refusing to overwrite an existing feature record.');
    if nargin < 5
        relative_target_method = '';
    end
    assert(strlength(string(site_id)) > 0, 'Provide independent site_id.');
    S = load(feature_mat);
    required = {'phase', 'E', 'tau', 'time_s', 'scatter_valid'};
    for i = 1:numel(required)
        assert(isfield(S, required{i}), ['Missing ' required{i}]);
    end
    has_labels = isfield(S, 'ground_truth');
    G = struct();
    if has_labels
        assert(isfield(S, 'boundary_target') && strlength(string(relative_target_method)) > 0, 'Labeled export needs boundary_target and documented relative-density target derivation.');
        G = S.ground_truth;
        for name = {'layer_count', 'relative_density_target', 'layer_density_gt', ...
                  'layer_thickness_m_gt', 'density_profile_gt', 'depth_m'}
            assert(isfield(G, name{1}), ['Missing independent target ' name{1}]);
        end
    else
        assert(isfield(S, 'depth_m'), 'Unlabelled export needs calibrated depth_m; it is never inferred from time_s.');
        G.depth_m = S.depth_m;
        relative_target_method = 'not_available_unlabelled';
    end
    % A failed export leaves no partial record under the requested filename.
    destination = output_h5;
    folder = fileparts(destination);
    if isempty(folder)
        folder = pwd;
    end
    assert(isfolder(folder), 'Output directory must exist.');
    output_h5 = [tempname(folder) '.h5'];
    cleanup = onCleanup(@()delete_partial(output_h5));
    for name = {'phase', 'E', 'tau', 'time_s', 'scatter_valid'}
        write_array(output_h5, name{1}, S.(name{1}));
    end
    write_array(output_h5, 'depth_m', G.depth_m);
    if has_labels
        write_array(output_h5, 'boundary_target', S.boundary_target);
        for name = {'layer_count', 'relative_density_target', 'layer_density_gt', ...
                  'layer_thickness_m_gt', 'density_profile_gt'}
            write_array(output_h5, name{1}, G.(name{1}));
        end
    end
    h5writeatt(output_h5, '/', 'icemap_schema', 'icemap.features.v1');
    h5writeatt(output_h5, '/', 'has_labels', double(has_labels));
    h5writeatt(output_h5, '/', 'site_id', char(string(site_id)));
    h5writeatt(output_h5, '/', 'record_id', char(string(record_id)));
    if isfield(S, 'workflow_mode')
        h5writeatt(output_h5, '/', 'workflow_mode', S.workflow_mode);
    else
        h5writeatt(output_h5, '/', 'workflow_mode', 'unverified');
    end
    h5writeatt(output_h5, '/', 'relative_density_target_method', char(string(relative_target_method)));
    if isfield(G, 'boundary_target_method')
        h5writeatt(output_h5, '/', 'boundary_target_method', G.boundary_target_method);
    else
        h5writeatt(output_h5, '/', 'boundary_target_method', 'unspecified');
    end
    h5writeatt(output_h5, '/', 'axis_order', 'C: frames,samples; frames,layers');
    h5writeatt(output_h5, '/time_s', 'unit', 's, two-way propagation');
    h5writeatt(output_h5, '/depth_m', 'unit', 'm, from registered ice surface');
    if has_labels
        h5writeatt(output_h5, '/layer_density_gt', 'unit', 'g/cm^3');
        h5writeatt(output_h5, '/layer_thickness_m_gt', 'unit', 'm');
    end
    if isfield(S, 'params')
        h5writeatt(output_h5, '/', 'matlab_config_json', jsonencode(S.params));
    end
    if isfield(S, 'parameter_catalog')
        h5writeatt(output_h5, '/', 'parameter_catalog_json', jsonencode(S.parameter_catalog));
    end
    if isfield(S, 'params') && isfield(S.params, 'calibration_status')
        h5writeatt(output_h5, '/', 'calibration_status', S.params.calibration_status);
    end
    for name = {'execution_timing', 'execution_environment'}
        if isfield(S, name{1})
            h5writeatt(output_h5, '/', [name{1} '_json'], jsonencode(S.(name{1})));
        end
    end
    for name = {'sync_metadata', 'sensor_registration'}
        if isfield(S, name{1})
            h5writeatt(output_h5, '/', [name{1} '_json'], jsonencode(S.(name{1})));
        end
    end
    if isfield(S, 'acquisition_time_s')
        clock = S.acquisition_time_s(:);
        if numel(clock) ~= size(S.phase, 1) || any(~isfinite(clock)) || any(diff(clock) <= 0)
            error('IceMap:AcquisitionClock', 'Actual acquisition_time_s must be strictly increasing, one timestamp per observed frame.');
        end
        write_array(output_h5, 'acquisition_time_s', clock);
        h5writeatt(output_h5, '/acquisition_time_s', 'unit', 's, acquisition clock (not propagation ToF)');
    end
    quality_count = 0;
    sources = {'design1_metadata', 'design2_metadata'};
    names = {{'surface_jump_m', 'surface_confidence', 'surface_discontinuity', 'power_gain'}, ...
           {'rL_bins', 'relative_fit_residual', 'fit_residual_valid', 'attenuation_identifiable', 'psf_grid_edge', 'cross_frequency_mismatch'}};
    for group = 1:numel(sources)
        if ~isfield(S, sources{group})
            continue
        end
        for j = 1:numel(names{group})
            name = names{group}{j};
            if ~isfield(S.(sources{group}), name)
                continue
            end
            value = S.(sources{group}).(name);
            if size(value, 1) ~= size(S.phase, 1)
                error('IceMap:QualityExport', 'Quality diagnostic %s must have one row per frame.', name);
            end
            write_matrix(output_h5, ['quality/' name], value);
            quality_count = quality_count + 1;
        end
    end
    if quality_count > 0
        h5writeatt(output_h5, '/', 'quality_schema', 'icemap.quality.v1');
    end
    for item = {{'design1_metadata', 'discontinuity_evaluated', 'surface_discontinuity_evaluated'}, ...
              {'design2_metadata', 'mismatch_threshold_calibrated', 'cross_frequency_mismatch_evaluated'}}
        entry = item{1};
        if isfield(S, entry{1}) && isfield(S.(entry{1}), entry{2})
            write_matrix(output_h5, ['quality/' entry{3}], repmat(S.(entry{1}).(entry{2}), size(S.phase, 1), 1));
            h5writeatt(output_h5, '/', 'quality_schema', 'icemap.quality.v1');
        end
    end
    for name = {'experiment_metadata', 'calibration_provenance'}
        if isfield(S, name{1})
            if ~isstruct(S.(name{1})) || ~isscalar(S.(name{1}))
                error('IceMap:ExperimentMetadata', '%s must be one documented metadata struct.', name{1});
            end
            h5writeatt(output_h5, '/', [name{1} '_json'], jsonencode(S.(name{1})));
        end
    end
    if has_labels && isfield(G, 'boundary_times_s_gt')
        write_matrix(output_h5, 'boundary_times_s_gt', G.boundary_times_s_gt);
    end
    if has_labels && isfield(G, 'measurements')
        names = fieldnames(G.measurements);
        for i = 1:numel(names)
            write_array(output_h5, ['measurements/' names{i}], G.measurements.(names{i}));
        end
        h5writeatt(output_h5, '/', 'measurement_schema', 'icemap.repeated_measurements.v1');
    end
    if isfield(S, 'fingerprint_input_mode')
        h5writeatt(output_h5, '/', 'fingerprint_input_mode', S.fingerprint_input_mode);
    end
    if isfield(S, 'tau_events_s') && isfield(S, 'tau_event_bins') && isfield(S, 'tau_intervals_s')
        T = size(S.phase, 1);
        count = cellfun(@numel, S.tau_events_s);
        K = max([1; count(:)]);
        events = zeros(T, K);
        bins = zeros(T, K);
        valid = false(T, K);
        intervals = zeros(T, max(1, K - 1));
        for t = 1:T
            n = count(t);
            e = S.tau_events_s{t}(:).';
            b = S.tau_event_bins{t}(:).';
            d = S.tau_intervals_s{t}(:).';
            if numel(b) ~= n || numel(d) ~= max(n - 1, 0) || any(diff(e) <= 0) || any(~isfinite([e(:); b(:); d(:)]))
                error('IceMap:EventExport', 'Inconsistent event timestamps/bins/intervals in frame %d.', t);
            end
            events(t, 1:n) = e;
            bins(t, 1:n) = b - 1;
            valid(t, 1:n) = true;
            intervals(t, 1:max(n - 1, 0)) = d;
        end
        write_matrix(output_h5, 'tau_events_s', events);
        write_matrix(output_h5, 'tau_event_bins', bins);
        write_matrix(output_h5, 'tau_event_valid', valid);
        write_matrix(output_h5, 'tau_intervals_s', intervals);
        write_array(output_h5, 'tau_event_count', count);
        h5writeatt(output_h5, '/', 'icemap_event_schema', 'icemap.events.v1');
        h5writeatt(output_h5, '/tau_event_bins', 'index_base', 0);
    end
    if isfield(S, 'params') && isfield(S.params, 'scene')
        h5writeatt(output_h5, '/', 'scene_metadata_json', scene_to_json(S.params.scene));
    end
    versions = struct();
    methods = struct();
    for name = {'design1', 'design2', 'design3'}
        key = [name{1} '_metadata'];
        if isfield(S, key) && isfield(S.(key), 'algorithm_version')
            versions.(name{1}) = S.(key).algorithm_version;
        end
        if isfield(S, key) && isfield(S.(key), 'algorithm_method')
            methods.(name{1}) = S.(key).algorithm_method;
        end
    end
    if ~isfield(versions, 'design3') && isfield(S, 'algorithm_version')
        versions.design3 = S.algorithm_version;
    end
    if ~isfield(methods, 'design3') && isfield(S, 'algorithm_method')
        methods.design3 = S.algorithm_method;
    end
    if ~isempty(fieldnames(versions))
        h5writeatt(output_h5, '/', 'algorithm_versions_json', jsonencode(versions));
    end
    if ~isempty(fieldnames(methods))
        h5writeatt(output_h5, '/', 'algorithm_methods_json', jsonencode(methods));
    end
    if isfield(S, 'extra_features')
        if ~isstruct(S.extra_features) || ~isfield(S, 'extra_feature_valid') || ~isfield(S, 'extra_feature_units')
            error('IceMap:ExtraFeatureExport', 'Extra channels require validity masks and units.');
        end
        names = fieldnames(S.extra_features);
        for i = 1:numel(names)
            k = names{i};
            value = S.extra_features.(k);
            if ~isfield(S.extra_feature_valid, k) || ~isfield(S.extra_feature_units, k) || ~isequal(size(value), size(S.phase)) || ~isequal(size(S.extra_feature_valid.(k)), size(S.phase))
                error('IceMap:ExtraFeatureExport', 'Extended channel %s has inconsistent shape/mask/unit.', k);
            end
            valid = S.extra_feature_valid.(k);
            if ~isreal(value) || any(~isfinite(value(:))) || any(~ismember(valid(:), [0 1])) || any(value(~logical(valid)) ~= 0)
                error('IceMap:ExtraFeatureExport', 'Extended channel %s must be real finite and zero outside valid support.', k);
            end
            write_array(output_h5, ['extra_features/' k], value);
            write_array(output_h5, ['extra_feature_valid/' k], valid);
            h5writeatt(output_h5, ['/extra_features/' k], 'unit', S.extra_feature_units.(k));
        end
        h5writeatt(output_h5, '/', 'icemap_extra_schema', 'icemap.extra_features.v1');
    end
    assert(~isfile(destination), 'Output appeared during export; refusing overwrite.');
    [ok, message] = movefile(output_h5, destination);
    assert(ok, message);
end

function delete_partial(path)
    if isfile(path)
        delete(path);
    end
end

function write_matrix(filename, name, value)
    if ~isnumeric(value) && ~islogical(value)
        error('IceMap:MatrixExport', '%s must be numeric.', name);
    end
    if ~ismatrix(value) || ~isreal(value) || any(~isfinite(value(:)))
        error('IceMap:MatrixExport', '%s must be a finite real matrix.', name);
    end
    h5create(filename, ['/' name], fliplr(size(value)), 'Datatype', 'double');
    h5write(filename, ['/' name], double(value.'));
end

function write_array(filename, name, value)
    value = double(value);
    assert(all(isfinite(value(:))), ['Nonfinite ' name]);
    if isvector(value)
        h5create(filename, ['/' name], numel(value), 'Datatype', 'double');
        h5write(filename, ['/' name], value(:));
    else
        assert(ismatrix(value), 'Only vectors and matrices are accepted by feature schema v1.');
        h5create(filename, ['/' name], fliplr(size(value)), 'Datatype', 'double');
        h5write(filename, ['/' name], value.');
    end
end
