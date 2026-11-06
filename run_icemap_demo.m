function summary = run_icemap_demo(data_root, output_root, options)
    % Run raw HDF5 -> D1 v2 -> D2 v3 -> D3 v1 -> feature HDF5.
    % With the code and dataset in sibling directories, run_icemap_demo works
    % without arguments. A timestamped output directory is created beside them.
    %
    % Examples:
    %   summary = run_icemap_demo('path/to/icemap_dataset', 'path/to/demo_output');
    %   opts = struct('MaxFiles', 2, 'RunNetwork', true, ...
    %                 'PythonExecutable', 'path/to/torch310/python');
    %   summary = run_icemap_demo('path/to/icemap_dataset', 'path/to/output', opts);
    %   opts.Import = struct('MaxFrames', 4, 'RangeBins', 512, 'MaxChirps', 8);
    %   opts.MaxFiles = Inf;  % explicitly process every raw record
    %   opts.LabelMode = 'filename_nominal';
    %   opts.FilenameThicknessScaleM = 0.001;  % assumed filename unit -> m
    %   opts.LabelMode = 'none'; opts.NetworkEpochs = 0;  % no target metrics
    %
    % Missing calibration, synchronization, INS and targets use explicit ideal
    % assumptions. Source echoes are read without modification. The generated
    % records are marked pipeline_debug and cannot be used as accuracy evidence.
    % MATLAB requires Signal Processing Toolbox. The optional network step
    % requires Python with the dependencies in p4_design4/requirements.txt.
    root = fileparts(mfilename('fullpath'));
    addpath(fullfile(root, 'common'), fullfile(root, 'p0_prehandle'), ...
            fullfile(root, 'p1_design1'), fullfile(root, 'p2_design2'), fullfile(root, 'p3_design3'));
    if nargin < 1 || strlength(string(data_root)) == 0
        data_root = fullfile(fileparts(root), 'icemap_dataset');
    end
    if nargin < 2 || strlength(string(output_root)) == 0
        stamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss_SSS'));
        output_root = fullfile(fileparts(root), 'icemap_demo_output', stamp);
    end
    if nargin < 3
        options = struct();
    end
    options = demo_options(options);
    data_root = char(string(data_root));
    output_root = char(string(output_root));
    if ~isfolder(data_root)
        error('IceMap:DemoDataPath', 'Dataset directory does not exist: %s', data_root);
    end
    data_root = canonical_path(data_root);
    output_root = canonical_path(output_root);
    if strcmpi(output_root, data_root) || startsWith(lower(output_root), [lower(data_root) filesep])
        error('IceMap:DemoOutputPath', 'Demo outputs must be outside the raw dataset directory.');
    elseif strcmpi(output_root, root) || startsWith(lower(output_root), [lower(root) filesep])
        error('IceMap:DemoOutputPath', 'Demo outputs must be outside the submission code directory.');
    elseif isfolder(output_root)
        contents = dir(output_root);
        if any(~ismember({contents.name}, {'.', '..'}))
            error('IceMap:DemoOutputExists', 'Choose a new or empty output directory; existing results are not overwritten.');
        end
    elseif isfile(output_root)
        error('IceMap:DemoOutputPath', 'Output path names a file.');
    end
    files = dir(fullfile(data_root, '**', '*.h5'));
    if isempty(files)
        error('IceMap:DemoNoFiles', 'No .h5 files were found under %s.', data_root);
    end
    paths = fullfile({files.folder}, {files.name});
    paths = sort(paths);
    paths = paths(1:min(options.MaxFiles, numel(paths)));
    if exist('findpeaks', 'file') ~= 2 || exist('hilbert', 'file') ~= 2
        error('IceMap:DemoDependency', 'Signal Processing Toolbox functions findpeaks and hilbert are required.');
    end
    cfg = icemap_config(options.Processing);
    if ~strcmp(cfg.calibration_status, 'engineering_initial')
        error('IceMap:DemoCalibration', 'This ideal-value adapter requires calibration_status=engineering_initial.');
    end
    prepared_dir = fullfile(output_root, 'prepared_mat');
    matlab_dir = fullfile(output_root, 'features_mat');
    feature_dir = fullfile(output_root, 'features_h5');
    mkdir(prepared_dir);
    mkdir(matlab_dir);
    mkdir(feature_dir);
    fprintf('DEMONSTRATION: assumed calibration/INS and nominal targets; not an accuracy benchmark.\n');
    fprintf('Processing %d raw record(s) with D1 %s, D2 %s, D3 %s.\n', ...
            numel(paths), cfg.design1_version, cfg.design2_version, cfg.design3_version);
    entries = cell(numel(paths), 1);
    for i = 1:numel(paths)
        started = tic;
        record = import_radar_h5(paths{i}, options.Import);
        record.experiment_metadata.raw_record_index = i;
        cfg_import = cfg;
        cfg_import.scene.uav.altitude_agl_m = record.experiment_metadata.ideal_altitude_m;
        cfg_import.scene.uav.speed_m_s = record.calibration_provenance.adapter_options.SpeedMS;
        cfg_import.scene.uav.frame_interval_s = record.sync_metadata.frame_interval_s;
        [~, stem] = fileparts(paths{i});
        input_mat = fullfile(prepared_dir, sprintf('record_%04d.mat', i));
        feature_mat = fullfile(matlab_dir, sprintf('record_%04d.mat', i));
        feature_h5 = fullfile(feature_dir, sprintf('record_%04d.h5', i));
        save(input_mat, '-struct', 'record', '-v7.3');
        clear record;
        features = run_icemap(input_mat, feature_mat, cfg_import);
        [features, target_method, target_details] = demo_targets(features, stem, options);
        features.experiment_metadata.target_assumptions = target_details;
        save(feature_mat, '-struct', 'features', '-v7.3');
        export_features_h5(feature_mat, feature_h5, 'demo_unknown_site', stem, target_method);
        entries{i} = struct('source', paths{i}, 'prepared_mat', input_mat, ...
                            'feature_mat', feature_mat, 'feature_h5', feature_h5, ...
                            'frames', size(features.phase, 1), 'range_bins', size(features.phase, 2), ...
                            'has_nominal_labels', isfield(features, 'ground_truth'), ...
                            'target_assumptions', target_details, 'elapsed_seconds', toc(started));
        fprintf('[%d/%d] %s -> %d x %d fingerprints (%.2f s).\n', ...
                i, numel(paths), stem, entries{i}.frames, entries{i}.range_bins, entries{i}.elapsed_seconds);
        clear features;
    end
    script = fullfile(root, 'p4_design4', 'run_feature_demo.py');
    network_dir = fullfile(output_root, 'network_demo');
    network_command = strjoin({shell_quote(options.PythonExecutable), '-B', shell_quote(script), ...
                               '--data-dir', shell_quote(feature_dir), ...
                               '--output-dir', shell_quote(network_dir), ...
                               '--epochs', num2str(options.NetworkEpochs), ...
                               '--device', options.Device}, ' ');
    summary = struct('workflow_mode', 'pipeline_debug', 'data_root', data_root, ...
                     'output_root', output_root, 'options', options, 'records', {entries}, ...
                     'feature_directory', feature_dir, 'network_command', network_command, ...
                     'network_executed', false, 'network_status', [], ...
                     'evaluation_role', 'demonstration_only; no independent site split or accuracy claim');
    save(fullfile(output_root, 'demo_manifest.mat'), 'summary');
    if options.RunNetwork
        fprintf('Starting the optional network demonstration.\n');
        status = system(network_command, '-echo');
        summary.network_executed = true;
        summary.network_status = status;
        save(fullfile(output_root, 'demo_manifest.mat'), 'summary');
        if status ~= 0
            error('IceMap:DemoNetwork', ...
                  'Network demo exited with status %d. MATLAB features are saved. Check PythonExecutable/dependencies and rerun the printed command:\n%s', ...
                  status, network_command);
        end
    else
        fprintf('MATLAB stages and HDF5 export completed. Run the following in a terminal for Design4:\n%s\n', network_command);
    end
end

function options = demo_options(supplied)
    defaults = struct('MaxFiles', 2, 'Import', struct(), 'Processing', struct(), ...
                      'LabelMode', 'ideal', 'IdealThicknessM', [0.1 0.15], ...
                      'IdealDensityGcm3', [0.85 0.90], 'FilenameThicknessScaleM', 0.001, ...
                      'RunNetwork', false, 'PythonExecutable', 'python', ...
                      'NetworkEpochs', 1, 'Device', 'auto');
    if ~isstruct(supplied) || ~isscalar(supplied)
        error('IceMap:DemoOptions', 'Options must be one scalar struct.');
    end
    options = defaults;
    names = fieldnames(supplied);
    for i = 1:numel(names)
        if ~isfield(defaults, names{i})
            error('IceMap:DemoOption', 'Unknown demo option: %s.', names{i});
        end
        options.(names{i}) = supplied.(names{i});
    end
    value = options.MaxFiles;
    if ~isnumeric(value) || ~isreal(value) || ~isscalar(value) || isnan(value) || value < 1 || ...
            (isfinite(value) && value ~= fix(value))
        error('IceMap:DemoOption', 'MaxFiles requires a positive integer or Inf.');
    end
    for name = {'Import', 'Processing'}
        if ~isstruct(options.(name{1})) || ~isscalar(options.(name{1}))
            error('IceMap:DemoOption', '%s must be one scalar struct.', name{1});
        end
    end
    options.LabelMode = char(string(options.LabelMode));
    if ~any(strcmp(options.LabelMode, {'ideal', 'filename_nominal', 'none'}))
        error('IceMap:DemoLabelMode', 'LabelMode must be ideal, filename_nominal or none.');
    end
    for name = {'IdealThicknessM', 'IdealDensityGcm3'}
        value = options.(name{1});
        if ~isnumeric(value) || ~isreal(value) || ~isvector(value) || isempty(value) || any(~isfinite(value)) || any(value <= 0)
            error('IceMap:DemoTargets', '%s must contain finite positive layer values.', name{1});
        end
        options.(name{1}) = double(value(:).');
    end
    if numel(options.IdealThicknessM) ~= numel(options.IdealDensityGcm3)
        error('IceMap:DemoTargets', 'Ideal thickness and density must describe the same number of layers.');
    end
    value = options.FilenameThicknessScaleM;
    if ~isnumeric(value) || ~isreal(value) || ~isscalar(value) || ~isfinite(value) || value <= 0
        error('IceMap:DemoTargets', 'FilenameThicknessScaleM must be positive and finite.');
    end
    if ~islogical(options.RunNetwork) || ~isscalar(options.RunNetwork)
        error('IceMap:DemoOption', 'RunNetwork must be a logical scalar.');
    end
    value = options.NetworkEpochs;
    if ~isnumeric(value) || ~isreal(value) || ~isscalar(value) || ~isfinite(value) || value < 0 || value ~= fix(value)
        error('IceMap:DemoOption', 'NetworkEpochs must be a nonnegative integer; zero runs forward inference only.');
    end
    options.PythonExecutable = char(string(options.PythonExecutable));
    options.Device = char(string(options.Device));
    if ~any(strcmp(options.Device, {'auto', 'cpu', 'cuda'}))
        error('IceMap:DemoOption', 'Device must be auto, cpu or cuda.');
    elseif isempty(options.PythonExecutable)
        error('IceMap:DemoOption', 'PythonExecutable cannot be empty.');
    elseif strcmp(options.LabelMode, 'none') && options.NetworkEpochs > 0
        error('IceMap:DemoUnlabelled', 'LabelMode=none requires NetworkEpochs=0 for forward-only demonstration.');
    end
end

function [features, method, details] = demo_targets(features, stem, options)
    details = struct('mode', options.LabelMode, 'independent_measurements', false);
    method = '';
    if strcmp(options.LabelMode, 'none')
        details.description = 'Unlabelled forward demonstration; no target metrics.';
        return
    elseif strcmp(options.LabelMode, 'ideal')
        thickness = options.IdealThicknessM;
        density = options.IdealDensityGcm3;
        details.description = 'User-configurable constant ideal layers, unrelated to measured ice truth.';
    else
        tokens = regexp(stem, 'h_([0-9]+(?:\.[0-9]+)?)_rho_([0-9]+(?:\.[0-9]+)?)', 'tokens');
        if isempty(tokens)
            error('IceMap:DemoFilenameLabels', 'Filename has no h_<value>_rho_<value> pairs: %s', stem);
        end
        thickness = cellfun(@(x) str2double(x{1}), tokens) * options.FilenameThicknessScaleM;
        density = cellfun(@(x) str2double(x{2}), tokens);
        details.filename_thickness_scale_m = options.FilenameThicknessScaleM;
        details.description = 'Filename nominal values; thickness unit scale is an explicit assumption, not verified truth.';
    end
    K = features.params.max_layers;
    L = numel(thickness);
    if L < 1 || L > K || any(~isfinite([thickness density])) || any([thickness density] <= 0)
        error('IceMap:DemoTargets', 'Demo targets need 1..%d layers with finite positive thickness/density.', K);
    end
    T = size(features.phase, 1);
    N = numel(features.time_s);
    surface_bin = round(features.design1_metadata.surface_track_bins(1));
    surface_time = features.time_s(surface_bin);
    index = 1 + features.params.dielectric_density_coefficient * density;
    boundaries = surface_time + [0 cumsum(2 * thickness .* index / features.params.c0_m_s)];
    if boundaries(end) > features.time_s(end)
        error('IceMap:DemoTargetWindow', ...
              'Ideal bottom lies outside the assumed range window; enlarge range endpoints or reduce ideal thickness. Targets are not clipped.');
    elseif features.depth_m(end) < sum(thickness)
        error('IceMap:DemoDepthWindow', 'Import.DepthExtentM must cover the ideal total thickness %.6g m.', sum(thickness));
    end
    bins = round(interp1(features.time_s, 1:N, boundaries, 'linear'));
    if numel(unique(bins)) ~= L + 1 || any(~isfinite(bins))
        error('IceMap:DemoTargetResolution', 'Ideal interfaces collapse on the sample grid; increase Import.RangeBins.');
    end
    G = struct('layer_count', L * ones(T, 1), ...
               'relative_density_target', zeros(T, K), ...
               'layer_density_gt', zeros(T, K), 'layer_thickness_m_gt', zeros(T, K), ...
               'density_profile_gt', zeros(T, numel(features.depth_m)), ...
               'depth_m', features.depth_m, 'boundary_times_s_gt', zeros(T, K + 1), ...
               'boundary_target_method', 'demo_projected_ideal_layers_at_tracked_surface');
    G.relative_density_target(:, 1:L) = repmat(density / mean(density), T, 1);
    G.layer_density_gt(:, 1:L) = repmat(density, T, 1);
    G.layer_thickness_m_gt(:, 1:L) = repmat(thickness, T, 1);
    G.boundary_times_s_gt(:, 1:L + 1) = repmat(boundaries, T, 1);
    edges = [0 cumsum(thickness)];
    for j = 1:L
        support = features.depth_m >= edges(j) & features.depth_m < edges(j + 1);
        G.density_profile_gt(:, support) = density(j);
    end
    features.boundary_target = zeros(T, N);
    features.boundary_target(:, bins) = 1;
    features.ground_truth = G;
    method = 'demo_proxy_density_over_layer_mean';
    details.thickness_m = thickness;
    details.density_g_cm3 = density;
    details.relative_target_definition = method;
    details.surface_time_s = surface_time;
    details.boundary_times_s = boundaries;
end

function path = canonical_path(path)
    % Resolves relative segments even when the output directory does not exist.
    path = char(java.io.File(path).getCanonicalPath());
end

function quoted = shell_quote(value)
    % Paths are individual arguments; never interpolate unquoted user values.
    if any(value == char(10) | value == char(13) | value == char(0))
        error('IceMap:DemoCommandPath', 'Command paths cannot contain NUL/newline characters.');
    end
    if ispc
        if any(ismember(value, '"%!?&|<>^'))
            error('IceMap:DemoCommandPath', 'Windows command paths contain unsupported shell metacharacters. Run Python manually or choose a simple output path.');
        end
        quoted = ['"' value '"'];
    else
        quoted = ['''' strrep(value, '''', '''"''"''') ''''];
    end
end
