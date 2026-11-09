function run_opf_raw_features(raw_dir, match_csv, output_dir, options)
    % Process indexed raw observations through D1, D2, and D3 without labels.
    if nargin < 4
        options = struct();
    end
    defaults = struct('LowStepM', 0.001, 'HighStepM', 0.001, 'GridStepM', 0.001, ...
                      'ChirpStride', 8, 'MatchedOnly', false, 'Processing', struct());
    for field = fieldnames(options).'
        if ~isfield(defaults, field{1})
            error('IceMap:RecoveryOptions', 'Unknown recovery option %s.', field{1});
        end
        defaults.(field{1}) = options.(field{1});
    end
    options = defaults;
    steps = [options.LowStepM, options.HighStepM, options.GridStepM];
    if any(~isfinite(steps)) || any(steps <= 0) || options.ChirpStride < 1 || ...
            options.ChirpStride ~= fix(options.ChirpStride)
        error('IceMap:RecoveryGrid', 'Grid steps and chirp stride must be positive.');
    end
    root = fileparts(mfilename('fullpath'));
    addpath(fullfile(root,'common'), fullfile(root,'p1_design1'), ...
            fullfile(root,'p2_design2'), fullfile(root,'p3_design3'));
    settings = struct('surface_range_m',[0.60 1.18], 'shallow_end_s',9e-9, ...
                      'tof_min_prominence',0.015, 'tof_skeleton_sigma_bins',10, ...
                      'solver_tolerance',1e-6, 'attenuation_floor',0.05, ...
                      'deep_evidence_policy','unity');
    for field = fieldnames(options.Processing).'
        settings.(field{1}) = options.Processing.(field{1});
    end
    cfg = icemap_config(settings);
    matches = readtable(match_csv, 'Delimiter', ',', 'ReadVariableNames', true, ...
                        'TextType', 'string', 'VariableNamingRule', 'preserve');
    required_columns = {'key', 'raw_filename', 'status'};
    if ~all(ismember(required_columns, matches.Properties.VariableNames))
        error('IceMap:RecoveryManifest', 'The CSV must contain key, raw_filename, and status.');
    end
    if ~isfolder(output_dir)
        mkdir(output_dir);
    end
    for i = 1:height(matches)
        if ismissing(matches.raw_filename(i)) || contains(matches.raw_filename(i),"|") || ...
                (options.MatchedOnly && matches.status(i) ~= "unique_identifier")
            continue;
        end
        key = char(matches.key(i));
        destination = fullfile(output_dir,[key '.mat']);
        if isfile(destination)
            continue;
        end
        source = fullfile(raw_dir,matches.raw_filename(i));
        started = tic;
        [radarl, radarh, time_s, ins] = load_native_pair(source, options, cfg.c0_m_s);
        d1 = design1_select(radarl,radarh,time_s,ins,cfg);
        d2 = design2_select(d1.data_l,d1.data_h,d1.time_s,cfg);
        reference = struct('low',d1.geometry_low,'high',d1.geometry_high);
        d3 = design3_select(d2.data_l,d2.data_h,d2.time_s,cfg,reference);
        out = struct('phase',single(d3.phase), 'E',single(d3.E), 'tau',single(d3.tau), ...
                     'scatter_valid',d3.scatter_valid, 'time_s',d3.time_s, ...
                     'surface_track',single(d1.surface_track_bins - 1), ...
                     'shift_bins',single(d1.shift_bins), 'low_envelope',single(abs(d2.data_l)), ...
                     'high_envelope',single(high_envelope(d2.data_h)), ...
                     'raw_low_envelope',single(abs(radarl)), ...
                     'raw_high_envelope',single(high_envelope(radarh)), ...
                     'phase_coherence',single(d3.phase_coherence), ...
                     'scattering_coherence',single(d3.scattering_coherence), ...
                     'key',key, 'raw_filename',char(matches.raw_filename(i)), ...
                     'design1_version',cfg.design1_version, 'design2_version',cfg.design2_version, ...
                     'design3_version',cfg.design3_version, 'seconds',toc(started));
        if strcmp(cfg.design3_version,'v6')
            out.low_echo_vote = single(d3.extra_features.low_echo_vote);
            out.high_echo_vote = single(d3.extra_features.high_echo_vote);
        end
        partial = [destination '.partial'];
        save(partial,'-struct','out','-v7');
        movefile(partial,destination);
        fprintf('[%d/%d] %s: %d x %d, %.2f s\n', i,height(matches),key,size(out.phase,1),size(out.phase,2),toc(started));
    end
end

function [low, high, time_s, ins] = load_native_pair(path, options, c0)
    lo = h5info(path,'/X4M03_data_real');
    hi = h5info(path,'/BGT60TR13C_data_real');
    sl = lo.Dataspace.Size;
    sh = hi.Dataspace.Size;
    if numel(sl) ~= 2 || numel(sh) ~= 4 || sl(1) ~= sh(1) || sl(1) < 2
        error('IceMap:RecoveryShape','Expected corresponding low and high frame axes.');
    end
    T = sl(1);
    C = 1 + floor((sh(3)-1)/options.ChirpStride);
    low = read_complex(path,'X4M03',[1 1],sl,[1 1]);
    high = read_complex(path,'BGT60TR13C',[1 1 1 1], ...
                        [T sh(2) C sh(4)],[1 1 options.ChirpStride 1]);
    end_m = max((sl(2)-1)*options.LowStepM,(sh(4)-1)*options.HighStepM);
    grid = (0:round(end_m/options.GridStepM))*options.GridStepM;
    axis_l = (0:sl(2)-1)*options.LowStepM;
    axis_h = (0:sh(4)-1)*options.HighStepM;
    low = interp1(axis_l,low.',grid,'linear',0).';
    flat = reshape(high,[],sh(4));
    flat = interp1(axis_h,flat.',grid,'linear',0).';
    high = reshape(flat,T,sh(2),C,numel(grid));
    time_s = 2*grid/c0;
    clock = (0:T-1).'*0.05;
    ins = struct('z_delta_m',zeros(T,1),'x_m',0.5*clock,'y_m',zeros(T,1), ...
                 'timestamp_s',clock,'roll_rad',zeros(T,1),'pitch_rad',zeros(T,1));
end

function value = read_complex(path,prefix,start,count,stride)
    real_part = h5read(path,['/' prefix '_data_real'],start,count,stride);
    imag_part = h5read(path,['/' prefix '_data_imag'],start,count,stride);
    value = reshape(complex(double(real_part),double(imag_part)),count);
    if any(~isfinite(value(:)))
        error('IceMap:RecoveryFinite','Invalid complex samples in %s.',prefix);
    end
end

function value = high_envelope(input)
    [T,~,~,N] = size(input);
    value = reshape(sqrt(mean(mean(abs(input).^2,2),3)),T,N);
end
