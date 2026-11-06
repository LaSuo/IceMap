function out = design1(path_name, file_name, save_f1, save_f2, overrides)
    % File-compatible entry point. New callers should use design1_energy_ins.
    if nargin < 3
        save_f1 = 0;
    end
    if nargin < 4
        save_f2 = 0;
    end
    if nargin < 5
        overrides = struct();
    end
    root = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(root, 'common'));
    cfg = icemap_config(overrides);
    S = load(fullfile('data_h0', path_name, string(file_name) + '.mat'));
    assert(isfield(S, 'time_s'), 'Input MAT must contain calibrated time_s in seconds.');
    ins = S.ins;
    if ~isfield(ins, 'z_delta_m')
        require_config(cfg, {'ins_length_unit'});
        if strcmp(cfg.ins_length_unit, 'mm')
            scale = 1e-3;
        elseif strcmp(cfg.ins_length_unit, 'm')
            scale = 1;
        else
            error('ins_length_unit must be m or mm.');
        end
        ins.z_delta_m = double(ins.zv(:)) * scale;
        ins.x_m = double(ins.px(:)) * scale;
        ins.y_m = double(ins.py(:)) * scale;
    end
    out = design1_select(S.radarl, S.radarh, S.time_s, ins, cfg);
    if isfield(S, 'label')
        out.label = S.label;
        if isfield(S.label, 'label1')
            assert(strcmp(cfg.label1_coordinate, 'propagation_time'), ...
                   'Declare label1_coordinate=propagation_time before shifting legacy label1.');
            for k = 1:size(S.label.label1, 1)
                out.label.label1(k, :) = interp1(1:numel(S.time_s), double(S.label.label1(k, :)), ...
                                              (1:numel(S.time_s)) + out.shift_bins(k), 'nearest', 0);
            end
        end
        % Independent physical thickness/density/profile ground truth is unchanged.
    end
    folder = fullfile('data_h1', path_name);
    if ~isfolder(folder)
        mkdir(folder);
    end
    save(fullfile(folder, string(file_name) + '_D1.mat'), '-struct', 'out', '-v7.3');
    if save_f1
        ab = out;
        ab.data_l = out.geometry_low;
        ab.data_h = out.geometry_high;
        save(fullfile(folder, string(file_name) + '_D1a1.mat'), '-struct', 'ab', '-v7.3');
    end
    if save_f2
        ab = out;
        ab.data_l = S.radarl .* sqrt(out.power_gain);
        ab.data_h = S.radarh .* reshape(sqrt(out.power_gain), size(S.radarl, 1), 1, 1, 1);
        if isfield(S, 'label')
            ab.label = S.label;
        end
        ab.shift_bins = zeros(size(out.shift_bins));
        ab.delay_correction_s = zeros(size(out.delay_correction_s));
        save(fullfile(folder, string(file_name) + '_D1a2.mat'), '-struct', 'ab', '-v7.3');
    end
end
