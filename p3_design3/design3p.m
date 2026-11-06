function out = design3p(path_name, file_name, overrides)
    if nargin < 3
        overrides = struct();
    end
    root = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(root, 'common'));
    cfg = icemap_config(overrides);
    S = load(fullfile('data_h2', path_name, string(file_name) + '.mat'));
    out = design3_select(S.data_l, S.data_h, S.time_s, cfg);
    if isfield(S, 'label')
        out.label = S.label;
    end
    if isfield(S, 'tpf')
        out.design2_time_per_frame_s = S.tpf;
    end
    folder = fullfile('data_h3', path_name);
    if ~isfolder(folder)
        mkdir(folder);
    end
    save(fullfile(folder, string(file_name) + '_D3.mat'), '-struct', 'out', '-v7.3');
end
