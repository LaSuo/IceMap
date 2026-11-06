function out = design2(path_name, file_name, a_f1, a_f2, a_f3, a_f4, pre_mode, overrides)
    % File-compatible main path. Legacy alternate-iteration modes were not in mine.
    if nargin < 8
        overrides = struct();
    end
    if nargin < 7
        pre_mode = 0;
    end
    if nargin < 6
        a_f4 = 0;
    end
    if nargin < 5
        a_f3 = 0;
    end
    if nargin < 4
        a_f2 = 0;
    end
    if nargin < 3
        a_f1 = 0;
    end
    assert(~any([a_f1 a_f2 a_f3 a_f4]) && pre_mode == 0, ...
           'Legacy fusion ablations/alternate iterations are not the reference specification algorithm; use the backed-up version for historical comparisons.');
    root = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(root, 'common'));
    cfg = icemap_config(overrides);
    S = load(fullfile('data_h1', path_name, string(file_name) + '.mat'));
    out = design2_select(S.data_l, S.data_h, S.time_s, cfg);
    if isfield(S, 'label')
        out.label = S.label;
    end
    folder = fullfile('data_h2', path_name);
    if ~isfolder(folder)
        mkdir(folder);
    end
    save(fullfile(folder, string(file_name) + '_D2.mat'), '-struct', 'out', '-v7.3');
end
