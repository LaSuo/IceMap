function output_files = run_icemap_variants(input_mat, output_dir, versions1, versions2, overrides, versions3)
    % Explicit Cartesian research sweep; fails rather than overwriting a result.
    if nargin < 3
        versions1 = cellstr(compose('v%d', 1:7));
    end
    if nargin < 4
        versions2 = cellstr(compose('v%d', 1:7));
    end
    if nargin < 5
        overrides = struct();
    end
    if nargin < 6
        if isfield(overrides, 'design3_version')
            versions3 = {overrides.design3_version};
        else
            versions3 = {'v1'};
        end
    end
    assert(numel(unique(versions1)) == numel(versions1) && numel(unique(versions2)) == numel(versions2), 'Duplicate versions.');
    assert(~isempty(versions1) && ~isempty(versions2) && ~isempty(versions3), 'Empty version selection.');
    assert(numel(unique(versions3)) == numel(versions3), 'Duplicate D3 versions.');
    if ~isfolder(output_dir)
        mkdir(output_dir);
    end
    output_files = cell(numel(versions1), numel(versions2), numel(versions3));
    for i = 1:numel(versions1)
        for j = 1:numel(versions2)
            for k = 1:numel(versions3)
                out = fullfile(output_dir, ['d1_' versions1{i} '_d2_' versions2{j} '_d3_' versions3{k} '.mat']);
                assert(~isfile(out), 'IceMap:OutputExists', 'Refusing to overwrite existing sweep result: %s', out);
                options = overrides;
                options.design1_version = versions1{i};
                options.design2_version = versions2{j};
                options.design3_version = versions3{k};
                run_icemap(input_mat, out, options);
                output_files{i, j, k} = out;
            end
        end
    end
end
