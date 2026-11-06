root = fileparts(fileparts(mfilename('fullpath')));
folders = {'common', 'p1_design1', 'p2_design2', 'p3_design3'};
files = {fullfile(root, 'run_icemap.m'), fullfile(root, 'export_features_h5.m'), ...
       fullfile(root, 'run_icemap_variants.m'), fullfile(root, 'audit', 'test_matlab_invariants.m'), ...
       fullfile(root, 'audit', 'test_method_versions.m'), fullfile(root, 'audit', 'test_design3_methods.m')};
for k = 1:numel(folders)
    entries = dir(fullfile(root, folders{k}, '*.m'));
    for j = 1:numel(entries)
        files{end + 1} = fullfile(entries(j).folder, entries(j).name);
    end
end
records = struct('file', {}, 'diagnostics', {});
for k = 1:numel(files)
    records(k).file = files{k};
    records(k).diagnostics = checkcode(files{k}, '-struct', '-id');
end
fid = fopen(fullfile(root, 'audit', 'matlab_static_checks.json'), 'w', 'n', 'UTF-8');
fwrite(fid, jsonencode(records, 'PrettyPrint', true), 'char');
fclose(fid);
for k = 1:numel(records)
    fprintf('%s: %d static diagnostics\n', records(k).file, numel(records(k).diagnostics));
    for j = 1:numel(records(k).diagnostics)
        d = records(k).diagnostics(j);
        fprintf('  line %d: %s\n', d.line, d.message);
    end
end
