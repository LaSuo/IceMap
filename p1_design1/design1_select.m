function out = design1_select(low, high, time_s, ins, cfg)
    % Routing only: implementations are independent method files.
    catalog = algorithm_catalog();
    names = catalog.d1;
    id = find(strcmp(cfg.design1_version, compose('v%d', 1:numel(names))), 1);
    if isempty(id)
        error('IceMap:Version', 'Unsupported D1 version: %s', cfg.design1_version);
    end
    [low, high, time_s] = validate_echoes(low, high, time_s);
    ins = validate_ins(ins, size(low, 1));
    out = feval(names{id}, low, high, time_s, ins, cfg);
    if ~isstruct(out) || ~isequal(size(out.data_l), size(low)) || ~isequal(size(out.data_h), size(high))
        error('IceMap:StageShape', 'D1 %s returned malformed echoes.', names{id});
    elseif any(~isfinite(out.data_l(:))) || any(~isfinite(out.data_h(:))) || numel(out.shift_bins) ~= size(low, 1) || any(~isfinite(out.shift_bins))
        error('IceMap:StageFinite', 'D1 %s produced invalid samples or shifts.', names{id});
    end
    out.algorithm_version = cfg.design1_version;
    out.algorithm_method = names{id};
end
