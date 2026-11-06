function out = design2_select(low, high, time_s, cfg)
    catalog = algorithm_catalog();
    names = catalog.d2;
    id = find(strcmp(cfg.design2_version, compose('v%d', 1:numel(names))), 1);
    if isempty(id)
        error('IceMap:Version', 'Unsupported D2 version: %s', cfg.design2_version);
    end
    [low, high, time_s] = validate_echoes(low, high, time_s);
    out = feval(names{id}, low, high, time_s, cfg);
    if ~isstruct(out) || ~isequal(size(out.data_l), size(low)) || ~isequal(size(out.data_h), size(high))
        error('IceMap:StageShape', 'D2 %s returned malformed echoes.', names{id});
    elseif any(~isfinite(out.data_l(:))) || any(~isfinite(out.data_h(:)))
        error('IceMap:StageFinite', 'D2 %s produced nonfinite samples.', names{id});
    end
    out.algorithm_version = cfg.design2_version;
    out.algorithm_method = names{id};
end
