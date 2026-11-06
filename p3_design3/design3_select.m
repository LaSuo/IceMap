function out = design3_select(low, high, time_s, cfg)
    % Registry-only dispatcher; each feature method is implemented independently.
    catalog = algorithm_catalog();
    id = find(strcmp(cfg.design3_version, compose('v%d', 1:numel(catalog.d3))), 1);
    if isempty(id)
        error('IceMap:Version', 'Unknown D3 version: %s.', cfg.design3_version);
    end
    [low, high, time_s] = validate_echoes(low, high, time_s);
    out = feval(catalog.d3{id}, low, high, time_s, cfg);
    required = {'phase', 'E', 'tau', 'scatter_valid'};
    for i = 1:numel(required)
        k = required{i};
        if ~isfield(out, k) || ~isequal(size(out.(k)), size(low))
            error('IceMap:FingerprintShape', 'D3 %s output %s has incorrect shape.', catalog.d3{id}, k);
        end
        if ~isreal(out.(k)) || any(~isfinite(out.(k)(:)))
            error('IceMap:FingerprintFinite', 'D3 %s output %s is nonfinite or complex.', catalog.d3{id}, k);
        end
    end
    if any(out.E(:) < 0) || any(out.tau(:) < 0) || any(out.E(~out.scatter_valid) ~= 0)
        error('IceMap:FingerprintRange', 'E/tau must be nonnegative; invalid scatter bins must have zero E.');
    end
    out.algorithm_version = cfg.design3_version;
    out.algorithm_method = catalog.d3{id};
end
