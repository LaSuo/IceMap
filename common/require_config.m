function require_config(cfg, names)
    for k = 1:numel(names)
        if ~isfield(cfg, names{k}) || isempty(cfg.(names{k}))
            error('IceMap:CalibrationRequired', 'Set cfg.%s explicitly; the reference specification does not specify its numerical value.', names{k});
        end
    end
end
