function validate_method_options(options)
    if ~isstruct(options) || ~isscalar(options)
        error('IceMap:MethodOptions', 'Method options must be one scalar struct.');
    end
    names = fieldnames(options);
    for i = 1:numel(names)
        k = names{i};
        v = options.(k);
        if ~(isnumeric(v) || islogical(v)) || ~isreal(v) || ~isscalar(v) || ~isfinite(v)
            error('IceMap:MethodOptionValue', 'Method option %s must be a finite real scalar.', k);
        end
        if any(strcmp(k, {'require_convergence', 'allow_missing_surface_frames', 'attitude_correction', 'use_altitude_prior'})) && ~any(v == [0 1])
            error('IceMap:MethodOptionBoolean', '%s must be Boolean.', k);
        end
        if endsWith(k, {'max_iter', 'max_backtracks', 'max_missing_surface_frames', 'candidate_count', 'max_elements'}) && (v < 1 || v ~= round(v))
            error('IceMap:MethodOptionInteger', '%s must be a positive integer.', k);
        end
    end
end
