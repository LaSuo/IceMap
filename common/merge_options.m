function out = merge_options(defaults, options, path)
    % Strict recursive partial override; misspelled keys are errors.
    if nargin < 3
        path = 'options';
    end
    assert(isstruct(options) && isscalar(options), 'Options must be scalar structs.');
    out = defaults;
    names = fieldnames(options);
    for i = 1:numel(names)
        k = names{i};
        assert(isfield(defaults, k), ['Unknown option: ' path '.' k]);
        if isstruct(defaults.(k))
            out.(k) = merge_options(defaults.(k), options.(k), [path '.' k]);
        else
            out.(k) = options.(k);
        end
    end
end
