function values = parse_radar_numbers(line)
    % Strict numeric grammar; never evaluate text as MATLAB expressions.
    line = strtrim(line);
    if startsWith(line, '[') && endsWith(line, ']')
        line = line(2:end - 1);
    end
    pattern = '[+-]?(?:(?:\d+\.?\d*)|(?:\.\d+))(?:[eE][+-]?\d+)?';
    remainder = regexprep(line, pattern, '');
    if ~isempty(regexprep(remainder, '[\s,;]', ''))
        error('IceMap:NumericSyntax', 'Invalid token in radar numeric line.');
    end
    parts = regexp(line, pattern, 'match');
    values = str2double(parts);
    tokens = regexp(strtrim(line), '[\s,;]+', 'split');
    if isempty(parts) || numel(tokens) ~= numel(parts) || any(~isfinite(values))
        error('IceMap:NumericSyntax', 'Expected finite separated numeric values.');
    end
end
