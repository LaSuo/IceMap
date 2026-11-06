function cfg = read_radar_header(fid, count)
    cfg = struct();
    for k = 1:count
        line = fgetl(fid);
        if ~ischar(line)
            error('IceMap:RawHeader', 'EOF in radar header.');
        end
        split = strfind(line, ':');
        if isempty(split)
            continue
        end
        name = regexprep(strtrim(line(1:split(1) - 1)), '[^a-zA-Z0-9_]', '');
        if isempty(name) || ~isvarname(name)
            error('IceMap:RawHeader', 'Invalid header field.');
        end
        value = str2double(strtrim(line(split(1) + 1:end)));
        if isfinite(value)
            if isfield(cfg, name)
                error('IceMap:RawHeader', 'Duplicate numeric header field %s.', name);
            end
            cfg.(name) = value;
        end
    end
end
