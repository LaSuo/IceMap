function ins = validate_ins(ins, T)
    if ~isstruct(ins) || ~isscalar(ins)
        error('IceMap:INS', 'INS must be a scalar struct.');
    end
    for name = {'z_delta_m', 'x_m', 'y_m', 'roll_rad', 'pitch_rad', 'yaw_rad', 'timestamp_s'}
        k = name{1};
        mandatory = any(strcmp(k, {'z_delta_m', 'x_m', 'y_m'}));
        if ~isfield(ins, k)
            if mandatory
                error('IceMap:INSField', 'Required INS field %s is missing.', k);
            else
                continue
            end
        end
        value = ins.(k);
        if ~isnumeric(value) || ~isreal(value) || ~isvector(value) || numel(value) ~= T || any(~isfinite(value))
            error('IceMap:INSShape', 'INS.%s must contain T finite real SI-unit samples.', k);
        end
        ins.(k) = double(value(:));
        if strcmp(k, 'timestamp_s') && any(diff(value) <= 0)
            error('IceMap:INSClock', 'INS timestamps must strictly increase.');
        end
    end
end
