function validate_scene(s)
    groups = fieldnames(s);
    for g = 1:numel(groups)
        names = fieldnames(s.(groups{g}));
        for j = 1:numel(names)
            k = names{j};
            v = s.(groups{g}).(k);
            if isempty(v)
                continue
            end
            assert(isnumeric(v) && isreal(v) && all(isfinite(v(:))), ['Invalid scene field: ' k]);
            if strcmp(k, 'antenna_lever_arm_m')
                assert(numel(v) == 3, 'antenna_lever_arm_m must be a 3-vector.');
            else
                assert(isscalar(v), ['Scene metadata must be scalar: ' k]);
            end
            if any(strcmp(k, {'relative_humidity', 'liquid_water_fraction'}))
                assert(all(v >= 0 & v <= 1), 'IceMap:SceneRange', [k ' must be a fraction in [0,1].']);
            elseif contains(k, 'temperature_c')
                assert(v >= -273.15, 'Temperature below absolute zero.');
            elseif endsWith(k, '_rad') || strcmp(k, 'clock_offset_s') || strcmp(k, 'antenna_lever_arm_m')
                % Signed angles, offsets and body-frame lever arms are supported.
            else
                assert(all(v >= 0), ['Negative scene field: ' k]);
            end
        end
    end
end
