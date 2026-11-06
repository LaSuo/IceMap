function text = scene_to_json(scene)
    % Canonical cross-language encoding: unobserved MATLAB [] becomes JSON null.
    validate_scene(scene);
    groups = fieldnames(scene);
    for i = 1:numel(groups)
        names = fieldnames(scene.(groups{i}));
        for j = 1:numel(names)
            if isempty(scene.(groups{i}).(names{j}))
                scene.(groups{i}).(names{j}) = NaN;
            end
        end
    end
    text = jsonencode(scene, 'ConvertInfAndNaN', true);
end
