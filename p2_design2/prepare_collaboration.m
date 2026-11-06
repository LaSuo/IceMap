function [out, p] = prepare_collaboration(low, high, time_s, cfg)
    p = cfg.methods.d2;
    validate_method_options(p);
    names = fieldnames(p);
    for i = 1:numel(names)
        value = p.(names{i});
        assert((isnumeric(value) || islogical(value)) && isscalar(value) && isreal(value) && isfinite(value), 'IceMap:ResearchOption', 'Invalid D2 option: %s', names{i});
    end
    assert(p.admm_max_iter == round(p.admm_max_iter));
    assert(p.ridge_gradient_weight >= 0 && p.wiener_nsr > 0 && p.tv_weight >= 0 && p.admm_rho > 0);
    assert(p.admm_max_iter >= 1 && p.absolute_tolerance > 0 && p.relative_tolerance > 0);
    assert(p.noise_floor > 0 && p.snr_window_bins >= 1 && p.snr_window_bins == round(p.snr_window_bins) && p.penetration_decay_bins > 0);

    out = design2_multifrequency_collaboration(low, high, time_s, cfg);
end
