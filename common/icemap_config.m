function cfg = icemap_config(overrides)
    % Values explicitly reported in the reference specification. Other settings are engineering
    % choices or required calibration inputs, declared explicitly below.
    if nargin < 1
        overrides = struct();
    end
    cfg.c0_m_s = 299792458;
    cfg.dielectric_density_coefficient = 0.845; % rho in g/cm^3
    cfg.frequency_low_hz = 7.29e9;
    cfg.frequency_high_hz = 60e9;
    cfg.bandwidth_low_hz = 1.4e9;
    cfg.bandwidth_high_hz = 5.5e9;
    cfg.surface_sigma_initial_bins = 160;
    cfg.surface_sigma_final_bins = 80;
    cfg.surface_sigma_anneal_frames = 50;
    cfg.scattering_gamma = 1.5;
    cfg.assignment_temperature = 0.23;
    cfg.max_layers = 10;
    cfg.erf_windows = [30 2; 60 3; 90 4]; % depth x frames: phi, E, tau
    cfg.erf_beta = [1 1 1];
    cfg.stage_epochs = [20 20 40];
    cfg.loss_warmup_epochs = 5;
    cfg.loss_weights = [1 0.5 0.5 1 2 1]; % b,cnt,pre-rho,rho,h,p
    % User-authorized engineering initials; still require measurement calibration.
    initials = engineering_initials();
    initial_names = fieldnames(initials);
    for i = 1:numel(initial_names)
        cfg.(initial_names{i}) = initials.(initial_names{i});
    end
    cfg.calibration_status = 'engineering_initial'; % set measured only after actual calibration
    % Explicit engineering choices (the reference specification does not give these values).
    cfg.tracking_band = 'low';
    cfg.surface_min_peak_distance_bins = 1;
    cfg.surface_candidate_prominence = 0;
    cfg.spatial_neighborhood_frames = 8;
    cfg.min_cos_incidence = 0.15;
    cfg.psf_family = 'gaussian';
    cfg.deep_evidence_policy = 'error'; % 'unity' explicitly disables unidentified deep compensation.
    cfg.psf_support_sigmas = 4;
    cfg.max_operator_nonzeros = 5000000;
    cfg.attenuation_floor = 1e-4;
    cfg.attenuation_max_iter = 2000;
    cfg.solver_tolerance = 1e-8;
    cfg.phase_confidence_min = 0.25;
    cfg.phase_trend_bins = 101;
    cfg.coherence_depth_bins = 11;
    cfg.coherence_frame_radius = 1;
    cfg.coherence_floor = 1e-3;
    cfg.tof_smooth_bins = 7;
    cfg.tof_centroid_halfwidth_bins = 3;
    cfg.tof_skeleton_sigma_bins = 2;
    cfg.label1_coordinate = ''; % set 'propagation_time' to shift legacy label1
    cfg.ins_length_unit = ''; % legacy INS wrapper requires 'm' or 'mm'
    cfg.design1_version = 'v2';
    cfg.design2_version = 'v3';
    cfg.design3_version = 'v1';
    cfg.fingerprint_input_mode = 'dual_band'; % 'fused' is an explicit alternative experiment.
    cfg.methods = method_defaults();
    cfg.scene = scene_defaults();
    fields = fieldnames(overrides);
    for i = 1:numel(fields)
        if ~isfield(cfg, fields{i})
            error('IceMap:Config', 'Unknown config field: %s', fields{i});
        end
        if isstruct(cfg.(fields{i}))
            cfg.(fields{i}) = merge_options(cfg.(fields{i}), overrides.(fields{i}), fields{i});
        else
            cfg.(fields{i}) = overrides.(fields{i});
        end
    end
    versions = {'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7'};
    assert(any(strcmp(cfg.design1_version, versions)), 'IceMap:Version', 'Unknown D1 version.');
    assert(any(strcmp(cfg.design2_version, versions)), 'IceMap:Version', 'Unknown D2 version.');
    assert(any(strcmp(cfg.design3_version, {'v1', 'v2', 'v3', 'v4', 'v5'})), 'IceMap:Version', 'Unknown D3 version.');
    validate_scene(cfg.scene);
    validate_processing_config(cfg);
end
