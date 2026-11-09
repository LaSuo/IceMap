function out = design3_dual_band_echo_vote(low, high, time_s, cfg, reference)
    s = prepare_fingerprint_inputs(low, high, time_s, cfg);
    p = s.options;
    if nargin < 5 || isempty(reference)
        reference = struct();
    elseif ~isstruct(reference) || ~isscalar(reference)
        error('IceMap:EchoReferenceType', 'The uncompensated echo reference must be a scalar struct.');
    end
    if isempty(fieldnames(reference))
        ref_low = s.analytic_low;
        ref_high = s.analytic_high;
        use_reference = false;
    else
        require_config(reference, {'low', 'high'});
        [ref_low, ref_high] = validate_echoes(reference.low, reference.high, time_s);
        if ~isequal(size(ref_low),size(low)) || ~isequal(size(ref_high),size(high))
            error('IceMap:EchoReferenceShape', 'Uncompensated echoes must share the D2 grid.');
        end
        if isreal(ref_low)
            ref_low = hilbert(ref_low.').';
        end
        if isreal(ref_high)
            for t = 1:s.T
                x = reshape(ref_high(t,:,:,:), s.R*s.C, s.N);
                ref_high(t,:,:,:) = reshape(hilbert(x.').', 1,s.R,s.C,s.N);
            end
        end
        use_reference = true;
    end
    if p.echo_min_range_m < 0 || p.echo_max_range_m <= p.echo_min_range_m || ...
            p.echo_reference_weight < 0 || p.echo_reference_weight > 1 || ...
            p.echo_high_vote_weight < 0 || p.echo_high_vote_weight > 1 || ...
            ~is_positive_odd(p.echo_agreement_bins) || ~is_positive_odd(p.echo_temporal_window)
        error('IceMap:EchoVoteOptions', 'Invalid range, reference weight, or vote window.');
    end
    meters = s.time_s * cfg.c0_m_s / 2;
    gate = meters >= p.echo_min_range_m & meters <= p.echo_max_range_m;
    if nnz(gate) < 8
        error('IceMap:EchoVoteRange', 'The echo window needs at least eight observed bins.');
    end
    raw_low = abs(ref_low);
    raw_high = reshape(sqrt(mean(mean(abs(ref_high).^2, 2), 3)), s.T, s.N);
    pre_low = consensus_amplitude(raw_low, gate, p);
    pre_high = consensus_amplitude(raw_high, gate, p);
    post_low = consensus_amplitude(s.low_envelope, gate, p);
    post_high = consensus_amplitude(s.high_amplitude, gate, p);
    low_vote = pre_low .* (p.echo_reference_weight + ...
        (1-p.echo_reference_weight) .* min(1.5, post_low ./ max(pre_low, .04)));
    high_vote = pre_high .* (p.echo_reference_weight + ...
        (1-p.echo_reference_weight) .* min(1.5, post_high ./ max(pre_high, .04)));
    support = movmax(high_vote, p.echo_agreement_bins, 2, 'Endpoints', 'shrink');
    agreement = (1-p.echo_high_vote_weight) + p.echo_high_vote_weight * sqrt(min(1, support));
    score = low_vote .* agreement;
    score(:,~gate) = 0;
    threshold = repmat(max(0.018, .55*median(score(:,gate),2)), 1,s.N);
    [tau,bins,times,intervals,event_info] = build_event_fingerprint( ...
        score, pre_low, s.time_s, cfg, threshold, repmat(gate,s.T,1));

    [phase,phase_coherence,phase_valid,phase_info] = robust_phase_fingerprint(ref_high,cfg,p);
    mask = conv2(double(abs(phase) >= cfg.boundary_phase_threshold_rad & phase_valid), ...
        ones(1,2*cfg.boundary_dilation_bins+1),'same') > 0;
    [reliability,valid] = scattering_reliability(ref_high,mask,cfg);
    % Normalize on the observed band so deep inverse gains cannot set the scale.
    amplitude = pre_high ./ max(median(pre_high(:,gate),2), .025);
    E = log1p(amplitude) .* sqrt(reliability);
    valid = valid & repmat(gate,s.T,1) & raw_high > p.absolute_noise_floor;
    E(~valid) = 0;
    extras = struct('low_echo_vote',low_vote,'high_echo_vote',high_vote, ...
                    'dual_band_echo_agreement',score);
    valid_gate = repmat(gate,s.T,1);
    validity = struct('low_echo_vote',valid_gate,'high_echo_vote',valid_gate, ...
                      'dual_band_echo_agreement',valid_gate);
    units = struct('low_echo_vote','1','high_echo_vote','1','dual_band_echo_agreement','1');
    info = struct('uncompensated_reference_used',use_reference,'phase_estimator',phase_info, ...
                  'events',event_info,'echo_range_m',[p.echo_min_range_m p.echo_max_range_m]);
    out = assemble_fingerprint_output(s,phase,E,tau,phase_coherence,mask,valid,reliability, ...
        bins,times,intervals,extras,validity,units,info);
end

function value = consensus_amplitude(envelope, gate, p)
    smooth = movmean(envelope, 7, 2, 'Endpoints', 'shrink');
    norm = max(max(median(smooth(:,gate),1)), realmin);
    value = smooth / norm;
    value(:,~gate) = 0;
    consensus = movmedian(value,p.echo_temporal_window,1,'Endpoints','shrink');
    value = .65 * consensus + .35 * value;
end

function yes = is_positive_odd(value)
    yes = isscalar(value) && isfinite(value) && value >= 1 && mod(value,2) == 1;
end
