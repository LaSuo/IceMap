function out = assemble_fingerprint_output(s, phase, E, tau, phase_coherence, mask, scatter_valid, reliability, bins, times, intervals, extras, validity, units, diagnostics)
    out = struct('phase', phase, 'E', E, 'tau', tau, 'time_s', s.time_s, 'phase_coherence', phase_coherence, ...
               'boundary_mask', mask, 'scatter_valid', logical(scatter_valid), 'scattering_coherence', reliability, ...
               'tau_event_bins', {bins}, 'tau_events_s', {times}, 'tau_intervals_s', {intervals}, 'params', s.cfg, ...
               'extra_features', extras, 'extra_feature_valid', validity, 'extra_feature_units', units, 'diagnostics', diagnostics);
    names = fieldnames(extras);
    for i = 1:numel(names)
        k = names{i};
        value = extras.(k);
        if ~isequal(size(value), [s.T s.N]) || ~isreal(value) || any(~isfinite(value(:))) || ~isfield(validity, k) || ~isequal(size(validity.(k)), [s.T s.N]) || ~isfield(units, k)
            error('IceMap:ExtraFeature', 'Extended channel %s violates the finite real [T,N] contract.', k);
        end
        out.extra_features.(k)(~validity.(k)) = 0;
    end
end
