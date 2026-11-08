function plot_icemap_demo(features, echoes, record_id, output_png, show_figure)
    % Render the selected radar frames and fingerprints without using labels.
    time_ns = features.time_s * 1e9;
    frames = features.calibration_provenance.selected_frames;
    frames = frames(1):frames(2);
    d1 = features.design1_metadata;
    d2 = features.design2_metadata;
    fig = figure('Name', record_id, 'NumberTitle', 'off', 'Visible', 'off', ...
                 'Color', 'white', 'Position', [60 60 1380 900]);
    cleanup = onCleanup(@()close_hidden(fig));
    panels = tiledlayout(fig, 3, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(panels, record_id, 'Interpreter', 'none');
    subtitle(panels, ['Pipeline demo | Assumed ToF calibration | ' ...
                      'Input echoes shown after import and before D1']);
    xlabel(panels, 'Source frame index');
    ylabel(panels, 'Assumed two-way ToF (ns) in echo and fingerprint panels');

    low_ref = max(echoes.raw_low, [], 'all');
    high_ref = max(echoes.raw_high, [], 'all');
    band_limits = [-45 20];
    ax = heatmap_panel(panels, frames, time_ns, relative_db(echoes.raw_low, low_ref), ...
                       'Input low band | X4M03', 'dB re input peak', band_limits);
    hold(ax, 'on');
    plot(ax, frames, interp1(1:numel(time_ns), time_ns, d1.surface_track_bins), ...
         '.-', 'Color', [0.94 0.32 0.18], 'LineWidth', 1.4, 'DisplayName', 'D1 surface track');
    legend(ax, 'Location', 'northeast');
    heatmap_panel(panels, frames, time_ns, relative_db(echoes.raw_high, high_ref), ...
                  'Input high band | BGT60TR13C RMS', 'dB re input peak', band_limits);

    fused_title = 'D2 fused view | diagnostic';
    if strcmp(features.fingerprint_input_mode, 'fused')
        fused_title = 'D2 fused view | D3 input';
    end
    heatmap_panel(panels, frames, time_ns, relative_db(echoes.fused, max(echoes.fused, [], 'all')), ...
                  fused_title, 'dB re fused peak', [-45 0]);
    ax = heatmap_panel(panels, frames, time_ns, relative_db(echoes.aligned_low, low_ref), ...
                       'D2 aligned low band', 'dB re input peak', band_limits);
    mark_shallow_split(ax, features.params.shallow_end_s);
    ax = heatmap_panel(panels, frames, time_ns, relative_db(echoes.aligned_high, high_ref), ...
                       'D2 aligned high band | RMS', 'dB re input peak', band_limits);
    mark_shallow_split(ax, features.params.shallow_end_s);

    ax = nexttile(panels);
    residual = d2.relative_fit_residual;
    residual(~d2.fit_residual_valid) = NaN;
    plot(ax, frames, residual, '.-', 'LineWidth', 1.3, 'MarkerSize', 13);
    threshold = features.params.cross_frequency_max_relative_residual;
    fit_title = 'D2 fit residual';
    if ~isempty(threshold)
        yline(ax, threshold, '--', 'Threshold', 'LabelHorizontalAlignment', 'left', ...
              'HandleVisibility', 'off');
        fit_title = sprintf('D2 fit | %d/%d frames flagged', nnz(d2.cross_frequency_mismatch), numel(frames));
    end
    xlim(ax, [frames(1) - 0.5 frames(end) + 0.5]);
    ylim(ax, [0 1.05 * max([1; residual(:); threshold(:)], [], 'omitnan')]);
    ylabel(ax, 'Relative residual');
    title(ax, fit_title);
    legend(ax, {'Shallow', 'Deep'}, 'Location', 'southeast');
    grid(ax, 'on');

    phase_limit = max(features.params.boundary_phase_threshold_rad, max(abs(features.phase), [], 'all'));
    ax = heatmap_panel(panels, frames, time_ns, features.phase, ...
                       'Boundary fingerprint | phase residual', 'rad', [-phase_limit phase_limit]);
    colormap(ax, interp1([0 0.5 1], [0.12 0.30 0.76; 0.97 0.97 0.97; 0.12 0.30 0.76], ...
                         linspace(0, 1, 256)));

    % Display positive contrast against a shared local ToF background.
    % The exported fingerprint and the D2 residual panel retain their original values.
    log_scatter = log(max(features.E, realmin));
    log_scatter(~features.scatter_valid) = NaN;
    background = movmedian(median(log_scatter, 1, 'omitnan'), 31, 'omitnan');
    scatter_view = max(log_scatter - background, 0);
    scatter_view(~isfinite(scatter_view)) = 0;
    ax = heatmap_panel(panels, frames, time_ns, scatter_view, ...
                       'Density fingerprint | scattering', 'Local log contrast', [0 3]);
    colormap(ax, interp1([0 0.4 1], [0 0 0; 0.05 0.15 0.52; 0.30 0.65 1], ...
                         linspace(0, 1, 256)));
    ax = heatmap_panel(panels, frames, time_ns, features.tau, ...
                       'Delay fingerprint | candidate peaks', 'Event amplitude', []);
    hold(ax, 'on');
    counts = cellfun(@numel, features.tau_events_s);
    event_frames = repelem(frames(:), counts(:));
    event_times_ns = [features.tau_events_s{:}] * 1e9;
    plot(ax, event_frames, event_times_ns, 'wo', 'MarkerSize', 4);

    exportgraphics(panels, output_png, 'Resolution', 130);
    if show_figure
        fig.Visible = 'on';
        drawnow;
    end
end

function ax = heatmap_panel(panels, frames, time_ns, values, heading, units, limits)
    ax = nexttile(panels);
    imagesc(ax, frames, time_ns, values.');
    ax.YDir = 'reverse';
    colormap(ax, parula);
    if ~isempty(limits)
        clim(ax, limits);
    end
    if numel(frames) <= 10
        ax.XTick = frames;
    end
    bar = colorbar(ax);
    bar.Label.String = units;
    title(ax, heading);
end

function db = relative_db(amplitude, reference)
    db = 20 * log10(max(amplitude / max(reference, realmin), 1e-6));
end

function mark_shallow_split(ax, cutoff_s)
    yline(ax, cutoff_s * 1e9, '--w', 'D2 split', 'LabelHorizontalAlignment', 'left');
end

function close_hidden(fig)
    if isgraphics(fig) && strcmp(fig.Visible, 'off')
        close(fig);
    end
end
