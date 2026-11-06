function options = import_initials()
    % Temporary importer values, explicitly not measured sensor calibration.
    options.low_sample_rate_hz = 2.5e9;
    options.high_range_bin_m = 299792458 / (2 * 5.5e9);
    options.sync_tolerance_s = 0.02;
    options.high_clock_offset_s = 0;
    options.output_range_step_m = 0.001;
    options.calibration_status = 'engineering_initial';
end
