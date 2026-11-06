function report = check_matlab_environment()
    % MATLAB dependencies cannot be expressed in Python requirements files.
    report = struct('release', version('-release'), 'tested_release', '2024a', ...
                  'signal_processing_available', exist('findpeaks', 'file') == 2 && exist('hilbert', 'file') == 2, ...
                  'linear_solver_available', exist('decomposition', 'file') ~= 0 || exist('decomposition', 'builtin') ~= 0);
    if ~report.signal_processing_available
        error('IceMap:Dependency', 'Signal Processing Toolbox functions findpeaks and hilbert are required.');
    end
    if ~report.linear_solver_available
        error('IceMap:Dependency', 'This MATLAB release lacks decomposition.');
    end
    fprintf('MATLAB %s: signal processing and linear algebra dependencies available.\n', report.release);
end
