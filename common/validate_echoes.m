function [low, high, time_s] = validate_echoes(low, high, time_s)
    % No silent squeezing, time-grid repair, or NaN imputation.
    if ~isnumeric(low) || ~isnumeric(high) || issparse(low) || issparse(high)
        error('IceMap:EchoType', 'Echoes must be dense numeric real/complex arrays.');
    elseif isempty(low) || isempty(high)
        error('IceMap:EmptyEcho', 'Echo arrays cannot be empty.');
    elseif ~ismatrix(low) || ndims(high) > 4
        error('IceMap:EchoRank', 'Expected low [T,N], high [T,R,C,N]; do not squeeze axes.');
    end
    [T, R, C, N] = size(high);
    if N < 3 || T < 1 || R < 1 || C < 1 || ~isequal(size(low), [T N])
        error('IceMap:EchoShape', 'Low [%s] does not match high [%s].', num2str(size(low)), num2str(size(high)));
    elseif any(~isfinite(low(:))) || any(~isfinite(high(:)))
        error('IceMap:EchoFinite', 'Echo contains NaN/Inf; explicitly preprocess before this stage.');
    end
    if ~isnumeric(time_s) || ~isreal(time_s) || ~isvector(time_s) || numel(time_s) ~= N
        error('IceMap:TimeShape', 'time_s must be one real numeric N-vector in seconds.');
    end
    time_s = double(time_s(:).');
    dt = diff(time_s);
    if any(~isfinite(time_s)) || any(dt <= 0)
        error('IceMap:TimeOrder', 'time_s must be finite and strictly increasing.');
    elseif max(abs(dt - mean(dt))) > 1e-6 * mean(dt)
        error('IceMap:TimeUniform', 'Nonuniform propagation grid requires explicit resampling.');
    end
    low = double(low);
    high = double(high);
end
