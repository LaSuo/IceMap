function [H, D] = shallow_operator(h, N, Ns)
    offsets = -(numel(h) - 1) / 2:(numel(h) - 1) / 2;
    H = spdiags(repmat(h, N, 1), offsets, N, N);
    H = H(1:Ns, 1:Ns);
    D = diff(speye(Ns), 1, 1);
end
