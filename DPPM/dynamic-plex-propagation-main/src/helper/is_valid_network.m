function b = is_valid_network(a)
%IS_VALID_NETWORK Tests for valid binary unidrectional network
%   Detailed explanation goes here

if ~ismatrix(a) || size(a, 1) ~= size(a, 2)
    b = false;
elseif any(diag(a))
    b = false;
else
    b = issymmetric(double(a));
end

end

