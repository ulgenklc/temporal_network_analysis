function b = is_valid_dynamic_network(C)
%IS_VALID_DYNAMIC_NETWORK Tests for valid binary unidrectional network
%   Detailed explanation goes here

if 3 ~= ndims(C) || size(C, 1) ~= size(C, 2)
    b = false;
else
    b = true;
    for t = 1:size(C, 3)
        if ~issymmetric(double(C(:, :, t)))
            b = false;
            break;
        end
        if any(diag(C(:, :, t)))
            b = false;
            break;
        end
    end
end

end

