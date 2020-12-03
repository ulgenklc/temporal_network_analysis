function c = preserves_connectivity(a, i, j)
%PRESERVES_CONNECTIVITY Does removing edge break connectivity
%   Returns true if the graph connectivity is preserved after removing edge
%   i-j.

% hypothetically remove
a(i, j) = 0;
a(j, i) = 0;

n = size(a, 1);
init = zeros(1, n);
init(i) = 1;
aft = init * a * a;
if aft(j) % since undirected, only test one
    c = true;
    return
end

for j = 3:(n-1)
    aft = aft * a;
    if aft(j)
        c = true;
        return
    end
end

c = false;

end

