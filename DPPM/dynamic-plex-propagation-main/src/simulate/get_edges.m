function edges = get_edges(comm, con, num)
%GET_EDGES Returns random edges from adjacency matrix
%   Given a community `comm` with connectivity `con`, returns `num` of
%   random edges to preserve.

% enumerate potential edges
potential = zeros(0, 2);
n = size(con, 1);
for i = 1:(n-1)
    for j = (i+1):n
        if con(i, j)
            potential(end + 1, :) = comm([i j]);
        end
    end
end

% indices
idx = randperm(size(potential, 1), num);
edges = potential(idx, :);

end

