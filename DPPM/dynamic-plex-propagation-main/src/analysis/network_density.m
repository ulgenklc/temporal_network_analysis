function density = network_density(C)
%NETWORK_DENSITY Calculate density of network or dynamic network
%   Given a network (n x n) or a dynamic network (n x n x t), returns the
%   network density over time (either a scalar or a vector with t
%   elements).

% number of vertices
n = size(C, 1);

% max number of edges
max_edges = (n * (n - 1)) / 2; % division by cancels out, but left for readability

% count edges
density = squeeze(sum(sum(C, 1), 2)) / 2 / max_edges;

end
