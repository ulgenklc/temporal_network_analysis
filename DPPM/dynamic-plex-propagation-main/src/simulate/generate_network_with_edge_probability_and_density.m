function C = generate_network_with_edge_probability_and_density(edge_freq, densities)
%GENERATE_NETWORK_WITH_EDGE_PROBABILITY_AND_DENSITY Summary
%   Generates a dynamic network based on the probability of edges existing
%   between two vertices and matching a time course of densities.

% time steps
time_steps = length(densities);

% number of vertices
n = size(edge_freq, 1);

% generate an upper right triangular mask to extract edge probabilities
v = 1:n;
[x, y] = meshgrid(v, v);
ult = y < x;

% make into a column vector
edge_freq = reshape(edge_freq(ult), [], 1);
x = reshape(x(ult), [], 1);
y = reshape(y(ult), [], 1);
idx = 1:length(edge_freq);

% max edges
max_edges = (n * (n - 1)) / 2;
if max(densities) <= 1
    target_edges = round(max_edges * densities);
else
    target_edges = densities;
    target_edges(densities > max_edges) = max_edges;
end

% empty network
C = false(n, n, time_steps);
for t = 1:time_steps
	% copy edge probability matrix
	cur_edge_freq = edge_freq;
	
	% add edges up to target density
	for i = 1:target_edges(t)
        j = randsample(idx, 1, true, cur_edge_freq);
        k = x(j);
        l = y(j);
        cur_edge_freq(j) = 0;
        
        C(l, k, t) = true;
        C(k, l, t) = true;
	end
end

end
