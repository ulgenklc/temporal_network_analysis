function C = generate_network_with_edge_probability(p_edge, time_steps)
%GENERATE_NETWORK_WITH_EDGE_PROBABILITY Network based on edge probabilities
%   Generates a dynamic network based on the probability of edges existing
%   between two vertices.
%
%   C = generate_network_with_edge_probability(P_EDGE, TIME_STEPS) takes a 
%   square (N x N), symmetric matrix P_EDGE representing the probability of 
%   edges existing between two vertices (diagonal elements are ignored) and 
%   produces a logical dynamic network of size N x N x TIME_STEPS 
%   representing random connectivity chosen independently for each edge and 
%   each time step.

% number of vertices
n = size(p_edge, 1);

% generate an upper right triangular mask to extract edge probabilities
v = 1:n;
[x, y] = meshgrid(v, v);
ult = y < x;

% ignore rest of matrix
p_edge(~ult) = 0;

% empty network
C = false(n, n, time_steps);
for t = 1:time_steps
    % generate random numbers between 0 and 1
    r = rand(n, n) <= p_edge;
    C(:, :, t) = r | r';
end

end

