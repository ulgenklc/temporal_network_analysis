function [C, xy] = generate_network(p, width, height, time_steps)
%GENERATE_NETWORK Generate a timeseries of network adjacency matrices
%   Produces a timeseries of length `time_steps`, where ach step has an
%   adjacency matrix. Also generates a `width` by `height` set of nodes
%   that have spatial coordinates returned as `xy`. Each edge has a
%   probability of `p` of being connected, independent of the previous
%   timesteps and other edges.

% default values
if ~exist('p', 'var') || isempty(p)
    p = 0.1;
end
if ~exist('width', 'var') || isempty(width)
    width = 4;
end
if ~exist('height', 'var') || isempty(height)
    height = 4;
end
if ~exist('time_steps', 'var') || isempty(time_steps)
    time_steps = 50;
end

% build grid
xy = zeros(width * height, 2);
for x = 1:width
    s = (x - 1) * height + 1;
    e = x * height;
    xy(s:e, 1) = x * ones(height, 1);
    xy(s:e, 2) = 1:height;
end

n = width * height; % number of vertices
m = n * (n - 1) / 2; % potential number of edges
idx = zeros(m, 2); % index of potential edge -> coordinates in adjacency matrix
k = 1;
for i = 1:(n - 1)
    for j = (i+1):n
        idx(k, :) = [i j];
        k = k + 1;
    end
end

% build connectivity network
C = zeros(n, n, time_steps);
for t = 1:time_steps
    r = rand(1, m);
    for i = find(r < p)
        j = idx(i, 1); k = idx(i, 2);
        C(j, k, t) = 1;
        C(k, j, t) = 1;
    end
end

end
