function C = generate_network_with_degree(degree, edge_freq)
%GENERATE_NETWORK_WITH_DEGREE Summary
%   Generates a dynamic network based on a time series of degree 
%   values for each vertex. The DEGREE matrix should be a t by n matrix,
%   where n is the number of vertices and t is the number of time steps.

%   between two vertices and matching a time course of densities.

% number of vertices and number of time steps
n = size(degree, 2);
time_steps = size(degree, 1);

% edge frequency matrix
if ~exist('edge_freq', 'var') || isempty(edge_freq)
    edge_freq = ones(n, n);
end

% check edge matrix
if ~ismatrix(edge_freq) || n ~= size(edge_freq, 1)  || n ~= size(edge_freq, 2)
    error('The edge_freq matrix should be square and describe the same number of vertices as the degree matrix.');
end

% check maximum degree
if max(max(degree)) > (n-1)
    error('The maximum degree can not be larger than %d.', n-1);
end
if any(mod(sum(degree, 2), 2))
    error('The sum of the degree values should be an even number.');
end

% generate a grid
v = 1:n;
[x, y] = meshgrid(v, v);
edge_freq(x == y) = 0;

% empty network
C = false(n, n, time_steps);
for t = 1:time_steps
    % target degree
	target_degree = degree(t, :);
    
    % try multiple times
    success = false;
    while ~success
        % copy edge probability matrix
        cur_edge_freq = edge_freq;
        
        % start current degree
        diff_degree = target_degree;
        
        % current adjacency matrix
        a = false(n, n);
        
        while true
            % node requiring most
            [d, i] = max(diff_degree);
            if d == 0
                success = true;
                break;
            end
        
            % weights connecting to node i
            iter_p = cur_edge_freq(i, :);
            if ~any(iter_p)
                break
            end
            
            % get connected node
            j = randsample(v, 1, true, iter_p);

            % add edge
            a(i, j) = true;
            a(j, i) = true;

            % zero out edge frequency
            cur_edge_freq(i, j) = 0;
            cur_edge_freq(j, i) = 0;
            
            % decrement degree and zero out any unneeded row/column in freq matrix
            diff_degree(i) = d - 1;
            if diff_degree(i) == 0
                cur_edge_freq(i, :) = 0;
                cur_edge_freq(:, i) = 0;
            end
            diff_degree(j) = diff_degree(j) - 1;
            if diff_degree(j) == 0
                cur_edge_freq(j, :) = 0;
                cur_edge_freq(:, j) = 0;
            end
        end
    end
    
    C(:, :, t) = a;
    fprintf('%.1f%%\n', 100*t/time_steps);
end

end
