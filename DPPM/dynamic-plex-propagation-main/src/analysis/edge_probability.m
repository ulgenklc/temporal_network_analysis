function edge_prob = edge_probability(C)
%EDGE_PROBABILITY Calculate probabilities of all edges in dynamic network
%   Given a dynamic network (n x n x t), calculates the probability of each
%   edge. This function just takes the mean over time, but is included for
%   better understandability of code.

edge_prob = mean(C, 3);

end
