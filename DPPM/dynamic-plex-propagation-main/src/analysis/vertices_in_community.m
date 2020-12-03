function vert = vertices_in_community(vertices, dynamic_communities, community)
%VERTICES_IN_COMMUNITY Finds vertices in a specific dynamic community at each time step
%   Returns a logical matrix with columns for vertices and rows for time
%   steps, where a true value represents that a vertex was involved in
%   community.
%
%   VERT = VERTICES_IN_COMMUNITY(VERTICES, DYNAMIC_COMMUNITIES, COMMUNITY) takes the
%	a description of dynamic communities (VERTICES and DYNAMIC_COMMUNITIES) and extracts
%   the occurences of a single community in terms of the vertices involved at each time
%   time step.

% number of time stpes
t = length(vertices);

% number of vertices
n = size(vertices{1}, 2);

% return
vert = false(t, n);

% for each time step
for i = 1:t
	% occurences of community
    in = (community == dynamic_communities{i});
    % has any?
    if any(in)
        vert(i, :) = any(vertices{i}(in, :), 1);
    end
end

end

