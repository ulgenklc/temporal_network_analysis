function vertices = vertices_in_communities(cliq_and_plex, communities)
%VERTICES_IN_COMMUNITIES Vertices in each community
%   Combines cliques and plexes to get a full view of community in terms of
%   vertices. Returns a logical matrix of m by n rows, where m is the
%   number of communities identified and n is the number of vertices in the
%   original adjacency matrix.

% make vector that maps components to vertices
num_vertices = size(cliq_and_plex, 2);
num_communities = max(communities);
vertices = false(num_communities, num_vertices);
for i = 1:num_communities
    vertices(i, :) = any(cliq_and_plex(communities == i, :), 1);
end

end

