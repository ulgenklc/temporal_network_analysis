function sub_a = subgraph(a, v)
%SUBGRAPH Extracts a subgraph including vertices listed in v.
%   Extracts a subgraph from adjacency matrix a including vertices in v. Preserves all 
%   edges {i, j} in graph a iff i, j exist in v.

sub_a = a(v, v);

end
