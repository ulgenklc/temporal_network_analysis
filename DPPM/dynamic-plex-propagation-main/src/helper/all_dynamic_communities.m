function all_dyn_comms = all_dynamic_communities(dynamic_communities)
%ALL_DYNAMIC_COMMUNITIES Returns a list of all dynamic communities
%   Creates a vector of all dynamic community IDs, useful for enumerating dynamic
%   communities.
%
%   ALL_DYN_COMMS = ALL_DYNAMIC_COMMUNITIES(DYNAMIC_COMMUNITIES) takes a cell array
%   with dynamic communities at each time step. It returns ALL_DYN_COMMS, a vector of all 
%   unique dynamic communities.

all_dyn_comms = unique(horzcat(dynamic_communities{:}));

end
