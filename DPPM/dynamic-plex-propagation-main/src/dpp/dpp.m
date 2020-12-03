function [vertices, dynamic_communities] = dpp(A, k, m)
%DPP Dynamic plex propogation algorithm
%   Takes a series of adjacency matrices over time and identifies static
%   cliques and k-plexes at each time step. Communities are then linked
%   over time in a manner similar to the clique propogation method (CPM).
%   Two optional parameters determine the minimum k-plex size (m) and the
%   k value for the k-plexes (k).

% default values
if ~exist('k', 'var') || isempty(k)
    k = 2;
end
if ~exist('m', 'var') || isempty(m)
    m = k + 2;
end

% prepare variables
max_t = size(A, 3); % max time (number of adjacency matrices)
vertices = cell(1, max_t); % vertices in each community
dc = cell(1, max_t); % initial numbering of communities
tot = 0; % counter for initial numbering of communities
times = cell(1, max_t); % used to create a vector asscoiating communities with time steps
communities = [];

% for each time step
for t = 1:(max_t - 1)
    nchar = fprintf('%.1f%%', 100*t/max_t);
    if 1 < t
        % second and later iteration
        % reuse previous calculations
        
        % move previous values
        a1 = a2;
        cliq_and_plex1 = cliq_and_plex2;
        communities1 = communities2;
        
        % extract new time points
        a2 = squeeze(A(:, :, t + 1));
        
        % run bkstar
        % cliq1, color1, comps1
        [~, ~, cliq_and_plex2, communities2, dyn_communities] = dpp_iter(a1, a2, k, m, cliq_and_plex1, communities1);
    else
        % first iteration
        
        % extract two time points
        a1 = squeeze(A(:, :, t));
        a2 = squeeze(A(:, :, t + 1));
        
        % run bkstar
        % cliq1, color1, comps1
        [cliq_and_plex1, communities1, cliq_and_plex2, communities2, dyn_communities] = dpp_iter(a1, a2, k, m);
        vertices_in_comm1 = vertices_in_communities(cliq_and_plex1, communities1);
        
        % store special initial entries
        vertices{t} = vertices_in_comm1; % vertices in each community
        n = size(vertices_in_comm1, 1);
        dc{t} = (tot+1):(tot+n); % unique numbering (later clustered
        times{t} = t * ones(n, 1); % timestamps
        communities = [communities (tot+1):(tot+n)];
        tot = tot + n; % increment counter
    end
    
    % calculate vertices in second time step
    vertices_in_comm2 = vertices_in_communities(cliq_and_plex2, communities2);
    
    % store static communities
    vertices{t + 1} = vertices_in_comm2; % vertices in each community
    n = size(vertices_in_comm2, 1);
    dc{t + 1} = (tot+1):(tot+n); % unique numbering (later clustered
    times{t + 1} = (t + 1) * ones(n, 1); % timestamps
    communities = [communities (tot+1):(tot+n)];
    tot = tot + n; % increment counter
    
    % renumber communities as we go
    n = size(vertices{t}, 1);
    if n < length(dyn_communities)
        % for each community in time t
        for i = 1:n
    
            % merge into a single community
            communities = communities_merge(communities, [dc{t}(i) dc{t+1}(dyn_communities(n+1:end) == dyn_communities(i))]);
        end
    end
    
    fprintf(repmat('\b', 1, nchar));
end

% renumber communities
communities = communities_renumber(communities);

% make dynamic components
dynamic_communities = cell(1, max_t);
for t = 1:max_t
    dynamic_communities{t} = communities(dc{t});
end

% project the communities into the new numbering
%dynamic_communities = communities(horzcat(dc{:}));

% convert to structure
%s = struct('vertices', vertcat(vertices{:}), 'time', vertcat(times{:}), 'community', dynamic_communities');
