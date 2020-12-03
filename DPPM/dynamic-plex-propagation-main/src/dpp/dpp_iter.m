function [cliq_and_plex1, communities1, cliq_and_plex2, communities2, dyn_communities] = dpp_iter(a1, a2, k, m, cliq_and_plex1, communities1)
%DPP_ITER 
%   Performs calculations for dynamic plex propegations, identifying
%   components that span two time steps and connecting them into a dynamic
%   community.

% reuse previous calculations
if ~exist('cliq_and_plex1', 'var')
    [cliq_and_plex1, communities1] = dpp_single(a1, k, m);
end

% extract second time step
[cliq_and_plex2, communities2] = dpp_single(a2, k, m);

% number of communities in each time step
mc1 = max(communities1);
mc2 = max(communities2);

% initial numbering of dynamic communities
dyn_communities = 1:(mc1 + mc2);

% for each components
for i = 1:mc1
    % cliques and plexes in community i in first time step
    vals1 = find(communities1 == i);
    for j = 1:mc2
        % cliques and plexes in community j in second time step
        vals2 = find(communities2 == j);

        % for each clique/plex in the two communities
        found = false;
        for hi = vals1
            for hj = vals2
                % compare the overlap
                overlap = sum(and(cliq_and_plex1(hi, :), cliq_and_plex2(hj, :)));
                if overlap >= (m-1)
                    % merge the two communities
                    dyn_communities = communities_merge(dyn_communities, [i, j + mc1]);
                    
                    % mark as found, stop looking
                    found = true;
                    break;
                end
            end

            % found? stop looking
            if found
                break;
            end
        end
    end
end

end

