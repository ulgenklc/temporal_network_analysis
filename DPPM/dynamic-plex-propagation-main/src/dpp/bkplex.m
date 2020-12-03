function h = bkplex(a, k, minsize)
%BKPLEX Produces a list of maximal k-plexes larger than msize
%   Uses the Bron-Kerbosch algorithm to find the maximal k-plexes (relaxed
%   cliques such that each node is connected to n-k other nodes in the
%   k-plex), such that each k-plex is larger than or equal to size minsize.

% Standard cliques by setting k = 0.

    if ~exist('k', 'var') || isempty(k)
        k = 2;
    end
    if ~exist('minsize', 'var') || isempty(minsize)
        minsize = k + 2;
    end

    % dimensions
    n = size(a, 1);
    
    % list of k-plexes
    h = false(0, n);
    
    % start algorithm
    P = true(1, n);
    X = false(1, n);
    for i = 1:n
        % build new R for recursive call
        R = false(1, n); R(i) = true;
        
        % eliminate
        P(i) = false;
        
        % call
        bkplexhelp(R, P, X, 1);

        % eliminate
        X(i) = true;
    end

    function bkplexhelp(R, P, X, n_cur)
        if ~any(P) && ~any(X)
            if minsize <= n_cur
                h(end + 1, :) = R;
            end
        else
            % length
            n_new = n_cur + 1;
            
            % subgraph connections
            cur_con = sum(a(R, :), 1);
            
            % only explore direct connections: and(P, any(a(R, :), 1))
            for j = find(P)
                if 0 == cur_con(j)
                    continue
                end
                
                % build new R for recursive call
                R_new = R; R_new(j) = true;
                
                % calculates the list of neighbors that may expand the k-plex R
                % crit is a list of nodes with degree |R| - k
                con = cur_con + a(j, :);
                
                % get list of neighbors
                crit = and(con <= (n_new - k), R_new);
                ng = and(all(a(crit, :), 1), con > (n_new - k));
                ng(j) = false;
                
                % build values for recursive call
                P_new = and(P, ng);
                X_new = and(X, ng);
                
                % recursive
                bkplexhelp(R_new, P_new, X_new, n_new);
                
                % eliminate
                P(j) = false;
                X(j) = true;
            end
        end
    end

end
