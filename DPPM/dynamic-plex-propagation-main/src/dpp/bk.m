function h = bk(a, minsize)
%BK Produces a list of maximal cliques larger than msize
%   Uses the Bron-Kerbosch algorithm to find the maximal cliques, such that
%   each clique is larger than or equal to size minsize.

% Based in part on:
% http://www.mathworks.com/matlabcentral/fileexchange/47524-find-maximal-cliques-for-large---sparse-network/content/ELSclique.m

    if ~exist('minsize', 'var') || isempty(minsize)
        minsize = 4;
    end

    % dimensions
    n = size(a, 1);
    
    % list of cliques
    h = false(0, n);
    
    % degenerecy ordering
    ord = nan(1, n);       % to store a degeneracy ordering of the vertices
    a_temp = double(a); % copy
    for i = 1:n
        deg = sum(a_temp, 2);
        % get entry of lowest degree vertex 
        [~, idx] = min(deg);
        % remove entry
        a_temp(idx, :) = 0; a_temp(:, idx) = 0; a_temp(idx, idx) = inf;
        ord(i) = idx;
    end

    % start algorithm
    for i = 1:n
        % get vertex
        v = ord(i);
        % fill boolean vector
        R = false(1,n); R(v) = true;
        % neighbors of v
        neigh_v = logical(a(v, :));
        P = neigh_v; P(ord(1:i)) = false;
        X = neigh_v; X(ord(i:end)) = false;
        bkhelp(R, P, X);
    end

    function bkhelp(R, P, X)
        if ~any(P) && ~any(X)
            if minsize <= sum(R)
                h(end + 1, :) = R;
            end
        else
            pivots = find(or(P, X));
            p_counts = sum(a(pivots, P), 2);
            [~, ind] = max(p_counts);
            u_p = pivots(ind);
            
            for j = find(and(~a(u_p, :), P))
                % build values for recursive call
                R_new = R; R_new(j) = true;
                neigh_j = logical(a(j, :));
                P_new = and(P, neigh_j);
                X_new = and(X, neigh_j);
                
                % recursive
                bkhelp(R_new, P_new, X_new);
                
                % eliminate
                P(j) = false;
                X(j) = true;
            end
        end
    end
end
