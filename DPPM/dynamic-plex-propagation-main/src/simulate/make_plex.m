function con = make_plex(m, k)
%MAKE_PLEX Generates an adjacency matrix for a k-plex
%   Returns an m by m adjacency matrix that meets the minimum requirements
%   for a k-plex. It may have more edges.

% initial connectivity
con = ones(m, m) - diag(ones(1, m));
attempt = 10; % finite number of attempts
while attempt
    % remove connection to make k-plex
    can = find(sum(con, 1) > (m - k));
    if length(can) < 2
        break
    end
    
    % 
    idx = randperm(length(can), 2);
    i = can(idx(1));
    j = can(idx(2));
    
    % not connected?
    if ~con(i, j)
        attempt = attempt - 1;
        continue
    end
    
    % ensure connectivity is preserved (not interested in disconnected
    % plexes)
    if ~preserves_connectivity(con, i, j)
        attempt = attempt - 1;
        continue
    end
    
    % remove connection
    con(i, j) = 0;
    con(j, i) = 0;
end

end

