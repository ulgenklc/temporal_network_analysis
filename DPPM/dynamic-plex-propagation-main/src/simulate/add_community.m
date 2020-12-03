function C = add_community(C, m, k, preserve, start, stop)
%ADD_COMMUNITY Adds a dynamic community through time to 
%   Creates a `k`-plex of size `m` within the timeseries `C` starting at
%   time `start` and persisting to time `stop`. At each timestep, the
%   k-plex moves randomly, preserving `preserve` edges from the previous
%   iteration.

% default values
if ~exist('m', 'var') || isempty(m)
    m = 5;
end
if ~exist('k', 'var') || isempty(k)
    k = 2;
end
if ~exist('preserve', 'var') || isempty(preserve)
    preserve = 2; % number of edges to preserve
end
if ~exist('start', 'var') || isempty(start)
    start = 1;
end
if ~exist('stop', 'var') || isempty(stop)
    stop = size(C, 3);
end

% number of vertices
n = size(C, 1);

% initial community
comm = randperm(n, m);

% initial connectivity
con = make_plex(m, k);

% set starting connectivity
C(comm, comm, start) = con;

for t = (start + 1):stop
    % edges to preserve
    edges = get_edges(comm, con, preserve);
    
    % nodes to preserve
    to_preserve = unique(reshape(edges, 1, []));
    % remaining nodes
    remaining = setdiff(1:n, to_preserve);
    % establish new community
    comm = [to_preserve, remaining(randperm(n - length(to_preserve), m - length(to_preserve)))];
    
    % generate new connectivity
    con = make_plex(m, k);
    
    % re-add preserved edges
    remove = false;
    for i = 1:preserve
        j = find(comm == edges(i, 1));
        h = find(comm == edges(i, 2));
        if ~con(j, h)
            con(j, h) = 1;
            con(h, j) = 1;
            remove = true;
        end
    end
    if remove
        % could be more tightly connected then k-plex, remove extra edges
        con = to_plex(con, k);
    end
    
    C(comm, comm, t) = con;
end

end

