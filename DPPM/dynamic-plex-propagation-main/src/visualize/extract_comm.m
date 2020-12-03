function ret = extract_comm(C, vertices, dynamic_communities)
%EXTRACT_COMM Converts to plotting format
%   Converts the results of the dpp matrix into a format that can be used
%   by the plotting functions written by Wes Viles.

n = size(C, 1);
max_t = size(C, 3);
ret = cell(1, max_t);
for t = 1:max_t
    % make empty col
    cur_col = cell(n, n);
    
    % for each vertex
    for i = 1:n
        % no colors
        if isempty(vertices{t})
            continue
        end
        
        % get in color
        in_color = vertices{t}(:, i);
        if ~any(in_color)
            continue
        end
        
        % get communities
        com = dynamic_communities{t};
        
        % get connections
        con = C(i, :, t);
        
        % for each color that contains the vertex
        for j = find(in_color')
            % find other vertices that are in the same community and
            % connected
            col = and(vertices{t}(j, :), con);
            col(i) = false;
            
            % for each vertex
            for h = find(col)
                % add the community number
                cur_col{i, h} = [cur_col{i, h} com(j)];
            end
        end
    end
    
    % add to the list
    ret{t} = cur_col;
end

end

