function plot_vertices_over_time(vertices, dynamic_communities, time_axis)
%PLOT_VERTICES_OVER_TIME Plot the number of vertices in communities over time
% create a timeline of all  of the number of vertices in a dynamic component ever

% default values
if ~exist('time_axis', 'var') || isempty(time_axis)
    time_axis = 1:length(dynamic_communities);
end

% extract list of all dynamic communities
all_dyn_comms = all_dynamic_communities(dynamic_communities);

% number of dynamic communities
num_dynamic_communities = length(all_dyn_comms);

% make an index
index = cell(1, max(all_dyn_comms));
for t = 1:length(dynamic_communities)
    % current dynamic communities
    c = dynamic_communities{t};

    % add time step to index
    if ~isempty(c)
        index(c) = cellfun(@(x) [x t], index(c), 'UniformOutput', false);
    end
end

% make cell array for plot function
ca = cell(1, 2 * num_dynamic_communities);

% calculate timeline for each dynamic community
for i = 1:num_dynamic_communities
    cur_dynamic_comm = all_dyn_comms(i);
    idx = index{cur_dynamic_comm};
    cur_vertices = vertices_in_community(vertices(idx), dynamic_communities(idx), cur_dynamic_comm);
    num = sum(cur_vertices, 2);
    
    % add to cell array
    ca{i * 2 - 1} = time_axis(idx);
    ca{i * 2} = num;
end

% free up memory
clear index;

% configure plot
plot(ca{:});
xlim([time_axis(1) time_axis(end)]);
xlabel('Time');
ylabel('Number of Vertices');

end
