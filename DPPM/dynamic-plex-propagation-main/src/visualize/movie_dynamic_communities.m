function movie_dynamic_communities(file, C, vertices, dynamic_communities, xy, time_axis, frame_rate)
%MOVIE_DYNAMIC_COMMUNITIES Create a movie file depicting dynamic communities
%   Renders a movie showing dynamic communities and their evolution over time.
%
%   MOVIE_DYNAMIC_COMMUNITIES(FILE, C, VERTICES, DYNAMIC_COMMUNITIES, XY) saves a movie to
%   FILE depicting the dynamic communities (as described by VERTICES and 
%   DYNAMIC_COMMUNITIES) in the functional network C, where the position of each vertex is
%   described in matrix XY.
%
%   MOVIE_DYNAMIC_COMMUNITIES(..., TIME_AXIS) uses the timing information in TIME_AXIS to
%   label each frame and calculate the frame rate for the movie based on the timing. Note 
%   that the frame rate will not be less than 5 frames a second.
%
%   MOVIE_DYNAMIC_COMMUNITIES(..., TIME_AXIS, FRAME_RATE) explicitly sets the frame rate 
%   for the generated movie file.

% default values
if ~exist('time_axis', 'var') || isempty(time_axis)
    time_axis = 1:length(dynamic_communities);
end
if ~exist('frame_rate', 'var') || isempty(frame_rate)
	% calculate frame rate
	frame_rate = round(length(time_axis) / (time_axis(end) - time_axis(1)));
	% minimum frame rate
	if frame_rate < 5
		frame_rate = 5;
	end
end

% convert to plotting format
comm = extract_comm(C, vertices, dynamic_communities);

% prepare plotting
circ = generate_circmat(xy);

% start movie
wo = VideoWriter(file, 'MPEG-4');
wo.FrameRate = frame_rate;
open(wo);

% render each frame
for t=1:size(C, 3)
	bendplot(C(:,:,t), comm{t}, circ, xy);
	title(['Time ', num2str(time_axis(t))]);
	frame = getframe(gcf);
	writeVideo(wo, frame);
end

% close movie
close(wo);

end
