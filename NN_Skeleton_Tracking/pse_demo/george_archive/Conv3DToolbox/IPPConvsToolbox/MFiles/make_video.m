%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% A function that takes in paths and parameters to make a video out of png images.
%
% @file
% @author Matthew Zeiler
% @date Nov 16, 2010
%
% @param framepath the path where all the png frames are
% @param framename the name of the frames (must be followed by %05d frame_nume).
% @param outputpath the name of the output video (which will be put in the framepath folder)
% @param resolution a 1x2 matrix with the size information (defaults to 550x412)
% @param infps the frames per second of the input video
% @param outfps the frames per second of the output video
%
% @plotting_file @copybrief make_video.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = make_video(framepath,framename,outputname,resolution,infps,outfps)


if(strcmp(framepath(end),'/')==0)
    framepath = [framepath '/'];
end

if(nargin<3)
    outputname = [framename '.mp4'];
end
if(nargin<4)
    resolution = [550 412]
end
if(nargin<5)
    infps = 30
end
if(nargin<6)
    outfps = infps
end


if(strcmp(outputname(end-3:end),'.mp4')==0)
    outputname = [outputname '.mp4'];
end


if(isempty(resolution))
    resolutionstring = '';
else
    resolutionstring = ['-s ' num2str(resolution(1)) 'x' num2str(resolution(2)) ' '];
end

if(isempty(infps))
    infpsstring = '';
else
    infpsstring = ['-r ' num2str(infps) ' '];
end


if(isempty(outfps))
    outfpsstring = '';
else
    outfpsstring = ['-crf ' num2str(outfps) ' '];
end



evalstring = ['!ffmpeg -an ' infpsstring ' -i ' framepath framename '%05d.png -vcodec mpeg4 ' outfpsstring resolutionstring framepath outputname]
eval(evalstring)


end





























