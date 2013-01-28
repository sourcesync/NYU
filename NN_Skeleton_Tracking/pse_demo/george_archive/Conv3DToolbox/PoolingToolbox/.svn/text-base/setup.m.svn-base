%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @mainpage Documentation for Pooling Toolbox
%
% <div class="version_description">
% @author Matthew Zeiler (zeiler@cs.nyu.edu)
% @date Aug 16, 2010
% @version 1.0.0 -fisrst release of the Pooling Toolbox with Max, Avg, and 
% AbsAvg pooling included.
%
% </div>
%
% <div class="website_description">
% This toolbox includes code that implements a pooling (aka subsampling) of image planes
% over specified regions. Included are various types of pooling operations such as
% max pooling, average pooling, and absolute average pooling along with their
% reverse operations such as placing average back in each location or the 
% max back where it was taken from. Inlcuded are MATLAB M-file versions and 
% MUCH faster MEX file versions to compile. Finally a wrapper
% to place in your code has been provided so you can simply call that with the
% desired type of pooling type and size and it will do the rest. The MEX files 
% rely on OpenMP for multi-threading speed and the directories in the included
% mexopts.sh script may need to be modified for your setup.
%
% </div>
%
% <div class="instructions">
% @instructions Download and unzip this toolbox in the location of choice. To
% setup this toolbox, simply open matlab, cd to this directory, and type "setup"
% (without the quotes) in the command window. This sets up your path to include the required
% directories (you may want to add these paths into your startup.m file as well
% however) compiles the included MEX files (which are MUCH faster than the
% included M-files), and opens complete documentation for this toolbox. 
%
% </div>
%
% @license Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own
% risk.
%
% @bug Please report any bugs to zeiler@cs.nyu.edu.
%
%
% <div class="prev_versions">
% 
% </div>
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


fprintf('Fist make sure you are running this setup.m from its directory.\n');

fprintf('Adding paths now...\n')
addpath(genpath(pwd))

fprintf('Compiling MEX files now...\n')
run('./compilemex')


fprintf('Opening the HTML documentation in a browser now...\n');
try
    open([pwd '/Documentation/index.html'])
catch ME
    fprintf('Either path is incorrect to Documentation, or you are not running with java enabled\n');
end





