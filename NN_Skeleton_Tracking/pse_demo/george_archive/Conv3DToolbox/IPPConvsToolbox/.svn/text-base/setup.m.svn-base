%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @mainpage Documentation for IPP Convolution Toolbox
%
% <div class="version_description">
% @author Matthew Zeiler (zeiler@cs.nyu.edu)
% @date Apr 4, 2010
% @version 1.0.0 - This is the initial release of this toolbox and may contain
% unkown bugs.
%
% </div>
%
% <div class="website_description">
% This toolbox includes <a href="http://software.intel.com/en-us/intel-ipp/">
% Intel Performance Primitives (IPP)</a> based convolutions of
% various types. The standard implementation ipp_conv2 can replace the
% MATLAB conv2 directly with significant speed-ups. These convolutions
% automatically switch to FFTs when the kernel size becomes large enough.
% Additionally, ipp_conv2 and the other included files have been extended beyond
% the functionality of MATLAB's conv2 as they can take in multiple images and
% multiple kernels. See their documentation for further details.
%
% </div>
%
% <div class="instructions">
% @instructions Download and unzip this toolbox in the location of choice. To
% setup this toolbox, simply open matlab, cd to this directory, and type "setup"
% (without quotes) in the command window. This sets up your path to include the required
% directories (you may want to add these paths into your startup.m file as well
% however), tries to compile the IPP mex files included in this toolbox,
% and opens complete documentation for this toolbox. You will likely have to 
% modify the paths in the /MEX/compilemex.m file to suite your machine's setup.
%
% </div>
%
% Alternatively, you can compile the mex files by running compilemex.m from the
% MATLAB command window.
%
% The naming convention for some of the files is as follows:
% It is based on the operations done on dimensions 3 and 4 of filters used
% in a deconvolutional or convolutional network. I will explain here for the
% deconvolutional network case (the convolution network should just be using
% the opposite functions with permuted 3rd and 4th dimensions). The files
% are named as follows:<br>
% [ConvolutionType]_[each/sum][index]_[each/sum][index] where [ConvolutionType]
% is either valid or full (as in standard convolution literature), [each/sum]
% means if that dimension (index) is summed over or if there is a resulting
% output for each plane and [index] is the corresponding dimension of the filters
% you are working with. For example: valid_each3_sum4.m is used in the deconvolutional
% network to reconstruct down with valid convolutions by summing over the features
% maps (dimension 4 of the filters) to get a resulting sum for each of the input
% maps below (dimension 3 of the filters).<br>
% If you are using the old IPPConvToolbox, then the naming scheme was as follows:
% (included are backwards compatibility functions that call the new functions)
% ConvolutionType_index1_index2 where index1 is the indexing into the planes
% of input 1 and index 2 is the indexing into the planes of input 2. For
% example: valid_eachK_loopJ.cpp does 'valid' convolutions where for each plane
% K of the first input, it convolves that with each plane J of the second input.
% The resulting size for each functoin is the size of the plane-by-plane
% convolution (depends on type) in the first two dimensions, then by the number
% of input maps (first dimension of connectivity matrix passed in) and
% number of feature maps (second dimension of connectivity matrix passed in)
% in the remaining two dimensions of the 4-D output.
%
% Requirements:
%   \li A system with one or more multi-core Intel 64-bit CPU's
%   \li Up to date installations of:
%       1) Intel 64-bit C compiler (tested on version current)
%       2) Intel Integrated Performance Primitive (IPP) libraries (tested on version 5.3)
%   \li Matlab - to actually use the MEX file (tested on 7.5.0.338 (R2007b))
%
% Points to note:
%
% 1. These environment variables that need to be set in bash before running the mex file:
% export PATH="/opt/intel/cce/current/bin:${PATH}";
% export LD_LIBRARY_PATH="/opt/intel/cce/current/lib:${LD_LIBRARY_PATH}";
%
% 2.  The IPP libraries will automatically swap to using Fourier domain ...
% multiplication once the size of the kernel is above 15x15 or so.
%
%%%%%%%%%%%%%%%%%
% How to compile:
% Run this setup.m file in MATLAB.
%
% Note that you will likely need to change the paths in some of the commands to find
% (i) the IPP libraries (ii) Intel compiler on your system and (iii) the
% customized mexopts.sh file. See the compilemex.m file to change some of these
% paths.
%%%%%%%%%%%%%%%%%
%
% You will likely need to alter the mexopts.sh file that Matlab uses. We have included one with
% this package, but you may have to alter it. Then alter the -f option in the
% ./MEX/compilemex command to call it.
%
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
% @ipp_file @copybrief setup.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


fprintf('Fist make sure you are running this setup.m from its directory.\n');

fprintf('Adding paths now...\n')
addpath(genpath(strcat(pwd,'/MEX')))
addpath(genpath(strcat(pwd,'/M')))
addpath(genpath(strcat(pwd,'/Test')))

fprintf('Compiling the MEX files now...\n')

cd('./MEX')
try
    compilemex
catch ME
    fprintf('Check the IPP library paths in /MEX/compilemex.m and your environment variables, then please try again.\n');
end
cd('../')

fprintf('Opening the HTML documentation in a browser now...\n');
try
    open([pwd '/Documentation/index.html'])
catch ME
    fprintf('Either path is incorrect to Documentation, or you are not running with java enabled\n');
end





