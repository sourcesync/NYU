%CUCONV 2-D convolution : square images, square filters
%
%   CUCONV(A,B,X) performs 2-D convolution of filters B on images A
%   writing the result to X. This assumes 'valid' type of
%   convolution. This parallelizes over filters and images.
%   INPUTS:
%    A - imgPixels by numCases matrix representing numCases square images
%    which have been flattened into vectors
%    B - filterPixels by numFilters matrix representing numFilters square
%    filters which have been flattened into vectors
%    X - numFilters*numOutputs by numCases matrix representing the result
%    of convolving each of numFilters filters with each of numCases
%    images numOutputs is the number of pixels in the output (reflecting
%    the effect of convolution).
%
%
%   This uses Alex Krizhevsky's CUDA code to perform the convolution.
%
%   Note that, by the definition of convolution, each filter is rotated
%   (180 deg) before it is dot-multiplied by an image patch
%   This is implicit in Matlab's conv2()
%   However, Alex's code does not perform this rotation
%   So to achieve identical results to conv2() you must rotate your
%   filters ( i.e. fliplr(flipud()) ) before passing them to cuConv
%
