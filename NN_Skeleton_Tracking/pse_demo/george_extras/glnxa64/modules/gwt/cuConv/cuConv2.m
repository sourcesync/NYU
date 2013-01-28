%CUCONV2 2-D convolution : square images, square filters
% This routine convolves each image with a different set of filters,
% unlike CUCONV, which convolves every image with every
% filter. This is useful for computing derivatives with respect to
% weights in a convolutional net or RBM.
%
%
%   CUCONV2(A,B,X,filterSize) performs 2-D convolution of filters B on
%   images A writing the result to X. Each of the filters in B (columns)
%   are size filterSize x filterSize.
%   This assumes 'valid' type of convolution. It parallelizes over filters 
%   and images.
%   INPUTS:
%    A - imgPixels by numCases matrix representing numCases square images
%    which have been flattened into vectors
%    B - numFilters*filterPixels by numCases matrix representing numFilters square
%    filters (per case) which have been flattened into vectors
%    X - numFilters*numOutputs by numCases matrix representing the result
%    of convolving each of numFilters filters with each of numCases
%    images numOutputs is the number of pixels in the output (reflecting
%    the effect of convolution).
%
%   Limitations (note that zero padding can be used to get around these):
%     -Images and filters must be square
%     -numFilters must be a multiple of 16
%
%   This uses Alex Krizhevsky's CUDA code to perform the convolution.
%
%   Note that, by the definition of convolution, each filter is rotated
%   (180 deg) before it is dot-multiplied by an image patch
%   This is implicit in Matlab's conv2()
%   However, Alex's code does not perform this rotation
%   So to achieve identical results to conv2() you must rotate your
%   filters ( i.e. fliplr(flipud()) ) before passing them to cuConv2
%
