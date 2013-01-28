%CUGRIDTOMATRIX Extract blocks from square images and store as columns
%(like im2col but in parallel)
%
%   CUGRIDTOMATRIX(A,B,f) extracts every non-overlapping f*f block from
%   every image (columns of A) and stores the result as columns of B
%
%   INPUTS: 
%   A : (imgSize * imgSize) x numCases SQUARE images 
%   B :   (f*f) x regionsPerImage*numCases SQUARE result images
%   f : Region size (in each of x and y) 2<= p <= 16
%
%   NOTE: the (f*f) blocks are currently transposed before writing to B