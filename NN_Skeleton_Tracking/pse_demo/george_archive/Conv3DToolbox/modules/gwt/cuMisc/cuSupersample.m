%CUSUPERSAMPLE Upsampling by a constant factor, f
%
%   CUSUPERSAMPLE(A,B,f) Subsamples every image (columns of A) by a factor
%   of f by copying each element f*f times and storing the result in B 
%   INPUTS:
%     A : (imgSize * imgSize) x numCases SQUARE images
%     B : (imgSize*f)*(imgSize*f) x numCases SQUARE upsampled images
%     f : factor by which to upsample (in each of x and y) 2<= f <= 16
