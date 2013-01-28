%CUROTATE180 Batch rotation of many filters
%
%   CUROTATE180(A,B) rotates all filters (columns of A) and store the
%   result in B
%   This is equivalent to
%   flipud(fliplr(reshape(A(:,ii),filterSize,filterSize))) for every ii
%%   INPUTS:
%     A : (filterSize * filterSize) x numCases SQUARE filters
%     B : (filterSize * filterSize) x numCases SQUARE rotated filters
