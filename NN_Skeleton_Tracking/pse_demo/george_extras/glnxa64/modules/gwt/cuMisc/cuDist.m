%CUDIST Distance matrix (one for-loop).
%
%   CUDIST(X, Y, D) writes to D the distance matrix with all distances
%   between the points represented by the rows of X and Y.
%
%   CUDIST(X,D) is equivalent to CUDIST(X, X, D), but the former computes
%   the distance matrix faster.
%
%   Distance is Euclidean.
%
%   The calculation is done with one for-loop.