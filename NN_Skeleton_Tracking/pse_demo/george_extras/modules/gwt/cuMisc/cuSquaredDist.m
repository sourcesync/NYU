%CUSQUAREDDIST SQUARED Distance matrix (one for-loop).
%
%   CUSQUAREDDIST(X, Y, D) writes to D the distance matrix with all distances
%   between the points represented by the rows of X and Y.
%
%   CUSQUAREDDIST(X,D) is equivalent to CUSQUAREDDIST(X, X, D), but the
%   former computes the distance matrix faster.
%
%   Distance is Euclidean.
%
%   The calculation is done with one for-loop.
%
%   Two differences between cuSquaredDist and cuDist
%    1) We do not square the distances
%    2) The output, D, is not overwritten but is added to
%   These changes are useful in a high-dimensional setting, where we can
%   call cuSquaredDist many times for different blocks of dimensions
