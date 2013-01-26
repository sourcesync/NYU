function d = sdistmatbsxfun_t(x, y)
%DISTMAT1 Squared distance matrix (uses bsxfun).
%Same as distmatbsxfun_t but doesn't apply sqrt(d)
%   D = SDISTMATBSXFUN_T(X, Y) returns the distance matrix with all distances
%   between the points represented by the COLUMNS of X and Y.
%
%   SDISTMATBSXFUN_T(X) is equivalent to SDISTMATBSXFUN_T(X,X)
%
%   Distance is Euclidean.
%
% Uses a single loop over dimensions
% and makes use of bsxfun
% The idea comes from
% http://blog.accelereyes.com/blog/2010/04/05/converting-matlab-loops-to-gp
% u-code/
%
% This version assumes that data are in COLUMNS; not rows
error(nargchk(1, 2, nargin));

if nargin == 1                       % DISTMAT(X)
  
  if ndims(x) ~= 2
    error('Input must be a matrix.');
  end
  
  [n,m] = size(x);
  d = zeros(m, m);                  % initialise output matrix
    
  for ii=1:n
    d = d + bsxfun(@minus,x(ii,:),x(ii,:)').^2;
  end
  
  
else                                 % DISTMAT(X, Y)
  
  if ndims(x) ~= 2 | ndims(y) ~= 2
    error('Input must be two matrices.');
  end
  
  [nx, mx] = size(x);
  [ny, my] = size(y);
  
  if nx ~= ny
    error('Both matrices must have the same number of rows.');
  end
  
  m = mx;                   % number of rows in distance matrix
  n = my;                   % number of columns in distance matrix
  p = nx;                   % dimension of each point
  d = zeros(m, n);          % initialise output matrix

  for ii=1:nx
    d = d +bsxfun(@minus,y(ii,:),x(ii,:)').^2;
  end

  
end
