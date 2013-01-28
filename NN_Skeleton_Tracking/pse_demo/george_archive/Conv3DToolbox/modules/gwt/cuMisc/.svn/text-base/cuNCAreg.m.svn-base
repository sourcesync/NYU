%CUNCAREG Gradients for NCA regression
%
% Computes:
%   for kk=1:n
%     for jj=1:n
%       out(kk,:) = out(kk,:) + (in(:,kk)-in(:,jj)) * ...
%           w(jj,kk);
%     end
%   end
% Note that out has cases first, dims last
% So that writes are coalesced
% INPUT1: in (mxn) data matrix where m are the dim and n is the number of cases
% INPUT2: w  (nxn) weight matrix: columns correspond to n in the output
%               rows correspond to the cases that are summed over
% OUTPUT: out the (nxm) gradient matrix
%
% After execution, likely want to transpose the gradient matrix to match
% dimensions of input 
%
% Usage: cuNCAreg(in,w,out)
%
% Known bugs: Running with the setting 1600 cases, 2 dimensions
% causes Matlab to crash (even though 4096 cases, 2 dimensions is fine)
% If I "prime" it with many other calls before running the 1600-2 case
% then I can run 1600-2 without a crash