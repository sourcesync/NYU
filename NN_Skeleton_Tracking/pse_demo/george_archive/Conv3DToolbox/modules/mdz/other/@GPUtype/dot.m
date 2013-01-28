function [ out ] = dot( A,B )
%DOT Summary of this function goes here
%   Detailed explanation goes here
% sA = size(A);
% sB = size(B);
% if(sA(2)~=1)
%     A = A(:);
% end
% if(sB(2)~=1)
%     B = B(:);
% end

out = cublasSdot(numel(A),getPtr(A),1,getPtr(B),1);


end

