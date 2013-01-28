% Note: this just does vector norms for GPU variables.
function [ out ] = norm( in )
%NORM Summary of this function goes here
%   Detailed explanation goes here
% 
% if(size(in,2)~=1)
%     in = in(:);
% end

out = cublasSnrm2(numel(in),getPtr(in),1);


end

