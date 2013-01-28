%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% A wrapper for cudaBlass calls that do C = alpha*A*B' + beta*C;  Faster than
% doing the transpose then the multiple in matlab.
%
% @file
% @author Matthew Zeiler
% @date Apr 15, 2011
%
% @gpu_file @copybrief gpu_NmT.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief gpu_NmT.m
%
% @param A a matrix
% @param B a matrix to be transposed in the multiply.
% @param C output matrix (has to be preallocated*
% @param alpha optional scalar multiply on the matrix multiplicatoin.
% @param beta optional scalar time the previous C matrix.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = gpu_NmT(A,B,C,alpha,beta)

if(nargin<4)
    alpha = 1.0;
end
if(nargin<5)
    beta = 0.0;
end


cublasSgemm('n','t',size(A,1),size(B,1),size(A,2),alpha,getPtr(A),size(A,1),getPtr(B),size(B,1),beta,getPtr(C),size(C,1));

end

