%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% A wrapper for cudaBlass calls that do C = alpha*A'*B + beta*C; Faster than
% doing the transpose then the multiple in matlab.
%
% @file
% @author Matthew Zeiler
% @date Apr 15, 2011
%
% @gpu_file @copybrief gpu_TmN.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief gpu_TmN.m
%
% @param A a matrix to be transposed in the multiply.
% @param B a matrix.
% @param C output matrix (has to be preallocated*
% @param alpha optional scalar multiply on the matrix multiplicatoin.
% @param beta optional scalar time the previous C matrix.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = gpu_TmN(A,B,C,alpha,beta)

if(nargin<4)
    alpha = 1.0;
end
if(nargin<5)
    beta = 0.0;
end

% if(nargin<3)
%     C = zeros(size(A,2),size(B,2),GPUsingle);
% end
cublasSgemm('t','n',size(A,2),size(B,2),size(A,1),alpha,getPtr(A),size(A,1),getPtr(B),size(B,1),beta,getPtr(C),size(C,1));


end








