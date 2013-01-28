%CUTHREEWAY Generalized three-way outer product
%
%   CUTHREEWAY(A,B,C,D) takes, as inputs,  
%     A - nA by numCases matrix
%     B - nB by numCases matrix
%     C - nC by numCases matrix
%     and produces,
%     D - nA x nB x nC array where element 
%      D(a,b,c) = sum_i A(a,i) * B(b,i) * C(c,i) 
%
%   This is a required statistic for a three-way (Gated) RBM.
