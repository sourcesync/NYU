%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Train model and plot results using Convolutional NCA Regression %
% objective                                                       %
% For details, see: Pose-sensitive Embedding by Convolutional NCA %
% Regression                                                      %
% http://www.cs.nyu.edu/~gwtaylor/publications/                   %
%                      nips2010/gwtaylor_nips2010.pdf             %
%                                                                 %
% Make sure GPUmat is started                                     %
% and snapshot_path is set correctly in conv_drlim.m              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('./util')

% Train model (for fixed # of mini-batches)
conv_ncar
% Show example matches
plot_results