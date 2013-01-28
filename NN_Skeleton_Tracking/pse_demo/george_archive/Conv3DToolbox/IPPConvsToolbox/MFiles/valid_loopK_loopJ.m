%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% This is a backward compatibility file for doing the valid_each3_each4 operation.
% DO NOT USE THIS FUNCTION if coding up new software, use the newer valid_each3_each4.m
% @copybrief valid_each3_each4.m
% @see valid_each3_each4.m for full documentation and usage.
%
% @deprecated use valid_each3_each4.m instead.
% @file
% @author Matthew Zeiler
% @data Jun 14, 2011
%
% @conv_file @copybrief valid_each3_each4.m
% @see valid_each3_each4_gpu.m valid_each3_each4_ipp.cpp valid_each3_each4_3d.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief valid_each3_each4.m
%
% @param maps  n x m x num_feature_maps x num_images matrix of single precision floating point numbers.
% @param images  p x q x num_input_maps x num_images matrix of single precision floating point numbers.
% @param C  num_input_maps x num_feature_maps connectivity matrix.
% @param COMP_THEADS (optional and unused) specifies number of computation threads to parallelize over. Defaults to 4.
%
% @retval out (this is plhs[0]) A 4D matrix of (n+p-1) x (m+q-1) x num_input_maps x num_feature_maps x num_images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [out] = valid_loopK_loopJ(maps,F,C,COMP_THREADS)
out = valid_each3_each4(maps,F,C,COMP_THREADS);
end
