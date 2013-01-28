%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% This is a backward compatibility file for doing the full_each4_sum3 operation.
% DO NOT USE THIS FUNCTION if coding up new software, use the newer full_each4_sum3.m
% @copybrief full_each4_sum3.m
% @see full_each4_sum3.m for full documentation and usage.
%
% @deprecated
% @file
% @author Matthew Zeiler
% @data Jun 14, 2011
%
% @conv_file @copybrief full_each4_sum3.m
% @see full_each4_sum3_gpu.m full_each4_sum3_ipp.cpp full_each4_sum3_3d.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief full_each4_sum3.m
%
% @param maps  n x m x num_input_maps x num_images  matrix of single precision floating point numbers.
% @param F  p x q x num_input_maps x num_feature_maps [x num_images] matrix of single precision floating point numbers.
% @param C  num_input_maps x num_feature_maps connectivity matrix.
% @param COMP_THEADS (optional and unused) specifies number of computation threads to parallelize over. Defaults to 4.
%
% @retval out (this is plhs[0]) A 4D matrix of (n+p-1) x (m+q-1) x num_input_maps x num_images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [out] = full_eachJ_loopK(maps,F,C,COMP_THREADS)
out = full_each4_sum3(maps,F,C,COMP_THREADS);
end


