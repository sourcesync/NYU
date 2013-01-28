%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Average pools the absoluve value of the input maps within pool_size region.
% This just takes abs(input) and passes it into avg_pool. It will automatically
% use the compiled MEX version of avg_pool if available.
%
% @file
% @author Matthew Zeiler
% @date Aug 16, 2010
%
% @pooling_file @copybrief abs_avg_pool.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief abs_avg_pool.m
%
% @param input the planes to be normalized (xdim x ydim x n3 [x n4])
% @param pool_size a vector specifying the pooling sizein x and y dimensions.
% @param pooled_indices UNUSED
% @param COMP_THREADS sets the number of threads to use in MEX version of avg_pool
%
% @retval pooled the pooled output planes
% @retval indices [] for average pooling.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [pooled,indices] = abs_avg_pool(input,pool_size,pooled_indices,COMP_THREADS)

% Just wrapping average pooling by taking abs of the input images.
[pooled,indices] = avg_pool(abs(input),pool_size,pooled_indices,COMP_THREADS);

end
