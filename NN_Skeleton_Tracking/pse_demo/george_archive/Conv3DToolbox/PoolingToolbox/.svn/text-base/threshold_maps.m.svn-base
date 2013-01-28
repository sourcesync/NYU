%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Threshold the output of pooled feature maps in order to suppress pooling
% blocks that do not have strong activatiosn. The threshold is based on the
% histogram bins of the pooled maps.
%
% @file
% @author Matthew Zeiler
% @date Aug 2, 2010
%
% @pooling_file @copybrief threshold_maps.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief threshold_maps.m
%
% @param input the planes to be normalized (xdim x ydim x n3 [x n4])
% @param pool_size a vector specifying the pooling sizein x and y dimensions.
%
% @retval output the planes with only the maxes selected.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [pooled_maps] = threshold_maps(pooled_maps,pool_size)

% Form 1/10 the number of elements in the feature maps (afrer pooling) as bins.
nbins = floor((numel(pooled_maps)/size(pooled_maps,4)/prod(pool_size))/10);
binsize = (max(pooled_maps(:))-min(pooled_maps(:)))/nbins;
[counts,bins] = hist(pooled_maps(:),nbins);
[maxv,maxind] = max(counts);
cbin = bins(maxind);
pooled_maps = (pooled_maps>cbin+binsize/2 | pooled_maps<cbin-binsize/2).*pooled_maps;
fprintf('-thresh');


end