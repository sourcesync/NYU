%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% A simple helper function that gets the indices and pooled versions of the
% input maps (input as a cell array) for each layer of a hierarchical model.
% Think of the pooling for layer i happening to the input maps of layer i, that
% is y{i} first, then the deconvolution layer is above these pooled inputs.
%
% @file
% @author Matthew Zeiler
% @date Aug 26, 2010
%
% @pooltoolbox_file @copybrief pool_all_layers.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief pool_all_layers.m
%
% @param pool_types cell array of the type of pooling to be done ('Avg','AbsAvg','Max',or 'None')
% as a string
% @param maps cell array of the planes to be pooled (xdim x ydim x n3 [x n4])
% @param y the input maps at the first layer are treated like being pooled
% feature maps.
% @param pool_sizes cell array of the size of the pooling region [poolX x poolY] as a
% 2x1 matrix
% @param pooled_indices cell array of the indices of a previous pooling operation (optional).
% This currently only works for 'Max' pooling . Pass in [] if you don't want to
% use this but still want to set remaining parameters.
% @param COMP_THREADS tells the MEX versions of the pooling how many threads
% to split the computations over.
% @param layer pool all the layers up to the top most active one.
% 
% @retval pooled_maps the pooled maps (or maps if 'None' is the pooling type)
% @retval pooled_indices the indices returned by some pooling operations
% (such as the index of the max location for 'Max' pooling).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [pooled_maps,pooled_indices] = pool_all_layers(pool_types,maps,y,pool_sizes,pooled_indices,COMP_THREADS,layer)

pooled_maps = cell(1,layer+1);
if(nargin<7)
    layer = length(maps);
end
if(nargin<6)
    COMP_THREADS = 4;
end
if(nargin<5 || isempty(pooled_indices))
    pooled_indices = cell(1,layer+1);
end

% Pool each layer.
for i=1:layer+1
    if(i==1)
        [pooled_maps{i},pooled_indices{i}] = pool_wrapper(pool_types{i},y,pool_sizes{i},pooled_indices{i},COMP_THREADS);
    else
        % This update the PCA dictionary by default.
        [pooled_maps{i},pooled_indices{i}] = pool_wrapper(pool_types{i},maps{i-1},pool_sizes{i},pooled_indices{i},COMP_THREADS,1);
    end
end



end