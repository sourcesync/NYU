%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% A simple helper function that uses the indices and pooled versions of the
% input maps (input as a cell array) to unpool each layer of a hierarchical model.
% Think of the pooling for layer i happening to the input maps of layer i, that
% is y{i} first, then the deconvolution layer is above these pooled inputs.
%
% @file
% @author Matthew Zeiler
% @date Aug 26, 2010
%
% @pooltoolbox_file @copybrief unpool_all_layers.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief unpool_all_layers.m
%
% @param pool_types cell array of the type of pooling to be done ('Avg','AbsAvg','Max',or 'None')
% as a string
% @param maps cell array of the planes to be pooled (xdim x ydim x n3 [x n4])
% @param pooled_indices cell array of the indices of a previous pooling operation (optional).
% This currently only works for 'Max' pooling . Pass in [] if you don't want to
% use this but still want to set remaining parameters.
% @param pool_sizes cell array of the size of the pooling region [poolX x poolY] as a
% 2x1 matrix
% @param COMP_THREADS tells the MEX versions of the pooling how many threads
% to split the computations over.
% @param layer pool all the layers up to the top most active one.
%
% @retval pooled_maps the pooled maps (or maps if 'None' is the pooling type)
% @param y the input maps at the first layer are treated like being pooled
% feature maps.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [unpooled_maps,y] = unpool_all_layers(pool_types,maps,pooled_indices,pool_sizes,unpooled_sizes,COMP_THREADS,layer)

unpooled_maps = cell(1,layer);
if(nargin<7)
    layer = length(maps)-1;
end
if(nargin<6)
    COMP_THREADS = 4;
end
if(nargin<5)
    unpooled_sizes = cell(1,length(maps));
end

% Unpool each layer
for i=1:layer+1
    if(i==1)
        y = unpool_wrapper(pool_types{i},maps{1},pooled_indices{i},pool_sizes{i},unpooled_sizes{i},COMP_THREADS);
    else
        if(~isempty(maps{i}))
            unpooled_maps{i-1} = unpool_wrapper(pool_types{i},maps{i},pooled_indices{i},pool_sizes{i},unpooled_sizes{i},COMP_THREADS);
        end
    end
end



end