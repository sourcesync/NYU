%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Undoes the max pooling by placing the max back into it's indexed location.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @pooltoolbox_file @copybrief reverse_max_pool.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief reverse_max_pool.m
%
% @param pooled_maps the planes to be normalized (xdim x ydim x n3 [x n4])
% @param pooled_indices the indices where the max came from during max_pool
% @param pool_size a vector specifying the pooling sizein x and y dimensions.
% @param unpooled_size this is used to specify the correct size of the unpooled
% region. If this is not passed in then xdim*pool_size(1) x ydim*pool_size(2)
% will be used.
% @param COMP_THREADS UNUSED
% @retval unpooled_maps the unpooled output planes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [unpooled_maps] = reverse_max_pool3d(pooled_maps,pooled_indices,pool_size,unpooled_size,COMP_THREADS)
if(numel(pool_size)<3)
    error('unpool_wrapper.m: For 3D pooling you must input 3D pooling sizes.')
end
if(prod(pool_size(1:3))>65535)
    error('Cannot have product of pooling region sizes greater than 256');
end
if(size(pooled_maps,3)==1 && size(pooled_indices,3)==1 && numel(unpooled_size==2))
    unpooled_size(3)=1;
end
if(numel(unpooled_size)==2)
    dbstack
    error('Need third dimension of unpooled size now');
end
%%%%%%%%
% Faster Version
% Get the k switches.
k_indices = uint16(floor(single(pooled_indices)/(pool_size(1)*pool_size(2))));
k_indices = reshape(k_indices,[size(k_indices,1) size(k_indices,2) 1 size(k_indices,3) size(k_indices,4)]);
k_indices = repmat(k_indices,[1 1 pool_size(3) 1 1]);

% Now we have to keep only the spatial indices.
pooled_indices = uint16(mod(single(pooled_indices),(pool_size(1)*pool_size(2))));
pooled_indices = reshape(pooled_indices,[size(pooled_indices,1) size(pooled_indices,2) 1 size(pooled_indices,3) size(pooled_indices,4)]);
pooled_indices = repmat(pooled_indices,[1 1 pool_size(3) 1 1]);
pooled_indices = reshape(pooled_indices,[size(pooled_indices,1) size(pooled_indices,2) size(pooled_indices,3)*size(pooled_indices,4) size(pooled_indices,5)]);
pooled_indices = pooled_indices(:,:,1:unpooled_size(3),:);


% Repmat to apply logical multiplication.
pooled_maps = reshape(pooled_maps,[size(pooled_maps,1) size(pooled_maps,2) 1 size(pooled_maps,3) size(pooled_maps,4)]);
pooled_maps = repmat(pooled_maps,[1 1 pool_size(3) 1 1]);

% Get indices over k dimension groups.
ind = repmat(reshape([0:pool_size(3)-1],[1 1 pool_size(3) 1 1]),[size(pooled_maps,1) size(pooled_maps,2) 1 size(pooled_maps,4) size(pooled_maps,5)]);

% Place zeros elsewhere.
pooled_maps = squeeze(pooled_maps.*(ind==k_indices));

% Reshape back to the proper size.
pooled_maps = reshape(pooled_maps,[size(pooled_maps,1) size(pooled_maps,2) size(pooled_maps,3)*size(pooled_maps,4) size(pooled_maps,5)]);


pooled_maps = pooled_maps(:,:,1:unpooled_size(3),:);

% Finally unpool the maps spatially.
unpooled_maps = reverse_max_pool(pooled_maps,pooled_indices,pool_size,unpooled_size,COMP_THREADS);

end