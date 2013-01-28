%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Max pools the input maps within pool_size region. This uses the maximum
% absolute value within the pooling region.
%
% @file
% @author Matthew Zeiler
% @date Feb 11, 2010
%
% @pooltoolbox_file @copybrief max_pool3d.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief max_pool3d.m
%
% @param maps the planes to be normalized (xdim x ydim x n3 [x n4])
% @param pool_size a vector specifying the pooling sizein x and y  and kdimensions.
% @param pooled_indices indices from a previous pooling operation you want to use
% @param COMP_THREADS UNUSED
%
% @retval pooled_maps the pooled output planes
% @retval pooled_indices the indice within each pool region that was selected as max.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [pooled_maps,pooled_indices] = max_pool3d(maps,pool_size,pooled_indices,COMP_THREADS)



if(numel(pool_size)<3)
    error('pool_wrapper.m: For 3D pooling you must input 3D pooling sizes.')
end

if(prod(pool_size(1:3))>65536)
    error('Cannot have product of pooling region sizes greater than 256');
end

% Pool spatially and return new indices.
if(nargin<6 || isempty(pooled_indices))
    %%%%%%
    % Less memory and quicker implementation (no for loop)
    [pooled_maps,pooled_indices] = max_pool(maps,pool_size(1:2),[],COMP_THREADS);
    
    % Padd the pooled maps with zeros to be able to reshape.
    extramaps = mod(size(pooled_maps,3),pool_size(3));
    if(extramaps>0)
        addmaps = pool_size(3)-extramaps;
        pooled_maps = cat(3,pooled_maps,zeros([size(pooled_maps,1),size(pooled_maps,2),addmaps,size(pooled_maps,4)],'single'));
        % New
        pooled_indices = cat(3,pooled_indices,ones([size(pooled_indices,1),size(pooled_indices,2),addmaps,size(pooled_indices,4)],'uint16'));
    end
    
    % Instead of looping over groups, reshape and then do the max(abs());
    pooled_maps = reshape(pooled_maps,[size(pooled_maps,1) size(pooled_maps,2) pool_size(3) size(pooled_maps,3)/pool_size(3) size(pooled_maps,4)]);
    
    % Compute the maxes over groups in k dimension.
    maxes = max(pooled_maps,[],3);
    % This sets the k-indices.
    [pooled_maps,k_indices] = max(abs(pooled_maps),[],3);
    k_indices = k_indices-1;
    inds = (maxes==pooled_maps)-0.5;
    
    
    pooled_maps = pooled_maps.*sign(inds);
    pooled_maps = reshape(pooled_maps,[size(pooled_maps,1) size(pooled_maps,2) size(pooled_maps,4) size(pooled_maps,5)]);
    % They should be the same size.
    
    % The only difference is at this part where we want to store less indices.
    pooled_indices = reshape(pooled_indices,[size(pooled_indices,1) size(pooled_indices,2) pool_size(3) size(pooled_indices,3)/pool_size(3) size(pooled_indices,4)]);
    
    inds = repmat(k_indices,[1 1 pool_size(3) 1 1]);
    sel = reshape(0:pool_size(3)-1,[1 1 pool_size(3) 1 1]);
    sel = repmat(sel,[size(inds,1) size(inds,2) 1 size(inds,4) size(inds,5)]);
    
    % This should be 1 whever the pooled_indices should be taken from.
    inds = uint16(sel==inds);
    pooled_indices = pooled_indices.*inds;
    % Make the indices depend on the group, ie. 1...9 for k=1, 10..18 for k=2 if using 3x3 pooling.
    pooled_indices = pooled_indices+uint16(sel).*uint16(repmat(pool_size(1)*pool_size(2),size(sel)));
    % Zero out other indices.
    pooled_indices = pooled_indices.*inds;
    % Sum over the groups (since only one in each group will be nonzero).
    pooled_indices = uint16(reshape(sum(pooled_indices,3),size(pooled_maps)));
    
    %%%%%%%%%%
else % Use the pooling indices that were passed in.
    %%%%%
    % Less memory and quicker version.
    
    k_indices = uint16(floor(single(pooled_indices)/(pool_size(1)*pool_size(2))));
    k_indices = reshape(k_indices,[size(k_indices,1) size(k_indices,2) 1 size(k_indices,3) size(k_indices,4)]);
    k_indices = repmat(k_indices,[1 1 pool_size(3) 1 1]);
    
    % Now we have to keep only the spatial indices.
    spatial_indices = uint16(mod(single(pooled_indices),(pool_size(1)*pool_size(2))));
    
    spatial_indices = reshape(spatial_indices,[size(spatial_indices,1) size(spatial_indices,2) 1 size(spatial_indices,3) size(spatial_indices,4)]);
    spatial_indices = repmat(spatial_indices,[1 1 pool_size(3) 1 1]);
    spatial_indices = reshape(spatial_indices,[size(spatial_indices,1) size(spatial_indices,2) size(spatial_indices,3)*size(spatial_indices,4) size(spatial_indices,5)]);
    spatial_indices = spatial_indices(:,:,1:size(maps,3),:);
    
    % Pool spatially with the pooling indices passed in.
    pooled_maps = max_pool(maps,pool_size(1:2),spatial_indices,COMP_THREADS);
                    
    % Padd the pooled maps with zeros to be able to reshape.
    extramaps = mod(size(pooled_maps,3),pool_size(3));
    if(extramaps>0)
        addmaps = pool_size(3)-extramaps;
        pooled_maps = cat(3,pooled_maps,zeros([size(pooled_maps,1),size(pooled_maps,2),addmaps,size(pooled_maps,4)],'single'));
    end
    
    % Make the new maps (smaller in the kth dimension).
    %             new_maps = zeros(size(pooled_maps,1),size(pooled_maps,2),length(1:pool_size(3):size(pooled_maps,3)),size(pooled_maps,4),'single');
    pooled_maps = reshape(pooled_maps,[size(pooled_maps,1) size(pooled_maps,2) pool_size(3) size(pooled_maps,3)/pool_size(3) size(pooled_maps,4)]);
    
    % Like meshgrid, gives the 1:pool_size(3) in the 3rd dimension.
    ind = repmat(reshape(0:size(pooled_maps,3)-1,[1 1 size(pooled_maps,3) 1 1]),[size(pooled_maps,1) size(pooled_maps,2) 1 size(pooled_maps,4) size(pooled_maps,5)]);
    %             keyboard
    pooled_maps = pooled_maps.*(k_indices==ind);
    pooled_maps = sum(pooled_maps,3);
    pooled_maps = reshape(pooled_maps,[size(pooled_maps,1) size(pooled_maps,2) size(pooled_maps,4) size(pooled_maps,5)]);
    %%%%%%%%%%%%%%%%
    
end


end