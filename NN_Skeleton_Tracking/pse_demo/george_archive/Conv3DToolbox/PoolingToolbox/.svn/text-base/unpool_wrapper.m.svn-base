%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% A wrapper that abstracts out any type of unpooling operation (to invert
% pooling of planes over certain regions into larger regions). Currently implemented
% poooling types are 'Avg', 'AbsAvg', 'Max', and 'None'.
%
% @file
% @author Matthew Zeiler
% @date Jul 23, 2010
%
% @pooltoolbox_file @copybrief unpool_wrapper.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief unpool_wrapper.m
%
% @param pool_type the type of pooling to be done ('Avg','AbsAvg','Max','AbsMax','Max3','Max3Abs',or 'None')
% as a string
% @param pooled_maps the planes that were previously pooled (xdim x ydim x n3 [x n4])
% @param pooled_indices the indices returned by some pooling operations
% (such as the index of the max location for 'Max' pooling).
% @param pool_size the size of the pooling region [poolX x poolY x poolK] as a
% 3x1 matrix
% @param unpooled_size this is used to specify the correct size of the unpooled
% region. If this is not passed in then xdim*pool_size(1) x ydim*pool_size(2)
% will be used by the unpooling functions.
% @param COMP_THREADS tells the MEX versions of the pooling how many threads
% to split the computations over.
% @param layer the layer of feature maps you are pooling (needed for PCA).
%
% @retval unpooled_maps the unpooled maps (or pooled_maps if 'None' is the pooling
% type)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [unpooled_maps] = unpool_wrapper(pool_type,pooled_maps,pooled_indices,pool_size,unpooled_size,COMP_THREADS)

if(nargin<6)
    fprintf('\n\nunpool_wrapper.m: Setting COMP_TRHEADS TO 4 because this was not passed in\n\n');
    COMP_THREADS = 4;
end
if(nargin<5)
    unpooled_size = [];
end


% Do PCA unpooling before regular unpooling.
if(strcmp(pool_type(end-2:end),'PCA'))
    
    if(numel(pool_size)<3)
        error('unpool_wrapper.m: For ...PCA unpooling you must input 3D pooling sizes where the 3rd size is the PCA over k.')
    end
    
    pool_type_start = pool_type(1:end-3);
    
    
    % Just need indices for the pooling part and the dictionaries for the rest.
    pca_info = pooled_indices;
    % Avoid duplicate indices.
    pca_info.inds = [];
    % Get the indices.
    pooled_indices = pooled_indices.inds;
    
    % Undo the PCA on the pooled maps.
    pooled_maps = reverse_pca_maps(pooled_maps,pca_info,COMP_THREADS);
    
    
    %     figure(4), sdf(pooled_maps);
elseif(length(pool_type)>3 && strcmp(pool_type(end-3:end),'Zero'))
    % set the pooled_maps to zero in cases where the input to that region was zero on the way up.
    pooled_maps(pooled_indices==0) = 0;
    pooled_indices = pooled_indices-1; % Should have a min of 0 since unisnged.
    pool_type_start = pool_type(1:end-4);
elseif(~isempty(regexp(pool_type,'CN','ONCE')))
    pool_type_start = regexprep(regexprep(pool_type,'CN2',''),'CN','');
    pooled_maps = bsxfun(@times,pooled_maps,pooled_indices.CNSTDs);
%         pooled_maps = bsxfun(@plus,pooled_maps,pooled_indices.CNMeans);
    
    pooled_indices = pooled_indices.inds;
else
    pool_type_start = pool_type;
end

if(~isfloat(pooled_maps) || isa(pooled_maps,'gsingle'))
    switch(pool_type_start)
        case 'Max'
%             pooled_maps = single(pooled_maps);
%             pooled_indices = uint16(single(pooled_indices));
%                         unpooled_maps = reverse_max_pool(pooled_maps,pooled_indices,pool_size,unpooled_size,COMP_THREADS);
% unpooled_maps = GPUsingle(unpooled_maps);

            unpooled_maps = cuRevMaxPool(pooled_maps,pooled_indices,pool_size,unpooled_size,COMP_THREADS);
        case 'Max3'
            unpooled_maps = cuRevMaxPool3d(pooled_maps,pooled_indices,pool_size,unpooled_size,COMP_THREADS);
        case 'Avg'
            unpooled_maps = cuRevAvgPool(pooled_maps,[],pool_size,unpooled_size,COMP_THREADS);
        case 'Avg3'
            unpooled_maps = cuRevAvgPool3d(pooled_maps,[],pool_size,unpooled_size,COMP_THREADS);            
        case 'None'
            unpooled_maps = pooled_maps;
        otherwise
            error('Unpooling type %s not implemented on GPU yet.',pool_type_start);
    end
else
    
 % Make sure the indices are uint16 (as long as they are not conv indices (linear so need to be single).
    if(isempty(regexp(pool_type,'Conv','ONCE')) && isempty(regexp(pool_type,'MaxAbs','ONCE')) && isempty(regexp(pool_type,'Max3Abs','ONCE')))
        pooled_indices = uint16(pooled_indices);
    end
    
    
    
    switch(pool_type_start)
        case 'Max'      % Place max back in location
            unpooled_maps = reverse_max_pool(pooled_maps,pooled_indices,pool_size,unpooled_size,COMP_THREADS);
            % Place back in the middle.
            %         unpooled_maps = reverse_max_pool(pooled_maps,floor(prod(pool_size)/2).*ones(size(pooled_maps),'single'),pool_size,unpooled_size,COMP_THREADS);
            %                 unpooled_maps = reverse_avg_pool(pooled_maps./prod(pool_size),pooled_indices,pool_size,unpooled_size,COMP_THREADS);
            %         unpooled_maps = reverse_avg_pool(pooled_maps,pooled_indices,pool_size,unpooled_size,COMP_THREADS);
        case 'Max3'
            
            % Correct Max3 unpooling
            unpooled_maps = reverse_max_pool3d(pooled_maps,pooled_indices,pool_size,unpooled_size,COMP_THREADS);
        case 'ConvMax'
            %                pooled_maps = zeros([ceil((size(maps,1)+pool_size(1)-1)/pool_size(1)) ceil((size(maps,2)+pool_size(2)-1)/pool_size(4)) size(maps,3) size(maps,4)]);
            %                 pooled_maps = zeros(length(pool_size(1):pool_size(4):(size(maps,1)+pool_size(1)-1)),...
            %                     length(pool_size(1):pool_size(4):(size(maps,2)+pool_size(2)-1)),size(maps,3),size(maps,4),'single');
            %                 whos
            % pooled_indices
%             max(pooled_indices(:))
            pooled_indices = bsxfun(@plus,pooled_indices,...
                reshape([0:(unpooled_size(3)-1)]*prod(unpooled_size(1:2)),[1 1 unpooled_size(3) 1]));
            % Make it indices into the entire unpooled map for this batch size.
            pooled_indices = bsxfun(@plus,pooled_indices,...
                reshape([0:(size(pooled_maps,4)-1)]*prod(unpooled_size(1:3)),[1 1 1 size(pooled_maps,4)]));
            % pooled_indices
%             max(pooled_indices(:))
%             prod(unpooled_size(1:3))
%             unpooled_size
            unpooled_maps = zeros([unpooled_size(1:3) size(pooled_maps,4)],'single');
            [sorted,sinds] = sort(abs(pooled_maps(:))); % Sort them so the max value is inserted.
            unpooled_maps(pooled_indices(sinds(:))) = pooled_maps(sinds(:));
            %                 keyboard
            %                pooled_maps = zeros(size(maps),'single');
            %                pooled_maps(pooled_indices) = maps(pooled_indices);
            %                 pooled_maps(:) = maps(pooled_indices);
            %                 size(pooled_maps)
            %                pooled_maps = pooled_maps(pool_size(1):pool_size(4):end,pool_size(2):pool_size(4):end,:,:);
            
        case 'ConvMax3'
        case 'Max3NoK'
            % For removing the k-indices do this.
            pooled_indices = mod(pooled_indices,pool_size(3));
            unpooled_maps = reverse_max_pool3d(pooled_maps,pooled_indices,pool_size,unpooled_size,COMP_THREADS);
        case 'Max3OnlyK'
            % For keeping the k-indices (but 0 spatially).
            % Subtract off the spatial index.
            pooled_indices = pooled_indices - mod(pooled_indices,prod(pool_size(1:2)));
            unpooled_maps = reverse_max_pool3d(pooled_maps,pooled_indices,pool_size,unpooled_size,COMP_THREADS);
        case 'Max3NoInds'
            pooled_indices = zeros(size(pooled_indices),'uint16');
            unpooled_maps = reverse_max_pool3d(pooled_maps,pooled_indices,pool_size,unpooled_size,COMP_THREADS);
        case 'Avg3'
            % Unpool as Avg3 over all k indices .
            unpooled_maps = reverse_avg_pool(pooled_maps,pooled_indices,pool_size,unpooled_size,COMP_THREADS);
            unpooled_maps = reshape(unpooled_maps,[size(unpooled_maps,1) size(unpooled_maps,2) 1 size(unpooled_maps,3) size(unpooled_maps,4)]);
            unpooled_maps = repmat(unpooled_maps,[1 1 pool_size(3) 1 1])./prod(pool_size(1:3));
            unpooled_maps = reshape(unpooled_maps,[size(unpooled_maps,1) size(unpooled_maps,2) size(unpooled_maps,3)*size(unpooled_maps,4) size(unpooled_maps,5)]);            
            unpooled_maps = unpooled_maps(:,:,1:unpooled_size(3),:,:);
        case 'Avg3OnlyK'
            % Unpool as Avg with no K indices.
            % Place in right k map.
            unpooled_maps = reverse_max_pool3d(pooled_maps,pooled_indices,pool_size,unpooled_size,COMP_THREADS);
            % Then average pool and unpool again to spread out.
            [pooled_maps] = avg_pool(unpooled_maps,pool_size(1:2),pooled_indices,COMP_THREADS);
            unpooled_maps = reverse_avg_pool(pooled_maps,pooled_indices,pool_size,unpooled_size,COMP_THREADS);
        case 'MaxAbs'   % Replace the sign of the max value and its location.
            pooled_maps = pooled_maps.*sign(pooled_indices);
            pooled_indices = uint16(abs(pooled_indices)-1);
            unpooled_maps = reverse_max_pool(pooled_maps,pooled_indices,pool_size,unpooled_size,COMP_THREADS);
        case 'Max3Abs'   % Replace the sign of the max value and its location.
            pooled_maps = pooled_maps.*sign(pooled_indices);
            pooled_indices = uint16(abs(pooled_indices)-1);
            unpooled_maps = reverse_max_pool3d(pooled_maps,pooled_indices,pool_size,unpooled_size,COMP_THREADS);            
        case 'AbsMax'      % Place max back in location
            unpooled_maps = reverse_max_pool(pooled_maps,pooled_indices,pool_size,unpooled_size,COMP_THREADS);
        case 'AbsMax3'      % Place max back in location
            unpooled_maps = reverse_max_pool3d(pooled_maps,pooled_indices,pool_size,unpooled_size,COMP_THREADS);            
        case 'ProbMax'  % Place max back in sampled most probably location
            unpooled_maps = reverse_prob_max_pool(pooled_maps,pooled_indices,pool_size,unpooled_size);
        case 'Avg'      % Place average everywhere in region.
            unpooled_maps = reverse_avg_pool(pooled_maps,pooled_indices,pool_size,unpooled_size,COMP_THREADS);
        case 'AbsAvg'   % Place average everywhere in region (same as above).
            unpooled_maps = reverse_avg_pool(pooled_maps,pooled_indices,pool_size,unpooled_size,COMP_THREADS);
        case 'PNAvg'
            error('PNAvg unpooling not implemented yet')
        case 'SplitKMax'
            % Unpool in 2D first.
            kindices = floor(single(pooled_indices)/prod(pool_size(1:2)));
            sindices = uint16(single(pooled_indices) - kindices*prod(pool_size(1:2)));
            
            % The spatial indices to use have to be repmated by pool_size(3) to make sure we have the same number of maps.
            % Though on the first iteration, they were created incorrectly as number z maps (not 3d pooled maps).
            if(size(sindices,3)~=size(pooled_maps))
                ss = [size(sindices) 1 1 1 1 1 1];
                sindices = reshape(sindices,[ss(1) ss(2) 1 ss(3) ss(4)]);
                sindices = repmat(sindices,[1 1 pool_size(3) 1 1]);
                sindices = reshape(sindices,[ss(1) ss(2) ss(3)*pool_size(3) ss(4)]);
            end
            
            % Have to convert indices to 2D first though as they are 3D indices.
            unpooled_maps = reverse_max_pool(pooled_maps,sindices,pool_size,unpooled_size,COMP_THREADS);
            
            % Pool in 3D to reduce the number of maps (using the bottom up 3D indices).
            %         [pooled_maps] = max_pool3d(unpooled_maps,pool_size,pooled_indices,COMP_THREADS);
            
            % Unpool in 3D to get back to the unpooled size again (using the indices from the bottom up).
            %         unpooled_maps = reverse_max_pool3d(pooled_maps,pooled_indices,pool_size,unpooled_size,COMP_THREADS);
        case 'NormedMax'
            % Same as 'Max', just a 2D unpooling using already scaled pooled maps.
            unpooled_maps = reverse_max_pool(pooled_maps,pooled_indices,pool_size,unpooled_size,COMP_THREADS);
        case 'SplitMax' % This splits the max location into different maps (each of the pooled size)
            % This is only 2D because there is no different once split if you do 3D or not.
            
            %         pooled_maps = reverse_avg_pool(pooled_maps,pooled_indices,pool_size,[ceil(unpooled_size(1)/pool_size(1)) ceil(unpooled_size(2)/pool_size(2))],COMP_THREADS);
            
            % Number of repmating to do.
            num_new = prod(pool_size(1:2));
            %         sp = size(pooled_maps);
            
            %         %%%%
            %         % Make a smaller version with the original number of pooled maps.
            %         new_maps = zeros([size(pooled_maps,1) size(pooled_maps,2) size(pooled_maps,3)/num_new size(pooled_maps,4)],'single');
            %         % Recompile the maps as if we just did 2D Max pooling using in indices (shitty but correct way).
            %         for i=1:num_new
            %             new_maps = new_maps + pooled_maps(:,:,i:num_new:end,:).*(pooled_indices==i);
            %         end
            %         pooled_maps = new_maps;
            %         %%%%
            
            %%%%%%%%%%%%%%%%%
            % Repmate but keep all copies of same map in order.
            pooled_maps  = reshape(pooled_maps,[size(pooled_maps,1) size(pooled_maps,2) num_new size(pooled_maps,3)/num_new size(pooled_maps,4)]);
            % Get the new pooling indices based on the whatever map has the strongest activation.
            maxes = max(pooled_maps,[],3);
            [pooled_maps,pooled_indices] = max(abs(pooled_maps),[],3);
            inds = (maxes==pooled_maps)-0.5;
            pooled_maps = pooled_maps.*sign(inds);
            
            pooled_maps  = single(reshape(pooled_maps,[size(pooled_maps,1) size(pooled_maps,2) size(pooled_maps,4) size(pooled_maps,5)]));
            pooled_indices  = single(reshape(pooled_indices,size(pooled_maps)));
            
            %%%%%%%%%%%%
            % Fix the boundaries of pooling indices because max(,[],3) above doesn't know about them.
            % Get how big the last pooling regions are for the boundary in teh bottom and right sides.
            extrax = mod(unpooled_size(1),pool_size(1));
            extray = mod(unpooled_size(2),pool_size(2));
            % These mod's should be how far into the last pooling region you should place the max.
            if(extrax>0)
                pooled_indices(end,1:end,:,:) = single(extrax);
            end
            if(extray>0)
                pooled_indices(1:end,end,:,:) = single(extray);
            end
            %%%%%%%%%%%%
            
            
            %
            %         if(min(pooled_indices(:))==0)
            %             keyboard
            %         end
            %    if(max(pooled_indices(:))~=num_new && size(pooled_maps,3)==15)
            %         keyboard
            % end
            %
            % %         pooled_indices = round(pooled_indices);
            %
            %         if(size(pooled_maps,3)==15)
            % %         figure(1), hist(pooled_indices(:));
            %         pooled_indices
            %         dbstack
            %         pool_size
            %         size(pooled_maps)
            %         size(pooled_indices)
            %         whos
            %         unpooled_size
            %         end
            
            % Unpool as before.
            unpooled_maps = reverse_max_pool(pooled_maps,pooled_indices,pool_size(1:2),unpooled_size(1:2),COMP_THREADS);
            
            %         if(size(pooled_maps,3)==15)
            %             unpsize = size(unpooled_maps)
            %         end
        case 'Maxes3'
            unpooled_maps = reverse_max_pool(pooled_maps,pooled_indices,pool_size,unpooled_size,COMP_THREADS);
        case 'OldMax3'
            
            pooled_indices = permute(pooled_indices,[1 2 4 3]);
            pooled_maps = permute(pooled_maps,[1 2 4 3]);
            
            ind = [1:size(pooled_maps,1)*size(pooled_maps,2)*size(pooled_maps,3)]'  + vect_array(size(pooled_maps,1)*size(pooled_maps,2)*size(pooled_maps,3).*(pooled_indices(:,:,:,end)-1));
            temp_maps = zeros(size(pooled_maps),'single');
            temp_maps(ind') = pooled_maps(ind');
            pooled_maps = temp_maps;
            
            pooled_indices = permute(pooled_indices,[1 2 4 3]);
            pooled_maps = permute(pooled_maps,[1 2 4 3]);
            
            unpooled_maps = reverse_max_pool(pooled_maps,pooled_indices(:,:,1:end-1,:),pool_size,unpooled_size,COMP_THREADS);
            
            
        case 'Old2Max3'
            if(numel(pool_size)<3)
                error('unpool_wrapper.m: For 3D pooling you must input 3D pooling sizes.')
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%
            % if you want to play with reconstructing using avg or middle indices (this part removes k indices)
            %                fprintf('Must remove pool_indices=1 setting here\n');
            %                        pooled_indices = ones(size(pooled_indices),'single');
            %         fprintf('mid');
            %         middle = floor(pool_size(1)/2);
            %         middle = pool_size(1)*middle + middle + 1;
            %         pooled_indices(:,:,1:(size(pooled_indices,3)-size(pooled_maps,3)),:) = single(middle);
            %         % make sure the boundaries doesn't go over so set indice to 1.
            %         % Get how big the last pooling regions are for the boundary in teh bottom and right sides.
            %         extrax = mod(unpooled_size(1),pool_size(1));
            %         extray = mod(unpooled_size(2),pool_size(2));
            %         % These mod's should be how far into the last pooling region you should place the max.
            %         if(extrax>0)
            %             pooled_indices(end,1:end,1:(size(pooled_indices,3)-size(pooled_maps,3)),:) = single(extrax);
            %         end
            %         if(extray>0)
            %             pooled_indices(1:end,end,1:(size(pooled_indices,3)-size(pooled_maps,3)),:) = single(extray);
            %         end
            %         %%%%%%%%%%%%%%%%%%%%%%%%%
            %         %%%%%%%%%%%%%%%%%%%%%%%%%
            
            
            
            
            %         % Get the number of groups.
            %         numgroups = size(pooled_maps,3);
            %         % Get the original number of maps, k.
            %         nummaps = size(pooled_indices,3)-numgroups;
            %
            %         pooled_indices = permute(pooled_indices,[1 2 4 3]);
            %         pooled_maps = permute(pooled_maps,[1 2 4 3]);
            %
            %         % Make the new maps the original number of feautre maps, k.
            %         new_maps = zeros(size(pooled_maps,1),size(pooled_maps,2),size(pooled_maps,3),nummaps,'single');
            %
            %         % This is the number of elements in each plane over the dataset.
            %         sizek = size(pooled_maps,1)*size(pooled_maps,2)*size(pooled_maps,3);
            %
            %         % Loop over each map.
            %         for group=1:numgroups
            %             ind = [1:sizek]'  +...   % The offset of each pixel for the current plane for each image in the set.
            %                 vect_array(sizek.*(pooled_indices(:,:,:,nummaps+group)-1)) + ... % The pooling indices offset.
            %                 sizek*(group-1)*pool_size(3); % The groups offset.
            %
            %             % If no zero maps are used, then use this index into the pooled maps.
            %             ind2 = [1:sizek]'  +...   % The offset of each pixel for the current plane for each image in the set.
            %                 sizek*(group-1); % The groups offset.
            %
            %             new_maps(ind') = pooled_maps(ind2');
            %         end
            %
            %         pooled_maps = new_maps;
            %
            %         pooled_indices = permute(pooled_indices,[1 2 4 3]);
            %         pooled_maps = permute(pooled_maps,[1 2 4 3]);
            %
            %         % Throw away the group indices for the unpooling operation.
            %         pooled_indices = pooled_indices(:,:,1:size(pooled_maps,3),:);
            %
            %
            %         % Finally unpool the maps spatially.
            %         unpooled_maps = reverse_max_pool(pooled_maps,pooled_indices,pool_size,unpooled_size,COMP_THREADS);
            
            
            
            
            %%%%%%%%
            % Faster Version
            nummaps = size(pooled_maps,3);
            
            
            
            % Get the k switches.
            k_indices = pooled_indices(:,:,end-nummaps+1:end,:);
            k_indices = reshape(k_indices,[size(k_indices,1) size(k_indices,2) 1 size(k_indices,3) size(k_indices,4)]);
            k_indices = repmat(k_indices,[1 1 pool_size(3) 1 1]);
            
            % Repmat to apply logical multiplication.
            pooled_maps = reshape(pooled_maps,[size(pooled_maps,1) size(pooled_maps,2) 1 size(pooled_maps,3) size(pooled_maps,4)]);
            pooled_maps = repmat(pooled_maps,[1 1 pool_size(3) 1 1]);
            
            % Get indices over k dimension groups.
            ind = repmat(reshape([0:pool_size(3)-1],[1 1 pool_size(3) 1 1]),[size(pooled_maps,1) size(pooled_maps,2) 1 size(pooled_maps,4) size(pooled_maps,5)]);
            
            % Place zeros elsewhere.
            pooled_maps = squeeze(pooled_maps.*(ind==k_indices));
            
            % Reshape back to the proper size.
            pooled_maps = reshape(pooled_maps,[size(pooled_maps,1) size(pooled_maps,2) size(pooled_maps,3)*size(pooled_maps,4) size(pooled_maps,5)]);
            
            % Keep only the spatial indices.
            pooled_indices = pooled_indices(:,:,1:end-nummaps,:);
            pooled_maps = pooled_maps(:,:,1:size(pooled_indices,3),:);
            
            % Finally unpool the maps spatially.
            unpooled_maps = reverse_max_pool(pooled_maps,pooled_indices,pool_size,unpooled_size,COMP_THREADS);
            
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%
            % If you want to reconstructiusing average or middle indice. (comment out above line).
            %                         fprintf('AvgU\n');
            %                         unpooled_maps = reverse_avg_pool(pooled_maps,pooled_indices,pool_size(1:2),unpooled_size,COMP_THREADS);
            %                 fprintf('Unpooling with middle indice instead of Max3\n');
            %                 unpooled_maps = reverse_max_pool(single(pooled_maps),single(ones(size(pooled_indices),'single')),pool_size,unpooled_size,COMP_THREADS);
            %%%%%%%%%%%%%%%%%%%%
            
            
        case 'OldMax3Abs'
            if(numel(pool_size)<3)
                error('unpool_wrapper.m: For 3D pooling you must input 3D pooling sizes.')
            end
            
            %                 pooled_maps = pooled_maps.*sign(pooled_indices);
            %         pooled_indices = abs(pooled_indices);
            %         unpooled_maps = reverse_max_pool(pooled_maps,pooled_indices,pool_size,unpooled_size,COMP_THREADS);
            
            % Get the number of groups.
            numgroups = size(pooled_maps,3);
            % Get the original number of maps, k.
            nummaps = size(pooled_indices,3)-numgroups;
            
            pooled_indices = permute(pooled_indices,[1 2 4 3]);
            pooled_maps = permute(pooled_maps,[1 2 4 3]);
            
            % Make the new maps the original number of feautre maps, k.
            new_maps = zeros(size(pooled_maps,1),size(pooled_maps,2),size(pooled_maps,3),nummaps,'single');
            
            % This is the number of elements in each plane over the dataset.
            sizek = size(pooled_maps,1)*size(pooled_maps,2)*size(pooled_maps,3);
            
            % Loop over each map.
            for group=1:numgroups
                ind = [1:sizek]'  +...   % The offset of each pixel for the current plane for each image in the set.
                    vect_array(sizek.*(abs(pooled_indices(:,:,:,nummaps+group))-1)) + ... % The pooling indices offset.
                    sizek*(group-1)*pool_size(3); % The groups offset.
                % If no zero maps are used, then use this index into the pooled maps.
                ind2 = [1:sizek]'  +...   % The offset of each pixel for the current plane for each image in the set.
                    sizek*(group-1); % The groups offset.
                
                new_maps(ind) = pooled_maps(ind2).*vect_array(sign(pooled_indices(:,:,:,nummaps+group)));
            end
            
            pooled_maps = new_maps;
            
            pooled_indices = permute(pooled_indices,[1 2 4 3]);
            pooled_maps = permute(pooled_maps,[1 2 4 3]);
            
            % Throw away the group indices for the unpooling operation.
            pooled_indices = pooled_indices(:,:,1:size(pooled_maps,3),:);
            
            % Finally unpool the maps.
            unpooled_maps = reverse_max_pool(pooled_maps,abs(pooled_indices),pool_size,unpooled_size,COMP_THREADS);
            
            
            
            
        case 'Avg3'
            if(numel(pool_size)<3)
                error('unpool_wrapper.m: For Avg 3D pooling you must input 3D pooling sizes.')
            end
            
            
            %%%%%%%%
            % Faster Version
            nummaps = size(pooled_maps,3);
            
            try
                unp3 = unpooled_size(3);
            catch
                unp3 = nummaps;
            end
            
            
            % Repmat to undo the averaging.
            pooled_maps = reshape(pooled_maps,[size(pooled_maps,1) size(pooled_maps,2) 1 size(pooled_maps,3) size(pooled_maps,4)]);
            pooled_maps = repmat(pooled_maps,[1 1 pool_size(3) 1 1]);
            pooled_maps = reshape(pooled_maps,[size(pooled_maps,1) size(pooled_maps,2) size(pooled_maps,3)*size(pooled_maps,4) size(pooled_maps,5)]);
            pooled_maps = pooled_maps(:,:,1:unp3,:);
            
            
            % Finally unpool the maps spatially.
            unpooled_maps = reverse_avg_pool(pooled_maps,pooled_indices,pool_size,unpooled_size,COMP_THREADS);
        case 'Dess3'
            unpooled_maps = zeros([unpooled_size size(pooled_maps,4)],'single');
            unpooled_maps(1:pool_size(1):end,1:pool_size(2):end,1:pool_size(3):end,:) = pooled_maps;
        case 'None'     % Do nothing
            unpooled_maps = pooled_maps;
    end
    
    
end



end