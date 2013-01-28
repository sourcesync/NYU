%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% A wrapper that abstracts out any type of pooling operation (to reduce
% planes over certain regions into smaller regions). Currently implemented
% poooling types are 'Avg', 'AbsAvg', 'Max', and 'None'.
%
% @file
% @author Matthew Zeiler
% @date Jul 23, 2010
%
% @pooltoolbox_file @copybrief pool_wrapper.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief pool_wrapper.m
%
% @param pool_type the type of pooling to be done ('Avg','AbsAvg','Max','AbsMax','Max3','Max3Abs',or 'None')
% as a string
% @param maps the planes to be pooled (xdim x ydim x n3 [x n4])
% @param pool_size the size of the pooling region [poolX x poolY x poolK] as a
% 3x1 matrix
% @param pooled_indices the indices of a previous pooling operation (optional).
% This currently only works for 'Max' pooling . Pass in [] if you don't want to
% use this but still want to set remaining parameters.
% @param COMP_THREADS tells the MEX versions of the pooling how many threads
% to split the computations over.
% @param layer the layer of feature maps you are pooling (needed for PCA).
% @param UPDATE_PCA a flag to tell if you want ot update the PCA dictionary. 1 means
% update the covariance only, 2 means update the PCA dictionary as well.
%
% @retval pooled_maps the pooled maps (or maps if 'None' is the pooling type). The number of pooled_maps may be less if using 3D pooling. The size in dimension 3 will reflect this here.
% @retval pooled_indices the indices returned by some pooling operations
% (such as the index of the max location for 'Max' pooling). The number of pooled_maps may be less if using 3D pooling, but the size of pooled_indices in dimension 3 is the size(maps,3) + size(pooled_maps,3). The first dimensions are skipped by pool_size(3) during unpooling. The last size(pooled_maps,3) planes are the k indices.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [pooled_maps,pooled_indices] = pool_wrapper(pool_type,maps,pool_size,pooled_indices,COMP_THREADS,UPDATE_PCA)
if(nargin<6)
    UPDATE_PCA = 0;
end

if(nargin<5)
    fprintf('\npool_wrapper.m: Setting COMP_TRHEADS TO 4 because this was not passed in\n\n');
    COMP_THREADS=4;
end




if(strcmp(pool_type(end-2:end),'PCA'))
    pool_type_start = pool_type(1:end-3);
    % Just need indices for the pooling part and the dictionaries for the rest.
    pca_info = pooled_indices;
    % Avoid duplicate indices.
    pca_info.inds = [];
    % Get the indices if the input was not empty (if it was emtpy then just pool as normal).
    % If they were empty then we don't use the indices, just spatial pool, then pca (but don't update the pca).
    %     if(isempty(pooled_indices))
    %        pooled_indices = [];
    %        pca_info.UPDATE_PCA = 0;
    %     else
    pooled_indices = pooled_indices.inds;
    % Whether you want to update the PCA dictionary or not.
    pca_info.UPDATE_PCA = UPDATE_PCA;
    %     end
elseif(length(pool_type)>3 && strcmp(pool_type(end-3:end),'Zero'))
    pool_type_start = pool_type(1:end-4);
    
    if(~isempty(pooled_indices)) % If the pooled indices are passed in, have to remove the offset.
        % There are many ways we could treat this case (this is the forward propagation using the indices).
        % If the whole region was zero (so indice==0) then we could set the output to 0 automatically (cancel gradient in that regino).
        % OR we could just undo the indice==0 setting and pool as normal just taking the max gradient in that region (probably correct otherwise gradients of a region would never work after shrunk once.).
        % The indices in this second case shouldn't be used when returned anyways so we should be okay.
        % Second case here (let full forward_prop through):
        pooled_indices = pooled_indices-1; % -1 should go to 0 since unsigned.
    
%                 % First case (suppress forward_prop as well):
%                 USE_ZEROS = 1;
%                 zero_indices = pooled_indices; %Keep track of which were zero so we can set pooled_maps to zero at these locations after pooling
%                 % ie. cancel gradient after pooling.
%                 pooled_indices = pooled_indices-1;
    else
        USE_ZEROS=0;
    end
elseif(~isempty(regexp(pool_type,'CN','ONCE')))
    if(isempty(pooled_indices.inds) || nargin<4)
        INDICES_PROVIDED = 0;
    else
        INDICES_PROVIDED = 1;
    end
    CNMeans = pooled_indices.CNMeans;
    CNSTDs = pooled_indices.CNSTDs;
    pooled_indices = pooled_indices.inds;
    pool_type_start = regexprep(regexprep(pool_type,'CN2',''),'CN','');
else
    pool_type_start = pool_type;
end



% If they are gpu variables, only a select number of pooling is implemented.
if(~isfloat(maps) || isa(maps,'gsingle'))
    switch(pool_type_start)
        case 'Max'
            if(nargin<4 || isempty(pooled_indices))
                [pooled_maps,pooled_indices] = cuMaxPool(maps,pool_size);
            else
                [pooled_maps,pooled_indices] = cuMaxPool(maps,pool_size,pooled_indices);
            end
            
%             maps = single(maps);
%             if(nargin<4 || isempty(pooled_indices))
%                 [pooled_maps,pooled_indices] = max_pool(maps,pool_size,[],COMP_THREADS);
%             else
%                 pooled_indices = uint16(single(pooled_indices));
%             
%                 [pooled_maps,pooled_indices] = max_pool(maps,pool_size,pooled_indices,COMP_THREADS);
%             end
%                         pooled_maps = GPUsingle(pooled_maps);
%                         pooled_indices = GPUsingle(single(pooled_indices));
        case 'Max3'
            if(nargin<4 || isempty(pooled_indices))
                [pooled_maps,pooled_indices] = cuMaxPool3d(maps,pool_size);
            else
                [pooled_maps,pooled_indices] = cuMaxPool3d(maps,pool_size,pooled_indices);
            end
        case 'Avg'
            [pooled_maps] = cuAvgPool(maps,pool_size);
            pooled_indices = [];
        case 'Avg3'            
            [pooled_maps] = cuAvgPool3d(maps,pool_size);            
            pooled_indices = [];
        case 'None'
            pooled_maps = maps;
            pooled_indices = [];
        otherwise
            error('Pooling type %s Not Yet Implemented on GPU.',pool_type_start);
    end
else
    
    % Make sure the indices are uint16 (as long as they are not conv indices (linear so need to be single).
    if(isempty(regexp(pool_type,'Conv','ONCE')) && isempty(regexp(pool_type,'MaxAbs','ONCE')) && isempty(regexp(pool_type,'Max3Abs','ONCE')))
        pooled_indices = uint16(pooled_indices);
    end
    
    
    
    switch(pool_type_start)
        case 'Max'      % 2D max pooling Take the max (abs) value in region
            if(nargin<4 || isempty(pooled_indices))
                [pooled_maps,pooled_indices] = max_pool(maps,pool_size,[],COMP_THREADS);
            else
                [pooled_maps,pooled_indices] = max_pool(maps,pool_size,pooled_indices,COMP_THREADS);
            end
        case 'Max3'     % 3D max pooling
            if(nargin<4 || isempty(pooled_indices))
                %                         fprintf('Ignoring Indices Always\n');
                [pooled_maps,pooled_indices] = max_pool3d(maps,pool_size,[],COMP_THREADS);
            else
                %             fprintf('Ignoring input switches in Max!!!!!!\n\n\n\n\n\n\n\n\n\n');
                [pooled_maps,pooled_indices] = max_pool3d(maps,pool_size,pooled_indices,COMP_THREADS);
            end
        case 'ConvMax'
            if(nargin<4 || isempty(pooled_indices))
                % Window size is pool_size(4)
                newsize = [pool_size(4) pool_size(5) 1 1 1 1 ];
                [minim,maxim,minind,maxind] = minmaxfilt(maps,newsize,'both','full');
                select = maxim>=abs(minim); % logical indexing to choose maxes or mins.
                pooled_maps = maxim.*(select) + minim.*(~select);
                pooled_indices = single(maxind.*(select) + minind.*(~select));
                pooled_maps = pooled_maps(pool_size(4):pool_size(1):end,pool_size(5):pool_size(2):end,:,:);
                pooled_indices = pooled_indices(pool_size(4):pool_size(2):end,pool_size(5):pool_size(2):end,:,:);
                
                % Make sure the indices are independent of the number of images in teh batch (ie. they will be linear indices per image).
                pooled_indices = bsxfun(@minus,pooled_indices,...
                    reshape([0:(size(maps,4)-1)]*size(maps,1)*size(maps,2)*size(maps,3),[1 1 1 size(maps,4)]));
                % Subtract off each plane.
                pooled_indices = bsxfun(@minus,pooled_indices,...
                    reshape([0:(size(maps,3)-1)]*size(maps,1)*size(maps,2),[1 1 size(maps,3) 1]));
                % So by this point the pooled_indices only depend on the planar sizes (are linear inds into unpooled planes).
                
            else
                pooled_maps = zeros(length(pool_size(4):pool_size(1):(size(maps,1)+pool_size(4)-1)),...
                    length(pool_size(5):pool_size(1):(size(maps,2)+pool_size(5)-1)),size(maps,3),size(maps,4),'single');
                
                % Convert eh indices to be dependent on the batch size again.
                % Add back per plane.
                pooled_indices = bsxfun(@plus,pooled_indices,...
                    reshape([0:(size(maps,3)-1)]*size(maps,1)*size(maps,2),[1 1 size(maps,3) 1]));
                % Add bck per image.
                pooled_indices = bsxfun(@plus,single(pooled_indices),...
                    reshape([0:(size(maps,4)-1)]*size(maps,1)*size(maps,2)*size(maps,3),[1 1 1 size(maps,4)]));
                
                pooled_maps(:) = maps(pooled_indices);
            end
        case 'ConvMax3'
        case 'AbsMax'      % Take the max (abs) value in region
            if(nargin<4 || isempty(pooled_indices))
                [pooled_maps,pooled_indices] = max_pool(abs(maps),pool_size,[],COMP_THREADS);
            else
                [pooled_maps,pooled_indices] = max_pool(abs(maps),pool_size,pooled_indices,COMP_THREADS);
            end
                    case 'AbsMax3'      % Take the max (abs) value in region
            if(nargin<4 || isempty(pooled_indices))
                [pooled_maps,pooled_indices] = max_pool3d(abs(maps),pool_size,[],COMP_THREADS);
            else
                [pooled_maps,pooled_indices] = max_pool3d(abs(maps),pool_size,pooled_indices,COMP_THREADS);
            end
        case 'MaxAbs'  % Max pooling with rectification afterwards
            if(nargin<4 || isempty(pooled_indices))
                [pooled_maps,pooled_indices] = max_pool(maps,pool_size,[],COMP_THREADS);
                pooled_indices = single(pooled_indices+1).*sign(pooled_maps);
                pooled_maps = abs(pooled_maps);
            else
                pooled_maps = max_pool(abs(maps),pool_size,uint16(abs(pooled_indices)-1),COMP_THREADS);
            end
        case 'Max3Abs'  % Max pooling with rectification afterwards
            if(nargin<4 || isempty(pooled_indices))
                [pooled_maps,pooled_indices] = max_pool3d(maps,pool_size,[],COMP_THREADS);
                pooled_indices = single(pooled_indices+1).*sign(pooled_maps);
                pooled_maps = abs(pooled_maps);
            else
                pooled_maps = max_pool3d(abs(maps),pool_size,uint16(abs(pooled_indices)-1),COMP_THREADS);
                pooled_maps = pooled_maps.*sign(pooled_indices);
            end            
        case 'ProbMax'  % Same as max on way up
            [pooled_maps,pooled_indices] = max_pool(maps,pool_size);
        case 'Avg'      % Take the avg (abs) of values in region
            [pooled_maps,pooled_indices] = avg_pool(maps,pool_size,[],COMP_THREADS);
        case 'AbsAvg'   % Take abs values and average them in regions.
            [pooled_maps,pooled_indices] = abs_avg_pool(maps,pool_size,[],COMP_THREADS);
        case 'PNAvg'   % Separates positive and negative values and averages them in regions.
            % Just need to save the sign to unpool.
            pooled_indices = sign(maps);
            % Double the number of maps.
            num_maps = size(maps,3);
            maps = repmat(maps,[1 1 2 1 1]);
            maps(:,:,1:num_maps,:) = maps(:,:,1:num_maps,:).*(maps(:,:,1:num_maps,:)>0);
            maps(:,:,num_maps+1:end,:) = maps(:,:,num_maps+1:end,:).*(maps(:,:,num_maps+1:end,:)<0);
            
            
            [pooled_maps,pooled_indices] = avg_pool(maps,pool_size,[],COMP_THREADS);
        case 'PNAbsAvg'   % Separates positive and negative values and averages the abs values in regions.
            % Just need to save the sign to unpool.
            pooled_indices = sign(maps);
            % Double the number of maps.
            num_maps = size(maps,3);
            maps = repmat(maps,[1 1 2 1 1]);
            maps(:,:,1:num_maps,:) = maps(:,:,1:num_maps,:).*(maps(:,:,1:num_maps,:)>0);
            maps(:,:,num_maps+1:end,:) = maps(:,:,num_maps+1:end,:).*(maps(:,:,num_maps+1:end,:)<0);
            
            
            [pooled_maps,pooled_indices] = avg_pool(abs(maps),pool_size,[],COMP_THREADS);            
        case 'SplitKMax' % This splits the max k location into different maps.
            % Pool in 3D to get maxes over k.
            if(nargin<4 || isempty(pooled_indices))
                [pooled_maps,pooled_indices] = max_pool3d(maps,pool_size,[],COMP_THREADS);
            else
                [pooled_maps,pooled_indices] = max_pool3d(maps,pool_size,pooled_indices,COMP_THREADS);
            end
            unpooled_size = size(maps);
            % Unpool in 3D
            unpooled_maps = reverse_max_pool3d(pooled_maps,pooled_indices,pool_size,unpooled_size,COMP_THREADS);
            % Finally pool in 2D only to leave original number of maps.
            % This is only 2D because there is no different once split if you do 3D or not.
            [pooled_maps] = max_pool(unpooled_maps,pool_size,[],COMP_THREADS);
        case 'NormedMax'
            % This divides 2D max pooling by pool_size(3) for use when splitting modles from 3D to 2D to prevent too many summations.
            if(nargin<4 || isempty(pooled_indices))
                [pooled_maps,pooled_indices] = max_pool(maps,pool_size,[],COMP_THREADS);
            else
                [pooled_maps,pooled_indices] = max_pool(maps,pool_size,pooled_indices,COMP_THREADS);
            end
            % Normalize by the replications of the pooling size on the way up (therefore pooled maps will be at the correct scale for reconstruction).
            pooled_maps = pooled_maps/pool_size(3);
        case 'SplitMax' % This splits the max location into different maps (each of the pooled size)
            % This is only 2D because there is no different once split if you do 3D or not.
            if(nargin<4 || isempty(pooled_indices))
                [pooled_maps,pooled_indices] = max_pool(maps,pool_size,[],COMP_THREADS);
            else
                [pooled_maps,pooled_indices] = max_pool(maps,pool_size,pooled_indices,COMP_THREADS);
            end
            % Number of repmating to do.
            num_new = prod(pool_size(1:2));
            % Repmate but keep all copies of same map in order.
            pooled_maps  = reshape(pooled_maps,[size(pooled_maps,1) size(pooled_maps,2) 1 size(pooled_maps,3) size(pooled_maps,4)  ]);
            pooled_maps = repmat(pooled_maps,[1 1 num_new 1 1]);
            pooled_maps  = reshape(pooled_maps,[size(pooled_maps,1) size(pooled_maps,2) size(pooled_maps,3)*size(pooled_maps,4) size(pooled_maps,5) ]);
            
            % Now turn off elements that are not from that index.
            for i=1:num_new
                % Jumping every num_new maps.
                pooled_maps(:,:,i:num_new:end,:) = pooled_maps(:,:,i:num_new:end,:).*(pooled_indices==i);
            end
            
            %         [pooled_maps] = avg_pool(pooled_maps,pool_size,[],COMP_THREADS);
        case 'Maxes3'
            % Do 2D Max pooling then limit the number in each x,y resulting location to kdim/pool_size(3) elements on.
            % Only use the 2D indices to undo this (just 2D unpool).
            % This is an error because on forward prop with the correct indices it's not selecting the correct maxes in 3D.!!!!
            % Maybe this is good though, letting all feature to activate the gradients.
            if(nargin<4 || isempty(pooled_indices))
                [spatial_maps,pooled_indices] = max_pool(maps,pool_size(1:2),[],COMP_THREADS);
%                 tic
                [smaps,sinds] = sort(abs(spatial_maps),3);
                % Get the number of maxes to keep. 
                k = ceil(size(spatial_maps,3)/pool_size(3));
%                 smaps = smaps(:,:,end-k+1:end,:);
                sinds = sinds(:,:,end-k+1:end,:);
                % z here is the image index.
                [x,y,blah,z] = ndgrid(1:size(spatial_maps,1),0:(size(spatial_maps,2)-1),1:k,0:(size(spatial_maps,4)-1));
                sp = [size(spatial_maps) 1 1 1];
%                 keyboard
                lininds = x  + y*sp(1) + (sinds-1)*(sp(1)*sp(2)) + z*prod(sp(1:3));
                pooled_maps = zeros(size(spatial_maps),'single');
                pooled_maps(lininds) = spatial_maps(lininds);
                
                % Need meshgrids to get the linear indices.
%                 tsort = toc
                
                
%                 [pooled_maps,pooled_indices] = max_pool(maps,pool_size(1:2),[],COMP_THREADS);
% 
%                 tic
%                 % Keep the top kdim/pool_size(3) maxes.
%                 st = [size(pooled_maps) 1 1 1 1 1];
%                 % Get the maps as the first dimension.
%                 pooled_maps = permute(pooled_maps,[3 1 2 4]);
%                 % Mak the maps 1 x maps x 1 x x*y*images
%                 pooled_maps = reshape(pooled_maps,[1 st(3) 1 st(1)*st(2)*st(4)]);
%                 % Get one max per column over the feature maps and return that.
%                 pooled_maps = select_top_num_maxes(pooled_maps,ceil(st(3)/pool_size(3)),0);
%                 % Reshape into the permuted maps.
%                 pooled_maps = reshape(pooled_maps,[st(3) st(1) st(2) st(4)]);
%                 % Now permute back to original maps.
%                 pooled_maps = permute(pooled_maps,[2 3 1 4]);
%                 tmaxes=toc
%                 max(abs(pooled_maps1(:))-abs(pooled_maps(:)))
%                 keyboard
            else
                [pooled_maps,pooled_indices] = max_pool(maps,pool_size(1:2),pooled_indices,COMP_THREADS);
            end
        case 'OldMax3'     % Take the max (abs) value in region then over maps.
            % Pool spatially.
            if(nargin<4 || isempty(pooled_indices))
                [pooled_maps,pooled_indices] = max_pool(maps,pool_size,[],COMP_THREADS);
            else
                [pooled_maps,pooled_indices] = max_pool(maps,pool_size,pooled_indices,COMP_THREADS);
            end
            num_feature_maps = size(pooled_maps,3);
            % Max over feature maps.
            [maxes,mapinds] = max(abs(pooled_maps),[],3);
            %         pooled_maps==repmat(maxes,[1 1 size(pooled_maps,3) 1])
            %         keyboard
            
            % Make index to match mapind on.
            index = ones(1,1,num_feature_maps,1);
            index(:,:,1:num_feature_maps,:) = 1:num_feature_maps;
            index = repmat(index,[size(pooled_maps,1) size(pooled_maps,2) 1 size(pooled_maps,4)]);
            
            % Get the correct region indices.
            %         reginds = pooled_indices.*(repmat(mapinds,[1 1 num_feature_maps 1])==index);
            %         reginds = max(reginds,[],3);
            %         keyboard
            %         new_inds(:) = pooled_indices(logical(abs(pooled_maps)==repmat(maxes,[1 1 size(pooled_maps,3) 1])));
            %         pooled_indices = reshape(pooled_indices,size(maxinds));
            % Adjust indices to be over maps in each region as a column.
            %         pooled_indices = reginds+(mapinds-1)*prod(pool_size);
            %         keyboard
            % Keep the negatives.
            maxessize = size(maxes);
            unabsmaxes = max(pooled_maps,[],3);
            maxes(logical(unabsmaxes~=maxes)) = maxes(logical(unabsmaxes~=maxes)).*-1;
            %         spm = size(pooled_maps);
            %         keyboard
            pooled_maps = repmat(maxes,[1 1 num_feature_maps 1]).*(repmat(mapinds,[1 1 num_feature_maps 1])==index);
            
            %         pooled_maps = reshape(pooled_maps,spm);
            %         pooled_maps = maxes;
            
            % A VERY HACKY WAY OF KEEPING TRACK OF THE NUMBER OF FEATURE MAPS.
            %         pooled_indices(end+1,:,:,:) = 0;
            pooled_indices(:,:,end+1,:) = mapinds;
        case 'Old2Max3'
            
            if(numel(pool_size)<3)
                error('pool_wrapper.m: For 3D pooling you must input 3D pooling sizes.')
            end
            
            % Pool spatially.
            if(nargin<4 || isempty(pooled_indices))
                %             [pooled_maps,pooled_indices] = max_pool(maps,pool_size(1:2),[],COMP_THREADS);
                %
                %             % The correct method (collapsing the feature maps).
                %             new_maps = zeros(size(pooled_maps,1),size(pooled_maps,2),length(1:pool_size(3):size(pooled_maps,3)),size(pooled_maps,4),'single');
                %
                %             endmap = size(pooled_indices,3);
                %             group = 0;
                %
                %             % For each group, 'group', compute the max over the feature maps (dimension 3) in the group.
                %             for k=1:pool_size(3):size(pooled_maps,3)
                %                 group = group+1;            % index of group
                %
                %                 % Pool over feature maps with no zero planes (correct way).
                %                 [new_maps(:,:,group,:),pooled_indices(:,:,endmap+group,:)] = max(abs(pooled_maps(:,:,k:min(k+pool_size(3)-1,size(pooled_maps,3)),:)),[],3);
                %
                %                 % Have to keep the sign in the pooled maps (use k for the hack, group for the real way).
                %                 maxes = max(pooled_maps(:,:,k:min(k+pool_size(3)-1,size(pooled_maps,3)),:),[],3);
                %                 inds = (maxes==new_maps(:,:,group,:))-0.5;
                %                 new_maps(:,:,group,:) = new_maps(:,:,group,:).*sign(inds);
                %
                %             end
                %             pooled_maps = new_maps;
                
                
                %%%%%%
                % Less memory and quicker implementation (no for loop)
                [pooled_maps,pooled_indices] = max_pool(maps,pool_size(1:2),[],COMP_THREADS);
                
                % Padd the pooled maps with zeros to be able to reshape.
                extramaps = mod(size(pooled_maps,3),pool_size(3));
                if(extramaps>0)
                    addmaps = pool_size(3)-extramaps;
                    pooled_maps = cat(3,pooled_maps,zeros([size(pooled_maps,1),size(pooled_maps,2),addmaps,size(pooled_maps,4)],'single'));
                end
                
                % Instead of looping over groups, reshape and then do the max(abs());
                pooled_maps = reshape(pooled_maps,[size(pooled_maps,1) size(pooled_maps,2) pool_size(3) size(pooled_maps,3)/pool_size(3) size(pooled_maps,4)]);
                
                % Compute the maxes over groups in k dimension.
                maxes = max(pooled_maps,[],3);
                % This sets the k-indices.
                [pooled_maps,k_indices] = max(abs(pooled_maps),[],3);
                
                inds = (maxes==pooled_maps)-0.5;
                pooled_maps = pooled_maps.*sign(inds);
                pooled_maps = reshape(pooled_maps,[size(pooled_maps,1) size(pooled_maps,2) size(pooled_maps,4) size(pooled_maps,5)]);
                % They should be the same size.
                
                k_indices = reshape(k_indices-1,size(pooled_maps));
                pooled_indices = cat(3,pooled_indices,k_indices);
                
                %%%%%%%%%%
            else % Use the pooling indices that were passed in.
                %             % Pool spatially with the pooling indices passed in.
                %             pooled_maps = max_pool(maps,pool_size(1:2),pooled_indices(:,:,1:size(maps,3),:),COMP_THREADS);
                %
                %             nummaps = size(pooled_maps,3);
                %             group = 0;
                %
                %             pooled_indices = permute(pooled_indices,[1 2 4 3]);
                %             pooled_maps = permute(pooled_maps,[1 2 4 3]);
                %
                %             sizek = size(pooled_maps,1)*size(pooled_maps,2)*size(pooled_maps,3);
                %
                %             % Make the new maps (smaller in the kth dimension).
                %             new_maps = zeros(size(pooled_maps,1),size(pooled_maps,2),size(pooled_maps,3),length(1:pool_size(3):size(pooled_maps,4)),'single');
                %
                %             % For each group, 'group', compute the max over the feature maps (dimension 3) in the group.
                %             for k=1:pool_size(3):size(pooled_maps,4)
                %                 group = group+1;            % index of group
                %
                %                 ind = [1:sizek]'  +...   % The offset of each pixel for the current plane for each image in the set.
                %                     vect_array(sizek.*(pooled_indices(:,:,:,nummaps+group)-1)) + ... % The pooling indices offset.
                %                     sizek*(group-1)*pool_size(3); % The groups offset.
                %
                %                 ind2 = [1:sizek]'  +...   % The offset of each pixel for the current plane for each image in the set.
                %                     sizek*(group-1); % The groups offset.
                %
                %                 new_maps(ind2') = pooled_maps(ind');
                %
                %             end
                %
                %             pooled_maps = new_maps;
                %
                %             pooled_indices = permute(pooled_indices,[1 2 4 3]);
                %             pooled_maps = permute(pooled_maps,[1 2 4 3]);
                
                
                %%%%%
                % Less memory and quicker version.
                % Pool spatially with the pooling indices passed in.
                pooled_maps = max_pool(maps,pool_size(1:2),pooled_indices(:,:,1:size(maps,3),:),COMP_THREADS);
                nummaps = size(pooled_maps,3);
                
                % Padd the pooled maps with zeros to be able to reshape.
                extramaps = mod(size(pooled_maps,3),pool_size(3));
                if(extramaps>0)
                    addmaps = pool_size(3)-extramaps;
                    pooled_maps = cat(3,pooled_maps,zeros([size(pooled_maps,1),size(pooled_maps,2),addmaps,size(pooled_maps,4)],'single'));
                end
                
                % Make the new maps (smaller in the kth dimension).
                %             new_maps = zeros(size(pooled_maps,1),size(pooled_maps,2),length(1:pool_size(3):size(pooled_maps,3)),size(pooled_maps,4),'single');
                pooled_maps = reshape(pooled_maps,[size(pooled_maps,1) size(pooled_maps,2) pool_size(3) size(pooled_maps,3)/pool_size(3) size(pooled_maps,4)]);
                
                %             sizeI = size(pooled_indices);
                % Only need the k switches from this point on (use them to select on dim=3 from pooled_maps).
                k_indices = pooled_indices(:,:,nummaps+1:end,:);
                k_indices = reshape(k_indices,[size(k_indices,1) size(k_indices,2) 1 size(k_indices,3) size(k_indices,4)]);
                
                k_indices = repmat(k_indices,[1 1 size(pooled_maps,3) 1 1]);
                % Like meshgrid, gives the 1:pool_size(3) in the 3rd dimension.
                ind = repmat(reshape([0:size(pooled_maps,3)-1],[1 1 size(pooled_maps,3) 1 1]),[size(pooled_maps,1) size(pooled_maps,2) 1 size(pooled_maps,4) size(pooled_maps,5)]);
                %             keyboard
                pooled_maps = pooled_maps.*(k_indices==ind);
                pooled_maps = sum(pooled_maps,3);
                pooled_maps = reshape(pooled_maps,[size(pooled_maps,1) size(pooled_maps,2) size(pooled_maps,4) size(pooled_maps,5)]);
                %%%%%%%%%%%%%%%%
                
            end
        case 'OldMax3Abs'
            if(numel(pool_size)<3)
                error('pool_wrapper.m: For 3D pooling you must input 3D pooling sizes.')
            end
            
            % Pool spatially.
            if(nargin<4 || isempty(pooled_indices))
                [pooled_maps,pooled_indices] = max_pool(maps,pool_size(1:2),[],COMP_THREADS);
                
                % The correct method (collapsing the feature maps).
                new_maps = zeros(size(pooled_maps,1),size(pooled_maps,2),length(1:pool_size(3):size(pooled_maps,3)),size(pooled_maps,4),'single');
                
                endmap = size(pooled_indices,3);
                group = 0;
                
                % For each group, 'group', compute the max over the feature maps (dimension 3) in the group.
                for k=1:pool_size(3):size(pooled_maps,3)
                    group = group+1;            % index of group
                    
                    % Pool over feature maps with no zero planes (correct way).
                    [new_maps(:,:,group,:),pooled_indices(:,:,endmap+group,:)] = max(abs(pooled_maps(:,:,k:min(k+pool_size(3)-1,size(pooled_maps,3)),:)),[],3);
                    
                    % Have to keep the sign in the pooled maps (use k for the hack, group for the real way).
                    maxes = max(pooled_maps(:,:,k:min(k+pool_size(3)-1,size(pooled_maps,3)),:),[],3);
                    inds = (maxes==new_maps(:,:,group,:))-0.5;
                    %                 new_maps(:,:,group,:) = new_maps(:,:,group,:).*sign(inds);
                    
                    pooled_indices(:,:,endmap+group,:) = pooled_indices(:,:,endmap+group,:).*sign(inds);
                end
                pooled_maps = new_maps;
                
                %             pooled_indices = pooled_indices.*sign(pooled_maps);
                %             pooled_maps = abs(pooled_maps);
                
                %%%%%%%%%%
            else % Use the pooling indices that were passed in.
                % Pool spatially with the pooling indices passed in.
                pooled_maps = max_pool(maps,pool_size(1:2),pooled_indices(:,:,1:size(maps,3),:),COMP_THREADS);
                
                
                nummaps = size(pooled_maps,3);
                group = 0;
                
                pooled_indices = permute(pooled_indices,[1 2 4 3]);
                pooled_maps = permute(pooled_maps,[1 2 4 3]);
                
                sizek = size(pooled_maps,1)*size(pooled_maps,2)*size(pooled_maps,3);
                
                % Make the new maps (smaller in the kth dimension).
                new_maps = zeros(size(pooled_maps,1),size(pooled_maps,2),size(pooled_maps,3),length(1:pool_size(3):size(pooled_maps,4)),'single');
                
                % For each group, 'group', compute the max over the feature maps (dimension 3) in the group.
                for k=1:pool_size(3):size(pooled_maps,4)
                    group = group+1;            % index of group
                    
                    ind = [1:sizek]'  +...   % The offset of each pixel for the current plane for each image in the set.
                        vect_array(sizek.*(abs(pooled_indices(:,:,:,nummaps+group))-1)) + ... % The pooling indices offset.
                        sizek*(group-1)*pool_size(3); % The groups offset.
                    
                    ind2 = [1:sizek]'  +...   % The offset of each pixel for the current plane for each image in the set.
                        sizek*(group-1); % The groups offset.
                    
                    new_maps(ind2') = pooled_maps(ind');
                    
                end
                
                pooled_maps = new_maps;
                
                pooled_indices = permute(pooled_indices,[1 2 4 3]);
                pooled_maps = permute(pooled_maps,[1 2 4 3]);
                
            end
        case 'Avg3'
            if(numel(pool_size)<3)
                error('pool_wrapper.m: For Avg 3D pooling you must input 3D pooling sizes.')
            end
            
            %%%%%%
            % Less memory and quicker implementation (no for loop)
            pooled_maps = avg_pool(maps,pool_size(1:2),[],COMP_THREADS);
            
            
            % Padd the pooled maps with zeros to be able to reshape.
            extramaps = mod(size(pooled_maps,3),pool_size(3));
            if(extramaps>0)
                addmaps = pool_size(3)-extramaps;
                pooled_maps = cat(3,pooled_maps,zeros([size(pooled_maps,1),size(pooled_maps,2),addmaps,size(pooled_maps,4)],'single'));
            end
            
            % Instead of looping over groups, reshape and then do the max(abs());
            pooled_maps = reshape(pooled_maps,[size(pooled_maps,1) size(pooled_maps,2) pool_size(3) size(pooled_maps,3)/pool_size(3) size(pooled_maps,4)]);
            
            % Compute the maxes over groups in k dimension.
            pooled_maps = sum(pooled_maps,3);
            pooled_maps = reshape(pooled_maps,[size(pooled_maps,1) size(pooled_maps,2) size(pooled_maps,4) size(pooled_maps,5)]);
            
            
            % Now have to make it the mean.
            % Mean for the full groups.
            pooled_maps(:,:,1:(end-1),:) = pooled_maps(:,:,1:(end-1),:)./pool_size(3);
            % Mean for the last group;
            pooled_maps(:,:,end,:) = pooled_maps(:,:,end,:)./(pool_size(3)-extramaps);
            
            % Make sure these are empty.
            pooled_indices = [];
        case 'Dess3'
            pooled_maps = maps(1:pool_size(1):end,1:pool_size(2):end,1:pool_size(3):end,:);
        case 'None'     % Do nothing
            pooled_maps = maps;
            pooled_indices = [];
        otherwise
            fprintf('Not valid pooling type, so ignoring. (pool_wrapper.m)\n')
            pooled_maps = maps;
            pooled_indices = [];
    end
    
    % Make sure that the pooled_indices are always returned as uint16. (Conv needs singles for linear indexing).
    if(~isempty(pooled_indices) && strcmp(class(pooled_indices),'uint16')==0 &&...
            isempty(regexp(pool_type,'Conv')) && isempty(regexp(pool_type,'MaxAbs')) && isempty(regexp(pool_type,'Max3Abs')))
        error('pooled_indices created in pool_wrapper are of type %s instead of uint16 using type %s\n',class(pooled_indices),pool_type_start);
    end
end




% Can use this in combination with CN if you do BlahCNZero
if(length(pool_type)>3 && strcmp(pool_type(end-3:end),'Zero'))
    % For any pooling regions that were all zero, mark them with 0 indices
    % And therefore shift the rest of teh indices by 1 (this must be undone on the way down).
    % Also is undone at top of this function if indices are passed in.
    pooled_indices = pooled_indices+1;
    pooled_indices(pooled_maps==0) = 0;
    % This is case 1 where no matter what we zero out the maps (if indices were included);
%     if(USE_ZEROS)
%     fprintf(' Using zeros in FP ');
%     pooled_maps = pooled_maps.*(zero_indices~=0);
%     end
end


if(strcmp(pool_type(end-2:end),'PCA'))
    
    if(numel(pool_size)<3)
        error('pool_wrapper.m: For ...PCA pooling you must input 3D pooling sizes where the 3rd size is the PCA over k.')
    end
    temp_maps = pooled_maps;
    figure(1), sdf(temp_maps);
    [pooled_maps,pca_info] = pca_maps(pooled_maps,pca_info,COMP_THREADS);
    
    temp_maps2 = reverse_pca_maps(pooled_maps,pca_info,COMP_THREADS);
    figure(2), sdf(temp_maps2);
    figure(3), sdf(temp_maps-temp_maps2);
    keyboard
    
    
    % Store the new pooled_indices returned from above functions.
    pca_info.inds = pooled_indices;
    pooled_indices = pca_info;
    
elseif(~isempty(regexp(pool_type,'CN','ONCE'))) % If you want to contrast normalize the pooled maps.
    
    % Make the struct again.
    temp = pooled_indices;
    clear pooled_indices
    pooled_indices.inds = temp;
    pooled_indices.CNMeans = CNMeans;
    pooled_indices.CNSTDs = CNSTDs;
    
    if(~isempty(regexp(pool_type,'CN2','ONCE')))
        'blah'
    else
        
        
        % If not given indices, then must be doing the pooling stage so contrast normalize as well.
        if(INDICES_PROVIDED==0)
            k = fspecial('gaussian',pool_size(1:2),2);
            %     k = fspecial('gaussian',pool_size(1:2),2);
            %         k = 1; % to just average over dim3
            k = repmat(k,[1 1 size(pooled_maps,3)]); % Make multidimensional.
            k = k/sum(k(:)); % Make sure kernel summs to 1.
            for image=1:size(pooled_maps,4)
                %         fprintf('Contrast Normalizing Maps with Local CN: %10d\r',image);
                in = double(pooled_maps(:,:,:,image));
                %             figure(1), sdf(in);
                % Result for the summation of maps.
                if(size(k,1)>1)
                    fprintf('Conv')
                    lmn = zeros(size(in(:,:,1)));
                    for j=1:size(in,3)
                        lmn = lmn + rconv2(double(abs(in(:,:,j))),k(:,:,j));
                    end
                else
                    lmn = mean(abs(in),3);
                end
                % Subtract off the mean
                pooled_indices.CNMeans(:,:,:,image) = lmn;
                dimmean = in;
                %                     dimmean = bsxfun(@minus,in,lmn);
                
                dimmeansq = dimmean .* dimmean;
                
                if(size(k,1)>1)
                    lstd = zeros(size(in(:,:,1)));
                    for j=1:size(pooled_maps,3)
                        lstd = lstd + rconv2(dimmeansq(:,:,j),k(:,:,j));
                    end
                    lstd = sqrt(lstd);
                else
                    lstd = sqrt(mean(dimmeansq,3));
                end
                mstd = mean(lstd(:));
                lstd(lstd<mstd) = mstd;
                pooled_indices.CNSTDs(:,:,:,image) = lstd;
                
                pooled_maps(:,:,:,image) = single(bsxfun(@rdivide,dimmean,lstd));
                %             figure(2), sdf(pooled_maps(:,:,:,image));
                %             keyboard
            end
            
            
        else %Otherwise we're doing forward prop so just use the CNMeans and CNSTDs that are saved.
            %         pooled_maps = bsxfun(@minus,pooled_maps,CNMeans);
            pooled_maps = bsxfun(@rdivide,pooled_maps,CNSTDs);
        end
    end
    
    
    
end

end