%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Max pools the input maps within pool_size region ona GPU. This uses the maximum
% absolute value within the pooling region. NOTE: THE INDICES RETURNED BY
% THIS FUNCTION ARE NOT COMPATIBLE WITH THE CPU VERSIONS OF POOLING BECAUSE
% cuGridToMatrix works in row major format so they indices count in the opposite
% ordering within each block.
%
% @deprecated This has been deprecated as cuMaxPool is much faster and easier. 
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @pooltoolbox_file @copybrief max_pool_gpu.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief max_pool_gpu.m
%
% @param input the planes to be normalized (xdim x ydim x n3 [x n4])
% @param pool_size a vector specifying the pooling sizein x and y dimensions.
% @param indices indices from a previous pooling operation you want to use
% @param COMP_THREADS UNUSED
%
% @retval pooled the pooled output planes
% @retval indices the indice within each pool region that was selected as max.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [pooled,indices] = max_pool_gpu(input,pool_size,indices,COMP_THREADS)

if(isa(input,'gsingle')) % GPU Jacket implementation
    
    blocksx = ceil(size(input,1)/pool_size(1));
    blocksy = ceil(size(input,2)/pool_size(2));
    num_maps = size(input,3);
    num_ims = size(input,4);
    
    if(nargin<3 || isempty(indices))
        
%         pooled = zeros([blocksx blocksy num_maps num_ims],gsingle);
%         indices = zeros([blocksx blocksy num_maps num_ims],gsingle);
%         
%         for im=1:num_ims
%             for map=1:num_maps
%                 gbA = im2col(input(:,:,map,im),[pool_size(1) pool_size(2)],'distinct');
%                 [gabsmaxes,ginds] = max(abs(gbA),[],1);
%                 gmaxes=max(gbA,[],1);
%                 pooled(:,:,map,im) = reshape(-gabsmaxes.*(gmaxes~=gabsmaxes)+gmaxes.*(gmaxes==gabsmaxes),[blocksx blocksy 1 1]);
%                 indices(:,:,map,im) = reshape(ginds,[blocksx blocksy 1 1]);
%             end
%         end
%         
%         indices = indices-1;



[xdim ydim numplanes numcases] = size(input);

% fprintf(1,'WARNING: You should compile the MEX version of "max_pool.cpp",\n         found in the MEX subdirectory of PoolingToolbox, and put it in your matlab path.  It is MUCH faster.\n');
% indices = double(indices);
        indices = zeros(ceil(xdim/pool_size(1)),ceil(ydim/pool_size(2)),numplanes*numcases,gsingle);

% The pooled input planes (not dimensions 3 and 4 are reversed).
pooled = zeros(ceil(xdim/pool_size(1)),ceil(ydim/pool_size(2)),numplanes*numcases,gsingle);
rows = gsingle(1:pool_size(1));
cols = gsingle(1:pool_size(2));
rblocks = ceil(xdim/pool_size(1));
cblocks = ceil(ydim/pool_size(2));
blockel = pool_size(1)*pool_size(2); %number of elements in block
% x = zeros(blockel,numplanes*numcases,gsingle); %this is made double to work with randbinom


gfor bk=1:rblocks*cblocks
% jj =     
% ii = bk - (jj-1)*rblocks;
        [ii,jj] = ind2sub([rblocks cblocks],bk);


    % Get blocks of the image.
%     for ii=0:rblocks-1
%         for jj=0:cblocks-1
            % Get current block for each plane and case.
            x = reshape(input(min((ii-1)*pool_size(1)+rows,xdim),min((jj-1)*pool_size(2)+cols,ydim),:), ...
                blockel,numplanes*numcases);
            
            % Get most positive and most negative numbers (and their indices).
            [gabsmaxes,ginds] = max(abs(x),[],1);
            gmaxes = max(x,[],1);
%             inds=inds-1; % Start at 0.
%             maxes = minA; % Iitialize to the mins (and their indices).
            % If abs(minA) smaller than maxA elements then replace them.
%             gtind = maxA>=abs(minA);
%             maxes(gtind) = maxA(gtind);
%             inds(gtind) = maxind(gtind);
            
            % Set the indices for all cases.
            indices(ii,jj,:) = ginds;
            pooled(ii,jj,:) = -gabsmaxes.*(gmaxes~=gabsmaxes)+gmaxes.*(gmaxes==gabsmaxes);
%         end
%     end
gend
    indices = reshape(indices,size(indices,1),size(indices,2),numplanes,numcases);
    pooled = reshape(pooled,size(pooled,1),size(pooled,2),numplanes,numcases);
indices = indices-1;









    else
%         pooled = zeros([blocksx blocksy num_maps num_ims],gsingle);
%         
%         % Remove offset.
%         indices = indices+1;
%         for im=1:num_ims
%             for map=1:num_maps
%                                 gbA = im2col(input(:,:,map,im),[pool_size(1) pool_size(2)],'distinct');
%                 %                 [gabsmaxes,ginds] = max(abs(gbA),[],1);
%                 %                 gmaxes=max(gbA,[],1);
%                 ginds = reshape(indices(:,:,map,im),[1 blocksx*blocksy]);
%                 ginds = ginds+[0:(blocksx*blocksy-1)]*(pool_size(1)*pool_size(2));
%                 gmaxes = gbA(ginds);
%                 pooled(:,:,map,im) = reshape((gmaxes),[blocksx blocksy 1 1]);
% %                 indices(:,:,map,im) = reshape(ginds,[blocksx blocksy 1 1]);
%             end
%         end
%         indices = indices-1;
        




[xdim ydim numplanes numcases] = size(input);

% The pooled input planes (not dimensions 3 and 4 are reversed).
pooled = zeros(ceil(xdim/pool_size(1)),ceil(ydim/pool_size(2)),numplanes*numcases,gsingle);
rows = gsingle(1:pool_size(1));
cols = gsingle(1:pool_size(2));
rblocks = ceil(xdim/pool_size(1));
cblocks = ceil(ydim/pool_size(2));
blockel = pool_size(1)*pool_size(2); %number of elements in block

    indices = reshape(indices,size(pooled));
    indices = indices+1;
    % Precompute the linear inex starting points used below.
    lininds = blockel*[0:(numplanes*numcases-1)];
    
    
gfor bk=1:rblocks*cblocks
% jj =     
% ii = bk - (jj-1)*rblocks;
        [ii,jj] = ind2sub([rblocks cblocks],bk);


    % Get blocks of the image.
%     for ii=0:rblocks-1
%         for jj=0:cblocks-1
            % Get current block for each plane and case.
            x = reshape(input(min((ii-1)*pool_size(1)+rows,xdim),min((jj-1)*pool_size(2)+cols,ydim),:), ...
                blockel,numplanes*numcases);
            
            % Get most positive and most negative numbers (and their indices).
%             [gabsmaxes,ginds] = max(abs(x),[],1);
%             gmaxes = max(x,[],1);
%             inds=inds-1; % Start at 0.
%             maxes = minA; % Iitialize to the mins (and their indices).
            % If abs(minA) smaller than maxA elements then replace them.
%             gtind = maxA>=abs(minA);
%             maxes(gtind) = maxA(gtind);
%             inds(gtind) = maxind(gtind);
            
            % Set the indices for all cases.
%             indices(ii,jj,:) = ginds;
%             pooled(ii,jj,:) = -gabsmaxes.*(gmaxes~=gabsmaxes)+gmaxes.*(gmaxes==gabsmaxes);
            pooled(ii,jj,:) = x(lininds+reshape(indices(ii,jj,:),[1 size(indices,3)]));
%         end
%     end
gend
    pooled = reshape(pooled,size(pooled,1),size(pooled,2),numplanes,numcases);
    indices = indices-1;
    indices = reshape(indices,size(indices,1),size(indices,2),numplanes,numcases);

    
%     for ii=0:rblocks-1
%         for jj=0:cblocks-1
%             % Reshape so each pooling region is a column.
%             x(1:blockel,:) = reshape(input(min(ii*pool_size(1)+rows,xdim),min(jj*pool_size(2)+cols,ydim),:), ...
%                 blockel,numplanes*numcases);
%             
%             
%             
%             %         % Get most positive and most negative numbers (and their indices).
%             %         [maxA,maxind] = max(x);
%             %         [minA,inds] = min(x);
%             %         maxes = minA; % Iitialize to the mins (and their indices).
%             %         % If abs(minA) smaller than maxA elements then replace them.
%             %         gtind = maxA>=abs(minA);
%             %         maxes(gtind) = maxA(gtind);
%             %         inds(gtind) = maxind(gtind);
%             
%             % Set the indices for all cases.
%             %         indices(ii+1,jj+1,:) = inds;
%             % Select the elements at the pooled indices.
%             %         pooled(ii+1,jj+1,:) = x(sub2ind(size(x),vect_array(indices(ii+1,jj+1,:))',1:size(x,2)));
%             maxes = x(lininds+vect_array(indices(ii+1,jj+1,:)+1)'); % Add one because indices starts at 0 now. 
%             pooled(ii+1,jj+1,:) = maxes;
%             
%         end
%     end












    end
    
    
    
    
    
    
    
    
    
    
    
    
    
    
else % GPUmat implementation.
    
    [xdim ydim maps images] = size(input);
    % Get the number of cases (maps*num_images).
    numCases = size(input,3)*size(input,4);
    % Calculate the regions to be extracted per image.
    regionsPerImage = ceil(size(input,1)/pool_size(1))*ceil(size(input,2)/pool_size(2));
    blocksx = ceil(size(input,1)/pool_size(1));
    remx = blocksx*pool_size(1)-size(input,1);
    if(remx>0) % This means there is the need for padding with zeros.
        % Create a new array that has twice as mkeyboardany padded element (we will remove the first half of them to make it the same as the CPU version).
        padded = zeros((xdim+remx*2)*(ydim+remx*2),maps*images,GPUsingle);
        % Reshape the image planes into 2D.
        setSize(input,[size(input,1)*size(input,2) size(input,3)*size(input,4)]);
        % Pad.
        cuCopyInto(input,padded,remx);
        % Reshape back to planes.
        setSize(padded,[(xdim+remx*2) (ydim+remx*2) maps images]);
        % Ignore the first part of the padding (to match cpu pooling padding).
        padded = slice(padded,[remx+1,1,END],[remx+1,1,END],':',':');
        %     clear padded
    else
        % Probably slow memory copy here (but avoides reshaping happing to input which affects it's size outside this function).
        padded = input;
    end
    
    % Reshape each image plane into 2D.
    setSize(padded,[size(padded,1)*size(padded,2) size(padded,3)*size(padded,4)]);
    
    
    % Make the resulting gridded matrix.
    gridded = zeros(pool_size(1)*pool_size(2),numCases*regionsPerImage,GPUsingle);
    % Split the image planes into pooling regions.
    cuGridToMatrix(padded,gridded,pool_size(1));
    
    
    % If no indices were passed in then create them as zeros and pool.
    if(nargin<3 || isempty(indices))
        
        % For now (since NVMax2 hasn't been updated) have to transpose pooled.
        % So dimension 1 is no the number of cases and dimension 2 is the size of the pooling region.
        gridded = transpose(gridded);
        % Initialize the maxes and indices they are located at.
        pooled = zeros(size(gridded,1),1,GPUsingle);
        indices = zeros(size(gridded,1),1,GPUsingle);
        
        % Compute argmax on the abs maps (then use the indices to get the actual values.
        nvMax3(gridded,2,pooled,indices);
        % nvMax2(abs(gridded),2,pooled,indices);
        % nvMax2 doesn't look for max(abs(input)) so have to select it based on the indices provided.
        %     indices2 = colon(1,1,size(gridded,1),GPUsingle)'+(indices-1).*(size(gridded,1));
        %     pooled = gridded(indices2);
        setSize(indices,[blocksx blocksx maps images]);
        %     setSize(indices2,[blocksx blocksx maps images]);
        
        %%%%%%%
        %     Try and convert from row major GPU indices into CPU equivalent matlab indices.
        % Row indices
        GPUminus(indices,1,indices);
        %     xindices = floor((indices2)/pool_size(1));
        % Column indices (without remainder because gpuMat doesn't support that).
        %     yindices = indices - (xindices)*pool_size(1);
        % Now these are CPU equivalent indices (start at 0 now).
        %     indices3 = xindices + (yindices-1)*pool_size(1);
        
        indices = (indices)*pool_size(1) + ...
            floor(indices/pool_size(1))*(1-pool_size(1)*pool_size(1));
        %%%%%%%
        
        %     max(abs(va(single(indices3))-va(single(indices4))))
        %     keyboard
        
    else % The indices were provided, so theyindic should be the planar size of pooled already.
        %     setSize(indices,size(gridded));
        % Pooled maps are the single max by the number of cases.
        %     pooled = zeros(1,size(gridded,2),GPUsingle);
        
        
        % String out the indices
        setSize(indices,[1 size(gridded,2)]);
        
        
        %%%%%%%
        % Have to take the provided correct matlab linear indices into the patch
        % and make them the row major format the cuGridToMatrix spits out.
        %     yindices = floor((indices)/pool_size(1));
        % Column indices (without remainder because gpuMat doesn't support that).
        %     xindices = indices - (yindices)*pool_size(1);
        % Now these are CPU equivalent indices (start at 0 now).
        %     indices3 = yindices + (xindices)*pool_size(1)+1;
        indices2 = floor((indices)/pool_size(1))*(1-pool_size(1)*pool_size(1)) + 1 + indices*pool_size(1);
        %     max(abs(va(single(indices3))-va(single(indices2))))
        % keyboard
        %%%%%%%%
        
        % Indices should be the linear indices into the rows of gridded
        % (since gridded doesn't need to be transposed in this case, that's only for nvMax2).
        % Remember have to add 1 to indices as they are stored starting from 0.
        %     indices+1
        indices2 = colon(0,1,size(gridded,2)-1,GPUsingle).*(pool_size(1)*pool_size(2))+indices2;
        %     indices
        %     gridded
        %     keyboard
        pooled = gridded(indices2);
        % Reshape the indices back to normal planar sizes.
        setSize(indices,[blocksx blocksx maps images]);
        
        
    end
    % Reshape the pooled maps into planes.
    setSize(pooled,[blocksx blocksx maps images]);
    % Make sure you don't affect the size of the input maps outside this function.
    setSize(input,[xdim ydim maps images]);
    
    
    
end





end


