%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Undoes the max pooling by placing the max back into it's indexed location.
%
% @deprecated This has been deprecated as cuRevMaxPool is much faster and easier. 
%
% @file
% @author Matthew Zeiler
% @date Apr 9, 2011
%
% @pooltoolbox_file @copybrief reverse_max_pool_gpu.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief reverse_max_pool_gpu.m
%
% @param input the planes to be normalized (xdim x ydim x n3 [x n4])
% @param indices the indices where the max came from during max_pool
% @param pool_size a vector specifying the pooling sizein x and y dimensions.
% @param unpooled_size this is used to specify the correct size of the unpooled
% region. If this is not passed in then xdim*pool_size(1) x ydim*pool_size(2)
% will be used.
% @param COMP_THREADS UNUSED
% @retval unpooled the unpooled output planes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [unpooled] = reverse_max_pool_gpu(input,indices,pool_size,unpooled_size,COMP_THREADS)


if(isa(input,'gsingle'))
    %     blocksx = ceil(unpooled_size(1)/pool_size(1));
    %     blocksy = ceil(unpooled_size(2)/pool_size(2));
    %     %     num_maps = size(input,3);
    %     num_maps = unpooled_size(3);
    %     num_ims = size(input,4);
    %     unpooled = zeros([unpooled_size(1) unpooled_size(2) unpooled_size(3) num_ims],gsingle);
    %
    %     input = reshape(input,[1 blocksx*blocksy num_maps num_ims]);
    %
    %     % Remove offset.
    %     indices = indices+1;
    %     for im=1:num_ims
    %         for map=1:num_maps
    % %             gbA = im2col(input(:,:,map,im),[pool_size(1) pool_size(2)],'distinct');
    % gbA = gzeros([pool_size(1)*pool_size(2) blocksx*blocksy 1 1]);
    %             %                 [gabsmaxes,ginds] = max(abs(gbA),[],1);
    %             %                 gmaxes=max(gbA,[],1);
    %             ginds = reshape(indices(:,:,map,im),[1 blocksx*blocksy]);
    %             ginds = ginds+[0:(blocksx*blocksy-1)]*(pool_size(1)*pool_size(2));
    %             gbA(ginds) = input(:,:,map,im);
    % %             pooled(:,:,map,im)
    %             unpooled(:,:,map,im) = col2im(gbA,[pool_size(1) pool_size(2)],[unpooled_size(1) unpooled_size(2)],'distinct');
    %             %                 indices(:,:,map,im) = reshape(ginds,[blocksx blocksy 1 1]);
    %         end
    %     end
    %     indices = indices-1;
    %
    %
    %
    
    
    
    
    
    
    
    
    
    [pxdim pydim numplanes numcases] = size(input);
    xdim = unpooled_size(1);
    ydim = unpooled_size(2);
    % The pooled input planes (not dimensions 3 and 4 are reversed).
    unpooled = zeros(unpooled_size(1)*unpooled_size(2),numplanes*numcases,gsingle);
    rows = gsingle(1:pool_size(1));
    cols = gsingle(1:pool_size(2));
    rblocks = ceil(xdim/pool_size(1));
    cblocks = ceil(ydim/pool_size(2));
    blockel = pool_size(1)*pool_size(2); %number of elements in block
    
    input = reshape(input,size(input,1),size(input,2),numplanes*numcases);
    indices = reshape(indices,size(input));
    indices = indices+1;
    % Precompute the linear inex starting points used below.
    lininds = gsingle((unpooled_size(1)*unpooled_size(2))*[0:(numplanes*numcases-1)]);
    xdim = gsingle(xdim);
    ydim = gsingle(ydim);
    
    
    
    
    gfor bk=1:rblocks*cblocks
    [ii,jj] = ind2sub([rblocks cblocks],bk);
    
    % Get current block for each plane and case.
    %             x = reshape(input(min((ii-1)*pool_size(1)+rows,xdim),min((jj-1)*pool_size(2)+cols,ydim),:), ...
    %                 blockel,numplanes*numcases);
    
    %             x = zeros(blockel,numplanes*numcases,gsingle);
    %             x(lininds+reshape(indices(ii,jj,:),[1 size(indices,3)])) = input(ii,jj,:);
    yinds = reshape(floor((indices(ii,jj,:)-1)/pool_size(1)),[1 numplanes*numcases]);
    xinds = reshape(indices(ii,jj,:),[1 numplanes*numcases])-(yinds)*pool_size(1);
    %             keyboard
    %             unpooled(min((ii-1)*pool_size(1)+rows,xdim),min((jj-1)*pool_size(2)+cols,ydim),:) = reshape(x,[pool_size(1) pool_size(2) numplanes*numcases]);
    % A = (ii-1)*pool_size(1)+xinds+((jj-1)*pool_size(2)+yinds)*unpooled_size(1)+lininds;
    % keyboard
    unpooled((ii-1)*pool_size(1)+xinds+((jj-1)*pool_size(2)+yinds)*unpooled_size(1)+lininds) = reshape(input(ii,jj,:),[1 numplanes*numcases]);
    
    
    gend
    unpooled = reshape(unpooled,unpooled_size(1),unpooled_size(2),numplanes,numcases);
    %     indices = indices-1;
    indices = reshape(indices,size(indices,1),size(indices,2),numplanes,numcases);
    
    
    
    
    
    
    
    
    
    
else
    
    % Get the sizes of the pooled maps and other needed dimensions.
    [xdim ydim maps images] = size(input);
    % Get the number of pooled cases (unpooled_maps*num_images).
    numCases = unpooled_size(3)*size(input,4);
    % Calculate the regions to be extracted per image.
    regionsPerImage = ceil(unpooled_size(1)/pool_size(1))*ceil(unpooled_size(2)/pool_size(2));
    blocksx = ceil(unpooled_size(1)/pool_size(1));
    remx = blocksx*pool_size(1)-unpooled_size(1);
    
    % Initialize the unpooled maps on the GPU.
    unpooled = zeros((unpooled_size(1)+remx)*(unpooled_size(2)+remx),unpooled_size(3)*images,GPUsingle);
    % Make it into grid form before inserting.
    gridded = zeros(pool_size(1)*pool_size(2),numCases*regionsPerImage,GPUsingle);
    
    
    setSize(indices,[xdim*ydim*maps*images])
    setSize(input,[xdim*ydim*maps*images])
    
    % Have to take the provided correct matlab linear indices into the patch
    % and make them the row major format the cuGridToMatrix spits out.
    yindices = floor((indices)/pool_size(1));
    % Column indices (without remainder because gpuMat doesn't support that).
    xindices = indices - (yindices)*pool_size(1);
    % Now these are CPU equivalent indices (start at 0 now).
    indices2 = yindices + (xindices)*pool_size(1)+1;
    %XXXXXXXXXXXXXXXXXXX
    % for some reason the pooled_indices that are passed in are now reshaped afterwards.
    % even though below we setSize back to the original.
    % so indices2 creates a copy and then that is used for indexing then discarded upon leaving this function.
    %XXXXXXXXXXXXXXXXXXX
    
    % Convert the patch based indices into linear indices into the entire array.
    % Note: this assumes teh standard matlab CPU equivalent indices starting at 0.
    indices2 = indices2+colon(0,1,size(gridded,2)-1,GPUsingle)*pool_size(1)*pool_size(2);
    
    % Put the max into the correct location of each patch of zeros.
    gridded(indices2) = input;
    
    % Now convert the gridded version with the maxes inserted into the unpooeld version.
    cuMatrixToGrid(gridded,unpooled,pool_size(1));
    
    
    % Form into planes.
    setSize(unpooled,[(unpooled_size(1)+remx) (unpooled_size(2)+remx) unpooled_size(3) images]);
    
    
    % Finally make sure the returned size is the correct unpooled size.
    unpooled = slice(unpooled,[1,1,unpooled_size(1)],[1,1,unpooled_size(2)],[1,1,unpooled_size(3)],':');
    
    % Make sure the inidces nad input are the original sizes at the end.
    setSize(indices,[xdim ydim maps images]);
    setSize(input,[xdim ydim maps images]);
    
end

end
