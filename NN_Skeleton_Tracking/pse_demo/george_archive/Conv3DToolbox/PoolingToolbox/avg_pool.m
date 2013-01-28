%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Average pools the input maps within pool_size region.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @pooltoolbox_file @copybrief avg_pool.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief avg_pool.m
%
% @param input the planes to be normalized (xdim x ydim x n3 [x n4])
% @param pool_size a vector specifying the pooling sizein x and y dimensions.
% @param indices UNUSED
% @param THRESHOLD if you want to threshold the pooled maps so that some pooled
% regions can turn off completely. This is the value of the threshold, and it can
% have separate values for each plane (n3 x n4 matrix).
% @param COMP_THREADS UNUSED
% @retval pooled the pooled output planes
% @retval indices [] for average pooling.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [pooled,indices] = avg_pool(input,pool_size,indices,THRESHOLD,COMP_THREADS)

if(ndims(input)==3)
    [xdim ydim numplanes] = size(input);
    numcases = 1;
else
    [xdim ydim numplanes numcases] = size(input);
end

if(nargin<4)
    THRESHOLD = 0;
end





% fprintf(1,'WARNING: You should compile the MEX version of "avg_pool.cpp",\n         found in the MEX subdirectory of PoolingToolbox, and put it in your matlab path.  It is MUCH faster.\n');

% The pooled input planes (not dimensions 3 and 4 are reversed).
pooled = zeros(ceil(xdim/pool_size(1)),ceil(ydim/pool_size(2)),numplanes*numcases,'single');
% Store the indices for each plane.
% indices = zeros(size(im2col(input(:,:,1),pool_size,'distinct'),2),numplanes);
indices = [];

rows = 1:pool_size(1);
cols = 1:pool_size(2);
rblocks = ceil(xdim/pool_size(1));
cblocks = ceil(ydim/pool_size(2));
blockel = pool_size(1)*pool_size(2); %number of elements in block
x = zeros(blockel,numplanes*numcases,'single'); %this is made double to work with randbinom

input = reshape(input,size(input,1),size(input,2),size(input,3)*size(input,4));



% Get blocks of the image.
for ii=0:rblocks-1
    for jj=0:cblocks-1
        % Get the current block for each plane and case.
        x(1:blockel,:) = reshape(input(min(ii*pool_size(1)+rows,xdim),min(jj*pool_size(2)+cols,ydim),:), ...
            blockel,numplanes*numcases);

        % Compute mean of each block.
%         temp = 
        pooled(ii+1,jj+1,:) = mean(x,1);
    end
end


if(THRESHOLD>0)
    pooled = pooled.*(abs(pooled)>THRESHOLD);
end
        
pooled = reshape(pooled,size(pooled,1),size(pooled,2),numplanes,numcases);

% newinp = zeros(rblocks*pool_size(1)+1,cblocks*pool_size(2)+1,size(input,3),size(input,4));
% newinp(2:1+size(input,1),2:1+size(input,2),:,:) = input;
% input = newinp;
% 
% % Computes abs, cumsums and divide by pool regions to get means.
% input = cumsum(cumsum(input,1),2)/prod(pool_size);
% 
% 
% % input= padarray
% 
% % Now have to do d+a-b-c
% % d
% pooled = input(1+pool_size(1):pool_size(1):end,1+pool_size(2):pool_size(2):end,:,:);
% pooled = pooled + input(1:pool_size(1):end-pool_size(1)+1,1:pool_size(2):end-pool_size(2)+1,:,:);
% pooled = pooled - input(1:pool_size(1):end-pool_size(1)+1,1+pool_size(2):pool_size(2):end,:,:);
% pooled = pooled - input(1+pool_size(1):pool_size(1):end,1:pool_size(2):end-pool_size(2)+1,:,:);


end
