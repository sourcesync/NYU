%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Undoes the average pooling by placing the average back into each location
% within the pool region. Note there is no reverse_abs_avg_pool.m since we do
% not know which elements were positive/negative and thus you should use this
% function to undo the abs_avg_pool.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @pooltoolbox_file @copybrief reverse_avg_pool.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief reverse_avg_pool.m
%
% @param input the planes to be normalized (xdim x ydim x n3 [x n4])
% @param indices [] not used here.
% @param pool_size a vector specifying the pooling sizein x and y dimensions.
% @param unpooled_size this is used to specify the correct size of the unpooled
% region. If this is not passed in then xdim*pool_size(1) x ydim*pool_size(2)
% will be used.
% @param COMP_THREADS UNUSED
% @retval unpooled the unpooled output planes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [unpooled] = reverse_avg_pool(input,indices,pool_size,unpooled_size,COMP_THREADS)

if(ndims(input)==3)
    [xdim ydim numplanes] = size(input);
    numcases = 1;
else
    [xdim ydim numplanes numcases] = size(input);
end

% fprintf(1,'WARNING: You should compile the MEX version of "reverse_avg_pool.cpp",\n         found in the MEX subdirectory of PoolingToolbox, and put it in your matlab path.  It is MUCH faster.\n');


% The unpooled input planes.
% if(nargin<4)
unpooled = zeros(ceil(xdim*pool_size(1)),ceil(ydim*pool_size(2)),numplanes*numcases,'single');
% else % Will need this to be computed externally.
%     % Feature map sizes (unpooled sizes) are xdim+filter_size-1...
%     unpooled = zeros(unpooled_size(1),unpooled_size(2),numplanes*numcases,'single');
% end

input = reshape(input,xdim,ydim,numplanes*numcases);

rblocks = xdim;
cblocks = ydim;

rows = 1:pool_size(1);
cols = 1:pool_size(2);

for ii=0:rblocks-1
    for jj=0:cblocks-1
        % copy the average into all locations of the pool region.
        % Copy into corresponding location in output.
        unpooled(ii*pool_size(1)+rows,jj*pool_size(2)+cols,:) = repmat(input(ii+1,jj+1,:),[pool_size(1) pool_size(2) 1]);
    end
end
unpooled = reshape(unpooled,size(unpooled,1),size(unpooled,2),numplanes,numcases);

% The unpooled input planes.
if(nargin==4)
    % Feature map sizes (unpooled sizes) are xdim+filter_size-1...
    unpooled = unpooled(1:unpooled_size(1),1:unpooled_size(2),:,:);
end



end
