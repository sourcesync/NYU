%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% An alternative implemention of gpu 2D convolution that replicates the
% functionality of \c ipp_conv2.m . This relies on Graham Taylor's \c cuConv
% function for the actually CUDA convolutions on the gpu. This is simply a
% wrapper for that function so that you can pass in the inputs in planar form as
% opposed to vectorizing them. 
%
% This is no longer supported.
%
% @deprecated
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @gpu_file @copybrief gpu_conv2.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief gpu_conv2.m
% 
% @param A stack of image planes (3D) of type GPUsingle
% @param B stack of filter planes (3D) of type GPUsingle
% @param mode 'valid' or 'full' (valid is default if nothing is passed in)
% @retval output stack of convolved planes (4D) where the thrid dimension is for
% each filter and the fourth dimension is for each image plane.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [output] = gpu_conv2(A,B,mode)


if(nargin<3 || isempty(mode))
    mode = 'valid';
end

%% Get Sizes
filterSize = [size(B,1) size(B,2)];

numCases = size(A,3);
if(ndims(B)==4)  % Assume the third dimension is singleton.
    B = reshape(B,size(B,1),size(B,2),size(B,3)*size(B,4));
end
numFilters = size(B,3);

if((numFilters/16-floor(numFilters/16))>0)  % Means that is greater than a multiple of 16
    actFilters = (floor(numFilters/16)+1)*16;
    zeroFilters = actFilters-numFilters; % number of filters that will be zero.
else
    actFilters = numFilters;
    zeroFilters = 0;
end
% zeroFilters=0;
%% Reshape images depending on mode.
if(strcmp(mode,'valid'))
    imgSize = [size(A,1) size(A,2)];
    A = reshape(A,size(A,1)*size(A,2),numCases); % Make them into cuConv format
elseif(strcmp(mode,'full'))
    %     'in the full'
    % have to pad the images to get the full from the valid convolution.
    newA = zeros(size(A,1)+2*size(B,1)-2,size(A,2)+2*size(B,2)-2,size(A,3),GPUsingle);
    % since padarray is not in GPU mat have to do the insertion manually.
    if(size(A,3)==1)
        assign(1,newA,A,[size(B,1) 1 size(B,1)-1+size(A,1)],[size(B,2) 1 size(B,2)-1+size(A,2)])
    else
        assign(1,newA,A,[size(B,1) 1 size(B,1)-1+size(A,1)],[size(B,2) 1 size(B,2)-1+size(A,2)],[1 1 END])
    end
    imgSize = [size(newA,1) size(newA,2)];
    A = reshape(newA,size(newA,1)*size(newA,2),numCases); % Make them into cuConv format
else
    error('Input mode is not valid or full')
end

%% Reshape filters.
B = flipdim(flipdim(B,1),2); % To be like conv2.
B = reshape(B,filterSize(1)*filterSize(2),numFilters); % into cuConv format
if(zeroFilters~=0) % If there is padding needed
    % Pad with zeros for the extra zeroFilters dimension.
    B = [B zeros(filterSize(1)*filterSize(2),zeroFilters,GPUsingle)];
end



% %%%%%%%%
% %% If you want the multiple image multiple kernel case to convolve each
% %% filter with each image once (not each combination of them).
% if(numFilters > 1 && numFilters == numCases) % For the multiple multple case just extract the identity.
%     %% Setup output
%     numOutputsX = imgSize - filterSize + 1;
%     numOutputs = numOutputsX*numOutputsX;
%     output = zeros(numOutputsX,numOutputsX,numFilters,GPUsingle);
%
%     for i=1:numCases
%
%         tempoutput = zeros(numOutputs*actFilters,1, GPUsingle); %note this is
%         size(A)
%         size(B)
%         numCases
%         %% Do convolution on GPU
%         cuConv(A(:,i),B,tempoutput);
%
%         % output = transpose(output);
%         output(:,:,i) = reshape(tempoutput(numOutputs*(i-1)+1:numOutputs*i,:),numOutputsX,numOutputsX,1,1);
%
%

%     end
% else

%% Setup output
numOutputsX = imgSize(1) - filterSize(1) + 1;
numOutputsY = imgSize(2) - filterSize(2) + 1;
numOutputs = numOutputsX*numOutputsY;
output = zeros(actFilters*numOutputs,numCases, GPUsingle); %note this is

%% Do convolution on GPU
if(numFilters > 1 && numFilters == numCases) % For the multiple multple case just extract the identity.
    size(A)
    size(B)
    actFilters
    numFilters
    outy = zeros(actFilters*numOutputs,1,GPUsingle);
    for k=1:numFilters
        k
        cuConv(A(:,k),B(:,k),outy);
        output(:,:,k) = reshape(outy,numOutputsX,numOutputsY,1);
    end
else
    cuConv(A,B,output);
    
    % output = transpose(output);
    output = reshape(output,numOutputsX,numOutputsY,actFilters,numCases);
    
    % Need to remove all the extra filters we created (only numFilters are
    % good).
    output = output(:,:,1:numFilters,:);
end
% end

if(numFilters > 1 && numFilters == numCases) % For the multiple multple case just extract the identity.
    selector = GPUsingle(find(eye(numCases,numCases)));
    output = reshape(output,size(output,1),size(output,2),size(output,3)*size(output,4));
    output = output(:,:,selector);
end


end

%> @}