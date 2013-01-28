function targets = convolve3_bw(images,filters,numGroups,numImgsPerGroup,color,imgOrder)
  
  
  if color ~= 0
    error('Color not supported yet')
  end
  
  numFiltersPerGroup = size(filters,1)/numGroups;
  filterPixels = size(filters,2);
  
  filterSize = sqrt(filterPixels);
  assert(filterSize == floor(filterSize));
    
  if imgOrder==0
    %GROUP_FILTER_IMAGE
    imgPixels = size(images,2)/numImgsPerGroup;
  elseif imgOrder==1
    % IMAGE_GROUP_FILTER
    imgPixels = size(images,2)/numFiltersPerGroup;
  else
    error('Unsupported order')
  end
  
  imgSize = sqrt(imgPixels);  
  assert(imgSize == floor(imgSize));
  
  numOutputsX = imgSize - filterSize + 1;
  numOutputs = numOutputsX * numOutputsX;
    
  targets = zeros(numImgsPerGroup*numGroups,numOutputs,'single');
  
  if imgOrder==0
    
    for gg=1:numGroups
      for cc=1:numImgsPerGroup
        targetRowIdx = numImgsPerGroup*(gg-1)+cc;
        for kk=1:numFiltersPerGroup
          imgRowIdx=numFiltersPerGroup*(gg-1)+kk;
          imgColIdx=imgPixels*(cc-1)+1:imgPixels*cc;
          im = reshape(images(imgRowIdx,imgColIdx),imgSize,imgSize);

          % Note that in cuConv, cuConv2
          % Alex's code just dot-multiplies the filters AS-is
          % Instead of doing a flipud(fliplr(filter)) before the dot-multiply,
          % which is Matlab's definition of 2-D convolution
          % So to do the same thing as he is doing, I was rotating my
          % filters before applying convolution
          % However, in cuConv3 it appears he is indeed doing convolution
          % So no rotation is necessary
          f = reshape(filters(imgRowIdx,:),filterSize,filterSize);
          
          r = conv2(im,f,'valid');
          
          targets(targetRowIdx,:)=targets(targetRowIdx,:)+r(:)';
        end
      end
    end
        
  elseif imgOrder==1

    for gg=1:numGroups %sF(3)*sf(5) 
      for cc=1:numImgsPerGroup % 1
        targetRowIdx = numImgsPerGroup*(gg-1)+cc;
        for kk=1:numFiltersPerGroup % size(maps,3) or sF(4)
          imgRowIdx=targetRowIdx;
          imgColIdx=imgPixels*(kk-1)+1:imgPixels*kk;
          im = reshape(images(imgRowIdx,imgColIdx),imgSize,imgSize);

          filterRowIdx=numFiltersPerGroup*(gg-1)+kk;
          
          % Note that in cuConv, cuConv2
          % Alex's code just dot-multiplies the filters AS-is
          % Instead of doing a flipud(fliplr(filter)) before the dot-multiply,
          % which is Matlab's definition of 2-D convolution
          % So to do the same thing as he is doing, I was rotating my
          % filters before applying convolution
          % However, in cuConv3 it appears he is indeed doing convolution
          % So no rotation is necessary
          f = reshape(filters(filterRowIdx,:),filterSize,filterSize);
          
          r = conv2(im,f,'valid');
          
          targets(targetRowIdx,:)=targets(targetRowIdx,:)+r(:)';
        end
      end
    end
    
  else
    error('Order not supported')
  end
  
  
