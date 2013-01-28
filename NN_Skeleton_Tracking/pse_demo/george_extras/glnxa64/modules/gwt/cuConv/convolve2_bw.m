function targets = convolve2_bw(images,filters,filterSize)
  
  [numCases,imgPixels] = size(images);
  [nc, filterDims] = size(filters);
  
  filterPixels = filterSize*filterSize;
  
  assert(nc==numCases);
  assert( mod(filterDims,(filterPixels))==0 );
    
  imgSize = sqrt(imgPixels);
    
  numFilters = filterDims / (filterSize * filterSize);
  
  assert(imgSize == floor(imgSize));
  
  numOutputsX = imgSize - filterSize + 1;
  numOutputs = numOutputsX * numOutputsX;
  
  targets = zeros(numCases,numFilters*numOutputs,'single');
  
  for cc=1:numCases
    im = reshape(images(cc,:),imgSize,imgSize);
    for kk=1:numFilters
      %Note that Alex's code just dot-multiplies the filters AS-is
      %Instead of doing a flipud(fliplr(filter)) before the dot-multiply,
      %which is Matlab's definition of 2-D convolution
      %So to do the same thing as he is doing, I rotate my filters
      %f = fliplr(flipud(reshape(filters(kk,:),filterSize,filterSize)));
      f = fliplr(flipud(reshape(filters(cc,...
            (kk-1)*filterPixels+1:kk*filterPixels),filterSize,filterSize)));
      r = conv2(im,f,'valid');
      %store result in vector form
      targets(cc,(kk-1)*numOutputs+1:kk*numOutputs)=r(:);
    end
  end
  