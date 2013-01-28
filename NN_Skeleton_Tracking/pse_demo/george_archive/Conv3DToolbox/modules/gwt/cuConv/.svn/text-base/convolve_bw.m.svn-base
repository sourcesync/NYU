function targets = convolve_bw(images,filters)
  
  [numCases,imgPixels] = size(images);
  [numFilters, filterPixels] = size(filters);
  
  imgSize = sqrt(imgPixels);
  filterSize = sqrt(filterPixels);
  assert(imgSize == floor(imgSize));
  assert(filterSize == floor(filterSize));
  
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
      f = fliplr(flipud(reshape(filters(kk,:),filterSize,filterSize)));
      r = conv2(im,f,'valid');
      %store result in vector form
      targets(cc,(kk-1)*numOutputs+1:kk*numOutputs)=r(:);
    end
  end
  