function targets = subsample_cpu(images,factor)
  
  %Performs average pooling of many images
  %The images are the columns of images
  %They must be square
  [imgPixels,numImgs]=size(images);
  imgSize = sqrt(imgPixels);
  assert(floor(imgSize)==imgSize);
  
  numRegions = imgSize / factor;
  
  targets = zeros (numRegions*numRegions,numImgs);  
  divisor = factor*factor;
  
  for ii=1:numImgs
    %reshape each image into 2-d before applying im2col
    im = im2col( reshape( images(:,ii), imgSize,imgSize),[factor factor],'distinct');
    targets(:,ii)=(sum(im,1)/divisor);
  end
  
  
