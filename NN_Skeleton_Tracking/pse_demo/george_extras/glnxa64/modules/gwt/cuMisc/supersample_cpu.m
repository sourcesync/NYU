function targets = supersample_cpu(images,factor)
  
  %Performs up-sampling on many images by a fixed factor
  %The images are the columns of images
  %They must be square
  [imgPixels,numImgs]=size(images);
  imgSize = sqrt(imgPixels);
  assert(floor(imgSize)==imgSize);
  
  targetSize = factor*imgSize;
  targetPixels = targetSize*targetSize;
  
  targets = zeros(targetPixels,numImgs);
  
  for ii=1:numImgs
    %reshape each image into 2-d before applying upsampling
    im = kron( reshape( images(:,ii), imgSize,imgSize), ones(factor) );

    targets(:,ii)=im(:);
  end
  
  
