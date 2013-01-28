function targets = gridToMatrix_cpu(images,factor)

  %Like im2col but on vectorized inputs
  %The images are the columns of images
  %They must be square
  [imgPixels,numImgs]=size(images);
  imgSize = sqrt(imgPixels);
  assert(floor(imgSize)==imgSize);
  
  regionsPerImage = (imgSize/factor)*(imgSize/factor);
  
  targets = zeros (factor*factor,numImgs*regionsPerImage);  
  
  for ii=1:numImgs
    %reshape each image into 2-d before applying im2col
    im = im2col( reshape( images(:,ii), imgSize,imgSize),[factor ...
                        factor],'distinct');
    
    %note that im2col returns each block in col order
    %but Alex's code returns each block in row order
    im = reshape(im,factor,factor,regionsPerImage);
    im = permute(im,[2 1 3]);
    im = reshape(im,factor*factor,regionsPerImage);
    
    targets(:,(ii-1)*regionsPerImage+1:ii*regionsPerImage)=im;
    
  end
  
  
