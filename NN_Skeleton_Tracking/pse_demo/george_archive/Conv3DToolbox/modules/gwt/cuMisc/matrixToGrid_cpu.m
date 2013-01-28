function grid = matrixToGrid_cpu(matrix,imgSize,numImgs)

  %Like im2col but on vectorized inputs
  %The images are the columns of images
  %They must be square
  [factorPixels,numSquares]=size(matrix);
    
  factor = sqrt(factorPixels);
  assert(floor(factor)==factor);
  
  regionsPerImage = numSquares/numImgs;
  
  imgPixels = imgSize*imgSize;
  
  grid = zeros( imgPixels, numImgs);
  
  for ii=1:numImgs
    
    im = matrix(:,(ii-1)*regionsPerImage+1:ii*regionsPerImage);
    
    %col2im expects that each block is in col order
    %however, we want the blocks to be put back in row order
    %Since Alex's code assumes row order
    im = reshape(im,factor,factor,regionsPerImage);
    im = permute(im,[2 1 3]);
    im = reshape(im,factor*factor,regionsPerImage);
    
    g2d = col2im( im, [factor factor],[imgSize imgSize],'distinct');
    
    grid(:,ii) = g2d(:); %flatten into vector
    
  end
  
  
