function runme
%% run test
disp('* Start test');

imgSize=47;
%filterSize=16;
numFilters=64;
numCases=128;

maxFilterSize=21;

for filterSize=2:maxFilterSize
fprintf('filterSize %d/%d\n',filterSize,maxFilterSize);

filterPixels=filterSize*filterSize;
imgPixels=imgSize*imgSize;
numOutputsX = imgSize - filterSize + 1;
numOutputs = numOutputsX*numOutputsX;

%Assume that each row of filtersH is a filterSize*filterSize filter
%Flattened in ROW-MAJOR format
filtersH = rand(numFilters,filterPixels,'single');
%Assume that each row of imagesH is a imgSize*imgSize image
%Flattened in ROW-MAJOR format
imagesH = rand(numCases,imgPixels,'single');

%Note that even though Matlab is going to interpret images & filters in
%COLUMN-MAJOR format, both the image and filter will be transposed
%So the result of convolution will just be a transposed version of the
%(C/C++) result
%However, when we flatten this transposed result to vector form in
%COLUMN-MAJOR format (using Matlab) this will give us the equivalent of
%the (C/C++) proper result in ROW-MAJOR format
%Which is what we want!
fprintf('Starting convolution on CPU\n');
tic;targetsH = convolve_bw(imagesH,filtersH);toc;
fprintf('Done convolution on CPU\n');

%Unfortunately just like mxArrays, GPUsingle type stores data in
%COLUMN-MAJOR format
%Since all of Alex's convolutional code assumes ROW-MAJOR format we take
%the transpose of the data before making GPUsingles
%this ensures the data is in ROW-MAJOR form (i.e. cases are consecutive,
%within each case rows are consecutive, etc.)
filters = GPUsingle(transpose(filtersH));
images = GPUsingle(transpose(imagesH));
targets = zeros(numFilters*numOutputs,numCases, GPUsingle); %note this is
                                                            %transposed
                                                            %from the
                                                            %Matlab
                                                            %version
fprintf('Starting convolution on GPU\n');
tic;cuConv(images,filters,targets);GPUsync;toc;
fprintf('Done convolution on GPU\n');

targets_h = transpose(single(targets)); %move to CPU and transpose to
                                        %align with targetsH


fprintf('Showing some of CPU result:\n');
targetsH(1:3,1:6)

fprintf('Showing some of GPU result:\n');
targets_h(1:3,1:6)

tol=1e-3;
maxabsdiff = max(abs(targetsH(:)-targets_h(:))); 
fprintf('Max abs difference: %f\n', maxabsdiff );
assert(maxabsdiff<tol);
end
disp('* Test finished');
end