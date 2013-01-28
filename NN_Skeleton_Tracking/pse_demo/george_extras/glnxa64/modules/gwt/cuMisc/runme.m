function runme
%% run test
disp('* Start test');

imgSize=47;
filterSize=16;
numFilters=64;
numCases=128;

filterPixels=filterSize*filterSize;
imgPixels=imgSize*imgSize;
numOutputsX = imgSize - filterSize + 1;
numOutputs = numOutputsX*numOutputsX;


filters = GPUsingle(rand(numFilters,filterPixels));
images = GPUsingle(rand(numCases,imgPixels));
targets = zeros(numCases, numFilters*numOutputs, GPUsingle);

cuConv(images,filters,targets);

disp('* Test finished');
end