function runme
%% run test

testcuRand %rand
testcuRandn %randn
testcuBinarizeProbs %Bernoulli sampling

testcuSigmoid %sigmoid
testcuThreeway %three-way outer product
testcuDist %Euclidean distance computation

testcuCopyInto %zero-padding
testcuRotate180 %rotation of filters
testcuSubsample %average downsampling
testcuSupersample %up-sampling by a constant factor
testcuGridToMatrix %like im2col
testcuMatrixToGrid %like col2im
testcuSampleMultinomial %sample in parallel from many multinomials
testcuEltWiseDivideByVector2 % Divide a matrix element-wise by a vector
%end