function runme
%% run test


disp('* Start test: cuBinarizeProbs');
m=16;
n=16;
A_h = rand(m,n);

fprintf('Some data before binarizing :\n');
A_h(1:3,1:6)

A = GPUsingle(A_h); %uniform random data

cuBinarizeProbs(A); %randomly sample each unit


A_h = single(A); %bring back to CPU
fprintf('Showing some of GPU result (after Binarizing):\n');

A_h(1:3,1:6)

disp('* Test finished: cuBinarizeProbs');

%end