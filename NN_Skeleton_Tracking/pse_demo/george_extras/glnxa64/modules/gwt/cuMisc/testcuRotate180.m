function runme
%% run test
disp('* Start test: cuRotate180');
filterSize=9;
numFilters = 256;

Ah = randn(filterSize,filterSize,numFilters);

fprintf('Some data before copy :\n');
Ah(1:3,1:6,1)

fprintf('Starting computation on CPU\n');
tic; Bh = Ah(end:-1:1,end:-1:1,:);toc;
fprintf('Done computation on CPU\n');

Ah = reshape(Ah,filterSize*filterSize,numFilters); %collapse to vectors
A = GPUsingle(Ah); 

%to store the result
B = zeros(filterSize*filterSize,numFilters,GPUsingle);

fprintf('Starting computation on GPU\n');
tic;cuRotate180(A,B);GPUsync;toc; 
fprintf('Done computation on GPU\n');

B_h = single(B); %bring back to CPU
B_h = reshape(B_h,filterSize,filterSize,numFilters); %back to 2-d

fprintf('Showing some of CPU result:\n');
Bh(1:3,1:6,1)

fprintf('Showing some of GPU result:\n');
B_h(1:3,1:6,1)

tol=1e-3;
maxabsdiff = max(abs(Bh(:)-B_h(:))); 
fprintf('Max abs difference: %f\n', maxabsdiff );
assert(maxabsdiff<tol);

disp('* Test finished: cuRotate180');

%end