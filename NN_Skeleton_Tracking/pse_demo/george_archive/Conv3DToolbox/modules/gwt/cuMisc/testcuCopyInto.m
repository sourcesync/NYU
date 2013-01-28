function runme
%% run test
disp('* Start test: cuCopyInto');
imgSize=128;
numCases = 256;
paddingSize = 16;
Ah = randn(imgSize,imgSize,numCases);
Bh = zeros(imgSize+2*paddingSize,imgSize+2*paddingSize,numCases); %result

fprintf('Some data before copy :\n');
Ah(1:3,1:3,1)

fprintf('Starting computation on CPU\n');
tic;
Bh(paddingSize+1:end-paddingSize,paddingSize+1:end-paddingSize,:) = Ah;
toc;
fprintf('Done computation on CPU\n');

Ah = reshape(Ah,imgSize*imgSize,numCases); %collapse to vectors
A = GPUsingle(Ah); 

%to store the result
B = zeros((imgSize+2*paddingSize)*(imgSize+2*paddingSize),numCases,GPUsingle); 

fprintf('Starting computation on GPU\n');
tic;cuCopyInto(A,B,paddingSize);GPUsync;toc; 
fprintf('Done computation on GPU\n');

B_h = single(B); %bring back to CPU
B_h = reshape(B_h,imgSize+2*paddingSize, ...
              imgSize+2*paddingSize,numCases); %back to 2-d

fprintf('Showing some of CPU result (around image):\n');
Bh(paddingSize-1:paddingSize+2,paddingSize-1:paddingSize+4,1)

fprintf('Showing some of GPU result (around image):\n');
B_h(paddingSize-1:paddingSize+2,paddingSize-1:paddingSize+4,1)

tol=1e-3;
maxabsdiff = max(abs(Bh(:)-B_h(:))); 
fprintf('Max abs difference: %f\n', maxabsdiff );
assert(maxabsdiff<tol);

disp('* Test finished: cuCopyInto');

%end