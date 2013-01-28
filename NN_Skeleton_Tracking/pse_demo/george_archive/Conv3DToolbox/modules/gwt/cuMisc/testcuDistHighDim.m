function runme
% Because of the shared memory used by the cuDist kernel
% We can't use a single kernel call for > 4096 dims (4088 I found in
% practice) since the size of the shared memory needed is 4 bytes * number
% of dimensions and there is shared memory of 16,384 bytes/block
% However, since the Euclidean distance is just the sqrt of a dot product,
% we can just break up the dot product into multiple kernels (per blocks of
% dimensions)
% Here we demonstrate this use
%% run test
disp('* Start cuDist test: 1 input');

n=1024; %number of cases
m=256; %number of dimensions

maxDim=64; %split

Ah = randn(n,m,'single');

fprintf('Starting computation on CPU\n');
tic;Dh = distmat1(Ah);toc;
fprintf('Done computation on CPU\n');

% %Demonstrates how to do splitting on CPU
% Dh_s = zeros(n,n);
% for ii=1:ceil(m/maxDim) %loop over each split
%   startDim = (ii-1)*maxDim+1;
%   endDim = min(ii*maxDim,m);
%   %distmat1 returns dist not squared dist
%   %but we want to take square root after all dims have been factored in
%   Dh_s = Dh_s + (distmat1(Ah(:,startDim:endDim))).^2;
% end
% Dh_s = sqrt(Dh_s);

%Unfortunately just like mxArrays, GPUsingle type stores data in
%COLUMN-MAJOR format
%Since the kernel was written with row-major format in mind
%We transpose the data before making GPUsingles
%this ensures the data is in ROW-MAJOR form (i.e. cases are consecutive,
%within each case rows are consecutive, etc.)
A = GPUsingle(transpose(Ah));
D = zeros(n, n, GPUsingle); 

tic;
for ii=1:ceil(m/maxDim) %loop over each split
  startDim = (ii-1)*maxDim+1;
  endDim = min(ii*maxDim,m);
  cuSquaredDist(A(startDim:endDim,:),D); %note that D accumulates
end
GPUsqrt(D,D); %NOW apply the sqrt (after the sum)
GPUsync;toc
fprintf('Done convolution on GPU\n');
D_h = single(D);
  
fprintf('Showing some of CPU result:\n');
Dh(1:3,1:6,1)

fprintf('Showing some of GPU result:\n');
D_h(1:3,1:6,1)

tol=1e-3;
maxabsdiff = max(abs(Dh(:)-D_h(:))); 
fprintf('Max abs difference: %f\n', maxabsdiff );
assert(maxabsdiff<tol);

disp('* Test finished: 1 input');

%% run test
disp('* Start cuDist test: 2 inputs');

%note threads are over n2 so matrix B should be the bigger matrix
n1= 512; %number of cases (matrix 1)
n2 = 2048; %number of cases (matrix 2)
m=64; %number of dimensions

Ah = randn(n1,m,'single');
Bh = randn(n2,m,'single');

fprintf('Starting computation on CPU\n');
tic;Dh = distmat1(Ah,Bh);toc;
fprintf('Done computation on CPU\n');

%Unfortunately just like mxArrays, GPUsingle type stores data in
%COLUMN-MAJOR format
%Since the kernel was written with row-major format in mind
%We transpose the data before making GPUsingles
%this ensures the data is in ROW-MAJOR form (i.e. cases are consecutive,
%within each case rows are consecutive, etc.)
A = GPUsingle(transpose(Ah));
B = GPUsingle(transpose(Bh));
%note the output is also transposed
D = zeros(n2, n1, GPUsingle);

%tic;cuDist(A,B,D);GPUsync;toc;
tic;
for ii=1:ceil(m/maxDim) %loop over each split
  startDim = (ii-1)*maxDim+1;
  endDim = min(ii*maxDim,m);
  cuSquaredDist(A(startDim:endDim,:),B(startDim:endDim,:),D);
end
GPUsqrt(D,D); %NOW apply the sqrt (after the sum)
GPUsync;toc

fprintf('Done convolution on GPU\n');

D_h = transpose(single(D)); %move to CPU 

fprintf('Showing some of CPU result:\n');
Dh(1:3,1:6,1)

fprintf('Showing some of GPU result:\n');
D_h(1:3,1:6,1)

tol=1e-3;
maxabsdiff = max(abs(Dh(:)-D_h(:))); 
fprintf('Max abs difference: %f\n', maxabsdiff );
assert(maxabsdiff<tol);

disp('* Test finished: 2 inputs');

%end