function runme
%% run test
disp('* Start cuDist test: 1 input');

n=1024; %number of cases
m=64; %number of dimensions

Ah = randn(n,m,'single');

fprintf('Starting computation on CPU\n');
tic;Dh = distmat1(Ah).^2;toc;
fprintf('Done computation on CPU\n');

%Unfortunately just like mxArrays, GPUsingle type stores data in
%COLUMN-MAJOR format
%Since the kernel was written with row-major format in mind
%We transpose the data before making GPUsingles
%this ensures the data is in ROW-MAJOR form (i.e. cases are consecutive,
%within each case rows are consecutive, etc.)
A = GPUsingle(transpose(Ah));
D = zeros(n, n, GPUsingle);

tic;cuSquaredDist(A,D);GPUsync;toc;
fprintf('Done convolution on GPU\n');

D_h = single(D); %move to CPU 

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

tic;cuDist(A,B,D);GPUsync;toc;
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