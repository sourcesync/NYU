function runme
%% run test
disp('* Start test');

numCases = 10;
numA = 256;
numB = 256;
numC = 256;

Ah = rand(numA,numCases,'single');
Bh = rand(numB,numCases,'single');
Ch = rand(numC,numCases,'single');

fprintf('Starting computation on CPU\n');
tic;targetsH = threeway_m(Ah,Bh,Ch);toc;
fprintf('Done computation on CPU\n');

%Unfortunately just like mxArrays, GPUsingle type stores data in
%COLUMN-MAJOR format
%Since all of Alex's convolutional code assumes ROW-MAJOR format we take
%the transpose of the data before making GPUsingles
%this ensures the data is in ROW-MAJOR form (i.e. cases are consecutive,
%within each case rows are consecutive, etc.)
A = GPUsingle(Ah);
B = GPUsingle(Bh);
C = GPUsingle(Ch);
targets = zeros(numA,numB,numC, GPUsingle);

tic;cuThreeway(A,B,C,targets);GPUsync;toc;
fprintf('Done convolution on GPU\n');

targets_h = single(targets); %move to CPU 

fprintf('Showing some of CPU result:\n');
targetsH(1:3,1:6,1)

fprintf('Showing some of GPU result:\n');
targets_h(1:3,1:6,1)

tol=1e-3;
maxabsdiff = max(abs(targetsH(:)-targets_h(:))); 
fprintf('Max abs difference: %f\n', maxabsdiff );
assert(maxabsdiff<tol);

disp('* Test finished');

end