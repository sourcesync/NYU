% Use GPUmat compiler function to automatically generated loopmult_g
% which will matrix-multiply slices of A and B in parallel
function runme
%% run test
disp('* Start test');

numCases = 1024;
numA = 512;

Ah = rand(numA,numA, numCases,'single');
Bh = rand(numA,numCases,'single');

fprintf('Starting computation on CPU\n');
tic;targetsH = loopmult_m(Ah,Bh);toc;
fprintf('Done computation on CPU\n');

%Unfortunately just like mxArrays, GPUsingle type stores data in
%COLUMN-MAJOR format
%Since all of Alex's convolutional code assumes ROW-MAJOR format we take
%the transpose of the data before making GPUsingles
%this ensures the data is in ROW-MAJOR form (i.e. cases are consecutive,
%within each case rows are consecutive, etc.)
A = GPUsingle(Ah);
B = GPUsingle(Bh);

targets = zeros(numA, numCases, GPUsingle);
temp = zeros(numA, 1, GPUsingle); % hold intermediate result

% set up compiled function 'loopmult_g'
% looping will be done on GPU
GPUcompileStart('loopmult_g', '-f', A, B, temp, targets, numCases)
GPUfor it=1:numCases
  GPUmtimes(slice(A, ':', ':', it), slice(B, ':', it), temp)  
  assign(1, targets, temp, ':', it)
GPUend
GPUcompileStop

fprintf('Starting computation on GPU\n')
tic;
loopmult_g(A, B, temp, targets, numCases)
GPUsync;
toc

fprintf('Done computation on GPU\n');

targets_h = single(targets); %move to CPU 

fprintf('Showing some of CPU result:\n');
targetsH(1:3,1:6)

fprintf('Showing some of GPU result:\n');
targets_h(1:3,1:6)

tol=1e-3;
maxabsdiff = max(abs(targetsH(:)-targets_h(:))); 
fprintf('Max abs difference: %f\n', maxabsdiff );
assert(maxabsdiff<tol);

fprintf('* Test finished\n');

end