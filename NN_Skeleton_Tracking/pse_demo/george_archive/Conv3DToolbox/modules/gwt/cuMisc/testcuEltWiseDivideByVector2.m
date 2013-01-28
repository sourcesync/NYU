%function runme
%% run test
disp('* Start cuEltWiseDivideByVector2 row vector test *');

% try row vector divide
m=128; 
n=1024; 

Ah = randn(m,n,'single'); %data
vh = randn(1,n,'single'); %vec

fprintf('Starting computation on CPU\n');
tic;targetsH = bsxfun(@rdivide,Ah,vh);
fprintf('Done computation on CPU: %fs\n',toc);

%Note that cuNCAreg (like my other distance-based functions)
%has been written specifically with Matlab-style COLUMN-major format in mind
A = GPUsingle(Ah);
v = GPUsingle(vh);

targets = zeros(m,n, GPUsingle);

tic;
cuEltWiseDivideByVector2(A,v,targets)
GPUsync;
fprintf('Done convolution on GPU: %fs\n',toc);

targets_h = single(targets); %move to CPU 

fprintf('Showing some of CPU result:\n');
targetsH(1:3,1:6,1)

fprintf('Showing some of GPU result:\n');
targets_h(1:3,1:6,1)

tol=1e-3;
maxabsdiff = max(abs(targetsH(:)-targets_h(:))); 
fprintf('Max abs difference: %f\n', maxabsdiff );
assert(maxabsdiff<tol);

disp('* Test finished *');
disp('* Start cuEltWiseDivideByVector2 row vector test *');
%try col vctor divide

Ah = randn(m,n,'single'); %data
vh = randn(m,1,'single'); %vec

fprintf('Starting computation on CPU\n');
tic;targetsH = bsxfun(@rdivide,Ah,vh);
fprintf('Done computation on CPU: %fs\n',toc);

%Note that cuNCAreg (like my other distance-based functions)
%has been written specifically with Matlab-style COLUMN-major format in mind
A = GPUsingle(Ah);
v = GPUsingle(vh);

targets = zeros(m,n, GPUsingle);

tic;
cuEltWiseDivideByVector2(A,v,targets)
GPUsync;
fprintf('Done convolution on GPU: %fs\n',toc);

targets_h = single(targets); %move to CPU 

fprintf('Showing some of CPU result:\n');
targetsH(1:3,1:6,1)

fprintf('Showing some of GPU result:\n');
targets_h(1:3,1:6,1)

tol=1e-3;
maxabsdiff = max(abs(targetsH(:)-targets_h(:))); 
fprintf('Max abs difference: %f\n', maxabsdiff );
assert(maxabsdiff<tol);

disp('* Test finished *');

%end