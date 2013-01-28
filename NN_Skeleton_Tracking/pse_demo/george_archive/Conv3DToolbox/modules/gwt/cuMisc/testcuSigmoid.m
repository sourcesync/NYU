function runme
%% run test


disp('* Start test: cuSigmoid');
m=512;
n=512;
Ah = randn(m,n);

fprintf('Some data before sigmoid :\n');
Ah(1:3,1:6)


fprintf('Starting computation on CPU\n');
tic;Bh = 1./(1+exp(-Ah));toc;
fprintf('Done computation on CPU\n');


A = GPUsingle(Ah); %uniform random data

B = zeros(m,n,GPUsingle); %to store the result

fprintf('Starting computation on GPU\n');
tic;cuSigmoid(A,B);GPUsync;toc; %randomly sample each unit
fprintf('Done computation on GPU\n');

B_h = single(B); %bring back to CPU

fprintf('Showing some of CPU result:\n');
Bh(1:3,1:6,1)

fprintf('Showing some of GPU result:\n');
B_h(1:3,1:6,1)

tol=1e-3;
maxabsdiff = max(abs(Bh(:)-B_h(:))); 
fprintf('Max abs difference: %f\n', maxabsdiff );
assert(maxabsdiff<tol);

disp('* Test finished: cuSigmoid');

%end