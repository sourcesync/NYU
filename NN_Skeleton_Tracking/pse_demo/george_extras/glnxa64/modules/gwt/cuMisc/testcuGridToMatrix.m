function runme
%% run test
disp('* Start test: cuGridToMatrix');
imgBaseSize=32;
numCases = 128;

%only factors 2:16 are supported
%image must be square, smaller than 512x512 and a multiple of factor
for factor=2:16
  imgSize = imgBaseSize*factor;
  regionsPerImage = (imgSize/factor) * (imgSize/factor);
  
  fprintf('imgSize: %dx%d factor: %d\n',imgSize,imgSize,factor);

  Ah = randn(imgSize*imgSize,numCases);

%fprintf('Some data before copy :\n');
%Ah(1:3,1:3,1)

fprintf('Starting computation on CPU\n');
tic;Bh = gridToMatrix_cpu(Ah, factor);toc;
fprintf('Done computation on CPU\n');

A = GPUsingle(Ah); 

%to store the result
B = zeros(factor*factor,numCases*regionsPerImage,GPUsingle); 

fprintf('Starting computation on GPU\n');
tic;cuGridToMatrix(A,B,factor);GPUsync;toc; 
fprintf('Done computation on GPU\n');

B_h = single(B); %bring back to CPU

fprintf('Showing some of CPU result:\n');
Bh(1:min(3,factor),1:6)

fprintf('Showing some of GPU result:\n');
B_h(1:min(3,factor),1:6)

tol=1e-3;
maxabsdiff = max(abs(Bh(:)-B_h(:))); 
fprintf('Max abs difference: %f\n', maxabsdiff );
assert(maxabsdiff<tol);

end

disp('* Test finished: cuGridToMatrix');

%end