% function testnvMax()
disp('* Start nvMax test *');

% data
m=1024; 
n=512; 

Ah = randn(m,n,'single');

%test both row and column max
for axis=1:2
  tic; targetsH = max(Ah,[],axis);
  fprintf('Done computation on CPU: %fs\n',toc);
  
  A = GPUsingle(Ah);
  if axis==1
    targets = zeros(1,n,GPUsingle);
  else
    targets = zeros(m,1,GPUsingle);
  end
  
  tic;
nvMax(A,axis,targets)
GPUsync;
fprintf('Done convolution on GPU: %fs\n',toc);

targets_h = single(targets); %move to CPU 

fprintf('Showing some of CPU result:\n');
if axis==1
  targetsH(1,1:6)
else
  targetsH(1:6,1)
end
fprintf('Showing some of GPU result:\n');
if axis==1
  targets_h(1,1:6)
else
  targets_h(1:6,1)
end
  
% targets_h
% targetsH
tol=1e-3;
maxabsdiff = max(abs(targetsH(:)-targets_h(:))); 
fprintf('Max abs difference: %f\n', maxabsdiff );
% assert(maxabsdiff<tol);

end

disp('* Test finished *');


