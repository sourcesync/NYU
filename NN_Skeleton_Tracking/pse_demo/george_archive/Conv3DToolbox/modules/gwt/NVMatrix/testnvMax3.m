%function testnvMax()
disp('* Start nvMax test *');

% data
m=1024; 
n=512; 

Ah = randn(m,n,'single');

%only working right now for row max
for axis=2
  tic; 
  [targetsH,argmaxH] = max(abs(Ah),[],axis);
  %fix sign
  if axis==1
    targetsH = sign(Ah(sub2ind([m n],argmaxH',1:n))).*targetsH;
  else
    targetsH = sign(Ah(sub2ind([m n],1:m,argmaxH')))'.*targetsH;
  end

  fprintf('Done computation on CPU: %fs\n',toc);
  
  A = GPUsingle(Ah);
  if axis==1
    targets = zeros(1,n,GPUsingle);
    argmax = zeros(1,n,GPUsingle);
  else
    targets = zeros(m,1,GPUsingle);
    argmax = zeros(m,1,GPUsingle);
  end
  
  tic;
nvMax3(A,axis,targets,argmax)
GPUsync;
fprintf('Done convolution on GPU: %fs\n',toc);

targets_h = single(targets); %move to CPU 
argmax_h = single(argmax); %move to CPU 

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

fprintf('Showing some of CPU result:\n');
if axis==1
  %argmaxH(1,1:6)
else
  argmaxH(1:6,1)
end
fprintf('Showing some of GPU result:\n');
if axis==1
  %argmax_h(1,1:6)
else
  argmax_h(1:6,1)
end


tol=1e-3;
maxabsdiff = max(abs(targetsH(:)-targets_h(:))); 
fprintf('Max abs difference (max): %f\n', maxabsdiff );
assert(maxabsdiff<tol);

maxabsdiff = max(abs(argmaxH(:)-argmax_h(:))); 
fprintf('Max abs difference (argmax): %f\n', maxabsdiff );
assert(maxabsdiff<tol);


end

disp('* Test finished *');


