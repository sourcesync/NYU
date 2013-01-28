function runme
%% run test
disp('* Start cuNCAreg test');

n=4096; %number of cases
m=32; %number of dimensions

Ah = randn(m,n,'single'); %data
Wh = randn(n,n,'single'); %weights
fprintf('Starting computation on CPU\n');

targetsh = zeros(m,n,'single');
tic;
  % %Brute force version
  % for kk=1:n
  %   for jj=1:n
  %     targetsh(:,kk) = targetsh(:,kk) + (Ah(:,kk)-Ah(:,jj)) * ...
  %         Wh(jj,kk);
  %   end
  % end
  
  % More clever version, using bsxfun
  for dd=1:m
    targetsh(dd,:)=sum(Wh.*bsxfun(@minus,Ah(dd,:),Ah(dd,:)'),1);
  end
toc;
fprintf('Done computation on CPU\n');

%Note that cuNCAreg (like my other distance-based functions)
%has been written specifically with Matlab-style COLUMN-major format in mind
A = GPUsingle(Ah);
W = GPUsingle(Wh);
%targets = zeros(m, n, GPUsingle);

%We get coalesced writes if cases are sequential in memory So it is faster
%to compute this way (cases as rows, dims as cols) and transpose afterward
targets = zeros(n,m, GPUsingle);

tic;
cuNCAreg(A,W,targets)
targets=transpose(targets);
GPUsync;toc;
fprintf('Done convolution on GPU\n');

targets_h = single(targets); %move to CPU 

fprintf('Showing some of CPU result:\n');
targetsh(1:3,1:6,1)

fprintf('Showing some of GPU result:\n');
targets_h(1:3,1:6,1)

tol=1e-3;
maxabsdiff = max(abs(targetsh(:)-targets_h(:))); 
fprintf('Max abs difference: %f\n', maxabsdiff );
assert(maxabsdiff<tol);

disp('* Test finished');


%end