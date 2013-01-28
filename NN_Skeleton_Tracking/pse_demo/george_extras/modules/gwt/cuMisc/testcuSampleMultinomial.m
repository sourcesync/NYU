function runme
%% run test
disp('* Start test: cuSampleMultinomial (few)');

multinomials = 4096%128*25*96;
for nomials=[64 128 256 512 1024]
  fprintf('multinomials: %d, nomials: %d\n',multinomials,nomials);
  nomials = 49;

  multih = rand(nomials,multinomials);
  rh = rand(multinomials,1);

  s = sum(multih,1);
  s = s+1; %this takes care of the option of all being "off"

  multih = bsxfun(@rdivide,multih,s); %normalize the input

  fprintf('Starting computation on CPU\n');
  tic;targetsh=samplemultinomial_cpu(multih,rh);GPUsync;toc; 
  fprintf('Done computation on GPU\n');

  multi=GPUsingle(multih);
  r=GPUsingle(rh);

  targets = zeros(nomials,multinomials,GPUsingle);

  fprintf('Starting computation on GPU\n');
  tic;cuSampleMultinomial(multi,r,targets);GPUsync;toc; 
  fprintf('Done computation on GPU\n');

  targets_h=single(targets); %bring back to CPU

  fprintf('Showing some of CPU result:\n');
  targetsh(1:5,1:6)

  fprintf('Showing some of GPU result:\n');
  targets_h(1:5,1:6)

  tol=1e-3;
  diffs=targetsh(:)-targets_h(:);
  d = sum(diffs)/2;
  maxabsdiff = max(abs(targetsh(:)-targets_h(:))); 
  fprintf(['Number of distributions sampled differently: (this may be non-' ...
           'zero but only slightly) %f\n'], d );
  assert(d<tol);

end
disp('* Test finished: cuSampleMultinomial (few)');

disp('* Start test: cuSampleMultinomial (many)');

multinomials = 128*25*96;
for nomials=[4 8 12 16 32 48 64 80 96 112 128 144 160 176 192 208 224 240 ...
            256]
  fprintf('multinomials: %d, nomials: %d\n',multinomials,nomials);
  nomials = 49;

  multih = rand(nomials,multinomials);
  rh = rand(multinomials,1);

  s = sum(multih,1);
  s = s+1; %this takes care of the option of all being "off"

  multih = bsxfun(@rdivide,multih,s); %normalize the input

  fprintf('Starting computation on CPU\n');
  tic;targetsh=samplemultinomial_cpu(multih,rh);GPUsync;toc; 
  fprintf('Done computation on GPU\n');

  multi=GPUsingle(multih);
  r=GPUsingle(rh);

  targets = zeros(nomials,multinomials,GPUsingle);

  fprintf('Starting computation on GPU\n');
  tic;cuSampleMultinomial(multi,r,targets);GPUsync;toc; 
  fprintf('Done computation on GPU\n');

  targets_h=single(targets); %bring back to CPU

  fprintf('Showing some of CPU result:\n');
  targetsh(1:5,1:6)

  fprintf('Showing some of GPU result:\n');
  targets_h(1:5,1:6)

  tol=10;%1e-3;
  diffs=targetsh(:)-targets_h(:);
  d = sum(diffs)/2;
  maxabsdiff = max(abs(targetsh(:)-targets_h(:))); 
  fprintf(['Number of distributions sampled differently: (this may be non-' ...
           'zero but only slightly) %f\n'], d );
  assert(d<tol);

end
disp('* Test finished: cuSampleMultinomial (many)');

%end