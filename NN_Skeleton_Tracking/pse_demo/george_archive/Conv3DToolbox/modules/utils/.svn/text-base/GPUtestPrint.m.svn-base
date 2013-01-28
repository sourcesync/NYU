function GPUtestPrint
% GPUtestPrint Print GPUtest variable

global GPUtest

try
  type = GPUtest.type;
catch
  error('GPUtest not initialized. Please use GPUtestInit.');
end

%% Print configuration
disp(['* LOG file -> ' GPUtest.logFile]);

switch GPUtest.type
  case 1
    disp('* REAL/COMPLEX test ');
  case 2
    disp('* REAL test ');
  case 3
    disp('* COMPLEX test ');
    
end

disp('* Testing the following classes:');
for i=1:length(GPUtest.gpufun)
  disp(['   ' GPUtest.txtfun{i} ]);
  
end



disp(['* SINGLE tol -> ' num2str(GPUtest.tol.single)]);
disp(['* DOUBLE tol -> ' num2str(GPUtest.tol.double)]);

if (GPUtest.noCompareZeros==1)
  disp('* Error when comparing zeros');
else
  disp('* NO Error when comparing zeros');
end

if (GPUtest.stopOnError==1)
  disp('* STOP on error');
else
  disp('* NO stop on error');
end

if (GPUtest.fastMode==1)
  disp('* FAST MODE');
else
  disp('* NO fast mode');
end

if (GPUtest.memLeak==1)
  disp('* MEMORY LEAK CHECK');
else
  disp('* NO MEMORY LEAK CHECK');
end

if (GPUtest.checkPointers==1)
  disp('* POINTERS CHECK');
else
  disp('* NO POINTERS CHECK');
end

if (GPUtest.checkCompiler==1)
  disp('* COMPILER CHECK');
else
  disp('* NO COMPILER CHECK');
end

if (GPUtest.bigKernel==1)
  disp('* BIG KERNEL TEST');
else
  disp('* NO BIG KERNEL TEST');
end




