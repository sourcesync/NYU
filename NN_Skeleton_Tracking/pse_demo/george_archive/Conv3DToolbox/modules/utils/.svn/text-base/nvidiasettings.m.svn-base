% specify in CUDA_ROOT the directory where CUDA is installed

str = computer('arch');
switch str
  case 'win32'
    %%CUDA_ROOT = 'C:\CUDA';
    CUDA_ROOT = 'I:\CUDA\v4.0';
  case 'win64'
    CUDA_ROOT = 'D:\CUDA\v4.0';
  case 'glnx86'
    CUDA_ROOT = '/usr/local/cuda';
  case 'glnxa64'
    CUDA_ROOT = '/usr/local/pkg/cuda/4.0/cuda';
end


%CUDA_ROOT = 'C:\CUDA';

% SUPPORTED CUDA ARCHITECTURE
CUDA_ARCH = {'10','11','12','13','20','21','22','23','30'};

% check folders
if (~exist(CUDA_ROOT,'dir'))
  error('The specified CUDA_ROOT folder is invalid');
  
end

% check for lib and include
if (~exist([CUDA_ROOT filesep 'lib'],'dir'))
  error('The specified CUDA_ROOT folder is invalid');
end

if (~exist([CUDA_ROOT filesep 'include'],'dir'))
  error('The specified CUDA_ROOT folder is invalid');
end


disp('NVIDIA settings OK');
