function make(target)

make_cpp = 0;
make_cuda = 0;
make_install = 0;

switch target
  case 'all'
    make_cpp = 1;
    make_cuda = 1;
    make_install = 1;
    
  case 'cpp'
    make_cpp = 1;
  case 'cuda'
    make_cuda = 1;
  case 'install'
    make_install = 1;
  otherwise 
    error('Wrong option');
end

arch = computer('arch');
include = ['-I"' fullfile('..','include') '"'];

%% make .cpp files
if (make_cpp)

end

%% make cuda kernels
if (make_cuda)

end

%% make install
if (make_install)
  % copy mex files
  inpath = '.';
  outpath = fullfile('..','release',arch,'modules','utils');
  
  filesfilter = {'*.m'};
  makeinstall(filesfilter, inpath, outpath) 
   
end










end
