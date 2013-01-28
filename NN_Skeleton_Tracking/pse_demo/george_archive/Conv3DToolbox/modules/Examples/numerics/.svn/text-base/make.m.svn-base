function make(target)

make_cpp = 0;
make_cuda = 0;
make_install = 0;
make_debug = 0;

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
  case 'debug'
    make_debug = 1;
  otherwise
    error('Wrong option');
end

arch = computer('arch');
include = ['-I"' fullfile('..','..','include') '"'];

%% flags
if (make_debug)
  flags = '-g ';
else
  flags = '';
end

%% make .cpp files
if (make_cpp)
  
  lib = '';
  
  outdir = '.';
  
  common   = fullfile('..','..','common','GPUmat.cpp');
  
  infiles = {};
  infiles{end+1} = ['myplus.cpp ' common ];
  infiles{end+1} = ['mytimes.cpp ' common ];
  infiles{end+1} = ['myexp.cpp ' common ];
  infiles{end+1} = ['myslice1.cpp ' common ];
  infiles{end+1} = ['myslice2.cpp ' common ];
  % make
  makecpp(infiles, outdir, include, lib, flags);
end

%% make cuda kernels
if (make_cuda)
  makecuda('numerics',include);
end

%% make install
if (make_install)
  % copy mex files
  inpath = '.';
  outpath = fullfile('..','..','..','release',arch,'modules','Examples','numerics');
  
  filesfilters = {['*.' mexext], 'README', '*.cpp','*.cu', '*.cubin', '*.m', 'moduleinit.m'};
  makeinstall(filesfilters, inpath, outpath)
  
end