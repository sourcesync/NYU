function make(target,nvmex)


if(nargin<2)
    nvmex = 0;
end


if(nvmex) % Compile using nvmex.
    cmd = '!MATLAB=/opt/pkg/matlab/current ./nvmex -f nvopts.sh  -DUNIX -outdir . cuMaxPool.cu cuMaxPool_kernels.cu ../../common/GPUmat.cpp  -I"/usr/local/pkg/cuda/3.2/cuda/include" -I"../../include"  -L"/usr/local/pkg/cuda/3.2/cuda/lib64" -lcuda -lcudart -lcufft -lcublas';
    disp(cmd);
    eval(cmd)
    

else % Compile as GPUmat suggests.
    
    
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
%         infiles{end+1} = ['cuMaxPool.cpp ' common ];
%         infiles{end+1} = ['cuRevMaxPool.cpp ' common ];
% %         infiles{end+1} = ['cuMaxPool3d.cpp ' common ];
%         infiles{end+1} = ['cuRevMaxPool3d.cpp ' common ];
        infiles{end+1} = ['cuShrink.cpp ' common ];
%         infiles{end+1} = ['./@GPUtype/cuMax.cpp ' common ];
        %   infiles{end+1} = ['mytimes.cpp ' common ];
        %   infiles{end+1} = ['myexp.cpp ' common ];
        %   infiles{end+1} = ['myslice1.cpp ' common ];
        %   infiles{end+1} = ['myslice2.cpp ' common ];
        % make
        makecpp(infiles, outdir, include, lib, flags);
        
        % Make and install the 
        infiles = {};
        infiles{end+1} = ['./@GPUtype/max.cpp ' common ];
        infiles{end+1} = ['./@GPUtype/min.cpp ' common ];
        infiles{end+1} = ['./@GPUtype/sign.cpp ' common ];
        infiles{end+1} = ['./@GPUtype/padarray.cpp ' common ];
        makecpp(infiles, './@GPUtype/', include, lib, flags);
%         makecpp(infiles, './@GPUtype/', include, lib, flags);

    end
    
    %% make cuda kernels
    if (make_cuda)
        makecuda('cuOther',include);
    end
    
    % make install the max function into the 
    if (make_install)
      % copy mex files
      inpath = './@GPUtype/';
      outpath = fullfile('..','..','..','@GPUtype');
    
%       filesfilters = {['*.' mexext], 'README', '*.cpp','*.cu', '*.cubin', '*.m', 'moduleinit.m'};

      filesfilters = {['max.' mexext],['max.m'],['min.' mexext],['min.m'],['sign.' mexext],['sign.m'],...
          ['norm.m'],['dot.m'],['padarray.' mexext],['padarray.m']};
      makeinstall(filesfilters, inpath, outpath)
    
    end
    
    
end