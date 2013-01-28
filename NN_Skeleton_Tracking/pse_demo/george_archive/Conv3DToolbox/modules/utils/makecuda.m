function makecuda(base, include)

% load nvidia settings (CUDA_ROOT, CC_BIN)
nvidiasettings;

disp('**** MAKE CUDA ****');

%% Build kernel mykernel.cubin
disp('Building CUDA Kernel');
disp('* Calling nvcc');

%nvidiacmd = ['nvcc -arch sm_10 -maxrregcount=32 ' include ' -m32 -cubin -o "numerics.cubin" "numerics.cu"'];

%base = 'numerics';
arch = CUDA_ARCH;
for i=1:length(arch)
  outputfile = ['".' filesep base arch{i} '.cubin"'];
  inputfile  = ['".' filesep base '.cu"'];
  
  clinclude = '';
  switch computer
    case {'PCWIN64'}
      machine = '-m64';
      clinclude = locateCL;
    case {'PCWIN'}
      machine = '-m32';
    case {'GLNXA64'}
      machine = '-m64';
    otherwise
      machine = '-m32';
  end
  nvidiacmd = ['nvcc -arch sm_' arch{i} ' ' clinclude ' -maxrregcount=32 ' include ' ' machine ' -cubin -o  ' outputfile ' ' inputfile];
  
  disp(nvidiacmd);
  system(nvidiacmd);
end


%disp(nvidiacmd);
%system(nvidiacmd);

end

function y = splitPath(path)
if (ispc)
  c = ';';
end
if (isunix)
  c = ':';
end

y={};
[t,r] = strtok(path,c);
y{end+1} = t;
while ~isempty(r)
  [t,r] = strtok(r,c);
  y{end+1} = t;
end


end

function y=locateCL
path = getenv('PATH');


pathCell = splitPath(path);
CL_INCLUDE = '';

for i=1:length(pathCell)
  pi = pathCell{i};
  % check for CUDA dlls
  if (exist(fullfile(pi,'cl.exe'),'file'))
    if (exist(fullfile(pi,'..','include'),'dir'))
        CL_INCLUDE = [' -I"' fullfile(pi,'..','include') '" '];
    end
  end
  
end

y = CL_INCLUDE;

end



