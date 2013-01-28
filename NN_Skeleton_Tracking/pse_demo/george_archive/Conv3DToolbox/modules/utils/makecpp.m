function makecpp(infiles, outdir, include, lib, flags)

% load nvidia settings (CUDA_ROOT, CC_BIN)
nvidiasettings;

disp('**** MAKE CPP ****');
% Libraries and includes for CUDA
cudainclude = [' -I"' fullfile(CUDA_ROOT, 'include') '"'];
allinclude = [cudainclude ' ' include];
switch computer
  case {'PCWIN64'}
    libfolder = 'lib\x64';
  case {'PCWIN'}
    libfolder = 'lib\Win32';
  case {'GLNXA64'}
    libfolder = 'lib64';
  otherwise
    libfolder = 'lib';
end
% fullfile(CUDA_ROOT, libfolder)
cudalib  = [' -L"' fullfile(CUDA_ROOT, libfolder) '" -lcuda -lcudart -lcufft -lcublas'];
alllib = [cudalib ' ' lib];

% flags for mex compilation
if (isunix)
  mexflags = [flags ' -DUNIX'];
else
  mexflags = flags;
end


%% Build
for i=1:length(infiles)
  cmd = ['mex ' mexflags ' -outdir ' outdir ' ' infiles{i} ' ' allinclude ' ' alllib ];
  disp(cmd);
  eval(cmd);
end


end
