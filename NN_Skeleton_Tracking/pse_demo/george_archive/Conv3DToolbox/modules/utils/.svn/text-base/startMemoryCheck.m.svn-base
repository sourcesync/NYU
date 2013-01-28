function startMemoryCheck()

global GPUtest

try
  type = GPUtest.type;
catch
  error('GPUtest not initialized. Please use GPUtestInit.');
end


if (GPUtest.checkPointers==1)
  GPUmem;
  GPUsync;
  delete(fullfile('.','gpufree.dat'));
  delete(fullfile('.','gpumalloc.dat'));
  delete(fullfile('.','free.dat'));
  delete(fullfile('.','malloc.dat'));
  delete(fullfile('.','GPUtype.dat'));
  delete(fullfile('.','GPUtypedelete.dat'));
  delete(fullfile('.','GPUmanagerdelete.dat'));
end

