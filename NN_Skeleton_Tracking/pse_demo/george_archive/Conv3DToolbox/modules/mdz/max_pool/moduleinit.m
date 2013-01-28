function moduleinit(load)
% moduleinit Loads the user defined module. 

if(nargin<1)
    load=1; %load
end


[status,major,minor] = cudaGetDeviceMajorMinor(GPUgetActiveDeviceNumber);
cubin = ['cuMaxPool' num2str(major) num2str(minor) '.cubin'];

if(load)
%% check GPUmat version
disp('- Loading module MAXPOOL');
ver = 0.270;
gver = GPUmatVersion;
if (str2num(gver.version)<ver)
  warning(['MODULE EXAMPLES_NUMERICS requires GPUmat version ' num2str(ver) ' or higher. UNABLE TO LOAD MODULE']);
  return;
end


GPUuserModuleLoad('max_pool',['.' filesep cubin])
disp(['  -> ' cubin ]);

else % unload    
    GPUuserModuleUnload('max_pool')
end
    
    
end
  
