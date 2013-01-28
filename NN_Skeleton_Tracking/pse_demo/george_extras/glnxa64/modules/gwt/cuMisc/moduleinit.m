function moduleinit
% moduleinit Loads the user defined module. 

%% check GPUmat version
disp('- Loading module MISC');
ver = 0.251;
gver = GPUmatVersion;
if (str2num(gver.version)<ver)
  warning(['MODULE MISC requires GPUmat version ' num2str(ver) ' or higher. UNABLE TO LOAD MODULE']);
  return;
end

[status,major,minor] = cudaGetDeviceMajorMinor(0);
cubin = ['misc' num2str(major) num2str(minor) '.cubin'];
disp(['  -> ' cubin ]);
GPUuserModuleLoad('misc',['.' filesep cubin])

%% check GPUmat version
disp('- Loading module SUBSAMPLE');
ver = 0.251;
gver = GPUmatVersion;
if (str2num(gver.version)<ver)
  warning(['MODULE SUBSAMPLE requires GPUmat version ' num2str(ver) ' or higher. UNABLE TO LOAD MODULE']);
  return;
end

[status,major,minor] = cudaGetDeviceMajorMinor(0);
cubin = ['subsample' num2str(major) num2str(minor) '.cubin'];
disp(['  -> ' cubin ]);
GPUuserModuleLoad('subsample',['.' filesep cubin])

end
  
