function moduleinit
% moduleinit Loads the user defined module. 

%% check GPUmat version
disp('- Loading module CONV');
ver = 0.251;
gver = GPUmatVersion;
if (str2num(gver.version)<ver)
  warning(['MODULE CONV requires GPUmat version ' num2str(ver) ' or higher. UNABLE TO LOAD MODULE']);
  return;
end

[status,major,minor] = cudaGetDeviceMajorMinor(0);
cubin = ['conv' num2str(major) num2str(minor) '.cubin'];
disp(['  -> ' cubin ]);
GPUuserModuleLoad('conv',['.' filesep cubin])

%% check GPUmat version
disp('- Loading module CONV2');
ver = 0.251;
gver = GPUmatVersion;
if (str2num(gver.version)<ver)
  warning(['MODULE CONV2 requires GPUmat version ' num2str(ver) ' or higher. UNABLE TO LOAD MODULE']);
  return;
end

[status,major,minor] = cudaGetDeviceMajorMinor(0);
cubin = ['conv2' num2str(major) num2str(minor) '.cubin'];
disp(['  -> ' cubin ]);
GPUuserModuleLoad('conv2',['.' filesep cubin])

%% check GPUmat version
disp('- Loading module CONV3');
ver = 0.251;
gver = GPUmatVersion;
if (str2num(gver.version)<ver)
  warning(['MODULE CONV3 requires GPUmat version ' num2str(ver) ' or higher. UNABLE TO LOAD MODULE']);
  return;
end

[status,major,minor] = cudaGetDeviceMajorMinor(0);
cubin = ['conv3' num2str(major) num2str(minor) '.cubin'];
disp(['  -> ' cubin ]);
GPUuserModuleLoad('conv3',['.' filesep cubin])

end
  
