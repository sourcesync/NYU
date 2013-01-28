function moduleinit(load)
% moduleinit Loads the user defined module. 

if(nargin<1)
    load=1; %load
end

[status,major,minor] = cudaGetDeviceMajorMinor(0);
cubin = ['conv4' num2str(major) num2str(minor) '.cubin'];
cubin2 = ['conv5' num2str(major) num2str(minor) '.cubin'];
cubin3 = ['conv6' num2str(major) num2str(minor) '.cubin'];
cubin4 = ['conv7' num2str(major) num2str(minor) '.cubin'];
cubin5 = ['conv8' num2str(major) num2str(minor) '.cubin'];

if(load)

%% check GPUmat version
disp('- Loading module CONV4');
ver = 0.251;
gver = GPUmatVersion;
if (str2num(gver.version)<ver)
  warning(['MODULE CONV4 requires GPUmat version ' num2str(ver) ' or higher. UNABLE TO LOAD MODULE']);
  return;
end
GPUuserModuleLoad('conv4',['.' filesep cubin])
disp(['  -> ' cubin ]);


%% check GPUmat version
disp('- Loading module CONV5');
ver = 0.251;
gver = GPUmatVersion;
if (str2num(gver.version)<ver)
  warning(['MODULE CONV5 requires GPUmat version ' num2str(ver) ' or higher. UNABLE TO LOAD MODULE']);
  return;
end
GPUuserModuleLoad('conv5',['.' filesep cubin2])
disp(['  -> ' cubin2 ]);


%% check GPUmat version
disp('- Loading module CONV6');
ver = 0.251;
gver = GPUmatVersion;
if (str2num(gver.version)<ver)
  warning(['MODULE CONV6 requires GPUmat version ' num2str(ver) ' or higher. UNABLE TO LOAD MODULE']);
  return;
end
GPUuserModuleLoad('conv6',['.' filesep cubin3])
disp(['  -> ' cubin3 ]);




%% check GPUmat version
disp('- Loading module CONV7');
ver = 0.251;
gver = GPUmatVersion;
if (str2num(gver.version)<ver)
  warning(['MODULE CONV7 requires GPUmat version ' num2str(ver) ' or higher. UNABLE TO LOAD MODULE']);
  return;
end
GPUuserModuleLoad('conv7',['.' filesep cubin4])
disp(['  -> ' cubin4 ]);



%% check GPUmat version
disp('- Loading module CONV8');
ver = 0.251;
gver = GPUmatVersion;
if (str2num(gver.version)<ver)
  warning(['MODULE CONV8 requires GPUmat version ' num2str(ver) ' or higher. UNABLE TO LOAD MODULE']);
  return;
end
GPUuserModuleLoad('conv8',['.' filesep cubin5])
disp(['  -> ' cubin5 ]);



else % unload
        GPUuserModuleUnload('conv4')
        GPUuserModuleUnload('conv5')
        GPUuserModuleUnload('conv6')
        GPUuserModuleUnload('conv7')    
        GPUuserModuleUnload('conv8')    
end

end
  
