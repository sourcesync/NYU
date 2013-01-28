%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% This does a whos of the Matlab workspace then moves all the variables that
% are (GPUmat variables) GPUsingle to singles and GPUdouble to doubles on CPU.
% The variable left int he workspace calls AllGPUvars has the name and type of
% each GPUmat variable that was converted. Use this with moveAll2GPU to
% move these back (after a save for example).
%
% @file
% @author Matthew Zeiler
% @date May 19, 2011
%
% @gpu_file @copybrief moveAll2CPU.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Allvars = whos;
fprintf('Moving: ');
% If this is the second time you're calling it then append the variables to AllGPUvars.
if(exist('AllGPUvars','var'))
    gpucount = length(AllGPUvars);
else
    gpucount = 1;
end
for ii=1:numel(Allvars)
    eval(sprintf('GPUsing = isa(%s,''GPUsingle'');',Allvars(ii).name))
    if(GPUsing)
        fprintf('%s, ',Allvars(ii).name);
        % Save these names for moving back to GPU.
        AllGPUvars(gpucount).name = Allvars(ii).name;
        AllGPUvars(gpucount).type = 'GPUsingle';
        gpucount = gpucount+1;
        eval(sprintf('%s = single(%s);',Allvars(ii).name,Allvars(ii).name));
    end
    eval(sprintf('GPUdoub = isa(%s,''GPUdouble'');',Allvars(ii).name))
    if(GPUdoub)
        fprintf('%s, ',Allvars(ii).name);
        % Save these names for moving back to GPU.
        AllGPUvars(gpucount).name = Allvars(ii).name;
        AllGPUvars(gpucount).type = 'GPUdouble';
        gpucount = gpucount+1;
        eval(sprintf('%s = double(%s);',Allvars(ii).name,Allvars(ii).name));
    end
    
    eval(sprintf('GPUcell = isa(%s,''cell'');',Allvars(ii).name))
    if(GPUcell)
        eval(sprintf('GPUcell = isa(%s{1},''GPUsingle'') || isa(%s{1},''GPUdouble'');',Allvars(ii).name,Allvars(ii).name))
        if(GPUcell)
            fprintf('%s, ',Allvars(ii).name);
            AllGPUvars(gpucount).name = Allvars(ii).name;
            AllGPUvars(gpucount).type = 'GPUcell';
            eval(sprintf('cellLength = length(%s);',Allvars(ii).name))
            AllGPUvars(gpucount).cellLength = cellLength;
            AllGPUvars(gpucount).celltype = cell(1,cellLength);
            for cc=1:cellLength
                eval(sprintf('GPUsing = isa(%s{%d},''GPUsingle'');',Allvars(ii).name,cc))
                if(GPUsing)
                    AllGPUvars(gpucount).celltype{cc} = 'GPUsingle';
                    eval(sprintf('%s{%d} = single(%s{%d});',Allvars(ii).name,cc,Allvars(ii).name,cc));
                end
                eval(sprintf('GPUdoub = isa(%s{%d},''GPUdouble'');',Allvars(ii).name,cc))
                if(GPUdoub)
                    AllGPUvars(gpucount).celltype{cc} = 'GPUdouble';
                    eval(sprintf('%s = double(%s);',Allvars(ii).name,cc,Allvars(ii).name,cc));
                end
            end
            gpucount = gpucount+1;
        end
    end
end
fprintf(' to CPU\n');
clear GPUdoub GPUsing gpucount Allvars ii