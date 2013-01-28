%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% This does moves all the variables in AllGPUvars back to the GPU.
% This should be called after moveALL2CPU.m to generate the AllGPUvars stuct.
%
% @file
% @author Matthew Zeiler
% @date May 19, 2011
%
% @gpu_file @copybrief moveAll2GPU.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Moving: ');
for ii=1:numel(AllGPUvars)
    fprintf('%s, ',AllGPUvars(ii).name);
    if(strcmp(AllGPUvars(ii).type,'GPUcell'))        
        for cc=1:AllGPUvars(ii).cellLength
            eval(sprintf('%s{%d} = %s(%s{%d});',AllGPUvars(ii).name,cc,AllGPUvars(ii).celltype{cc},AllGPUvars(ii).name,cc));
        end
    else % Just cast GPU single or double        
        eval(sprintf('%s = %s(%s);',AllGPUvars(ii).name,AllGPUvars(ii).type,AllGPUvars(ii).name));
    end
end
fprintf(' to GPU\n');
clear ii AllGPUvars