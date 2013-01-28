%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% A wrapper for GPUmat setSize() which makes sure that trailing 1's in the 
% size vector are not used in setSize and therefore gives equivalent sizes
% to reshape in matlab. Note: this is a low_level function that modifies the
% size paramaters of the variable input. The variable will remain this size
% outside any function calls.
%
% @file
% @author Matthew Zeiler
% @date Apr 15, 2011
%
% @gpu_file @copybrief mySetSize.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief mySetSize.m
%
% @param A the variable to change the size of
% @param sz the new size to make it.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = mySetSize(A,sz)

lastsize = find(sz>1,1,'last');
% Special case where input is rectangle with second dimension 1 (column-vector).
if(lastsize==1 && length(sz)==2)
else
sz = sz(1:lastsize);
end
setSize(A,sz);

end



