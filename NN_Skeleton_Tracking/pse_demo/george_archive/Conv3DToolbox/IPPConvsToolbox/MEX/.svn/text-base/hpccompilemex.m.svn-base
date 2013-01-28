%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Compiles all the required IPP Library based files used in the IPP
% implementation of the Deconvoltuional Network. If you are interested in just
% the ipp_conv2 then you can ignore the other generated files (or delete them). 
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
% 
% @inparam These paths must be set within: 
% \li \e MEXOPTS_PATH path to the included mexopts.sh file.
% \li \e IPP_INCLUDE_PATH path to em64t IPP include directory.
% \li \e IPP_LIB64_PATH path to em64t IPP lib directory.
% \li \eIPP_LIB_PATH path to cce IPP lib directory.
%
% @ipp_file @copybrief compilemex.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

MEXOPTS_PATH = './hpcmexopts.sh';
IPP_INCLUDE_PATH =  '-I/share/apps/intel/Compiler/11.1/046/ipp/em64t/include';
IPP_LIB64_PATH = '-L/share/apps/intel/Compiler/11.1/046/ipp/em64t/lib';
IPP_LIB_PATH =  '-L/share/apps/intel/Compiler/11.1/046/lib/intel64';

%IPP_INCLUDE_PATH =  '-I/opt/intel/ipp/current/em64t/include';
%IPP_LIB64_PATH = '-L/opt/intel/ipp/current/em64t/lib';
%IPP_LIB_PATH =  '-L/opt/intel/cce/current/lib';

% Compile ipp_conv2.cpp
exec_string = strcat({'mex -f '},MEXOPTS_PATH,{' '},{IPP_INCLUDE_PATH},{' '}, {IPP_LIB64_PATH},{' '},{IPP_LIB_PATH},{' '},{'-lguide  -lippiemergedem64t -lippimergedem64t  -lippcoreem64t  -lippsemergedem64t   -lippsmergedem64t  -lstdc++ ipp_conv2.cpp'});
eval(exec_string{1});
% Compile valid_eachK_loopJ.cpp 
exec_string = strcat({'mex -f '},MEXOPTS_PATH,{' '},{IPP_INCLUDE_PATH},{' '}, {IPP_LIB64_PATH},{' '},{IPP_LIB_PATH},{' '},{'-lguide  -lippiemergedem64t -lippimergedem64t  -lippcoreem64t  -lippsemergedem64t   -lippsmergedem64t  -lstdc++ valid_eachK_loopJ.cpp'});
eval(exec_string{1});
% Compile full_eachJ_loopK.cpp
exec_string = strcat({'mex -f '},MEXOPTS_PATH,{' '},{IPP_INCLUDE_PATH},{' '}, {IPP_LIB64_PATH},{' '},{IPP_LIB_PATH},{' '},{'-lguide  -lippiemergedem64t -lippimergedem64t  -lippcoreem64t  -lippsemergedem64t   -lippsmergedem64t  -lstdc++ full_eachJ_loopK.cpp'});
eval(exec_string{1});
% Compile valid_loopK_loopJ.cpp 
exec_string = strcat({'mex -f '},MEXOPTS_PATH,{' '},{IPP_INCLUDE_PATH},{' '}, {IPP_LIB64_PATH},{' '},{IPP_LIB_PATH},{' '},{'-lguide  -lippiemergedem64t -lippimergedem64t  -lippcoreem64t  -lippsemergedem64t   -lippsmergedem64t  -lstdc++ valid_loopK_loopJ.cpp '});
eval(exec_string{1});
