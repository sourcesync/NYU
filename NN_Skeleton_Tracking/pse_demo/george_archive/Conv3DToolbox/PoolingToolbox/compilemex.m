%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Compiles all the subsampling mex files which are 10-1000x faster than the
% included M-files.
%
% @file
% @author Matthew Zeiler
% @date Aug 14, 2010
% 
% @inparam These paths must be set within: 
% \li \e MEXOPTS_PATH path to the included mexopts.sh file.
%
% @pooltoolbox_file @copybrief compilemex.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


MEXOPTS_PATH = './mexopts.sh';

% Compile max_pool.cpp
exec_string = strcat({'mex -f '},MEXOPTS_PATH,{' '},{'-lguide  -lstdc++ max_pool.cpp'});
eval(exec_string{1});

% Compile max_pool.cpp
exec_string = strcat({'mex -f '},MEXOPTS_PATH,{' '},{'-lguide  -lstdc++ max_pool3d.cpp'});
eval(exec_string{1});

% Compile reverse_max_pool.cpp
exec_string = strcat({'mex -f '},MEXOPTS_PATH,{' '},{'-lguide  -lstdc++ reverse_max_pool.cpp'});
eval(exec_string{1});


% Compile reverse_max_pool.cpp
exec_string = strcat({'mex -f '},MEXOPTS_PATH,{' '},{'-lguide  -lstdc++ reverse_max_pool3d.cpp'});
eval(exec_string{1});



% Compile avg_pool.cpp
exec_string = strcat({'mex -f '},MEXOPTS_PATH,{' '},{'-lguide  -lstdc++ avg_pool.cpp'});
eval(exec_string{1});


% Compile reverse_avg_pool.cpp
exec_string = strcat({'mex -f '},MEXOPTS_PATH,{' '},{'-lguide  -lstdc++ reverse_avg_pool.cpp'});
eval(exec_string{1});


