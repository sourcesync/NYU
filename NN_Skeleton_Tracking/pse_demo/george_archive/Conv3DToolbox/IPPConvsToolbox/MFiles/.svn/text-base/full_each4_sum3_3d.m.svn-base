%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% This is a switching function between gpu, ipp, and a non-IPP based convolution
% for the IPPToolbox to provide backwards
% compatibility if people do not have IPP libraries, though this is MUCH slower
% to use than the IPP libaries version.
% \f[
%   out(:,:,:,k,i) = \sum_j g_{j,k} \times (y_j^i \oplus_{full} f^{j(i)}_k)
% \f]
% where for each input map (prhs[0]) j (ie y(:,:,:,j,i)), it is convolved with each filter (prhs[1])
% over all k for that specific j (ie f(:,:,:,j,k,i)) using a 'full' convolution if  the
% binary connectivity matrix has g(j,k) == 1. .
%
% In terms of the input arguments this would be written as:
% \f[
%  OUTPUT(:,:,:,j,k,i) = \sum_j CONMAT(j,k) \times (MAPS(:,:,:,j,i) \oplus_{full} FILTERS(:,:,j,k,i))
% \f]
%
% This is commonly used in the inference and learning filters of a Deconvolutional
% Network.
% \f[
%   out(:,:,:,k,i) = \sum_j g_{j,k} \times ((\sum_k g_{j,k}(z(:,:,:,k,i) \oplus_{valid} f^{j(i)}_k)) \oplus_{full} f(:,:,:,j,k,i)^*)
% \f]
% where% is the flipup(fliplr()) operation in MATLAB.
%
% An example MATLAB usage could be:<br>
% out = full_each4_sum3( y , Fflip , conmat, [COMP_THREADS] );
%
% @file
% @author Matthew Zeiler
% @date May 16, 2011
%
% @conv_file @copybrief full_each4_sum3_3d.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief full_each4_sum3_3d.m
%
% @param maps  X x Y x T x num_input_maps x num_images matrix of single precision floating point numbers.
% @param F  fx x fy x fT x num_input_maps x num_feature_maps x num_images matrix of single precision floating point numbers.
% @param C  num_input_maps x num_feature_maps connectivity matrix.
% @param COMP_THEADS (optional and unused) specifies number of computation threads to parallelize over. Defaults to 4.
%
% @retval out (this is plhs[0]) A 4D matrix of (X+fx-1) x (Y+fy-1) x (T+ft-1) x num_feature_maps x num_images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = full_each4_sum3_3d(maps,F,C,COMP_THREADS)

% This isn't even used though.
if(nargin<4)
    dbstack
    fprintf('\n\n No comp_threads in valid_eachK_loopJ\n\n');
    COMP_THREADS=4;
end
% We fill permute the output at the end before returning to time is in the right place.
so = [size(maps,1)+size(F,1)-1 size(maps,2)+size(F,2)-1 size(F,5) size(maps,3)+size(F,3)-1 size(maps,5)];
% Depth in the time dimension of the input maps (need this for reshaping below). 
timemaps = size(maps,3);

if(~isfloat(maps))
    out = zeros(so,GPUsingle);
else
    out = zeros(so,'single');
end

% Make it so we can slice time and then use those frames as multiple separate cases.
maps = permute(maps,[1 2 4 3 5]);
% Due to full convolution, just need entire maps each time.
if(~isfloat(maps))
    setSize(maps,[size(maps,1) size(maps,2) size(maps,3) size(maps,4)*size(maps,5)]);
else
    maps = reshape(maps,[size(maps,1) size(maps,2) size(maps,3) size(maps,4)*size(maps,5)]);
end


% Do each slice of filter at a time as normal.
for time=1:size(F,3)
    % get the current time slice of the filter fx x fy x num_input_maps x num_feature_maps
    % Since we are input it into the output array in "reverse" order, we don't need to flip the filter.
    if(~isfloat(F))
        ft = slice(F,':',':',[time],':',':');
        setSize(ft,[size(ft,1) size(ft,2) size(ft,4) size(ft,5)]);
    else
        ft = F(:,:,time,:,:);
        ft = reshape(ft,[size(ft,1) size(ft,2) size(ft,4) size(ft,5)]);        
    end
    
    % This slice of filter goes from time:(end-size(F,3)+time)
    % We don't slice the maps according to time because we are doing a full convolutions
    % So we need the convolution over the entire maps through time and then add it in different locations
    % to the output maps.
%     mapt = maps(:,:,:,time:(end-size(F,3)+time),:);
    
    % convolve over num_input_maps x num_cases 2D images.
    % outt is outX x outY x num_feature_maps x num_cases
    outt = full_each4_sum3(maps,ft,C,COMP_THREADS);
    
    if(~isfloat(outt))
        setSize(outt,[so(1) so(2) so(3) timemaps so(5)]);
    else
        outt = reshape(outt,[so(1) so(2) so(3) timemaps so(5)]);
    end
    
    out(:,:,:,time:(timemaps+time-1),:) = out(:,:,:,time:(timemaps+time-1),:) + outt;
    clear outt
end
% Get time back int he third dimension.
out = permute(out,[1 2 4 3 5]);





