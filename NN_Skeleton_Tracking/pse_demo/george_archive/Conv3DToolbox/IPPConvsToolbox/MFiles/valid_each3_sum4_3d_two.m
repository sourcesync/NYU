%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% This is a 3D version of the valid_each3_sum4.m function. Since it relies
% on that function, it automatically switchest between gpu, ipp and a non-IPP based convolution
% for the IPPToolbox to provide backwards
% compatibility if people do not have IPP libraries or GPUs, though this is MUCH slower
% to use than the IPP libaries version.
% \f[
%    out(:,:,:,j,i) = \sum_k g_{j,k} \times (z_k^i \oplus_{valid} f^{j(i)}_k)
% \f]
% where for each input map (prhs[0]) k (ie z(:,:,k,i)), it is convolved with each filter (prhs[1])
% over all j for that specific k (ie f(:,:,j,k,i)) using a 'valid' convolution if  the
% binary connectivity matrix has g(j,k) == 1. .
%
% In terms of the input parameters this would be written as:
% \f[
%  OUTPUT(:,:,:,j,i) = \sum_k CONMAT(j,k) \times (MAPS(:,:,:,k,i) \oplus_{valid} FILTERS(:,:,:,j,k,i))
% \f]
%
% This is commonly used in the inference and learning filters of a Deconvolutional
% Network.See the example usage below.
% \f[
%  out(:,:,j,i) = \sum_k g_{j,k} \times (z_k^i \oplus_{valid} f^{j(i)}_k)
% \f]
% An example usage could be:<br>
% out = valid_each3_sum4_3d( feature_maps , filters , conmat, [COMP_THREADS] );
%
% @file
% @author Matthew Zeiler
% @data May 16, 2011
%
% @conv_file @copybrief valid_each3_sum4_3d.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief valid_each3_sum4_3d.m
%
% @param maps X x Y x T x num_feature_maps x num_cases dimensional images.
% @param F fx x fy x ft x num_input_maps x num_feature_maps dimensional filters.
% @param C num_input_maps x num_feature_maps connectivity matrix of ones and zeros.
% @param COMP_THREADS the number of threads to use for computation.
%
% @retval out X-fx+1 x Y-fy+1 x T-ft+1 x num_feature_maps x num_cases output maps.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = valid_each3_sum4_3d(maps,F,C,COMP_THREADS)

% This isn't even used though.
if(nargin<4)
    dbstack
    fprintf('\n\n No comp_threads in valid_each3_sum4\n\n');
    COMP_THREADS=4;
end
% We fill permute the output at the end before returning to time is in the right place.
so = [size(maps,1)-size(F,1)+1 size(maps,2)-size(F,2)+1 size(F,4) size(maps,3)-size(F,3)+1 size(maps,5)];
if(~isfloat(maps))
    out = zeros(so,GPUsingle);
else
    out = zeros(so,'single');
end

% Make it so we can slice time and then use those frames as multiple separate cases.
if(size(maps,5)==1)
    maps = permute(maps,[1 2 4 3]);
else
    maps = permute(maps,[1 2 4 3 5]);
end

% Do each slice of filter at a time as normal.
for time=1:size(F,3)
    % get the current time slice of the filter fx x fy x num_input_maps x num_feature_maps
    if(~isfloat(F))
        ft = slice(F,':',':',[size(F,3)-time+1],':',':');
        setSize(ft,[size(ft,1) size(ft,2) size(ft,4) size(ft,5)]);
    else
        ft = F(:,:,size(F,3)-time+1,:,:);
        ft = reshape(ft,[size(ft,1) size(ft,2) size(ft,4) size(ft,5)]);
    end
    
    % This slice of filter goes from time:(end-size(F,3)+time)
    if(~isfloat(maps))
        mapt = slice(maps,':',':',':',[time 1 (END-size(F,3)+time)],':');
        setSize(mapt,[size(mapt,1) size(mapt,2) size(mapt,3) size(mapt,4)*size(mapt,5)]);
    else
        mapt = maps(:,:,:,time:(end-size(F,3)+time),:);
        mapt = reshape(mapt,[size(mapt,1) size(mapt,2) size(mapt,3) size(mapt,4)*size(mapt,5)]);
    end
    
   
    outt = valid_each3_sum4(mapt,ft,C,COMP_THREADS);
            
    
    if(so(5)==1)
        if(~isfloat(outt))
            setSize(outt,[so(1) so(2) so(3) so(4)]);
            GPUplus(out,outt,out);
        else
            outt = reshape(outt,[so(1) so(2) so(3) so(4)]);
            out = out+outt;
        end
    else
        if(~isfloat(outt))
            setSize(outt,[so(1) so(2) so(3) so(4) so(5)]);
            GPUplus(out,outt,out);
        else
            outt = reshape(outt,[so(1) so(2) so(3) so(4) so(5)]);
            out = out + outt;
        end
    end
    clear outt
    
end
% Get time back int he third dimension.
if(so(5)==1)
    out = permute(out,[1 2 4 3]);
else
    out = permute(out,[1 2 4 3 5]);
end



