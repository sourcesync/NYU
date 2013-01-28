%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% This 3D convolution function is a switching function between gpu, ipp, and a non-IPP based convolution
% for the IPPToolbox to provide backwards
% compatibility if people do not have IPP libraries, though this is MUCH slower
% to use than the IPP libaries version.
% \f[
% out(:,:,:,j,k,i) = g_{j,k} \times (z_k^i \oplus_{valid} y_j^i)
% \f]
% where for each feature map (prhx[0]) k (ie z(:,:,:,k,i)), it is convolved with
% each input map (prhs[1]) j (ie y(:,:,:,j,i)) using a 'valid' convolution if  the
% binary connectivity matrix has g(j,k) != 0. .
%
% Note: the this function does the convolution of each k with each j creating
% an output with size: outsize1 x outsize2 x num_input_maps(aka J) x
% num_feature_maps(aka K) result. See the example usage below.
%
% In terms of the input parameters this would be written as:
% \f[
%  OUTPUT(:,:,:,j,k,i) = CONMAT(j,k) \times (MAPS(:,:,:,k,i) \oplus_{valid} IMAGES(:,:,:,j,i))
% \f]
%
% This function a commonly used operation when inferring the feature maps and
% learning filters. An example usage is like:
% out = valid_each3_each4( feature_maps , images , conmat, [COMP_THREADS] );
%
% @file
% @author Matthew Zeiler
% @data May 17, 2011
%
% @conv_file @copybrief valid_each3_each4_3d.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief valid_each3_each4_3d.m
%
% @param maps  n x m x Tm x num_feature_maps x num_images matrix of single precision floating point numbers.
% @param images  p x q x Ti x num_input_maps x num_images matrix of single precision floating point numbers.
% @param C  num_input_maps x num_feature_maps connectivity matrix.
% @param COMP_THEADS (optional and unused) specifies number of computation threads to parallelize over. Defaults to 4.
%
% @retval out (this is plhs[0]) A 4D matrix of (n-p+1) x (m-q+1) x (Tm-Ti+1) x (x num_input_maps x num_feature_maps x num_images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = valid_each3_each4_3d(maps,images,C,COMP_THREADS)

% This isn't even used though.
if(nargin<4)
    dbstack
    fprintf('\n\n No comp_threads in valid_eachK_loopJ\n\n');
    COMP_THREADS=4;
end
% We fill permute the output at the end before returning to time is in the right place.
so = [size(maps,1)-size(images,1)+1 size(maps,2)-size(images,2)+1 size(images,4) size(maps,4) size(maps,3)-size(images,3)+1 size(maps,5)];
if(~isfloat(maps))
    out = zeros(so,GPUsingle);
else
    out = zeros(so,'single');
end

% Make it so we can slice time and then use those frames as multiple separate cases.
maps = permute(maps,[1 2 4 3 5]);
images = permute(images,[1 2 4 3 5]);


% Do each slice of filter at a time as normal.
for time=1:size(images,4)
    
    if(~isfloat(images))
        ft = slice(images,':',':',':',size(images,4)-time+1,':');
        % This slice of filter goes from time:(end-size(F,3)+time)
        mapt = slice(maps,':',':',':',[time 1 (END-size(images,4)+time)],':');
    else
        % get the current time slice of the filter fx x fy x num_input_maps x num_cases
        ft = images(:,:,:,size(images,4)-time+1,:);
        % This slice of filter goes from time:(end-size(F,3)+time)
        mapt = maps(:,:,:,time:(end-size(images,4)+time),:);
    end
    
    
    % Need to repmat of the size of mapt's time dimension so that when treated as separate cases
    % the 2D version of valid_each3_each4 just works.
    ft = repmat(ft,[1 1 1 size(mapt,4) 1]);
    
    if(~isfloat(maps))
        setSize(ft,[size(ft,1) size(ft,2) size(ft,3) size(ft,4)*size(ft,5)]);
        setSize(mapt,[size(mapt,1) size(mapt,2) size(mapt,3) size(mapt,4)*size(mapt,5)]);
    else
        ft = reshape(ft,[size(ft,1) size(ft,2) size(ft,3) size(ft,4)*size(ft,5)]);
        mapt = reshape(mapt,[size(mapt,1) size(mapt,2) size(mapt,3) size(mapt,4)*size(mapt,5)]);
    end
    
    
    % convolve over num_input_maps x num_feature_maps x num_cases 2D images.
    % outt is outX x outY x num_feature_maps x num_cases
    outt = valid_each3_each4(mapt,ft,C,COMP_THREADS);
    
    if(~isfloat(outt))
        setSize(outt,[so(1) so(2) so(3) so(4) so(5) so(6)]);
        GPUplus(out,outt,out);
    else
        outt = reshape(outt,[so(1) so(2) so(3) so(4) so(5) so(6)]);
        % This is the sum in the time convolution.
        out = out + outt;
    end
end
% Get time back in the third dimension.
out = permute(out,[1 2 5 3 4 6]);