%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% This is a CPU/GPU implementation that valid convolves each feature
% map plane with each image plane and returns all of these resulting convolutions
% as follows:
% \f[
% out(:,:,j,k,i) = g_{j,k} \times (z_k^i \oplus_{valid} y_j^i)
% \f]
% where for each feature map (prhx[0]) k (ie z(:,:,k,i)), it is convolved with
% each input map (prhs[1]) j (ie y(:,:,j,i)) using a 'valid' convolution if  the
% binary connectivity matrix has g(j,k) != 0. This is used for gradient computations
% in conv and deconv nets to give for each input map and each feature map
% a valid convolution result to update the filter that makes that connection.
% It returns one such set of results for each image case (dimension 4 of images
% and maps arguments). The naming convention for this function is based on the operations
% done on dimensions 3 and 4 of the resulting output (size of 4 dimensional
% filters). So dimension 3 is the size of the number of images planes and 
% dimension 4 is the size of the number of maps planes.
%
% Note: the this function does the convolution of each k with each j creating
% an output with size: outsize1 x outsize2 x num_input_maps(aka J) x
% num_feature_maps(aka K) x num_images result. See the example usage below.
%
% In terms of the input parameters this would be written as:
% \f[
%  OUTPUT(:,:,j,k,i) = CONMAT(j,k) \times (MAPS(:,:,k,i) \oplus_{valid} IMAGES(:,:,j,i))
% \f]
%
% This function is commonly used in the filter learning of both Deconvolutional
% Netowrk and Convolutional Networks. An example usage in a Deconvolutional
% Network could be:<br>
% out = valid_each3_each4( feature_maps , images , conmat, [COMP_THREADS] );
%
% @file
% @author Matthew Zeiler
% @data Mar 11, 2010
%
% @conv_file @copybrief valid_each3_each4.m
% @see valid_each3_each4_gpu.m valid_each3_each4_ipp.cpp valid_each3_each4_3d.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief valid_each3_each4.m
%
% @param maps  n x m x num_feature_maps x num_images matrix of single precision floating point numbers.
% @param images  p x q x num_input_maps x num_images matrix of single precision floating point numbers.
% @param C  num_input_maps x num_feature_maps connectivity matrix.
% @param COMP_THEADS (optional and unused) specifies number of computation threads to parallelize over. Defaults to 4.
%
% @retval out (this is plhs[0]) A 4D matrix of (n+p-1) x (m+q-1) x num_input_maps x num_feature_maps x num_images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [out] = valid_each3_each4(maps,images,C,COMP_THREADS)

% This isn't even used though.
if(nargin<4)
    dbstack
    fprintf('\n\n No comp_threads in valid_eachK_loopJ\n\n');
    COMP_THREADS=4;
end




%%%%%%%%%%
% Try running on the gpu if the maps are not floats (already on GPU)
%%%%%%%%%%
if(~isfloat(maps))
    % Use GPU
    %   out = zeros(size(maps,1)-size(images,1)+1,size(maps,2)-size(images,2)+1,size(images,3),size(maps,3),size(maps,4),GPUsingle);
    %    % Don't need spcial case with multiple filters.
    %    if(size(maps,4)>1)
    %    for im=1:size(maps,4)
    %        assign(1,out,valid_each3_each4_gpu(slice(maps,':',':',':',im),slice(images,':',':',':',im),C,COMP_THREADS),':',':',':',':',im);
    %    end
    %    else
    % Feed in the whole lot of maps and images to be parallelized over.
    
    out = valid_each3_each4_gpu(maps,images,C,COMP_THREADS);
% out = cuConv7_2(maps,images,C,0,0);

    %    end
    % elseif(exist(strcat('ipp_conv2.',mexext),'file'))
else
    
    if(size(maps,4)~=size(images,4))
        error('valid_each3_each4.m: there must be the same number of maps and images.')
    end
    
    
    try
        out = valid_each3_each4_ipp(maps,images,C,COMP_THREADS);
    catch        
        fprintf(1,'WARNING: You should compile the MEX version of "valid_each3_each4.cpp",\n         found in the IPPConvsToolbox, and put it in your matlab path.  It is MUCH faster.\n');
        if(size(C,1)~=size(images,3))
            error('valid_each3_each4.m: connectivity matrix first dimension does not match second inputs third dimension.')
        end
        if(size(C,2)~=size(maps,3))
            error('valid_each3_each4.m: connectivity matrix second dimension does not match first inputs third dimension.')
        end
        
        
        % Initialize the running sum for each feature map.
        out = zeros(size(maps,1)-size(images,1)+1,size(maps,2)-size(images,2)+1,size(images,3),size(maps,3),size(maps,4),'single');
        for im=1:size(maps,4)
            for j=1:size(images,3)
                for k = 1:size(maps,3)
                    if(C(j,k)~=0)
                        % Place in correct location so when conctemp(:) is used below it will be
                        % the correct vectorized form for dfz.
%                         if(size(maps,1)>100)
%                         temp = real(ifft2(fft2(maps(:,:,k,im)).*fft2(images(:,:,j,im),size(maps,1),size(maps,2))));
%                         out(:,:,j,k,im) = temp(size(images,1):end,size(images,2):end);
%                         else
                        
                        out(:,:,j,k,im) = C(j,k)*conv2(maps(:,:,k,im),images(:,:,j,im),'valid');
%                         end
                        %                             keyboard
                    end
                end
            end
        end
        out = single(out);
    end
end

end

