%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% This is a CPU/GPU implementation that valid convolves each set of image
% planes with corresponding filters as follows:
% \f[
%    out(:,:,j,k,i) = g_{j,k} \times (z_k^i \oplus_{valid} f^{j(i)}_k)
% \f]
% where for each input map (prhs[0]) k (ie z(:,:,k,i)), it is convolved with each filter (prhs[1])
% over all j for that specific k (ie f(:,:,j,k,i)) using a 'valid' convolution if  the
% binary connectivity matrix has g(j,k) == 1. This is summing over the feature
% maps (k) to give one resulting sum per input map (j). This is done separately
% for each image case (i, dimension 4 of the maps argument). Additionally, if 
% this filters have only 4 dimensions (typical) then they are used for each
% image case, but it does support different sets of filters for each image case
% as well if filters have 5 dimensions. The naming convention for this function
% is based on the operations done on dimensions 3 and 4 of the input filters.
% Each dimension 4 plane is summed over resulting in dimension 3 number of 
% output planes.
%
% In terms of the input parameters this would be written as:
% \f[
%  OUTPUT(:,:,j,k,i) = CONMAT(j,k) \times (MAPS(:,:,k,i) \oplus_{valid} FILTERS(:,:,j,k,i))
% \f]
%
% This is commonly used in the inference and learning filters of a Deconvolutional
% Network to reconstruct down the network. This can also be used for a 
% convolutional network to do the forward pass. An example usage in a 
% deconvolutional network could be:<br>
% out = valid_each3_sum4( feature_maps , filters , conmat, [COMP_THREADS] );
%
% @file
% @author Matthew Zeiler
% @data Aug 26, 2010
%
% @conv_file @copybrief valid_each3_sum4.m
% @see valid_each3_sum4_gpu.m valid_each3_sum4_ipp.cpp valid_each3_sum4_3d.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief valid_each3_sum4.m
%
% @param maps  n x m x num_input_maps x num_images  matrix of single precision floating point numbers.
% @param F  p x q x num_input_maps x num_feature_maps [x num_images] matrix of single precision floating point numbers.
% @param C  num_input_maps x num_feature_maps connectivity matrix.
% @param COMP_THEADS (optional and unused) specifies number of computation threads to parallelize over. Defaults to 4.
%
% @retval out (this is plhs[0]) A 4D matrix of (n+p-1) x (m+q-1) x num_input_maps x num_images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [out] = valid_each3_sum4(maps,F,C,COMP_THREADS)

% This isn't even used though.
if(nargin<4)
    dbstack
    fprintf('\n\n No comp_threads in valid_each3_sum4\n\n');
    COMP_THREADS=4;
end



%%%%%%%%%%
% Try running on the gpu if the maps are not floats (already on GPU)
%%%%%%%%%%
if(~isfloat(maps))
    % Use GPU
    out = valid_each3_sum4_gpu(maps,F,C,COMP_THREADS);
    
else
    % If sending in a batch of images make it as though there are more feature maps.
%     try
%         % If sending in a batch of images make it as though there are more feature maps.
%         if(size(maps,4)>size(F,5))
%             F = repmat(F,[1 1 1 1 size(maps,4)]);
%         end
%         % Use IPP libraries
%         out = sum(sparse_valid_each3_sum4(maps,F,C,COMP_THREADS),4);
%         out = reshape(out,[size(out,1) size(out,2) size(out,3) size(out,5)]);
%     catch
%         fprintf('sparse_valid_each3_sum4.cpp failed, trying IPP version\n');
        
        try
            % If sending in a batch of images make it as though there are more feature maps.
            if(size(maps,4)>size(F,5))
                F = repmat(F,[1 1 1 1 size(maps,4)]);
            end
            % Use IPP libraries
            out = sum(valid_each3_sum4_ipp(maps,F,C,COMP_THREADS),4);
            out = reshape(out,[size(out,1) size(out,2) size(out,3) size(out,5)]);
        catch
            fprintf('IPP version of valid_each3_sum4 failed, trying MATLAB version.\n');
            %     fprintf(1,'WARNING: You should compile the MEX version of "valid_each3_sum4.cpp",\n         found in the IPPConvsToolbox, and put it in your matlab path.  It is MUCH faster.\n');
            if(size(C,1)~=size(F,3))
                error('valid_each3_sum4.m: connectivity matrix first dimension does not match second inputs third dimension.')
            end
            if(size(C,2)~=size(F,4))
                error('valid_each3_sum4.m: connectivity matrix second dimension does not match second inputs fourth dimension.')
            end
            if(size(C,2)~=size(maps,3))
                error('valid_each3_sum4.m: connectivity matrix second dimension does not match first inputs third dimension.')
            end
                                    
            % Initialize the running sum for each feature map.
            out = zeros(size(maps,1)-size(F,1)+1,size(maps,2)-size(F,2)+1,size(F,3),size(maps,4),'single');
            for im=1:size(maps,4)
                for j=1:size(F,3)
                    for k = 1:size(F,4)
                        if(C(j,k)~=0)
                            % Place in correct location so when conctemp(:) is used below it will be
                            % the correct vectorized form for dfz.
                            if(size(F,5)>1)
                                out(:,:,j,im) = out(:,:,j,im) + C(j,k)*conv2(maps(:,:,k,im),F(:,:,j,k,im),'valid');
                            else
                                out(:,:,j,im) = out(:,:,j,im) + C(j,k)*conv2(maps(:,:,k,im),F(:,:,j,k),'valid');
                            end
                        end
                    end
                end
            end
            out = single(out);
    end
end


end

