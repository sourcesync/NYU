%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% This is a CPU/GPU implementation that full convolves each set of image planes 
% with corresponding filters as follows:
% \f[
%   out(:,:,j,k,i) = g_{j,k} \times (y_j^i \oplus_{full} f^{j(i)}_{k}))
% \f]
% where for each input map (prhs[0]) j (ie y(:,:,j,i)), it is convolved with each filter (prhs[1])
% over all k for that specific j (ie f(:,:,j,k,i)) using a 'full' convolution if  the
% binary connectivity matrix has g(j,k) == 1. This is summing over the input maps (j)
% to give one resulting sum per feature map (k). This is done separately for
% each image case (i, dimension 4 of the maps argument). Additionally, if 
% this filters have only 4 dimensions (typical) then they are used for each
% image case, but it does support different sets of filters for each image case
% as well if filters have 5 dimensions. The naming convention for this function
% is based on the operations done on dimensions 3 and 4 of the input filters.
% Each dimension 3 plane is summed over resulting in dimension 4 number of 
% output planes.
%
% This function switches between gpu, ipp, and a non-IPP based convolution
% automatically to provide backwards compatibility if people do not have 
% IPP libraries or GPUmat installed. If using the CPU, IPP will be MUCH faster
% than the naive MATLAB implementation.
%
% In terms of the input arguments this would be written as:
% \f[
%  OUTPUT(:,:,j,k,i) = CONMAT(j,k) \times (MAPS(:,:,j,i) \oplus_{full} FILTERS(:,:,j,k,i))
% \f]
%
% This is commonly used in the inference and learning filters of a Deconvolutional
% Network to push the gradients up the network. This can also be used for a
% convolutional network to do backpropagation. An example usage in a deconvolutional
% network could be:<br>
% out = full_each4_sum3( y , Fflip , conmat, [COMP_THREADS] );
%
% @file
% @author Matthew Zeiler
% @date Aug 26, 2010
%
% @conv_file @copybrief full_each4_sum3.m
% @see full_each4_sum3_gpu.m full_each4_sum3_ipp.cpp full_each4_sum3_3d.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief full_each4_sum3.m
%
% @param maps  n x m x num_input_maps x num_images matrix of single precision floating point numbers.
% @param F  p x q x num_input_maps x num_feature_maps x num_images matrix of single precision floating point numbers.
% @param C  num_input_maps x num_feature_maps connectivity matrix.
% @param COMP_THEADS (optional and unused) specifies number of computation threads to parallelize over. Defaults to 4.
%
% @retval out (this is plhs[0]) A 4D matrix of (n+p-1) x (m+q-1) x num_feature_maps x num_images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [out] = full_each4_sum3(maps,F,C,COMP_THREADS)

% This isn't even used though.
if(nargin<4)
        dbstack
    fprintf('\n\n No comp_threads in valid_eachK_loopJ\n\n');
    COMP_THREADS=4;
end



%%%%%%%%%%
% Try running on the gpu if the maps are not floats (already on GPU)
%%%%%%%%%%
if(isa(maps,'gsingle') || ~isfloat(maps))
    % Use GPU
    %XXXXXXXXXXXXXXXXXXXXX
    % Doesn't work with multiple DIFFERENT filters as of now. 
    %XXXXXXXXXXXXx
%     if(size(F,5)==size(maps,4) && size(maps,4)>1) % There were many different versions of filters passed in, so just loop for now.
%         for im=1:size(F,5)
%         temp = full_each4_sum3_gpu(slice(maps,':',':',':',im),slice(F,':',':',':',':',im),C,COMP_THREADS);
%         if(im==1)
%             out = zeros([size(temp,1) size(temp,2) size(temp,3) size(temp,4)*size(maps,4)],GPUsingle);
%         end
%         assign(1,out,temp,':',':',':',im);
%         end
%     else
    out = full_each4_sum3_gpu(maps,F,C,COMP_THREADS);
%     end
else
    try
        % If sending in a batch of images make it as though there are more feature maps.
        if(size(maps,4)>size(F,5))
           F = repmat(F,[1 1 1 1 size(maps,4)]);
        end
        
        % Use IPP libraries
%         A = full_each4_sum3_ipp(maps,F,single(C),COMP_THREADS);
%         keyboard
%         'blah'
% figure(1), sdf(A(:,:,:,1));
% norm(va(A(:,:,:,1)))
        out = sum(full_each4_sum3_ipp(maps,F,C,COMP_THREADS),3);
        out = reshape(out,[size(out,1) size(out,2) size(out,4) size(out,5)]);
    catch
%         keyboard
%         if(RESHAPE_AFTER)
%             maps = reshape(maps,smaps);
%             C = C(1:size(C,1)/smaps(4),:);
%             if(multF)
%                 F = reshape(F,[sF(1) sF(2) sF(3) sF(5) sF(4)]);
%                 size(F)
%                 F = permute(F,[1 2 3 5 4]);
%                 size(F)
%             else
%                 F = F(:,:,1:sF(3),:);
%             end
%         end
        
        % Use matlab implmentation.
            fprintf(1,'WARNING: You should compile the MEX version of "full_each4_sum3.cpp",\n         found in the IPPConvsToolbox, and put it in your matlab path.  It is MUCH faster.\n');
        if(size(C,1)~=size(maps,3))
            error('full_eachK_loopK.m: connectivity matrix first dimension does not match first inputs third dimension.')
        end
        if(size(C,1)~=size(F,3))
            error('full_eachK_loopK.m: connectivity matrix first dimension does not match second inputs third dimension.')
        end
        if(size(C,2)~=size(F,4))
            error('full_eachK_loopK.m: connectivity matrix first dimension does not match second inputs fourth dimension.')
        end
        
        % Initialize the running sum for each feature map.
        out = zeros(size(maps,1)+size(F,1)-1,size(maps,2)+size(F,2)-1,size(F,4),size(maps,4),'single');        
        for im=1:size(maps,4)
            for j=1:size(F,3)
                for k = 1:size(F,4)
                    if(C(j,k)~=0)
                        if(k==size(F,4) && im==1)
                            temp(:,:,j,im) = conv2(maps(:,:,j,im),F(:,:,j,k),'full');
                        end
                        % Place in correct location so when conctemp(:) is used below it will be
                        % the correct vectorized form for dfz.
                        if(size(F,5)>1)
                            out(:,:,k,im) = out(:,:,k,im) + C(j,k)*conv2(maps(:,:,j,im),F(:,:,j,k,im),'full');
                        else
                            out(:,:,k,im) = out(:,:,k,im) + C(j,k)*conv2(maps(:,:,j,im),F(:,:,j,k),'full');
                        end
                    end
                end
            end
        end

        out = single(out);
    end    
end

end

