%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% This is a GPUmat implementation for doing the valid_each3_each4 operation:
% @copybrief valid_each3_each4.m
% @see valid_each3_each4.m for full documentation and usage. 
%
% @file
% @author Matthew Zeiler
% @data Apr 1, 2011
%
% @conv_file @copybrief valid_each3_each4_gpu.m
% @gpu_file @copybrief valid_each3_each4_gpu.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief valid_each3_each4.m
%
% @param maps  n x m x num_feature_maps  matrix of single precision floating point numbers.
% @param images  p x q x num_input_maps matrix of single precision floating point numbers.
% @param C  num_input_maps x num_feature_maps connectivity matrix.
% @param COMP_THEADS (optional and unused) specifies number of computation threads to parallelize over. Defaults to 4.
%
% @retval out (this is plhs[0]) A 4D matrix of (n+p-1) x (m+q-1) x num_input_maps x num_feature_maps
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [out] = valid_each3_each4_gpu(maps,images,C,COMP_THREADS)


% if(isa(maps,'gsingle'))
%     
%     feature_maps = size(maps,3);
%     input_maps = size(images,3);
%     num_images = size(maps,4);
%     sz = [size(maps,1)-size(images,1)+1 size(maps,2)-size(images,2)+1];
%     out = gzeros([sz(1:2) input_maps*feature_maps num_images]);
%     
%     gfor linind=1:input_maps*num_images
% %         [jt,im] = ind2sub([input_maps num_images],linind);
%         im = floor((linind-1)/input_maps)+1;
%         jt = linind - (im-1)*input_maps;
%     for kt=1:feature_maps
% %     linind
% %     [jt,kt,im] = ind2sub([input_maps feature_maps num_images],linind);
% 
%     out(:,:,(kt-1)*input_maps+jt,im) = conv2(maps(:,:,kt,im),images(:,:,jt,im),'valid');
%     end
%     gend
%     
% else % GPUmat implementation
    
    % % May have to flip the filters to be equivalent???
    % simages = size(images);
    % simages(3) = size(images,3);
    % setSize(images,[simages(1)*simages(2) simages(3)]);
    % ImFlip = clone(images);
    % cuRotate180(ImFlip,images);
    % setSize(images,simages);
    % clear ImFlip
    %
    % % The k index is the number of cases.
    % numCases = size(maps,3);
    % % The j index is the number of filters.
    % numFilters = size(images,3);
    %
    % smaps = size(maps);
    % setSize(maps,[size(maps,1)*size(maps,2) numCases]);
    %
    % if((numFilters/16-floor(numFilters/16))>0)  % Means that is greater than a multiple of 16
    %     actFilters = (floor(numFilters/16)+1)*16;
    %     zeroFilters = actFilters-numFilters; % number of filters that will be zero.
    % else
    %     actFilters = numFilters;
    %     zeroFilters = 0;
    % end
    %
    % setSize(images,[simages(1)*simages(2) numFilters]);
    % % Add zero filters so the total number of filters is 16.
    % if(zeroFilters~=0)
    %     images = [images zeros(simages(1)*simages(2),zeroFilters,GPUsingle)];
    % end
    %
    %
    % % Initialize the running sum for each feature map.
    % out = zeros((smaps(1)-simages(1)+1)*(smaps(2)-simages(2)+1)*actFilters,numCases,GPUsingle);
    %
    % cuConv(maps,images,out);
    %
    % setSize(out,[(smaps(1)-simages(1)+1) (smaps(2)-simages(2)+1) actFilters numCases]);
    % % out = out(:,:,1:numFilters,:);
    %
    % out = slice(out,':',':',[1,1,numFilters],':');
    % % This should be quicker.
    % % out = assign(0,out,out,[1,1,END],[1,1,END],[1,1,numFilters],[1,1,END]);
    %
    % % Set sizes back to originals so outside this function they are fine.
    % setSize(maps,smaps);
    % setSize(images,simages);
    
    
    
    
%     %%%%%%%%%%%%%%%%%%%%%
%     % The above method relies on only one image (dim4) in each of the filters of and
%     % images to be passed in, and therefore does 16x extra work often
%     % Below is a more parallel approach that feeds in batches of images (multipeles of 16) to the cuConv2 function (instead of cuConv).
%     %%%%%%%%%%%%%%%%%%%%%    
%     % May have to flip the filters to be equivalent??? (not for cuConv3)
%     simages = size(images);
%     smaps = size(maps);    
%     % We select only one input map at a time (maybe this can be the size(images,4) possibly).
%     numGroups = size(maps,4);
%     numImgsPerGroup = size(maps,3); % Going to slice up the input maps one at a time.
%     % This is what cuConv3 sums over so set to 1 always.
%     numFiltersPerGroup = 1;
%     
%     % Initialize the resulting filters (a 5D matrix).
%     out = zeros((smaps(1)-simages(1)+1),(smaps(2)-simages(2)+1),size(images,3),size(maps,3),numGroups,GPUsingle);
%     smallout = zeros((smaps(1)-simages(1)+1)*(smaps(2)-simages(2)+1),numFiltersPerGroup*numImgsPerGroup*numGroups,GPUsingle);
%     smallimages = zeros(size(images,1),size(images,2),1,numGroups,GPUsingle);
%     smallmapsize = [size(maps,1)*size(maps,2)*numImgsPerGroup numGroups];
%     
%     % Each feature map will be operated on over numCases and numFilters images.
%     % This is because we need every image with every map convolved.
%     % for im=1:size(images,4) % Loop over each image (size(images,4) and size(maps,4))
%     for current_map = 1:size(images,3) % Slice each input map sepearately as numImgsPerGroup
% 
%         smallmaps = maps;
%         % Reshape to 1 filter per group (so no summation occurs).
%         setSize(smallmaps,smallmapsize);
%         % Reshape the smallout output maps.
%         setSize(smallout,[(smaps(1)-simages(1)+1)*(smaps(2)-simages(2)+1) numGroups*numImgsPerGroup]);
%         setSize(smallimages,[size(images,1) size(images,2) 1 numGroups]);
%         
%         % This is now the number of input maps (will not be summed over though). 
%         numFiltersPerGroup = size(smallimages,3);
%         
%         % Get single input map at a time (for one image).
%         if(size(images,4)>1)
% %             smallimages = slice(images,':',':',current_map,':');
%             assign(0,smallimages,images,':',':',current_map,':');
%         else
%             smallimages = slice(images,':',':',current_map);
%         end
%         % cuConv2 wants filterpix*numFilters by numCases matrix.
%         setSize(smallimages,[size(images,1)*size(images,2) numGroups]);
%                 
%         % Do the convolutions.
%         cuConv3(smallmaps,smallimages,smallout,numGroups,0,0);
%         
%         GPUsync;
%         
%         % out = out(:,:,1:numFilters,:);
%         if(size(images,4)>1)
%             % Reshape to  F(1) x F(2) x 1 x F(4) x
%             % Might need a permute here.
%             setSize(smallout,[(smaps(1)-simages(1)+1) (smaps(2)-simages(2)+1) 1 numImgsPerGroup numGroups]);
%             assign(1,out,smallout,':',':',current_map,':',':');
%         else
%             % Reshape to  F(1) x F(2) x 1 x F(4) x
%             % Might need a permute here.
%             setSize(smallout,[(smaps(1)-simages(1)+1) (smaps(2)-simages(2)+1) 1 numImgsPerGroup]);
%             assign(1,out,smallout,':',':',current_map,':');
%             
%         end
%         GPUsync;
%     end
%     % Set sizes back to originals so outside this function they are fine.
%     setSize(maps,smaps);
%     setSize(images,simages);
%     %%%%%%%%%%%%%%%%%%%%%
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
% 
% %%%%%%%%%%%%%%%%%%%%%
% % The above method relies on only one image (dim4) in each of the filters of and
% % images to be passed in, and therefore does 16x extra work often
% % Below is a more parallel approach that feeds in batches of images (multipeles of 16) to the cuConv2 function (instead of cuConv).
% %%%%%%%%%%%%%%%%%%%%%
% % May have to flip the filters to be equivalent??? (not for cuConv3)
% simages = size(images);
% smaps = size(maps);
% % We select only one input map at a time (maybe this can be the size(images,4) possibly).
% numGroups = size(maps,4);
% numImgsPerGroup = size(maps,3); % Going to slice up the input maps one at a time.
% % This is what cuConv3 sums over so set to 1 always.
% numFiltersPerGroup = size(images,3);
% 
% % Initialize the resulting filters (a 5D matrix).
% % out = zeros((smaps(1)-simages(1)+1)*(smaps(2)-simages(2)+1),size(images,3)*size(maps,3)*numGroups,GPUsingle);
% 
% %%%
% % Low level allocations
% out = GPUsingle();
% setReal(out);
% setSize(out,[(smaps(1)-simages(1)+1)*(smaps(2)-simages(2)+1),size(images,3)*size(maps,3)*numGroups]);
% GPUallocVector(out);
% 
% 
% % Reshape to 1 filter per group (so no summation occurs).
% setSize(maps,[size(maps,1)*size(maps,2) numImgsPerGroup*numGroups]);
% setSize(images,[size(images,1)*size(images,2) numFiltersPerGroup*numGroups]);
% 
% % Do the convolutions.
% % if(sum(sum(C))==numel(C)) % Full connectivity
% %     cuConv5(maps,images,out,numGroups,0,0);
% % else % Sparse connectivity
%     cuConv7(maps,images,out,C,numGroups,0,0);
% % end
% 
% mySetSize(out,[(smaps(1)-simages(1)+1) (smaps(2)-simages(2)+1) numFiltersPerGroup numImgsPerGroup numGroups]);
% % Set sizes back to originals so outside this function they are fine.
% mySetSize(maps,smaps);
% mySetSize(images,simages);
% %%%%%%%%%%%%%%%%%%%%%





% We select only one input map at a time (maybe this can be the size(images,4) possibly).
% numGroups = size(maps,4);
% numImgsPerGroup = size(maps,3); % Going to slice up the input maps one at a time.
% This is what cuConv3 sums over so set to 1 always.
% numFiltersPerGroup = size(images,3);

% Reshape to 1 filter per group (so no summation occurs).
% No more initialization, rehspaing or anything, just convs.
% C = zeros(size(images,3),size(maps,3),GPUsingle);
% Checks if C is all ones or sparse
out = cuConv7_2(maps,images,C,0,0);




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    %
    %
    %
    %
    %
    % %%%%%%%%%%%%%%%%%%
    % % FFT
    % endsize = [size(maps,1)-size(images,1)+1,size(maps,2)-size(images,2)+1,size(images,3)];
    %
    % if(size(maps,1)>size(images,1))
    %     padsize = [size(maps,1) size(maps,2)];
    %     padamount = [size(maps,1)-size(images,1) size(maps,2)-size(images,2)];
    %
    %     padmaps = 0;
    %     % Allocate memory on GPU for the padded size
    %     newA = zeros(padsize(1),padsize(2),size(images,3),GPUsingle);
    % else
    %     padsize = [size(images,1) size(images,2)];
    %     padamount = [size(images,1)-size(maps,1) size(images,2)-size(maps,2)];
    %     padmaps = 1;
    %     % Allocate memory on GPU for the padded size
    %     newA = zeros(padsize(1),padsize(2),size(maps,3),GPUsingle);
    % end;
    %
    %
    % % Initialize the running sum for each feature map.
    % out = zeros(endsize(1),endsize(2),size(images,3),size(maps,3),GPUsingle);
    %
    % for j=1:size(images,3)
    %     for k = 1:size(maps,3)
    %         if(C(j,k)~=0)
    %             % Place in correct location so when conctemp(:) is used below it will be
    %             % the correct vectorized form for dfz.
    %             % Do convolution as FFT's on gpu
    %             if(padmaps) % Maps have to be made larger with zeros
    %                 % Put the maps in the new A variable middle.
    %                 assign(1,newA,maps,[floor(padamount(1)/2)+1 1 END-floor(padamount(1)/2)-1],[floor(padamount(2)/2)+1 1 END-floor(padamount(2)/2)-1],[1 1 END]);
    %                 % Do the convolution.
    %                 out(:,:,j,k) = slice(real(ifft2(fft2(newA(:,:,k)).*fft2(images(:,:,j)))),...
    %                     [END-endsize(1)+1 1 END],[END-endsize(1)+1 1 END]);
    %             else % Images have to be made larger with zeros
    %                 % Put the images in the new A variable middle.
    %                                 assign(1,newA,images,[1 1 END-padamount(1)],[1 1 END-padamount(2)],[1 1 END]);
    %
    % %                 assign(1,newA,images,[floor(padamount(1)/2)+1 1 END-floor(padamount(1)/2)-1],[floor(padamount(2)/2)+1 1 END-floor(padamount(2)/2)-1],[1 1 END]);
    %                 % Do the convolution.
    % %                 temp = real(ifft2(fft2(maps(:,:,k)).*fft2(newA(:,:,j))))
    % %                 temp = slice(real(ifft2(fft2(maps(:,:,k)).*fft2(newA(:,:,j)))),...
    % %                     [END-endsize(1)+1 1 END],[END-endsize(1)+1 1 END]);
    % %                 temp2 = conv2(single(maps(:,:,k)),single(images(:,:,j)),'valid');
    % %                 single(temp(1:10,1:10))
    % %                 temp2(1:10,1:10)
    % %                 figure(1), sdf(single(temp));
    % %                 figure(2), sdf(temp2);
    % %                 fprintf('Max Error: %f, Total Error: %f\n',max(abs(single(temp(:))-temp2(:))),norm(single(temp(:))-temp2(:)));
    % %                 keyboard
    %                 out(:,:,j,k) = slice(real(ifft2(fft2(maps(:,:,k)).*fft2(newA(:,:,j)))),...
    %                     [END-endsize(1)+1 1 END],[END-endsize(1)+1 1 END]);
    %             end
    %             %               out(:,:,j,k) = C(j,k)*conv2(maps(:,:,k),images(:,:,j),'valid');
    %         end
    %     end
    % end
    % out = single(out);
    %
    %
    
    
    
% end



end