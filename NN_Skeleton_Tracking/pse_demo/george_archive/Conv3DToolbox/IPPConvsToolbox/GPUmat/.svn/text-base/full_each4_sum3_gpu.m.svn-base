%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% This is a GPUmat implementation for doing the full_each4_sum3 operation:
% @copybrief full_each4_sum3.m
% @see full_each4_sum3.m for full documentation and usage.
%
% @file
% @author Matthew Zeiler
% @date Apr 1, 2011
%
% @conv_file @copybrief full_each4_sum3_gpu.m
% @gpu_file @copybrief full_each4_sum3_gpu.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief full_each4_sum3.m
%
% @param maps  n x m x num_input_maps [x num_images] matrix of single precision floating point numbers.
% @param F  p x q x num_input_maps x num_feature_maps [x num_images] matrix of single precision floating point numbers.
% If F is only 4D in size it is repmatted along dim-5 by num_images to use the
% same sets of filters for each case.
% @param C  num_input_maps x num_feature_maps connectivity matrix.
% @param COMP_THEADS (optional and unused) specifies number of computation threads to parallelize over. Defaults to 4.
%
% @retval out (this is plhs[0]) A 4D matrix of (n+p-1) x (m+q-1) x num_feature_maps [x num_images]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [out] = full_each4_sum3_gpu(maps,F,C,COMP_THREADS)

% if(isa(maps,'gsingle'))
%
%
% input_maps = size(maps,3);
% images = size(maps,4);
% % Single filter.
% if(size(F,3)==input_maps)
%     feature_maps = size(F,4);
% elseif(size(F,4)==images) % Multiple filters.
%     feature_maps = size(F,3)/input_maps;
% end
% sz = [size(maps,1)+size(F,1)-1 size(maps,2)+size(F,2)-1];
% out = zeros([ sz(1) sz(2) feature_maps*images input_maps],gsingle);
%
% % If there are different filters for each image dimension of maps (dim 4 of maps) then index differently.
% % First the filters are reshaped as reshape(F,[s(1) s(2) input_maps*feature_maps])
% % The filters in this case have been repmated by num_images over 4th dimension (because Jacket can't have more than 4 dims)
% if(size(F,3)>input_maps)
%
%  % Split up over whole batch and
% gfor linind2=1:input_maps*images
%     [jt,im] = ind2sub([input_maps images],linind2);
% % Sum over this dimension.
% for kt=1:feature_maps
%     linind = sub2ind([feature_maps images],kt,im);
%     out(:,:,linind,jt) = conv2(maps(:,:,jt,im),F(:,:,jt+(kt-1)*input_maps,im),'full');
% end
% gend
% else % Single filter
% % Split up over whole batch and
% gfor linind2=1:input_maps*images
%     [jt,im] = ind2sub([input_maps images],linind2);
% % Sum over this dimension.
% for kt=1:feature_maps
%     linind = sub2ind([feature_maps images],kt,im);
%     out(:,:,linind,jt) = conv2(maps(:,:,jt,im),F(:,:,jt,kt),'full');
% end
% gend
% end
% out = sum(out,4);
% out = reshape(out,[sz(1) sz(2) feature_maps images]);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% else % GPUmat implmentation



% %%%%%%%%%%%%%%%%%%%%
% % cuConv2 single image
% % May have to flip the filters to be equivalent???
% % Fflip = clone(F);
% % size(Fflip)
% sF = size(F);
% F = reshape(F,sF(1)*sF(2),sF(3)*sF(4));
% Fflip = clone(F);
% cuRotate180(Fflip,F);
% F = reshape(F,sF);
% clear Fflip
%
% % The j index is the number of cases.
% numCases = size(F,3);
% % The k index is the number of filters.
% numFilters = size(F,4);
%
% % Permute filters so the number of cases is last dimension
% F = permute(F,[1 2 4 3]);
%
% % The maps are the input maps to a layer so j is the numCases
% smaps = size(maps);
% smaps(3) = size(maps,3);
%
% %%%%%
% % Full convolution padding, have to pad the images to get the full from the valid convolution.
% newMaps = zeros((smaps(1)+2*sF(1)-2)*(smaps(2)+2*sF(2)-2),smaps(3),GPUsingle);
% cuCopyInto(reshape(maps,size(maps,1)*size(maps,2),size(maps,3)),newMaps,sF(1)-1);
% %%%%%
%
%
% if((numFilters/16-floor(numFilters/16))>0)  % Means that is greater than a multiple of 16
%     actFilters = (floor(numFilters/16)+1)*16;
%     zeroFilters = actFilters-numFilters; % number of filters that will be zero.
% else
%     actFilters = numFilters;
%     zeroFilters = 0;
% end
%
% F = reshape(F,sF(1)*sF(2)*numFilters,numCases);
% % Add zero filters so the total number of filters is 16.
% if(zeroFilters~=0)
%     F = [F; zeros(sF(1)*sF(2)*zeroFilters,numCases,GPUsingle)];
% end
%
% % Initialize the running sum for each feature map.
% out = zeros((smaps(1)+sF(1)-1)*(smaps(2)+sF(2)-1)*actFilters,numCases,GPUsingle);
% cuConv2(newMaps,F,out,sF(1));
% out = reshape(out,(smaps(1)+sF(1)-1),(smaps(2)+sF(2)-1),actFilters,numCases);
% out = out(:,:,1:numFilters,:);
% out = permute(out,[1 2 4 3]);







%%%%%%%%%%%%%%%%%%%%%%%%%%
% % code based on the inverse of valid_eachK_loopJ_gpu
% %%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%
% % Permuted CPU sizes [1 2 4 3] so number of images is dim 3.
% % % Do this to use the optimized code (first case below) which has less permutes.
% if(size(F,5)~=size(maps,4) || size(F,5)==1)
%     %     F = permute(F,[1 2 4 3]);
% %     F = repmat(F,[1 1 1 1 size(maps,4)]);
%     DIFF_F = 0;
% else % Different F were passed in.
%     DIFF_F = 1;
% end
% 
% 
% % if(sum(C(:))~=numel(C))
% %     GPUtimes(F,repmat(reshape(C,[1 1 size(C,1) size(C,2) 1]),[size(F,1) size(F,2) 1 1 size(F,5)]),F);
% % end
% 
% %%%%%%%%%%%%%%%%%
% % Working copy that gives 3.5x speedup with 16 images.
% sF = size(F);
% sF(3) = size(F,3);
% sF(4) = size(F,4);
% sF(5) = size(maps,4);
% % Sent in a batch of feature maps.
% numImgsPerGroup = 1; % size(maps,4);
% % The k index is the number of cases. (feature maps)
% numFiltersPerGroup = sF(3);
% % The j index is the number of filters in each group. (input maps)
% numGroups = sF(4)*sF(5);
% 
% smaps = size(maps);
% 
% % Initialize the running sum for each feature map.
% % out = zeros([(size(maps,1)+sF(1)-1)*(size(maps,2)+sF(2)-1)*sF(4) sF(5)],GPUsingle);
% 
% %%%%%
% % Allocate the output array (but don't touch every element). VERY LOW LEVEL
% out = GPUsingle();
% setReal(out);
% setSize(out,[(size(maps,1)+sF(1)-1)*(size(maps,2)+sF(2)-1)*sF(4) sF(5)]);
% GPUallocVector(out);
% 
% 
% 
% %%%%%
% % Full convolution padding, have to pad the images to get the full from the valid convolution.
% newmaps = zeros((smaps(1)+2*sF(1)-2)*(smaps(2)+2*sF(2)-2),numImgsPerGroup*numFiltersPerGroup*sF(5),GPUsingle);
% % newmaps = GPUsingle();
% % setReal(newmaps);
% % setSize(newmaps,[(smaps(1)+2*sF(1)-2)*(smaps(2)+2*sF(2)-2),numImgsPerGroup*numFiltersPerGroup*sF(5)]);
% % GPUallocVector(newmaps);
% 
% setSize(maps,[smaps(1)*smaps(2),numFiltersPerGroup*sF(5)]);
% cuCopyInto(maps,newmaps,sF(1)-1);
% setSize(newmaps,[(smaps(1)+2*sF(1)-2) (smaps(2)+2*sF(2)-2) numFiltersPerGroup sF(5)]);
% %%%%%
% 
% % Reshape and repmat the maps by size of sF(4) (ie. numGroups to reconstruct).
% setSize(newmaps,[size(newmaps,1)*size(newmaps,2)*numImgsPerGroup numFiltersPerGroup*numGroups])
% setSize(F,[sF(1)*sF(2) numGroups*numFiltersPerGroup]);
% 
% % if(sF(5)>1)
% %     cuConv4(newmaps,F,out,numGroups,0,1);
% % Use the quicker routing if all ones.
% % if(sum(sum(C))==numel(C))
% %     cuConv4(newmaps,F,out,numGroups,0,DIFF_F);
% % else
%     cuConv6(newmaps,F,out,C,numGroups,0,DIFF_F);
% % end
% 
% % Don't need to permute in this case for some reason.
% mySetSize(out,[(smaps(1)+sF(1)-1) (smaps(2)+sF(2)-1) sF(4) sF(5)]);
% mySetSize(maps,smaps);
% if(DIFF_F)
% mySetSize(F,sF);
% else
%     mySetSize(F,sF(1:4));
% end
%%%%%%%%%%%%%%%%%%%%%%











%%%%%%%%%%%%%%%%%%%%%%%%%%
% Permuted CPU sizes [1 2 4 3] so number of images is dim 3.
% % Do this to use the optimized code (first case below) which has less permutes.
if(size(F,5)~=size(maps,4) || size(F,5)==1)
    DIFF_F = 0;
else % Different F were passed in.
    DIFF_F = 1;
end
% Pad with zeros and do valid convolutions
out = cuConv6_2(padarray(maps,size(F,1)-1),F,C,0,DIFF_F);
%%%%%%%%%%%%%%%%%%%%%%














%
%
%
%
% if(size(maps,4)==size(F,5) && size(maps,4)>1) % If there are different filters for each image case then loop over them separately.
%
%
%     for im=1:size(F,5)
%         %%%%%%%%%%%%%%%%%
%         % Working copy that gives 3.5x speedup with 16 images.
%         sF = [size(F) 1 1 1 1];
%         % sF(3) = size(F,3);
%         % sF(4) = size(F,4);
%         % Sent in a batch of feature maps.
%         numImgsPerGroup = 1; %size(maps,4);
%         % The j index is the number of filters in each group (matches size(maps,3)).
%         numFiltersPerGroup = sF(3);
%         % The k index is the number of groups of filters.
%         numGroups = sF(4);
%
%         % Get the maps and filters for 1 image.
%         maps1 = slice(maps,':',':',':',im);
%         F1 = slice(F,':',':',':',':',im);
%
%         smaps = size(maps1);
%         if(size(smaps,2)==4)
%             maps1 = permute(maps1,[1 2 4 3]);
%         end
%
%
%         setSize(maps1,[size(maps1,1)*size(maps1,2) numImgsPerGroup*numFiltersPerGroup])
%         %%%%%
%         % Full convolution padding, have to pad the images to get the full from the valid convolution.
%         if(im==1)
%             newMaps = zeros((smaps(1)+2*sF(1)-2)*(smaps(2)+2*sF(2)-2),numImgsPerGroup*numFiltersPerGroup,GPUsingle);
%         else
%             setSize(newMaps,[(smaps(1)+2*sF(1)-2)*(smaps(2)+2*sF(2)-2) numImgsPerGroup*numFiltersPerGroup]);
%         end
%         cuCopyInto(maps1,newMaps,sF(1)-1);
%         setSize(newMaps,[(smaps(1)+2*sF(1)-2)*(smaps(2)+2*sF(2)-2)*numImgsPerGroup numFiltersPerGroup]);
%         %%%%%
%         if(numGroups>1)
%             newMaps = repmat(newMaps,[1 numGroups]);
%         end
%
%
%         setSize(F1,[sF(1)*sF(2) numGroups*numFiltersPerGroup]);
%
%         % Initialize the output of the convolution for the single image.
%         if(im==1)
%             out1 = zeros((smaps(1)+sF(1)-1)*(smaps(2)+sF(2)-1),numGroups*numImgsPerGroup,GPUsingle);
%         else
%            setSize(out1,[ (smaps(1)+sF(1)-1)*(smaps(2)+sF(2)-1) numGroups*numImgsPerGroup]);
%         end
%
%         cuConv3(newMaps,F1,out1,numGroups,0,0);
%
%
%         % Get the correct size back for the single image.
%
% %         setSize(out1,[(smaps(1)+sF(1)-1) (smaps(2)+sF(2)-1) numImgsPerGroup numGroups]);
% %         out1 = permute(out1,[1 2 4 3]);
%         setSize(out1,[(smaps(1)+sF(1)-1) (smaps(2)+sF(2)-1) numGroups numImgsPerGroup]);
%         % Make the entire output array.
%         if(im==1)
%             out = zeros([(smaps(1)+sF(1)-1) (smaps(2)+sF(2)-1) numGroups size(F,5)],GPUsingle);
%         end
%         % Place the single result into the total result.
%         assign(1,out,out1,':',':',':',im);
%     end
% %     setSize(maps,smaps);
%     setSize(F,sF);
%
% else
%
%     sF = size(F);
%     sF(3) = size(F,3);
%     sF(4) = size(F,4);
%     % Sent in a batch of feature maps.
%     numImgsPerGroup = size(maps,4);
%     % The j index is the number of filters in each group (matches size(maps,3)).
%     numFiltersPerGroup = sF(3);
%     % The k index is the number of groups of filters.
%     numGroups = sF(4);
%
%     smaps = size(maps);
%     if(size(smaps,2)==4)
%         maps = permute(maps,[1 2 4 3]);
%     end
%
%
%     setSize(maps,[size(maps,1)*size(maps,2) numImgsPerGroup*numFiltersPerGroup])
%     %%%%%
%     % Full convolution padding, have to pad the images to get the full from the valid convolution.
%     newMaps = zeros((smaps(1)+2*sF(1)-2)*(smaps(2)+2*sF(2)-2),numImgsPerGroup*numFiltersPerGroup,GPUsingle);
%     cuCopyInto(maps,newMaps,sF(1)-1);
%     setSize(newMaps,[(smaps(1)+2*sF(1)-2)*(smaps(2)+2*sF(2)-2)*numImgsPerGroup numFiltersPerGroup]);
%     %%%%%
%     if(numGroups>1)
%         newMaps = repmat(newMaps,[1 numGroups]);
%     end
%
%
%     setSize(F,[sF(1)*sF(2) numGroups*numFiltersPerGroup]);
%
%     % Initialize the running sum for each feature map.
%     out = zeros((smaps(1)+sF(1)-1)*(smaps(2)+sF(2)-1),numGroups*numImgsPerGroup,GPUsingle);
%
%     cuConv3(newMaps,F,out,numGroups,0,0);
%
%
%     % Get the correct size back.
%     setSize(out,[(smaps(1)+sF(1)-1) (smaps(2)+sF(2)-1) numImgsPerGroup numGroups]);
%     out = permute(out,[1 2 4 3]);
%     setSize(out,[(smaps(1)+sF(1)-1) (smaps(2)+sF(2)-1) numGroups numImgsPerGroup]);
%     % Make the filters the original sizes.
%     setSize(F,sF);
%     setSize(maps,smaps);
% end
%
%



% end





end