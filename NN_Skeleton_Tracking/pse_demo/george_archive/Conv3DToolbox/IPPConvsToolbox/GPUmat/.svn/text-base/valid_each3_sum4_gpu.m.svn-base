%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% This is a GPUmat implementation for doing the valid_each3_sum4 operation:
% @copybrief valid_each3_sum4.m
% @see valid_each3_sum4.m for full documentation and usage. 
%
% @file
% @author Matthew Zeiler
% @data Apr 1, 2011
%
% @conv_file @copybrief valid_each3_sum4_gpu.m
% @gpu_file @copybrief valid_each3_sum4_gpu.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief valid_each3_sum4.m
%
% @param maps  n x m x num_input_maps x num_images  matrix of single precision floating point numbers.
% @param F  p x q x num_input_maps x num_feature_maps [x num_images] matrix of single precision floating point numbers.
% can have num_images as 5th dimension but if it doesn't then it repmats by num_images for you 
% to use the same set of filters for each image.
% @param C  num_input_maps x num_feature_maps connectivity matrix.
% @param COMP_THEADS (optional and unused) specifies number of computation threads to parallelize over. Defaults to 4.
%
% @retval out (this is plhs[0]) A 4D matrix of (n+p-1) x (m+q-1) x num_input_maps x num_images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [out] = valid_each3_sum4_gpu(maps,F,C,COMP_THREADS)
% GPU Jacket implementation.
% if(isa(maps,'gsingle')) 
% 
% 
% feature_maps = size(maps,3);
% images = size(maps,4);
% % Single filter.
% if(size(F,4)==feature_maps)
%     input_maps = size(F,3);
% elseif(size(F,4)==images) % Different filters.
%     input_maps = size(F,3)/feature_maps;
% end
% 
% 
% sz = [size(maps,1)-size(F,1)+1 size(maps,2)-size(F,2)+1];
% % Create the gpu output array (going to sum over dim 4 in the end).
% out = zeros([ sz(1) sz(2) input_maps*images feature_maps],gsingle);
% 
% % If there are different filters for each image dimension of maps (dim 4 of maps) then index differently.
% % First the filters are reshaped as reshape(F,[s(1) s(2) input_maps*feature_maps])
% % The filters in this case have been repmated by num_images over 4th dimension (because Jacket can't have more than 4 dims)
% if(size(F,3)>input_maps)
% % Split up over whole batch and 
% gfor linind=1:input_maps*images
%     [jt,im] = ind2sub([input_maps images],linind);
% % Sum over this dimension.
% for kt=1:feature_maps    
%     % [jt,im,kt] = ind2sub([input_maps images feature_maps],linind);
% 
% 
%     out(:,:,linind,kt) = conv2(maps(:,:,kt,im),F(:,:,jt+(kt-1)*input_maps,im),'valid'); 
% end
% gend
% 
% else % Just a single filter to use it for all image dimension of maps. 
%     % In this case the fitlers have not been reshaped.
%     % Split up over whole batch and 
% gfor linind=1:input_maps*images
% % Sum over this dimension.
% for kt=1:feature_maps    
%     % [jt,im,kt] = ind2sub([input_maps images feature_maps],linind);
%     [jt,im] = ind2sub([input_maps images],linind);
% 
%     out(:,:,linind,kt) = conv2(maps(:,:,kt,im),F(:,:,jt,kt),'valid'); 
% end
% end
%     
% gend
% out = sum(out,4);
% out = reshape(out,[sz(1) sz(2) input_maps images]);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% 
% else % GPUmat implementation.


% % May have to flip the filters to be equivalent???
% % Fflip = clone(F);
% % size(Fflip)
% sF = size(F);
% setSize(F,[sF(1)*sF(2) sF(3)*sF(4)]);
% Fflip = clone(F);
% cuRotate180(Fflip,F);
% setSize(F,sF);
% % F = flipdim(flipdim(F,1),2);
% % F = slice(F,[END -1 1],[END -1 1],':',':');
% % clear Fflip
%
% % The k index is the number of cases.
% numCases = sF(4);
% % The j index is the number of filters.
% numFilters = sF(3);
%
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
% % tic
% % setSize(F,[sF(1)*sF(2)*numFilters numCases]);
% % Add zero filters so the total number of filters is 16.
% if(zeroFilters~=0)
%     newF = zeros(sF(1),sF(2),actFilters,numCases,GPUsingle);
%     assign(1,newF,F,':',';',[1 1 numFilters],';');
%     F = newF;
% end
% setSize(F,[sF(1)*sF(2)*actFilters numCases]);
% % GPUsync
% % t=toc;
% % cuCopyInto
%
% % Initialize the running sum for each feature map.
%
% out = zeros((smaps(1)-sF(1)+1)*(smaps(2)-sF(2)+1)*actFilters,numCases,GPUsingle);
%
% % tic;
% cuConv2(maps,F,out,sF(1));
% % GPUsync;
% % t=toc
%
% setSize(out,[(smaps(1)-sF(1)+1) (smaps(2)-sF(2)+1) actFilters numCases]);
% out = slice(out,':',':',[1 1 numFilters],':');
% % out = out(:,:,1:numFilters,:);

%
%
% GPUsync;
% t=toc



% %%%%%%%%%%%%%%%%%%%%%%
% % cuConv2 with 2.2x Layer 1 speedup for 16 images.
% % May have to flip the filters to be equivalent???
% % Fflip = clone(F);
% % size(Fflip)
% sF = size(F);
% setSize(F,[sF(1)*sF(2) sF(3)*sF(4)]);
% Fflip = clone(F);
% cuRotate180(Fflip,F);
% setSize(F,sF);
%
% % For each set of feature maps (one for each image) replicate the filters
% % To be applied to them.
% numGroups = size(maps,4);
% if(numGroups>1)
%     F = repmat(F,[1 1 1 numGroups]);
% end
%
% % sF = size(F);
% % The k index is the number of cases.
% numCases = sF(4);
% % The j index is the number of filters.
% numFilters = sF(3)
%
%
% smaps = size(maps);
% setSize(maps,[size(maps,1)*size(maps,2) numCases*numGroups]);
%
% if((numFilters/16-floor(numFilters/16))>0)  % Means that is greater than a multiple of 16
%     actFilters = (floor(numFilters/16)+1)*16;
%     zeroFilters = actFilters-numFilters % number of filters that will be zero.
% else
%     actFilters = numFilters;
%     zeroFilters = 0
% end
% % tic
% % setSize(F,[sF(1)*sF(2)*numFilters numCases]);
% % Add zero filters so the total number of filters is 16.
% if(zeroFilters~=0)
%     newF = zeros(sF(1),sF(2),actFilters,numCases*numGroups,GPUsingle);
%     assign(1,newF,F,':',':',[1 1 numFilters],':');
%     F = newF;
% end
% setSize(F,[sF(1)*sF(2)*actFilters numGroups*numCases]);
%
%
% % Initialize the running sum for each feature map.
%
% out = zeros((smaps(1)-sF(1)+1)*(smaps(2)-sF(2)+1)*actFilters,numCases*numGroups,GPUsingle);
%
% % tic;
% cuConv2(maps,F,out,sF(1));
% % GPUsync;
% % t=toc
%
% setSize(out,[(smaps(1)-sF(1)+1) (smaps(2)-sF(2)+1) actFilters numCases numGroups]);
% out = slice(out,':',':',[1 1 numFilters],':',':');
% % out = out(:,:,1:numFilters,:,:);







%%%%%%%%%%%%%%%%%%%%%%%%%%
% Permuted CPU sizes [1 2 4 3] so number of images is dim 3.
% % Do this to use the optimized code (first case below) which has less permutes.
% if(size(F,5)~=size(maps,4) || size(F,5)==1)
%     DIFF_F = 0;
% else
%     DIFF_F = 1;
% %     size(F)
% end


% origF = F*1;
% if(sum(C(:))~=numel(C))
%     GPUtimes(F,repmat(reshape(C,[1 1 size(C,1) size(C,2) 1]),[size(F,1) size(F,2) 1 1 size(F,5)]),F);
% end

% %%%%%%%%%%%%%%%%%
% % Working copy that gives 3.5x speedup with 16 images.
% sF = [size(F)];
% sF(3) = size(F,3);
% sF(4) = size(F,4);
% sF(5) = size(maps,4);
% % Sent in a batch of feature maps.
% numImgsPerGroup = 1; % size(maps,4);
% % The k index is the number of cases. (feature maps)
% numFiltersPerGroup = sF(4);
% % The j index is the number of filters in each group. (input maps)
% numGroups = sF(3)*sF(5);
% if(sF(3)>1 || sF(4)>1)
%     if(DIFF_F) % Different filters means 5 dimensions.
%         F = permute(F,[1 2 4 3 5]);
%     else
%         F = permute(F,[1 2 4 3]);
%     end
% end
% smaps = size(maps);
% 
% % % Initialize the running sum for each feature map.
% % out = zeros([(size(maps,1)-sF(1)+1)*(size(maps,2)-sF(2)+1)*sF(3) sF(5)],GPUsingle);
% 
% %%%%
% % Low level allocations
% out = GPUsingle();
% setReal(out);
% setSize(out,[(size(maps,1)-sF(1)+1)*(size(maps,2)-sF(2)+1)*sF(3) sF(5)]);
% GPUallocVector(out);
% 
% 
% % Reshape and repmat the maps by size of sF(3) (ie. numGroups to reconstruct).
% % setSize(newmaps,[size(maps,1)*size(maps,2)*numImgsPerGroup numFiltersPerGroup*numGroups])
% mySetSize(maps,[size(maps,1)*size(maps,2)*numImgsPerGroup numGroups])
% mySetSize(F,[sF(1)*sF(2) numGroups*numFiltersPerGroup]);
% 
% % If all ones use the quicker routine.
% % if(sum(sum(C))==numel(C))
% %     cuConv4(maps,F,out,numGroups,0,DIFF_F);
% % else
%     cuConv6(maps,F,out,C',numGroups,0,DIFF_F);
% % end
% % Don't need to permute in this case for some reason.
% mySetSize(out,[(smaps(1)-sF(1)+1) (smaps(2)-sF(2)+1) sF(3) sF(5)]);
% mySetSize(maps,smaps);
% if(DIFF_F)
%     mySetSize(F,sF);
% else
%     mySetSize(F,sF(1:4));
% end
% 
% % end


% 
% 
% %%%%%%%%%%%%%%%%%
% % Do this to use the optimized code (first case below) which has less permutes.
if(size(F,5)~=size(maps,4) || size(F,5)==1)
    DIFF_F = 0;
else
    DIFF_F = 1;
end
% 
% if(size(F,3)>1 || size(F,4)>1)
%     if(DIFF_F) % Different filters means 5 dimensions.
%         F = permute(F,[1 2 4 3 5]);
%     else
%         F = permute(F,[1 2 4 3]);
%     end
% end
% out = cuConv6_2(maps,F,C',0,DIFF_F);



% if(size(F,3)>1 || size(F,4)>1)
%     if(DIFF_F) % Different filters means 5 dimensions.
% %         F = permute(F,[1 2 4 3 5]);
%     else
%         F = permute(F,[1 2 4 3]);
%     end
% end

% This does the reconstruction convolutions without permutation.
out = cuConv8_2(maps,F,C,0,DIFF_F);






%
%
% % If there is a 5D vector of filters coming in, they must be different for each image.
% if(size(F,5)==size(maps,3) && size(F,5)>1) % && size(F,3)==1)
% %
%     %%%%%%%%%%%%%%%%%
%     % Working copy that gives 3.5x speedup with 16 images.
%     sF = size(F);
%     sF(3) = size(F,3);
%     sF(4) = size(F,4);
%     sF(5) = size(F,5);
%     % Sent in a batch of feature maps.
%     numImgsPerGroup = 1; % size(maps,4);
%     % The k index is the number of cases. (feature maps)
%     numFiltersPerGroup = sF(4);
%     % The j index is the number of filters in each group. (input maps)
%     numGroups = sF(3)*size(maps,3);
%     if(sF(3)>1 || sF(4)>1)
%         F = permute(F,[1 2 4 3 5]);
% % F = permute(F,[1 2 3 5 4]);
%     end
%
%     smaps = size(maps);
%
%     % Initialize the running sum for each feature map.
%     out = zeros([(size(maps,1)-sF(1)+1)*(size(maps,2)-sF(2)+1) sF(3)*sF(5)],GPUsingle);
%
%
%
%     % Reshape and repmat the maps by size of sF(3) (ie. numGroups to reconstruct).
% %     if(sF(3)>1)
% %        newmaps = repmat(maps,[1 1 1 sF(3) 1]);
% %            setSize(newmaps,[size(maps,1)*size(maps,2)*numFiltersPerGroup numImgsPerGroup*numGroups])
%
% %     else
%         setSize(maps,[size(maps,1)*size(maps,2)*numImgsPerGroup numFiltersPerGroup*size(maps,3)]);
% %     end
%     if(sF(3)>1)
%         newmaps = repmat(maps,[1 sF(3)]);
%     end
%
%
%     %
%     setSize(F,[sF(1)*sF(2) numGroups*numFiltersPerGroup]);
%     if(sF(3)>1)
%             cuConv3(newmaps,F,out,numGroups,0,0);
%     else
%     cuConv3(maps,F,out,numGroups,0,0);
%     end
%
%
%     % Get the correct size back.
% %     if(sF(3)>1)
%         % Could stop here if images were xdim x ydim x num_images x num_input_planes.
%         setSize(out,[(smaps(1)-sF(1)+1) (smaps(2)-sF(2)+1) sF(5) sF(3)]);
%         out = permute(out,[1 2 4 3]);
% %     end
%     % Don't need to permute in this case for some reason.
%     setSize(out,[(smaps(1)-sF(1)+1) (smaps(2)-sF(2)+1) sF(3) sF(5)]);
%     setSize(maps,smaps);
%     setSize(F,sF);
% else
%
%
%     %%%%%%%%%%%%%%%%%
%     % Working copy that gives 3.5x speedup with 16 images.
%     sF = size(F);
%     sF(3) = size(F,3);
%     sF(4) = size(F,4);
%     % Sent in a batch of feature maps.
%     numImgsPerGroup = size(maps,4);
%     % The k index is the number of cases. (feature maps)
%     numFiltersPerGroup = sF(4);
%     % The j index is the number of filters in each group. (input maps)
%     numGroups = sF(3);
%
%     smaps = size(maps);
%     if(size(smaps,2)==4) % if it is a 4D array.
%         maps = permute(maps,[1 2 4 3]);
%     end
%
%     setSize(maps,[size(maps,1)*size(maps,2)*numImgsPerGroup numFiltersPerGroup])
%     if(numGroups>1)
%         newmaps = repmat(maps,[1 numGroups]);
%     end
%
%     %     setSize(maps,[size(maps,1)*size(maps,2)*numFiltersPerGroup numImgsPerGroup])
%     %     if(numGroups>1)
%     %         maps = repmat(maps,[1 numGroups]);
%     %     end
%
%     if(sF(3)>1 || sF(4)>1)
%         if(size(F,5)>1)
%             F = permute(F,[1 2 4 3 5]);
%         else
%             F = permute(F,[1 2 4 3]);
%         end
%     end
%     setSize(F,[sF(1)*sF(2) numGroups*numFiltersPerGroup]);
%
%     % Initialize the running sum for each feature map.
%     out = zeros((smaps(1)-sF(1)+1)*(smaps(2)-sF(2)+1),numGroups*numImgsPerGroup,GPUsingle);
%
%     if(numGroups>1)
%     cuConv3(newmaps,F,out,numGroups,0,0);
%     else
%         cuConv3(maps,F,out,numGroups,0,0);
%     end
%
%
%     % Get the correct size back.
%     if(numGroups>1)
%         setSize(out,[(smaps(1)-sF(1)+1) (smaps(2)-sF(2)+1) numImgsPerGroup numGroups]);
%         out = permute(out,[1 2 4 3]);
%     end
%     setSize(out,[(smaps(1)-sF(1)+1) (smaps(2)-sF(2)+1) numGroups numImgsPerGroup]);
%     setSize(F,sF);
%     setSize(maps,smaps);
% end
% %%%%%%%%%%%%%%%%%%%%%





% %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Standard CPU sizes.
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% % Do this to use the optimized code (first case below) which has less permutes.
% if(size(F,5)~=size(maps,4))
%     F = repmat(F,[1 1 1 1 size(maps,4)]);
% end
%
%
% % If there is a 5D vector of filters coming in, they must be different for each image.
% if(size(F,5)==size(maps,4) && size(F,5)>1) % && size(F,3)==1)
%
%     %%%%%%%%%%%%%%%%%
%     % Working copy that gives 3.5x speedup with 16 images.
%     sF = size(F);
%     sF(3) = size(F,3);
%     sF(4) = size(F,4);
%     sF(5) = size(F,5);
%     % Sent in a batch of feature maps.
%     numImgsPerGroup = 1; % size(maps,4);
%     % The k index is the number of cases. (feature maps)
%     numFiltersPerGroup = sF(4);
%     % The j index is the number of filters in each group. (input maps)
%     numGroups = sF(3)*size(maps,4);
%     if(sF(3)>1 || sF(4)>1)
%         F = permute(F,[1 2 4 5 3]);
%     end
%
%     smaps = size(maps);
%
%     % Initialize the running sum for each feature map.
%     out = zeros([(size(maps,1)-sF(1)+1)*(size(maps,2)-sF(2)+1) sF(3)*sF(5)],GPUsingle);
%
%
%
%     % Reshape and repmat the maps by size of sF(3) (ie. numGroups to reconstruct).
%     setSize(maps,[size(maps,1)*size(maps,2)*numImgsPerGroup numFiltersPerGroup*sF(5)])
%     if(numGroups>1)
%         newmaps = repmat(maps,[1 sF(3)]);
%     end
%
%     %
%     setSize(F,[sF(1)*sF(2) numGroups*numFiltersPerGroup]);
%
%     cuConv3(newmaps,F,out,numGroups,0,0);
%
%
%     % Get the correct size back.
%     if(sF(3)>1)
%         % Could stop here if images were xdim x ydim x num_images x num_input_planes.
%         setSize(out,[(smaps(1)-sF(1)+1) (smaps(2)-sF(2)+1) sF(5) sF(3)]);
%         out = permute(out,[1 2 4 3]);
%     end
%     % Don't need to permute in this case for some reason.
%     setSize(out,[(smaps(1)-sF(1)+1) (smaps(2)-sF(2)+1) sF(3) sF(5)]);
%     setSize(maps,smaps);
%     setSize(F,sF);
% else
%
%
%     %%%%%%%%%%%%%%%%%
%     % Working copy that gives 3.5x speedup with 16 images.
%     sF = size(F);
%     sF(3) = size(F,3);
%     sF(4) = size(F,4);
%     % Sent in a batch of feature maps.
%     numImgsPerGroup = size(maps,4);
%     % The k index is the number of cases. (feature maps)
%     numFiltersPerGroup = sF(4);
%     % The j index is the number of filters in each group. (input maps)
%     numGroups = sF(3);
%
%     smaps = size(maps);
%     if(size(smaps,2)==4) % if it is a 4D array.
%         maps = permute(maps,[1 2 4 3]);
%     end
%
%     setSize(maps,[size(maps,1)*size(maps,2)*numImgsPerGroup numFiltersPerGroup])
%     if(numGroups>1)
%         newmaps = repmat(maps,[1 numGroups]);
%     end
%
%     %     setSize(maps,[size(maps,1)*size(maps,2)*numFiltersPerGroup numImgsPerGroup])
%     %     if(numGroups>1)
%     %         maps = repmat(maps,[1 numGroups]);
%     %     end
%
%     if(sF(3)>1 || sF(4)>1)
%         if(size(F,5)>1)
%             F = permute(F,[1 2 4 3 5]);
%         else
%             F = permute(F,[1 2 4 3]);
%         end
%     end
%     setSize(F,[sF(1)*sF(2) numGroups*numFiltersPerGroup]);
%
%     % Initialize the running sum for each feature map.
%     out = zeros((smaps(1)-sF(1)+1)*(smaps(2)-sF(2)+1),numGroups*numImgsPerGroup,GPUsingle);
%
%     if(numGroups>1)
%     cuConv3(newmaps,F,out,numGroups,0,0);
%     else
%         cuConv3(maps,F,out,numGroups,0,0);
%     end
%
%
%     % Get the correct size back.
%     if(numGroups>1)
%         setSize(out,[(smaps(1)-sF(1)+1) (smaps(2)-sF(2)+1) numImgsPerGroup numGroups]);
%         out = permute(out,[1 2 4 3]);
%     end
%     setSize(out,[(smaps(1)-sF(1)+1) (smaps(2)-sF(2)+1) numGroups numImgsPerGroup]);
%     setSize(F,sF);
%     setSize(maps,smaps);
% end


end


