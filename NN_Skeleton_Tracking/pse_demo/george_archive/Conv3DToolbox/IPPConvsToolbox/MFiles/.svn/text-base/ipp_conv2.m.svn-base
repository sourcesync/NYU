%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Non-IPP wrapper for 2-D image convolutions to be compatible with the version
% that uses the IPP libraries in the IPPConvToolbox.
%
% There are several cases based on the input.
% Case 1 (c images and single kernel):
% \f[
% out(:,:,c) = image(:,:,c) \oplus kernel
% \f]
%
% Case 2 (single image and c kernels):
% \f[
% out(:,:,c) = image \oplus kernel(:,:,c)
% \f]
%
% Case 3 (c images and c kernels where c>= 1):
% \f[
% out(:,:,c) = image(:,:,c) \oplus kernel(:,:,c)
% \f]
%
% @file
% @author Matthew Zeiler (zeiler@cs.nyu.edu)
% @data Aug 26, 2010
%
% @ipp_file @copybrief ipp_conv2.cpp
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief full_eachJ_loopK.m
%
% @param maps n x m x c  matrix of single precision floating point numbers.
% @param maps2 i x j x d matrix of single precision floating point numbers.
% @param mode a string either 'valid' or 'full' indicating the type of convolution.
% @param COMP_THEADS (optional and unused) specifies number of computation threads to parallelize over. Defaults to 4.
%
% @retval out (this is plhs[0]) A 3D matrix of outSize x outSize x c  (depends on different cases described in the deatiled documentation)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [out] = ipp_conv2(maps,maps2,mode,COMP_THREADS)

% This isn't even used though.
if(nargin<4)
    COMP_THREADS=4;
end

fprintf(1,'WARNING: You should compile the MEX version of "ipp_conv2.cpp",\n         found in the IPPConvsToolbox, and put it in your matlab path.  It is MUCH faster.\n');



% Initialize the running sum for each feature map.
if(strcmp(mode,'full'))
    out = zeros(size(maps,1)+size(maps2,1)-1,size(maps,2)+size(maps2,2)-1,max(size(maps,3),size(maps2,3)),'single');
else
    out = zeros(size(maps,1)-size(maps2,1)+1,size(maps,2)-size(maps2,2)+1,max(size(maps,3),size(maps2,3)),'single');
end



% Convolve each plane with each filter.
for j=1:max(size(maps,3),size(maps2,3))
    out(:,:,j) = conv2(maps(:,:,min(j,size(maps,3))),maps2(:,:,min(j,size(maps2,3))),mode);
end


end

