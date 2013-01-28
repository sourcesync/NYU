function runme
%run test

imgSize=150;
numImgsPerGroup=128;
numGroups=1;
numFiltersPerGroup=48;
imgOrder=0;
color=0; %bw

maxFilterSize=38; % there is a special case for filter sizes > 37

for order=0:1 % test both orderings: GROUP_FILTER_IMAGE, IMAGE_GROUP_FILTER
    
    fprintf('* Start test with ordering %d *\n',order);
    
    for filterSize=7:7
        
        
        if filterSize>21
            tol=1; % error can be quite big for large filter sizes, see Alex's C++ tests
        else
            
            % Had to raise the tolerance here compared to cuConv and cuConv2
            % Likely the reason is that the result is a sum of many convolutions
            % Numerical error adds up
            tol=1e-1;
        end
        
        
        fprintf('filterSize %d/%d\n',filterSize,maxFilterSize);
        
        filterPixels=filterSize*filterSize;
        imgPixels=imgSize*imgSize;
        numOutputsX = imgSize - filterSize + 1;
        numOutputs = numOutputsX*numOutputsX;
        
        %Assume that each row of filtersH is a numFilters*filterSize*filterSize filter
        %Flattened in ROW-MAJOR format
        filtersH = rand(numFiltersPerGroup*numGroups,filterPixels,'single');
        %Assume that each row of imagesH is a imgSize*imgSize image
        %Flattened in ROW-MAJOR format
        if imgOrder==0
            % GROUP_FILTER_IMAGE
            imagesH = rand(numFiltersPerGroup*numGroups,numImgsPerGroup*imgPixels,'single');
        elseif imgOrder==1
            % IMAGE_GROUP_FILTER
            imagesH = rand(numImgsPerGroup*numGroups,numFiltersPerGroup*imgPixels,'single');
        else
            error('Unsupported ordering')
        end
        
        %Note that even though Matlab is going to interpret images & filters in
        %COLUMN-MAJOR format, both the image and filter will be transposed
        %So the result of convolution will just be a transposed version of the
        %(C/C++) result
        %However, when we flatten this transposed result to vector form in
        %COLUMN-MAJOR format (using Matlab) this will give us the equivalent of
        %the (C/C++) proper result in ROW-MAJOR format
        %Which is what we want!
        
        fprintf('Starting convolution on CPU\n');
        tic;targetsH = convolve3_bw(imagesH,filtersH,numGroups,numImgsPerGroup,color,imgOrder);
        fprintf('Done convolution on CPU: %fs\n',toc);
        
        %Unfortunately just like mxArrays, GPUsingle type stores data in
        %COLUMN-MAJOR format
        %Since all of Alex's convolutional code assumes ROW-MAJOR format we take
        %the transpose of the data before making GPUsingles
        %this ensures the data is in ROW-MAJOR form (i.e. cases are consecutive,
        %within each case rows are consecutive, etc.)
        filters = GPUsingle(transpose(filtersH));
        images = GPUsingle(transpose(imagesH));
        targets = zeros(numOutputs,numImgsPerGroup*numGroups, GPUsingle); %note this is
        %transposed
        %from the
        %Matlab
        %version
        fprintf('Starting convolution on GPU\n');
        tic;cuConv3(images,filters,targets,numGroups,color,order);GPUsync;
        fprintf('Done convolution on GPU: %fs\n',toc);
        
        targets_h = transpose(single(targets)); %move to CPU and transpose to
        %align with targetsH
        
        fprintf('Showing some of CPU result:\n');
        targetsH(1:3,1:6)
        
        fprintf('Showing some of GPU result:\n');
        targets_h(1:3,1:6)
        
        
        maxabsdiff = max(abs(targetsH(:)-targets_h(:)));
        fprintf('Max abs difference: %f\n', maxabsdiff );
        assert(maxabsdiff<tol);
        
    end
    
    fprintf('* End test with ordering %d *\n',order);
    
end
