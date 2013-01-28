#include "GPUkernel.hh"
#include "conv.cuh"

#define MUL24 __mul24

//To use templates we must have C++ kernels
//extern "C" {

/*
 * This function uses block size 16x16.
 * Works for filters up to 37x37.
 */
template<int filterSize, bool checkBounds, int stride, bool gfi>
        __global__ void conv5_bw_fit_16x16(float* imgs, float* filters, float* targets,
        const int imgSize, const int numFiltersPerGroup, const int numGroups) {
    const int shImgSizeX = filterSize + 15, shImgSizeY = filterSize + 15;
    __shared__ float shImg[shImgSizeY][shImgSizeX];
    __shared__ float shFilter[filterSize][filterSize];
    
//    const int numImgsPerGroup = gridDim.x / (numGroups*numFiltersPerGroup); // This is sF(4) since numGroups is now sF(5) and numFiltersPerGroup is sF(3)
//    const int numImgsPerGroup = gridDim.x / numGroups; // This is sF(3) since numGroups is now sF(5)
    
//    const int imgIdxInGroup = blockIdx.x % numImgsPerGroup;
//    const int imgIdxInGroup = 0; // This should be 0 always like beforei fnumImgsPerGroup is 1.
//    const int groupIdx = blockIdx.x; // This is the target index used for targets as it should be [1:sF(3)*sF(4)*sF(5)]
//    const int ImgGroupIdx = blockIdx.x / numImgsPerGroup; // This is dividing [1:sF(3)*sF(5)]/sF(3) so it ranges [1:sF(5)]
//    const int InpMapIdx = blockIdx.x % numFiltersPerGroup; // Fitlers per group are sF(3) so this is the input map index.
//    const int FeatMapIdx = (blockIdx.x/numFiltersPerGroup) % numImgsPerGroup; // The index of the feature maps.
//    const int ImgIdx = blockIdx.x / (numFiltersPerGroup*numImgsPerGroup); // The index of the numCases (number of images).
    
//    const int outputPart = blockIdx.y;
//    const int numOutputsX = imgSize - filterSize + 1;
//    const int numOutputs = MUL24(numOutputsX, numOutputsX);
//    const int outputPartY = outputPart / DIVUP(numOutputsX, 16);
//    const int outputPartX = outputPart % DIVUP(numOutputsX, 16);
    
    // imgSize is just one of the squared dimensions.
//    const int imgPixels = MUL24(imgSize, imgSize);
    
//    const int shImgPixels = MUL24(shImgSizeX, shImgSizeY); // size of shared buffer for image
//    const int filterPixels = filterSize * filterSize;
    
    
// Since grid is limited to 65535 in size in each dim, have to split this way.
    const int numImgsPerGroup = gridDim.x / numGroups; // Is number of feature maps
    const int FeatMapIdx = blockIdx.x % numImgsPerGroup; // The index of the feature maps.
    const int ImgIdx = blockIdx.x / numImgsPerGroup; // The index of the numCases (number of images).
    
    
    
    const int numOutputsX = imgSize - filterSize + 1;
    const int numOutputs = MUL24(numOutputsX, numOutputsX);
    const int numPartsX = DIVUP(numOutputsX, 16);
    const int numParts = numPartsX*numPartsX;
    
    // Assume the y dimension does the output Parts in groups first for a given InpMapIdx
    const int outputPart = blockIdx.y % numParts; // Due to 65535 dimension size, have to split up grid in y dimension by num input maps as well.
    const int InpMapIdx = blockIdx.y / numParts;
    
    const int outputPartY = outputPart / numPartsX;
    const int outputPartX = outputPart % numPartsX;
    const int imgPixels = MUL24(imgSize, imgSize); // size of image
    const int filterPixels = MUL24(filterSize, filterSize);
    
    // Now this is linear outpu (target) index based on lockIdx.x and .y
    const int groupIdx = InpMapIdx + FeatMapIdx*numFiltersPerGroup + ImgIdx*(numImgsPerGroup*numFiltersPerGroup);
    
    
    const int tidx = threadIdx.y * 16 + threadIdx.x; // thread's index within the 16x16 "plate" of threads
    
    
    // Changes made:
    // Took out any use of imgIdxInGroup from targets and filters because that used to be zero and it worked fine before.
    
    if (gfi) { // Actually I think this is the one I'm using when 0 is passed in.
        //     imgs += MUL24(MUL24(groupIdx, numImgsPerGroup/stride), numFiltersPerGroup) * imgPixels
        //           + MUL24(imgIdxInGroup/stride, imgPixels)
        //          + MUL24(outputPartY, imgSize) * 16 + outputPartX * 16; // hid acts for conv rbm
        //   imgs += MUL24(FeatMapIdx, numImgsPerGroup) * imgPixels
        //       + MUL24(outputPartY, imgSize) * 16 + outputPartX * 16;
        imgs += FeatMapIdx * imgPixels
                + MUL24(ImgIdx, numImgsPerGroup) * imgPixels
                + MUL24(outputPartY, imgSize) * 16 + outputPartX * 16;
    } else { // This is the case I use (imgOrder==0)
        // The looping below loops over the number of Filters per group.
        // So we want a new group index for this that doesn't
        //imgs += MUL24(MUL24(imgIdxInGroup/stride, numGroups), numFiltersPerGroup) * imgPixels
        //      + MUL24(groupIdx, numImgsPerGroup) * imgPixels
        //      + MUL24(outputPartY, imgSize) * 16 + outputPartX * 16; // hid acts for conv rbm
        // Use the slower incrementing ImgGroupIdx [1:sF(5)] so jusmp by numFiltersPerGroup each time (then lool 1:numFiltersPerGroup below).
        imgs += FeatMapIdx * imgPixels
                + MUL24(ImgIdx, numImgsPerGroup) * imgPixels
                + MUL24(outputPartY, imgSize) * 16 + outputPartX * 16;
    }
    // target (outputs) should stay the same as before as they are [outx outy numGroups].
    // They are kind of funny though as they are split into groups of 16 pixels I believe.
    //targets += MUL24(groupIdx, numImgsPerGroup) * numOutputs
    //         + MUL24(imgIdxInGroup, numOutputs)
    //         + MUL24(outputPartY, numOutputsX) * 16 + outputPartX * 16
    //         + MUL24(threadIdx.y, numOutputsX) + threadIdx.x;
    // We can just go in by blockIdx for the outputs.
    targets += groupIdx * numOutputs
            + MUL24(outputPartY, numOutputsX) * 16 + outputPartX * 16
            + MUL24(threadIdx.y, numOutputsX) + threadIdx.x;
    
    if (filterSize <= 16)
        filters += tidx;
    // Have to do each filter with each image and skip whole groups of filters when we change cases.
    filters += InpMapIdx * filterPixels * stride
            + MUL24(ImgIdx, numFiltersPerGroup) * filterPixels * stride;
    //+ MUL24(imgIdxInGroup % stride, filterPixels);
    //const float* lastFilter = filters + filterPixels * stride * numFiltersPerGroup; // bad pointer
    // Don't do the summation by setting the numFiltersPerGroup here to be 1.
    const float* lastFilter = filters + filterPixels * stride * 1; // bad pointer
    float prod = 0;
    const bool compute = !checkBounds || (outputPartX * 16 + threadIdx.x < numOutputsX && outputPartY * 16 + threadIdx.y < numOutputsX);
    const int cmpX = imgSize - outputPartX * 16, cmpY = imgSize - outputPartY*16;
    do { // loop over all image/filter pairs (image = hidden activations in conv rbm)
        __syncthreads();
        
        /*
         * It might seem strange to have all these ifs explicitly in the loops rather than
         * just looping from x = threadIdx.x to min(shImgSizeX, cmpX), but this makes the loop bounds
         * compile-time constants, which allows the compiler to unroll the inner loop.
         */
        // Load image piece into shmem
        if (checkBounds) {
            int y;
            for (y = 0; y < shImgSizeY - 16; y += 16) {
                const int loadY = threadIdx.y + y;
                if (loadY < cmpY) {
                    int x;
                    for (x = 0; x < shImgSizeX - 16; x += 16) {
                        const int loadX = threadIdx.x + x;
                        if (loadX < cmpX) {
                            shImg[loadY][loadX] = imgs[MUL24(loadY, imgSize) + loadX];
                        }
                    }
                    const int loadX = threadIdx.x + x;
                    if (loadX < shImgSizeX && loadX < cmpX) {
                        shImg[loadY][loadX] = imgs[MUL24(loadY, imgSize) + loadX];
                    }
                }
            }
            const int loadY = threadIdx.y + y;
            if (loadY < shImgSizeY && loadY < cmpY) {
                int x;
                for (x = 0; x < shImgSizeX - 16; x += 16) {
                    const int loadX = threadIdx.x + x;
                    if (loadX < cmpX) {
                        shImg[loadY][loadX] = imgs[MUL24(loadY, imgSize) + loadX];
                    }
                }
                const int loadX = threadIdx.x + x;
                if (loadX < shImgSizeX && loadX < cmpX) {
                    shImg[loadY][loadX] = imgs[MUL24(loadY, imgSize) + loadX];
                }
            }
        } else { // turns out this is faster than computing indices using division/mod
            int y;
            for (y = 0; y < shImgSizeY - 16; y += 16) {
                const int loadY = threadIdx.y + y;
                int x;
                for (x = 0; x < shImgSizeX - 16; x += 16) {
                    const int loadX = threadIdx.x + x;
                    shImg[loadY][loadX] = imgs[MUL24(loadY, imgSize) + loadX];
                }
                const int loadX = threadIdx.x + x;
                if (loadX < shImgSizeX) {
                    shImg[loadY][loadX] = imgs[MUL24(loadY, imgSize) + loadX];
                }
            }
            const int loadY = threadIdx.y + y;
            if (loadY < shImgSizeY) {
                int x;
                for (x = 0; x < shImgSizeX - 16; x += 16) {
                    const int loadX = threadIdx.x + x;
                    shImg[loadY][loadX] = imgs[MUL24(loadY, imgSize) + loadX];
                }
                const int loadX = threadIdx.x + x;
                if (loadX < shImgSizeX) {
                    shImg[loadY][loadX] = imgs[MUL24(loadY, imgSize) + loadX];
                }
            }
        }
        
        // Load filter into shmem
        if (filterSize <= 16) {
            if (tidx < filterPixels)
                shFilter[0][tidx] = filters[0];
        } else {
            #pragma unroll
                    for (int y = 0; y < filterSize; y += 16) {
                const int loadY = threadIdx.y + y;
                if (loadY < filterSize) {
                    for (int x = 0; x < filterSize; x += 16) {
                        const int loadX = threadIdx.x + x;
                        if (loadX < filterSize) {
                            shFilter[loadY][loadX] = filters[MUL24(loadY, filterSize) + loadX];
                        }
                    }
                }
                    }
        }
        
        __syncthreads();
        
        if (compute) {
            const float* myShFilter = &shFilter[filterSize - 1][filterSize - 1];
            const float* myShImg = &shImg[threadIdx.y][threadIdx.x];
            
            if(filterSize < 16) {
                #pragma unroll // commented to speed up compiling
                        for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {
                        prod += myShFilter[0] * myShImg[0];
                        
                        myShFilter--;
                        myShImg++;
                    }
                    myShImg += 15;
                        }
            } else {
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {
                        prod += myShFilter[0] * myShImg[0];
                        
                        myShFilter--;
                        myShImg++;
                    }
                    myShImg += 15;
                }
            }
        }
        if (gfi) {
            imgs += MUL24(numImgsPerGroup/stride, imgPixels);
            //   imgs += imgPixels;
        } else { // My 0 gcase
            imgs += imgPixels;
        }
        filters += filterPixels * stride;
    } while (filters != lastFilter);
    
    if (compute) {
        targets[0] = prod;
    }
}


/*
 * This function uses block size 16x16.
 * Use for filters > 37x37.
 */
template<bool checkOutputBounds, bool checkFilterBounds, int stride, bool gfi>
        __global__ void conv5_bw_nofit_16x16(float* imgs, float* filters, float* targets,
        const int imgSize, const int filterSize, const int numFiltersPerGroup, const int numGroups) {
    const int shImgSizeX = 16 * 2 - 1, shImgSizeY = 16 * 2 - 1;
    __shared__ float shImg[shImgSizeY][shImgSizeX];
    __shared__ float shFilter[16][16];
    
//    const int imgIdxInGroup = blockIdx.x % numImgsPerGroup; // Is 0
//    const int groupIdx = blockIdx.x / numImgsPerGroup; // Is blockIdx.x [1:numGroups], want to mod this by num_input_maps
//    const int numImgsPerGroup = gridDim.x / (numGroups*numFiltersPerGroup); // This is sF(3) since numGroups is now sF(5)
    //const int imgIdxInGroup = blockIdx.x % numImgsPerGroup;
    //   const int imgIdxInGroup = 0; // This should be 0 always like beforei fnumImgsPerGroup is 1.
    //  const int groupIdx = blockIdx.x; // This is the index used for targets and filters as it should be [1:sF(3)*sF(5)]
    //const int ImgGroupIdx = blockIdx.x / numImgsPerGroup; // This is dividing [1:sF(3)*sF(5)]/sF(3) so it ranges [1:sF(5)]
//    const int InpMapIdx = blockIdx.x % numFiltersPerGroup; // Fitlers per group are sF(3) so this is the input map index.
    
    // Since grid is limited to 65535 in size in each dim, have to split this way.
    const int numImgsPerGroup = gridDim.x / numGroups; // Is number of feature maps
    const int FeatMapIdx = blockIdx.x % numImgsPerGroup; // The index of the feature maps.
    const int ImgIdx = blockIdx.x / numImgsPerGroup; // The index of the numCases (number of images).
    
    
    
    const int numOutputsX = imgSize - filterSize + 1;
    const int numOutputs = MUL24(numOutputsX, numOutputsX);
    const int numPartsX = DIVUP(numOutputsX, 16);
    const int numParts = numPartsX*numPartsX;
    
    // Assume the y dimension does the output Parts in groups first for a given InpMapIdx
    const int outputPart = blockIdx.y % numParts; // Due to 65535 dimension size, have to split up grid in y dimension by num input maps as well.
    const int InpMapIdx = blockIdx.y / numParts;
    
    const int outputPartY = outputPart / numPartsX;
    const int outputPartX = outputPart % numPartsX;
    const int imgPixels = MUL24(imgSize, imgSize); // size of image
    const int filterPixels = MUL24(filterSize, filterSize);
    
    // Now this is linear outpu (target) index based on lockIdx.x and .y
    const int groupIdx = InpMapIdx + FeatMapIdx*numFiltersPerGroup + ImgIdx*(numImgsPerGroup*numFiltersPerGroup);
    
    
    
    if (gfi) { // Actually I think this is the one I'm using when 0 is passed in.
        //     imgs += MUL24(MUL24(groupIdx, numImgsPerGroup/stride), numFiltersPerGroup) * imgPixels
        //           + MUL24(imgIdxInGroup/stride, imgPixels)
        //          + MUL24(outputPartY, imgSize) * 16 + outputPartX * 16; // hid acts for conv rbm
        //   imgs += MUL24(FeatMapIdx, numImgsPerGroup) * imgPixels
        //       + MUL24(outputPartY, imgSize) * 16 + outputPartX * 16;
        imgs += FeatMapIdx * imgPixels
                + MUL24(ImgIdx, numImgsPerGroup) * imgPixels
                + MUL24(outputPartY, imgSize) * 16 + outputPartX * 16;
    } else { // This is the case I use (imgOrder==0)
        // The looping below loops over the number of Filters per group.
        // So we want a new group index for this that doesn't
        //imgs += MUL24(MUL24(imgIdxInGroup/stride, numGroups), numFiltersPerGroup) * imgPixels
        //      + MUL24(groupIdx, numImgsPerGroup) * imgPixels
        //      + MUL24(outputPartY, imgSize) * 16 + outputPartX * 16; // hid acts for conv rbm
        // Use the slower incrementing ImgGroupIdx [1:sF(5)] so jusmp by numFiltersPerGroup each time (then lool 1:numFiltersPerGroup below).
        imgs += FeatMapIdx * imgPixels
                + MUL24(ImgIdx, numImgsPerGroup) * imgPixels
                + MUL24(outputPartY, imgSize) * 16 + outputPartX * 16;
    }
    // target (outputs) should stay the same as before as they are [outx outy numGroups].
    // They are kind of funny though as they are split into groups of 16 pixels I believe.
    //targets += MUL24(groupIdx, numImgsPerGroup) * numOutputs
    //         + MUL24(imgIdxInGroup, numOutputs)
    //         + MUL24(outputPartY, numOutputsX) * 16 + outputPartX * 16
    //         + MUL24(threadIdx.y, numOutputsX) + threadIdx.x;
    // We can just go in by blockIdx for the outputs.
    targets += groupIdx * numOutputs
            + MUL24(outputPartY, numOutputsX) * 16 + outputPartX * 16
            + MUL24(threadIdx.y, numOutputsX) + threadIdx.x;
    
    
    // Have to do each filter with each image and skip whole groups of filters when we change cases.
    filters += InpMapIdx * filterPixels * stride
            + MUL24(ImgIdx, numFiltersPerGroup) * filterPixels * stride;
    //+ MUL24(imgIdxInGroup % stride, filterPixels);
    //const float* lastFilter = filters + filterPixels * stride * numFiltersPerGroup; // bad pointer
    // Don't do the summation by setting the numFiltersPerGroup here to be 1.
    const float* lastFilter = filters + filterPixels * stride * 1; // bad pointer
    float prod = 0;
    const bool compute = !checkOutputBounds || (outputPartX * 16 + threadIdx.x < numOutputsX && outputPartY * 16 + threadIdx.y < numOutputsX);
    const int cmpX = imgSize - outputPartX * 16, cmpY = imgSize - outputPartY * 16;
    
    float* shFilterLoad = &shFilter[15 - threadIdx.y][15 - threadIdx.x];
    float* shImgLoad = &shImg[threadIdx.y][threadIdx.x];
    do { // loop over all image/filter pairs (image = hidden activations in conv rbm)
        for (int fY = 0; fY < filterSize; fY += 16) {
            for (int fX = 0; fX < filterSize; fX += 16) {
                __syncthreads();
                
                // Load image piece into shmem
                // this must exist cause f > 37 ==> i > 37
                
                if (!checkOutputBounds || threadIdx.x + fX < cmpX && threadIdx.y + fY < cmpY)
                    shImgLoad[0] = imgs[MUL24(threadIdx.y + fY, imgSize) + threadIdx.x + fX];
                if (!checkOutputBounds || threadIdx.x + fX + 15 < cmpX && threadIdx.y + fY < cmpY)
                    shImgLoad[15] = imgs[MUL24(threadIdx.y + fY, imgSize) + threadIdx.x + fX + 15];
                if (!checkOutputBounds || threadIdx.x + fX < cmpX && threadIdx.y + fY + 15 < cmpY)
                    shImgLoad[15 * shImgSizeX] = imgs[MUL24(threadIdx.y + fY + 15, imgSize) + threadIdx.x + fX];
                if (!checkOutputBounds || threadIdx.x + fX + 15 < cmpX && threadIdx.y + fY + 15 < cmpY)
                    shImgLoad[15 * shImgSizeX + 15] = imgs[MUL24(threadIdx.y + fY + 15, imgSize) + threadIdx.x + fX + 15];
                
                // Load filter piece into shmem
                
                const int rotFx = threadIdx.x + filterSize - fX - 16, rotFy = threadIdx.y + filterSize - fY - 16;
                if (checkFilterBounds)
                    shFilterLoad[0] = 0;
                if (!checkFilterBounds || (rotFx >= 0 && rotFy >= 0))
                    shFilterLoad[0] = filters[MUL24(filterSize, rotFy) + rotFx];
                
                __syncthreads();
                
                if (compute) {
                    const float* myShFilter = &shFilter[0][0];
                    const float* myShImg = &shImg[threadIdx.y][threadIdx.x];
                    
                    // TODO: uncomment this in final version!
                    #pragma unroll // commented to speed up compiling
                            for (int i = 0; i < 16; i++) {
                        for (int j = 0; j < 16; j++) {
                            prod += myShFilter[0] * myShImg[0];
                            
                            myShFilter++;
                            myShImg++;
                        }
                        myShImg += 15;
                            }
                }
            }
        }
        
        if (gfi) {
            imgs += MUL24(numImgsPerGroup/stride, imgPixels);
        } else {
            imgs += imgPixels;
        }
        filters += filterPixels * stride;
    } while (filters != lastFilter);
    
    if (compute) {
        targets[0] = prod;
        // targets[0] = myout;
    }
}



/* Templates need to be instantiated
 * See: http://forums.nvidia.com/index.php?showtopic=31953&pid=178825&mode=threaded&start=0#entry178825
 */
__host__ void dummyTemplateInstantiator(void){
    
    conv5_bw_fit_16x16<2, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<2, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<2, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<2, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<2, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<2, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<2, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<2, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<3, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<3, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<3, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<3, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<3, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<3, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<3, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<3, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<4, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<4, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<4, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<4, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<4, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<4, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<4, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<4, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<5, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<5, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<5, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<5, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<5, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<5, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<5, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<5, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<6, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<6, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<6, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<6, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<6, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<6, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<6, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<6, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<7, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<7, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<7, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<7, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<7, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<7, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<7, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<7, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<8, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<8, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<8, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<8, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<8, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<8, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<8, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<8, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<9, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<9, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<9, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<9, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<9, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<9, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<9, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<9, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<10, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<10, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<10, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<10, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<10, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<10, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<10, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<10, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<11, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<11, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<11, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<11, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<11, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<11, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<11, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<11, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<12, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<12, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<12, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<12, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<12, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<12, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<12, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<12, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<13, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<13, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<13, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<13, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<13, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<13, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<13, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<13, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<14, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<14, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<14, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<14, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<14, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<14, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<14, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<14, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<15, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<15, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<15, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<15, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<15, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<15, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<15, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<15, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<16, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<16, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<16, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<16, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<16, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<16, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<16, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<16, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<17, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<17, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<17, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<17, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<17, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<17, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<17, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<17, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<18, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<18, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<18, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<18, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<18, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<18, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<18, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<18, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<19, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<19, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<19, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<19, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<19, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<19, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<19, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<19, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<20, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<20, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<20, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<20, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<20, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<20, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<20, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<20, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<21, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<21, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<21, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<21, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<21, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<21, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<21, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<21, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<22, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<22, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<22, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<22, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<22, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<22, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<22, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<22, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<23, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<23, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<23, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<23, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<23, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<23, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<23, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<23, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<24, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<24, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<24, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<24, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<24, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<24, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<24, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<24, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<25, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<25, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<25, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<25, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<25, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<25, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<25, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<25, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<26, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<26, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<26, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<26, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<26, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<26, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<26, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<26, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<27, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<27, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<27, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<27, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<27, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<27, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<27, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<27, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<28, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<28, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<28, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<28, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<28, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<28, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<28, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<28, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<29, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<29, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<29, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<29, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<29, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<29, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<29, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<29, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<30, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<30, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<30, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<30, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<30, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<30, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<30, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<30, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<31, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<31, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<31, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<31, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<31, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<31, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<31, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<31, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<32, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<32, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<32, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<32, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<32, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<32, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<32, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<32, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<33, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<33, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<33, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<33, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<33, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<33, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<33, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<33, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<34, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<34, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<34, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<34, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<34, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<34, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<34, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<34, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<35, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<35, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<35, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<35, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<35, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<35, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<35, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<35, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<36, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<36, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<36, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<36, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<36, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<36, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<36, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<36, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_fit_16x16<37, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<37, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<37, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<37, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<37, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<37, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<37, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    conv5_bw_fit_16x16<37, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0);
    
    conv5_bw_nofit_16x16<false, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv5_bw_nofit_16x16<false, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv5_bw_nofit_16x16<false, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv5_bw_nofit_16x16<false, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv5_bw_nofit_16x16<false, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv5_bw_nofit_16x16<false, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv5_bw_nofit_16x16<false, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv5_bw_nofit_16x16<false, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv5_bw_nofit_16x16<true, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv5_bw_nofit_16x16<true, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv5_bw_nofit_16x16<true, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv5_bw_nofit_16x16<true, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv5_bw_nofit_16x16<true, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv5_bw_nofit_16x16<true, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv5_bw_nofit_16x16<true, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv5_bw_nofit_16x16<true, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    
}


//}
