#include "GPUkernel.hh"
#include "conv.cuh"

#define MUL24 __mul24

// This is supposed to sum over dimension 4 of the filters.

// The naming is assuming summation over numFeatureMaps for each num_feature_map output. 

// numFeatureMaps was numFiltersPerGroup (this is what is summed over).
// numInputMaps was numImagesPerGroup
// num_cases was numGroups

//To use templates we must have C++ kernels
//extern "C" {

/*
 * This function uses block size 16x16.
 * Works for filters up to 37x37.
 */
template<int filterSize, bool checkBounds, int stride, bool gfi>
        __global__ void conv8_bw_fit_16x16(float* imgs, float* filters, float* targets, float* conmat,
        const int imgSize, const int numFeatureMaps, const int num_cases) {
    const int shImgSizeX = filterSize + 15, shImgSizeY = filterSize + 15;
    __shared__ float shImg[shImgSizeY][shImgSizeX];
    __shared__ float shFilter[filterSize][filterSize];
    
    const int numInputMaps = gridDim.x / num_cases;
    const int outputIdx = blockIdx.x; // This is the index used for targets and filters this ranges from 1:(num_cases*numInputMaps)
    const int caseIdx = blockIdx.x / numInputMaps; // This is current case we're working on
    
    const int outputPart = blockIdx.y;
    const int numOutputsX = imgSize - filterSize + 1;
    const int numOutputs = (numOutputsX * numOutputsX);
    const int dived = DIVUP(numOutputsX, 16);
    const int outputPartY = outputPart / dived;
    const int outputPartX = outputPart - dived*outputPartY;
    const int tidx = threadIdx.y * 16 + threadIdx.x; // thread's index within the 16x16 "plate" of threads
    
    // imgSize is just one of the squared dimensions.
    const int imgPixels = imgSize * imgSize;
    
    const int filterPixels = filterSize * filterSize;
    
    // Changes made:
    // Took out any use of imgIdxInGroup from targets and filters because that used to be zero and it worked fine before.
    // These are the feature maps (so conv8 skipped over numInputMaps each time caseIdx==case changed)
    // This should not change as we didn't permut the maps before.
    imgs += (caseIdx * numFeatureMaps) * imgPixels
            + (outputPartY * imgSize + outputPartX) * 16;
    
    // target (outputs) should stay the same as before as they are [outx outy num_cases].
    // They are kind of funny though as they are split into groups of 16 pixels I believe.
    targets += outputIdx * numOutputs 
            + (outputPartY * numOutputsX + outputPartX) * 16
            + (threadIdx.y * numOutputsX) + threadIdx.x;
    
    if (filterSize <= 16)
        filters += tidx;
    ////////
    // This should changed from conv8 and the increments as well used later
    ////////
    // This is now used for indicating reuse of filter or use different filters for each case.
    if(!gfi){ // This is when 1 is passed in (use different filters per case)
        const int inputMapIdx = blockIdx.x - numInputMaps*caseIdx; // Index for resusing filters (this is just a mod).
        filters += (caseIdx*numFeatureMaps*numInputMaps + inputMapIdx) * filterPixels * stride;
        // Start the conmat at the row corresponding to the correct input map.
        conmat += (blockIdx.x - numInputMaps*caseIdx); // Just the row of the conmat (input map we're on).
    }else{ // When 0 is passed in (reuse numFeatureMaps*numInputMaps filters per case).
        const int inputMapIdx = blockIdx.x - numInputMaps*caseIdx; // Index for resusing filters (this is just a mod).
        filters += (inputMapIdx) * filterPixels * stride;
        // Start the conmat at the row corresponding to the correct input map.
        conmat += (inputMapIdx); // Just the row of the conmat (input maps we're on).
    }
    
    

    
    
    const float* lastFilter = filters + (filterPixels * stride * numFeatureMaps*numInputMaps); // bad pointer
    float prod = 0;
    bool compute = !checkBounds || (outputPartX * 16 + threadIdx.x < numOutputsX && outputPartY * 16 + threadIdx.y < numOutputsX);
    const int cmpX = imgSize - outputPartX * 16, cmpY = imgSize - outputPartY*16;
    do { // loop over all image/filter pairs (image = hidden activations in conv rbm)
        __syncthreads();
        
        // Make sure we want to do this convolution.
        if(conmat[0]>0){
            
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
                                shImg[loadY][loadX] = imgs[(loadY * imgSize) + loadX];
                            }
                        }
                        const int loadX = threadIdx.x + x;
                        if (loadX < shImgSizeX && loadX < cmpX) {
                            shImg[loadY][loadX] = imgs[(loadY * imgSize) + loadX];
                        }
                    }
                }
                const int loadY = threadIdx.y + y;
                if (loadY < shImgSizeY && loadY < cmpY) {
                    int x;
                    for (x = 0; x < shImgSizeX - 16; x += 16) {
                        const int loadX = threadIdx.x + x;
                        if (loadX < cmpX) {
                            shImg[loadY][loadX] = imgs[(loadY * imgSize) + loadX];
                        }
                    }
                    const int loadX = threadIdx.x + x;
                    if (loadX < shImgSizeX && loadX < cmpX) {
                        shImg[loadY][loadX] = imgs[(loadY * imgSize) + loadX];
                    }
                }
            } else { // turns out this is faster than computing indices using division/mod
                int y;
                for (y = 0; y < shImgSizeY - 16; y += 16) {
                    const int loadY = threadIdx.y + y;
                    int x;
                    for (x = 0; x < shImgSizeX - 16; x += 16) {
                        const int loadX = threadIdx.x + x;
                        shImg[loadY][loadX] = imgs[(loadY * imgSize) + loadX];
                    }
                    const int loadX = threadIdx.x + x;
                    if (loadX < shImgSizeX) {
                        shImg[loadY][loadX] = imgs[(loadY * imgSize) + loadX];
                    }
                }
                const int loadY = threadIdx.y + y;
                if (loadY < shImgSizeY) {
                    int x;
                    for (x = 0; x < shImgSizeX - 16; x += 16) {
                        const int loadX = threadIdx.x + x;
                        shImg[loadY][loadX] = imgs[(loadY * imgSize) + loadX];
                    }
                    const int loadX = threadIdx.x + x;
                    if (loadX < shImgSizeX) {
                        shImg[loadY][loadX] = imgs[(loadY * imgSize) + loadX];
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
                                shFilter[loadY][loadX] = filters[(loadY * filterSize) + loadX];
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
            
        } // Ignored the above if the filter plane was zero.
        
        // Increase to the next column in the same row of the conmat.
        conmat +=numInputMaps; // Skip to next column.
        
        // if (gfi) {
        //     imgs += MUL24(numInputMaps/stride, imgPixels);
        //   imgs += imgPixels;
        // } else { // My 0 gcase
        imgs += imgPixels;
        // }
        filters += (filterPixels * stride)*numInputMaps; // Skip over the num_input_maps planes.
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
        __global__ void conv8_bw_nofit_16x16(float* imgs, float* filters, float* targets, float* conmat,
        const int imgSize, const int filterSize, const int numFeatureMaps, const int num_cases) {
    const int shImgSizeX = 16 * 2 - 1, shImgSizeY = 16 * 2 - 1;
    __shared__ float shImg[shImgSizeY][shImgSizeX];
    __shared__ float shFilter[16][16];
    
    const int numInputMaps = gridDim.x / num_cases; // This is sF(3) since num_cases is now sF(5), ie. num_cases
    const int outputIdx = blockIdx.x; // This is the index used for targets and filters as it should be [1:sF(3)*sF(5)]
    const int caseIdx = blockIdx.x / numInputMaps; // This is dividing [1:sF(3)*sF(5)]/sF(3) so it ranges [1:sF(5)]
    
    
    const int outputPart = blockIdx.y;
    const int numOutputsX = imgSize - filterSize + 1;
    const int numOutputs = (numOutputsX * numOutputsX);
    const int dived = DIVUP(numOutputsX, 16);
    const int outputPartY = outputPart / dived;
    const int outputPartX = outputPart - dived*outputPartY;
    const int imgPixels = (imgSize * imgSize); // size of image
    const int filterPixels = (filterSize * filterSize);
    
    imgs += (caseIdx * numFeatureMaps) * imgPixels
            + (outputPartY * imgSize + outputPartX) * 16;
    
    // target (outputs) should stay the same as before as they are [outx outy num_cases].
    // They are kind of funny though as they are split into groups of 16 pixels I believe.
    targets += outputIdx * numOutputs
            + (outputPartY * numOutputsX + outputPartX) * 16
            + (threadIdx.y * numOutputsX) + threadIdx.x;
    
//     // This is now used for indicating reuse of filter or use different filters for each case.
//     if(gfi){ // This is when 1 is passed in (use different filters per case)
//         filters += (outputIdx * numFeatureMaps) * filterPixels * stride;
//         // Start the conmat at the row corresponding to the correct input map.
//         conmat += (blockIdx.x - numInputMaps*caseIdx)*numFeatureMaps;
//     }else{ // When 0 is passed in (reuse numFeatureMaps*numInputMaps filters per case).
//         const int inputMapIdx = blockIdx.x - numInputMaps*caseIdx; // Index for resusing filters.
//         filters += (inputMapIdx*numFeatureMaps) * filterPixels * stride;
//         // Start the conmat at the row corresponding to the correct input map.
//         conmat += (inputMapIdx)*numFeatureMaps;
//     }
    ////////
    // This should changed from conv8 and the increments as well used later
    ////////
    // This is now used for indicating reuse of filter or use different filters for each case.
    if(!gfi){ // This is when 1 is passed in (use different filters per case)
        const int inputMapIdx = blockIdx.x - numInputMaps*caseIdx; // Index for resusing filters (this is just a mod).
        filters += (caseIdx*numFeatureMaps*numInputMaps + inputMapIdx) * filterPixels * stride;
        // Start the conmat at the row corresponding to the correct input map.
        conmat += (blockIdx.x - numInputMaps*caseIdx); // Just the row of the conmat (input map we're on).
    }else{ // When 0 is passed in (reuse numFeatureMaps*numInputMaps filters per case).
        const int inputMapIdx = blockIdx.x - numInputMaps*caseIdx; // Index for resusing filters (this is just a mod).
        filters += (inputMapIdx) * filterPixels * stride;
        // Start the conmat at the row corresponding to the correct input map.
        conmat += (inputMapIdx); // Just the row of the conmat (input maps we're on).
    }
    
    
    const float* lastFilter = filters + ((filterPixels * stride) * numFeatureMaps*numInputMaps); // bad pointer, hope nothing rolls over...
    float prod = 0;
    bool compute = !checkOutputBounds || (outputPartX * 16 + threadIdx.x < numOutputsX && outputPartY * 16 + threadIdx.y < numOutputsX);
    const int cmpX = imgSize - outputPartX * 16, cmpY = imgSize - outputPartY * 16;
    
    float* shFilterLoad = &shFilter[15 - threadIdx.y][15 - threadIdx.x];
    float* shImgLoad = &shImg[threadIdx.y][threadIdx.x];
    do { // loop over all image/filter pairs (image = hidden activations in conv rbm)
        
        // Hack way of having connectivity matrices ignoring some convolutions.
        // Before calling cuconv8 you have to set filter planes to zero that you don't want to use.
        //if(filters[0]==0){
        //    compute = false;
        //}
        
        if(conmat[0]>0){
            
            
            
            for (int fY = 0; fY < filterSize; fY += 16) {
                for (int fX = 0; fX < filterSize; fX += 16) {
                    __syncthreads();
                    
                    // Load image piece into shmem
                    // this must exist cause f > 37 ==> i > 37
                    
                    if (!checkOutputBounds || threadIdx.x + fX < cmpX && threadIdx.y + fY < cmpY)
                        shImgLoad[0] = imgs[((threadIdx.y + fY)* imgSize) + threadIdx.x + fX];
                    if (!checkOutputBounds || threadIdx.x + fX + 15 < cmpX && threadIdx.y + fY < cmpY)
                        shImgLoad[15] = imgs[((threadIdx.y + fY)* imgSize) + threadIdx.x + fX + 15];
                    if (!checkOutputBounds || threadIdx.x + fX < cmpX && threadIdx.y + fY + 15 < cmpY)
                        shImgLoad[15 * shImgSizeX] = imgs[((threadIdx.y + fY + 15)* imgSize) + threadIdx.x + fX];
                    if (!checkOutputBounds || threadIdx.x + fX + 15 < cmpX && threadIdx.y + fY + 15 < cmpY)
                        shImgLoad[15 * shImgSizeX + 15] = imgs[((threadIdx.y + fY + 15)* imgSize) + threadIdx.x + fX + 15];
                    
                    // Load filter piece into shmem
                    
                    const int rotFx = threadIdx.x + filterSize - fX - 16, rotFy = threadIdx.y + filterSize - fY - 16;
                    if (checkFilterBounds)
                        shFilterLoad[0] = 0;
                    if (!checkFilterBounds || (rotFx >= 0 && rotFy >= 0))
                        shFilterLoad[0] = filters[(filterSize * rotFy) + rotFx];
                    
                    __syncthreads();
                    
                    if (compute && conmat[0]>0) {
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
            
        }
        
        // Increase to the next column in the same row of the conmat.
        conmat += numInputMaps;
        
        //if (gfi) {
        //    imgs += MUL24(numInputMaps/stride, imgPixels);
        //} else {
        imgs += imgPixels;
        //}
        filters += (filterPixels * stride)*numInputMaps; // Skip over the num_input_maps planes.
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
    
    conv8_bw_fit_16x16<2, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<2, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<2, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<2, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<2, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<2, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<2, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<2, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<3, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<3, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<3, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<3, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<3, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<3, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<3, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<3, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<4, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<4, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<4, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<4, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<4, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<4, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<4, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<4, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<5, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<5, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<5, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<5, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<5, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<5, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<5, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<5, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<6, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<6, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<6, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<6, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<6, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<6, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<6, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<6, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<7, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<7, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<7, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<7, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<7, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<7, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<7, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<7, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<8, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<8, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<8, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<8, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<8, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<8, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<8, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<8, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<9, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<9, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<9, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<9, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<9, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<9, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<9, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<9, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<10, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<10, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<10, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<10, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<10, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<10, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<10, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<10, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<11, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<11, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<11, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<11, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<11, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<11, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<11, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<11, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<12, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<12, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<12, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<12, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<12, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<12, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<12, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<12, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<13, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<13, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<13, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<13, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<13, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<13, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<13, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<13, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<14, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<14, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<14, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<14, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<14, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<14, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<14, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<14, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<15, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<15, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<15, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<15, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<15, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<15, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<15, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<15, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<16, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<16, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<16, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<16, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<16, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<16, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<16, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<16, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<17, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<17, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<17, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<17, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<17, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<17, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<17, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<17, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<18, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<18, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<18, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<18, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<18, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<18, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<18, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<18, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<19, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<19, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<19, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<19, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<19, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<19, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<19, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<19, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<20, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<20, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<20, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<20, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<20, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<20, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<20, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<20, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<21, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<21, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<21, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<21, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<21, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<21, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<21, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<21, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<22, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<22, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<22, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<22, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<22, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<22, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<22, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<22, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<23, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<23, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<23, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<23, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<23, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<23, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<23, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<23, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<24, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<24, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<24, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<24, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<24, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<24, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<24, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<24, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<25, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<25, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<25, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<25, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<25, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<25, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<25, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<25, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<26, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<26, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<26, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<26, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<26, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<26, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<26, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<26, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<27, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<27, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<27, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<27, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<27, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<27, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<27, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<27, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<28, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<28, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<28, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<28, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<28, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<28, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<28, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<28, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<29, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<29, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<29, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<29, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<29, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<29, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<29, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<29, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<30, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<30, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<30, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<30, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<30, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<30, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<30, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<30, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<31, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<31, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<31, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<31, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<31, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<31, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<31, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<31, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<32, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<32, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<32, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<32, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<32, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<32, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<32, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<32, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<33, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<33, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<33, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<33, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<33, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<33, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<33, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<33, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<34, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<34, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<34, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<34, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<34, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<34, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<34, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<34, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<35, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<35, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<35, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<35, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<35, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<35, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<35, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<35, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<36, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<36, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<36, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<36, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<36, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<36, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<36, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<36, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_fit_16x16<37, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<37, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<37, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<37, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<37, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<37, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<37, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    conv8_bw_fit_16x16<37, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0);
    
    conv8_bw_nofit_16x16<false, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0, 0);
    conv8_bw_nofit_16x16<false, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0, 0);
    conv8_bw_nofit_16x16<false, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0, 0);
    conv8_bw_nofit_16x16<false, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0, 0);
    conv8_bw_nofit_16x16<false, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0, 0);
    conv8_bw_nofit_16x16<false, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0, 0);
    conv8_bw_nofit_16x16<false, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0, 0);
    conv8_bw_nofit_16x16<false, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0, 0);
    conv8_bw_nofit_16x16<true, false, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0, 0);
    conv8_bw_nofit_16x16<true, false, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0, 0);
    conv8_bw_nofit_16x16<true, false, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0, 0);
    conv8_bw_nofit_16x16<true, false, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0, 0);
    conv8_bw_nofit_16x16<true, true, 1, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0, 0);
    conv8_bw_nofit_16x16<true, true, 1, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0, 0);
    conv8_bw_nofit_16x16<true, true, 3, false><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0, 0);
    conv8_bw_nofit_16x16<true, true, 3, true><<<0, 0, 0>>>(0, 0, 0, 0, 0, 0, 0, 0);
    
    
}


//}
