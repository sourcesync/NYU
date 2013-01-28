/*!
 * @brief MEX wrapper for reconstruction through Deconv Network with OpenMP implementation of
 * a sparse version of valid_each3_sum4 operation. If the input maps are sparse
 * then this may be much faster as it just places down the corresponding
 * filters under nonzero location instead of doing the complete convolution.
 * @copybrief valid_each3_sum4.m
 * @see valid_each3_sum4.m for full documentation and usage. 
 *
 * @file
 * @author Matthew Zeiler
 * @data Apr 6, 2011
 *
 * @conv_file @copybrief sparse_valid_each3_sum4.cpp
 * @see valid_each3_sum4_gpu.m valid_each3_sum4_ipp.cpp valid_each3_sum4_3d.m
 *
 */

#include <mex.h> // Mex header
#include <stdio.h>
// #include <ipp.h> // Intel IPP header
#include <math.h>
#include <string.h>

#ifdef MULTITHREADING_OMP
#include <omp.h> // OpenMP header
#endif

#define MAX_NUM_THREADS      4 // Max number of parallel threads that the ...
//code will try to use (Integer). Set to 1 if you want to use a single core ...
//    (typical speedup over Matlab's conv2/fft2 is 3.5x). Set >1 for ...
//    multithreading. This number should be less than the #cpus per ...
//    machine x #cores/cpu. i.e. a machine which two quad-core cpu's ...
//    could have upto 8 threads. Note that if there are fewer images than threads
//    then the code will automatically turn down the number of threads (since the extra ones do nothing except waste
//    resources.

// Input Arguments
#define	MAPS   	        prhs[0] // The stack of images you want to convolve: n x m x c x num_images matrix of single precision floating point numbers.
#define FILTERS         prhs[1] // The kernel: i x j x c x num_images matrix of single precision floating point numbers.
#define CONMAT          prhs[2] // Connectivity Matrix
#define INPUT_THREADS   prhs[3] // User can optionally input the number of computation threads to parallelized over.

// Output Arguments
#define	OUTPUT   	plhs[0] 
#ifndef max
    #define max(a,b) ( ((a) > (b)) ? (a) : (b) )
#endif
#ifndef min
    #define min(a,b) ( ((a) < (b)) ? (a) : (b) )
#endif

/*!
 * @copybrief valid_each3_sum4.m
 *
 * @param MAPS (this is prhs[0]) n x m x num_feature_maps x num_images  matrix of single precision floating point numbers.
 * @param FILTERS (this is prhs[1]) p x q x num_input_maps x num_feature_maps x num_images matrix of single precision floating point numbers.
 * @param CONMAT (this is prhs[2]) num_input_maps x num_feature_maps connectivity matrix.
 * @param INPUT_THEADS (this is prhs[3] and is optional) specifies number of computation threads to parallelize over. Defaults to 4.
 *
 * @retval out (this is plhs[0]) A 4D matrix of (n-p+1) x (m-q+1) x num_input_maps x num_images
 */
void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray*prhs[] )
        
{
    unsigned int image_x, image_y, kernel_x, kernel_y, output_x, output_y;
    int status, num_kernel_images, num_image_maps, num_images, num_kernels, num_kernels2, im, i, j, k, k_num_dims, num_dims, c_num_dims, number_threads, indi, indj;
    int num_input_maps, num_feature_maps;
    float *kernelp, *imagep, *outputp, *kernelp_base, *imagep_base, *outputp_base, *conmatp, *conmatp_base;
    int *number_threadsp;
    
//     IppStatus retval;
//     IppiSize output_size, kernel_size, image_size, conmat_size;
    const mwSize *imagedims;
    const mwSize *kerneldims;
    const mwSize *conmatdims;
    int outputdims[5];
    // 0 for multiple images single kernel
    // 1 for multiple kernels single image
    // 2 for multiple kernels multiple images
    
    // Check for proper number of arguments
    if (nrhs < 3) {
        mexErrMsgTxt("Three input arguments required at minimum.");
    } else if (nlhs > 1) {
        mexErrMsgTxt("Too many output arguments.");
    }
    
    // If number of threads was not input, then use default set abvoe.
    if (nrhs < 4)
        number_threads = MAX_NUM_THREADS;
    else
        number_threads = (int)mxGetScalar(INPUT_THREADS);
    
//     mexPrintf("%d is the number of threads\n",number_threads);
    
    // MAPS must be a single.
    if (mxIsSingle(MAPS) != 1)
        mexErrMsgTxt("Image must be a single precision (float), not double.");
    
    // MAPS must be 3-D stack of images
    // Uncomment if you only want this to run on stacks of images
    //if (mxGetNumberOfDimensions(MAPS) != 3)
    //  mexErrMsgTxt("Image must be a 3D array of images.");
    
    // Input must be a single.
    if (mxIsSingle(FILTERS) != 1)
        mexErrMsgTxt("Kernel must be a single precision (float), not double.");
    
    // Get dimensions of image and kernel
    num_dims = mxGetNumberOfDimensions(MAPS);
    imagedims = (mwSize*) mxCalloc(num_dims, sizeof(mwSize));
    imagedims = mxGetDimensions(MAPS);
    
//     imagedims[0] = imagedims[0];
//     imagedims[1] = imagedims[1];
    
    k_num_dims = mxGetNumberOfDimensions(FILTERS);
    kerneldims = (mwSize*) mxCalloc(k_num_dims, sizeof(mwSize));
    kerneldims = mxGetDimensions(FILTERS);
    
//     kerneldims[0] = kerneldims[0];
//     kerneldims[1] = kerneldims[1];
    
    c_num_dims = mxGetNumberOfDimensions(CONMAT);
    conmatdims = (mwSize*) mxCalloc(c_num_dims, sizeof(mwSize));
    conmatdims = mxGetDimensions(CONMAT);
    
    num_input_maps = conmatdims[0];
    num_feature_maps = conmatdims[1];
//     conmatdims[0] = conmatdims[0];
//     conmatdims[1] = conmatdims[1];
    
    if (num_dims == 2){
        num_image_maps = 1;
        num_images = 1;
    }else if(num_dims == 3){
        num_image_maps = imagedims[2];
        num_images = 1;
    }else{
        num_image_maps = imagedims[2];
        num_images = imagedims[3];
    }
    
    
    if(k_num_dims == 2){
        num_kernels = 1;
        num_kernels2 = 1;
        num_kernel_images = 1;
    }else if(k_num_dims == 3){
        num_kernels = kerneldims[2];
        num_kernels2 = 1;
        num_kernel_images = 1;
    }else if(k_num_dims == 4){
        num_kernels = kerneldims[2];
        num_kernels2 = kerneldims[3];
        num_kernel_images = 1;
    }else{
        num_kernels = kerneldims[2];
        num_kernels2 = kerneldims[3];
        num_kernel_images = kerneldims[4]; // Must match num_image_maps.
    }
    
    if(num_input_maps != num_kernels){
        mexErrMsgTxt("Connectivity Matrix first dimension doesn't match number of the second input's 3rd dimension.");
    }
    if(num_feature_maps != num_image_maps){
        mexErrMsgTxt("Connectivity Matrix first dimension doesn't match number of the first input's 3rd dimension.");
    }
    if(num_feature_maps != num_kernels2){
        mexErrMsgTxt("Connectivity Matrix second dimension doesn't match number of the second input's 4th dimension.");
    }
    if(num_kernel_images != num_images){
        mexErrMsgTxt("Number of input images (dim 4) does not match the number of kernels (dim 5).");
    }
    
//     mexPrintf("%d images at %d by %d\n", num_image_maps, imagedims[0], imagedims[1]);
//     mexPrintf("%d x %d kernels at %d by %d\n", num_kernels,kerneldims[3], kerneldims[0], kerneldims[1]);
//     mexPrintf("Conmat is %d by %d\n",num_input_maps,num_feature_maps);
    //mexPrintf("Mixing Type: %d\n", MIXING_TYPE);
    
    // Get pointers to MAPS and FILTERS
    imagep_base = (float*) mxGetData(MAPS);
    kernelp_base = (float*) mxGetData(FILTERS);
    conmatp_base = (float*) mxGetData(CONMAT);
    
    
    
    // *****************************************************************************************************
    // Main part of code
    // Always a valid convolution.
    // Create output matrix of appropriate size
    outputdims[0]  = imagedims[0]  - kerneldims[0] + 1;
    outputdims[1] = imagedims[1] - kerneldims[1] + 1;
//     outputdims[1] = outputdims[1];
//     outputdims[0] = outputdims[0];
    
    // Arrange the outpu dimensions like F' (outsize,outsize,num_feature_maps,num_input_maps)
    outputdims[2] = num_input_maps;
    outputdims[3] = num_feature_maps; // This is what should be summed over outside with sum(...,4)
    outputdims[4] = num_images;
    OUTPUT = mxCreateNumericArray(5, outputdims, mxSINGLE_CLASS, mxREAL);
    
    
    
    
    // Check matrix generated OK
    if (OUTPUT==NULL)
        mexErrMsgTxt("Could not allocate output array");
    
    // Get pointer to output matrix
    outputp_base = (float*) mxGetData(OUTPUT);
    
    
    ///////////////////////////////////////////////////////////////////////
    // Setup openMP for core inner loop of convolutions.
    // Have to ensure the pointers are private variables to each thread.
//#pragma omp parallel for num_threads(number_threads) shared(imagep_base, kernelp_base, outputp_base, output_size, kernel_size, image_size, outputdims) private(i, imagep, kernelp, outputp)
    
    if(num_images==1){
        int x,y,kx,ky,outx,outy=0;
        float inval = 0;
        // Main loop over all images in stack. Stuff inside this loop will be multi-threaded out to different cores.
        for (j=0;j<num_input_maps;j++){
            
            #pragma omp parallel for num_threads(number_threads) shared(j,outputp) private(k, x,y,kx,ky,outx,outy,inval, indi, indj, imagep, kernelp, conmatp)
            for(k=0;k<num_feature_maps;k++){
                // Update the pointer in the connectivity matrix.
                //         mexPrintf("Size of float: %d\n",sizeof(float));
                conmatp = conmatp_base + j+k*conmatdims[0];
                
//                 if(*conmatp != 0){
                    // Looping over the image planes with k now.
                    imagep = imagep_base + (k*imagedims[0]*imagedims[1]);
                    // set pointer offset for kernel
                    kernelp = kernelp_base + (j*kerneldims[0]*kerneldims[1]) + (k*num_input_maps*kerneldims[0]*kerneldims[1]);
                    // set pointer offset for output in it's 4D array.
                    outputp = outputp_base + (j*outputdims[0]*outputdims[1]) + (k*num_input_maps*outputdims[0]*outputdims[1]);
                    
                    // Do the sparse convolution here.
                    // Note: the pointers have already been shifted to the correct plane.
                    // Have to handle the boundary conditions separately.
                    // Do do the boundaries we will look over the 0 to kerneldims[0] first and last columns
                    // and the 0 to kerneldims[1] first and last rows, do the multiplication with the filters
                    // and this is the boundary pixel.
                    // Actually a better way is to use the same logic, but don't check the boundary once you are far enough inwards.
                    for(x=0;x<imagedims[0];x++){
                        for(y=0;y<imagedims[1];y++){
                            // Only plop down a filter if there is an activation in input map.
                            inval = imagep[x+y*imagedims[0]];
//             mexPrintf("x,y = %d,%d; kx,ky=%d,%d; outx,outy=%d,%d,offsetx,offsety=%d,%d\n", x, y, kx, ky, outx, outy, offsetx, offsety);
                            if(inval!=0){
                                // Loop over the region in the output image where teh filter should be plopped down.
                                // This does automatic bound checking.
                                for(kx=max(kerneldims[0]-x-1, 0);kx<min(kerneldims[0], imagedims[0]-x);kx++){
                                    for(ky=max(kerneldims[1]-y-1, 0);ky<min(kerneldims[1], imagedims[0]-y);ky++){
//                         outputp_base[x+y*outputdims[0]] = 1;
                                        outx = x+kx-kerneldims[0]+1;
                                        outy = y+ky-kerneldims[1]+1;
//                                                     mexPrintf("kx,ky=%d,%d\n",kx,ky);
//                         if(outx<0 || outy<0 || outx>=outputdims[0] || outy>=outputdims[1]){
//                                                                                     mexPrintf("Inside: outx,outy=%d,%d\n",outx,outy);
//                                                      mexPrintf("No inside as outx,outy=%d,%d\n",outx,outy);
                                        outputp[outx+outy*outputdims[0]] += inval*kernelp[kx+ky*kerneldims[0]];
                                    }
                                }
                            }
                        }
//                     }
                    
                }
            }
        }
    }else{
        int x,y,kx,ky,outx,outy=0;
        float inval = 0;
        #pragma omp parallel for num_threads(number_threads) shared(outputp) private(im, j, k, x,y,kx,ky,outx,outy,inval,indi, indj, imagep, kernelp, conmatp)
        for(im=0;im<num_images;im++){ // The 4th dimension of the feature maps. 
            // Main loop over all images in stack. Stuff inside this loop will be multi-threaded out to different cores.
            for (j=0;j<num_input_maps;j++){
                for(k=0;k<num_feature_maps;k++){
                    // Update the pointer in the connectivity matrix.
                    // mexPrintf("Size of float: %d\n",sizeof(float));
                    conmatp = conmatp_base + j+k*conmatdims[0];
                    
//                     if(*conmatp != 0){
                        // Looping over the image planes with k now, skipping whole sets of feature maps.
                        imagep = imagep_base + (k*imagedims[0]*imagedims[1]) + (im*num_feature_maps*imagedims[0]*imagedims[1]);
                        // set pointer offset for kernel
                        kernelp = kernelp_base + (j*kerneldims[0]*kerneldims[1]) + (k*num_input_maps*kerneldims[0]*kerneldims[1]) + (im*num_feature_maps*num_input_maps*kerneldims[0]*kerneldims[1]);
                        // set pointer offset for output in it's 4D array.
                        outputp = outputp_base + (j*outputdims[0]*outputdims[1]) + (k*num_input_maps*outputdims[0]*outputdims[1]) + (im*num_input_maps*num_feature_maps*outputdims[0]*outputdims[1]);
                        
                        // Do the sparse convolution here.
                        // Note: the pointers have already been shifted to the correct plane.
                        // Have to handle the boundary conditions separately.
                        // Do do the boundaries we will look over the 0 to kerneldims[0] first and last columns
                        // and the 0 to kerneldims[1] first and last rows, do the multiplication with the filters
                        // and this is the boundary pixel.
                        // Actually a better way is to use the same logic, but don't check the boundary once you are far enough inwards.
                        for(x=0;x<imagedims[0];x++){
                            for(y=0;y<imagedims[1];y++){
                                // Only plop down a filter if there is an activation in input map.
                                inval = imagep[x+y*imagedims[0]];
//             mexPrintf("x,y = %d,%d; kx,ky=%d,%d; outx,outy=%d,%d,offsetx,offsety=%d,%d\n", x, y, kx, ky, outx, outy, offsetx, offsety);
                                if(inval!=0){
                                    // Loop over the region in the output image where teh filter should be plopped down.
                                    // This does automatic bound checking.
                                    for(kx=max(kerneldims[0]-x-1, 0);kx<min(kerneldims[0], imagedims[0]-x);kx++){
                                        for(ky=max(kerneldims[1]-y-1, 0);ky<min(kerneldims[1], imagedims[0]-y);ky++){
//                         outputp_base[x+y*outputdims[0]] = 1;
                                            outx = x+kx-kerneldims[0]+1;
                                            outy = y+ky-kerneldims[1]+1;
//                                                     mexPrintf("kx,ky=%d,%d\n",kx,ky);
//                         if(outx<0 || outy<0 || outx>=outputdims[0] || outy>=outputdims[1]){
//                                                                                     mexPrintf("Inside: outx,outy=%d,%d\n",outx,outy);
//                                                      mexPrintf("No inside as outx,outy=%d,%d\n",outx,outy);
                                            outputp[outx+outy*outputdims[0]] += inval*kernelp[kx+ky*kerneldims[0]];
                                        }
                                    }
                                }
                            }
//                         }
                    }
                }
            }
        }
    }
    
//
// // Parse any error (can't put inside inner loop as it will stop ...
// //  multithreading)
// if (retval!=ippStsNoErr){
//     mexPrintf("Error performing convolution\n");
//
//     if (retval==ippStsNullPtrErr)
//         mexErrMsgTxt("Pointers are NULL\n");
//
//     if (retval==ippStsSizeErr)
//         mexErrMsgTxt("Sizes negative or zero\n");
//
//     if (retval==ippStsStepErr)
//         mexErrMsgTxt("Steps negative or zero\n");
//
//     if (retval==ippStsMemAllocErr)
//         mexErrMsgTxt("Memory allocation error\n");
//
// }
    
    
// Free up
// mxFree(modep);
    
    
    return;
    
}

