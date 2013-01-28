/*!
 * @brief MEX wrapper for 2-D image convolutions that involves an image with many
 * element that are zero (but it is of single type, not MATLAB sparse type).
 *
 * There are several cases based on the input.
 * Case 1 (c images and single kernel):
 * \f[
 * out(:,:,c) = image(:,:,c) \oplus kernel
 * \f]
 *
 * Case 2 (single image and c kernels):
 * \f[
 * out(:,:,c) = image \oplus kernel(:,:,c)
 * \f]
 *
 * Case 3 (c images and c kernels where c>= 1):
 * \f[
 * out(:,:,c) = image(:,:,c) \oplus kernel(:,:,c)
 * \f]
 *
 * @file
 * @author Matthew Zeiler (zeiler@cs.nyu.edu)
 * @data Mar 11, 2011
 *
 * @ipp_file @copybrief sparse_conv2.cpp
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
#define	IMAGE   	    prhs[0] // The stack of images you want to convolve: n x m x c matrix of single precision floating point numbers.
#define KERNEL          prhs[1] // The kernel: i x j x c matrix of single precision floating point numbers.
#define MODE            prhs[2] // String: either "full" or "valid". Unless string is "full", code will default to "valid".
#define INPUT_THREADS   prhs[3] // User can optionally input the number of computation threads to parallelized over.

// Output Arguments
#define	OUTPUT   	plhs[0] // Convolved stack of images. If in valid mode, this will be (n-i+1) x (m-j+1) x c  matrix of single ...
//  precision floating point numbers. If in full mode, this will be  (n+i-1) x (m+j-1) x c  matrix of single ...
//  precision floating point numbers.

#ifndef max
    #define max(a,b) ( ((a) > (b)) ? (a) : (b) )
#endif
#ifndef min
    #define min(a,b) ( ((a) < (b)) ? (a) : (b) )
#endif

/*!
 * @copybrief sparse_conv2.cpp
 *
 * @param IMAGE (this is prhs[0]) n x m x c  matrix of single precision floating point numbers.
 * @param KERNEL (this is prhs[1]) i x j x d matrix of single precision floating point numbers.
 * @param MODE (this is prhs[2]) a string either 'valid' or 'full' indicating the type of convolution.
 * @param INPUT_THEADS (this is prhs[3] and is optional) specifies number of computation threads to parallelize over. Defaults to 4.
 *
 * @retval out (this is plhs[0]) A 3D matrix of outSize x outSize x c  (depends on different cases described in the deatiled documentation)
 */
void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray*prhs[] )
        
{
    unsigned int image_x, image_y, kernel_x, kernel_y, output_x, output_y;
    int status, buflen, num_images, num_kernels, i, k_num_dims, num_dims, number_threads;
    char *modep;
    float *kernelp, *imagep, *outputp, *kernelp_base, *imagep_base, *outputp_base;
    int *number_threadsp;
    char default_mode[] = "valid";
    
//     IppStatus retval;
//     IppiSize output_size, kernel_size, image_size;
    const mwSize *imagedims;
    const mwSize *kerneldims;
    int outputdims[3];
    int MIXING_TYPE;
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
    
    //mexPrintf("%d is the number of threads\n",number_threads);
    
    // IMAGE must be a single.
    if (mxIsSingle(IMAGE) != 1)
        mexErrMsgTxt("Image must be a single precision (float), not double.");
    
    // IMAGE must be 3-D stack of images
    // Uncomment if you only want this to run on stacks of images
    //if (mxGetNumberOfDimensions(IMAGE) != 3)
    //  mexErrMsgTxt("Image must be a 3D array of images.");
    
    // Input must be a single.
    if (mxIsSingle(KERNEL) != 1)
        mexErrMsgTxt("Kernel must be a single precision (float), not double.");
    
    // Get dimensions of image and kernel
    num_dims = mxGetNumberOfDimensions(IMAGE);
    imagedims = (mwSize*) mxCalloc(num_dims+5, sizeof(mwSize));
    imagedims = mxGetDimensions(IMAGE);
    
//     imagedims[0] = imagedims[0];
//     imagedims[1] = imagedims[1];
    
    k_num_dims = mxGetNumberOfDimensions(KERNEL);
    kerneldims = (mwSize*) mxCalloc(k_num_dims+5, sizeof(mwSize));
    kerneldims = mxGetDimensions(KERNEL);
    
//     kerneldims[0] = kerneldims[0];
//     kerneldims[1] = kerneldims[1];
    
    if (num_dims == 2)
        num_images = 1;
    else
        num_images = imagedims[2];
    
    if (k_num_dims == 2)
        num_kernels = 1;
    else
        num_kernels = kerneldims[2];
    
    // Setup how the kernelp and imagep pointers should increment below.
    if(num_kernels == 1)
        if(num_images == 1)
            MIXING_TYPE = 2; // single image single kernel (same as multiples)
        else
            MIXING_TYPE = 0; // multiple images single kernel
    else{
        if(num_images == 1)
            MIXING_TYPE = 1; // multiple kernels single image
        else{
            // Ensure the number of kernels and images are equivalent.
            if(num_images != num_kernels)
                mexErrMsgTxt("Number of Kernels does not match number of Images in multiple image multiple kernel case.");
            MIXING_TYPE = 2; // multiple kernels multiple images
        }
    }
    
//     mexPrintf("%d images at %d by %d\n", num_images, imagedims[0], imagedims[1]);
//     mexPrintf("%d kernels at %d by %d\n", num_kernels, kerneldims[0], kerneldims[1]);
    //mexPrintf("Mixing Type: %d\n", MIXING_TYPE);
    
    // Get pointers to IMAGE and KERNEL
    imagep = (float*) mxGetData(IMAGE);
    kernelp = (float*) mxGetData(KERNEL);
    
    // MODE must be a string.
    if (mxIsChar(MODE) != 1)
        mexErrMsgTxt("Mode must be a string.");
    
    // Input must be a row vector.
    if (mxGetM(MODE) != 1)
        mexErrMsgTxt("Mode must be a row vector.");
    
    // Get the length of the input string.
    buflen = (mxGetM(MODE) * mxGetN(MODE)) + 1;
    
    // Allocate memory for input and output strings.
    modep = (char*) mxCalloc(buflen, sizeof(char));
    
    // Copy the string output from FILENAME into a C string input_buf.
    status = mxGetString(MODE, modep, buflen);
    if (status != 0)
        mexWarnMsgTxt("Not enough space. String is truncated.");
    
    // *****************************************************************************************************
    // Main part of code
    
    //********************************************************************************
    // Decide Full or valid. Default is valid
    if (strcmp(modep, default_mode)){ // Full convolution
        
        // Create output matrix of appropriate size
        outputdims[0]  = imagedims[0]  + kerneldims[0] - 1;
        outputdims[1] = imagedims[1] + kerneldims[1] - 1;
        outputdims[1] = outputdims[1];
        outputdims[0] = outputdims[0];
        // Size of the output depends on number of images or kernels.
        switch(MIXING_TYPE){
            case 0 : // multiple images single kernel
                outputdims[2] = num_images;
                break;
            case 1 : // multiple kernels single image
                outputdims[2] = num_kernels;
                break;
            case 2 : // mutlple both
                outputdims[2] = num_images;
                break;
            default : // multiple both
                outputdims[2] = num_images;
                break;
        }
        
        OUTPUT = mxCreateNumericArray(3, outputdims, mxSINGLE_CLASS, mxREAL);
        
        // Check matrix generated OK
        if (OUTPUT==NULL)
            mexErrMsgTxt("Could not allocate output array");
        
        // Get pointer to output matrix
        outputp = (float*) mxGetData(OUTPUT);
        
        if (outputdims[2]<number_threads)
            number_threads= outputdims[2];
        
        // Setup openMP for core inner loop
        // Have to ensure the pointers are private variables to each thread.
        int im,x,y = 0;
//#pragma omp parallel for num_threads(number_threads) shared(imagep_base, kernelp_base, outputp_base, output_size, kernel_size, image_size, outputdims) private(i, imagep, kernelp, outputp)
#pragma omp parallel for num_threads(number_threads) private(im, x, y, imagep, kernelp, outputp)

// Main loop over all images in stack. Stuff inside this loop ...
// will be multi-threaded out to different cores.
for (im=0;im<outputdims[2];im++){
    // Don't put any Matlab functions (e.g. mx...) in here - I ...
    // think it kill the multithreading.
    
    // Change pointers depending on single or multiple images and kernels.
    switch(MIXING_TYPE){
        case 0 : {// multiple images single kernel
            // set pointer offset for input
            imagep = imagep_base + (im*imagedims[0]*imagedims[1]);
            kernelp = kernelp_base;
            // set pointer offset for output
            outputp = outputp_base + (im*outputdims[0]*outputdims[1]);
        }    break;
        case 1 : {// single images multiple kernels
            imagep = imagep_base;
            // set pointer offset for kernel
            kernelp = kernelp_base + (im*kerneldims[0]*kerneldims[1]);
            // set pointer offset for output
            outputp = outputp_base + (im*outputdims[0]*outputdims[1]);
        }    break;
        case 2 : {// multiple images multiple kernels
            // set pointer offset for input
            imagep = imagep_base + (im*imagedims[0]*imagedims[1]);
            // set pointer offset for kernel
            kernelp = kernelp_base + (im*kerneldims[0]*kerneldims[1]);
            // set pointer offset for output
            outputp = outputp_base + (im*outputdims[0]*outputdims[1]);
        }    break;
        default : {// multiple multple
            // set pointer offset for input
            imagep = imagep_base + (im*imagedims[0]*imagedims[1]);
            // set pointer offset for kernel
            kernelp = kernelp_base + (im*kerneldims[0]*kerneldims[1]);
            // set pointer offset for output
            outputp = outputp_base + (im*outputdims[0]*outputdims[1]);
        }    break;
    }
    
    //mexPrintf("Image: %d imagep: %p %f outputp: %p %f kernelp: %p %f\n", i, imagep, *imagep, outputp, *outputp, kernelp, *kernelp);
    
    // call IPP full convolution routine for 32-bit floating point matrices
//     retval = ippiConvFull_32f_C1R(imagep, sizeof(float)*imagedims[0], image_size, kernelp, sizeof(float)*kerneldims[0], kernel_size, outputp, sizeof(float)*outputdims[0]);
    
}

    }
    
    else{ //Valid convolution
        
        // Create output matrix of appropriate size
        outputdims[0]  = imagedims[0]  - kerneldims[0] + 1;
        outputdims[1] = imagedims[1] - kerneldims[1] + 1;
        outputdims[1] = outputdims[1];
        outputdims[0] = outputdims[0];
        // Size of the output depends on number of images or kernels.
        switch(MIXING_TYPE){
            case 0 :
                outputdims[2] = num_images;
                break;
            case 1 :
                outputdims[2] = num_kernels;
                break;
            case 2 :
                outputdims[2] = num_images;
                break;
            default :
                outputdims[2] = num_images;
                break;
        }
        //mexPrintf("Output Dimensions: %d , %d , %d\n", outputdims[0], outputdims[1], outputdims[2]);
        OUTPUT = mxCreateNumericArray(3, outputdims, mxSINGLE_CLASS, mxREAL);
        
        // Check matrix generated OK
        if (OUTPUT==NULL)
            mexErrMsgTxt("Could not allocate output array");
        
        // Get pointer to output matrix
        outputp = (float*) mxGetData(OUTPUT);
        
        if (outputdims[2]<number_threads)
            number_threads= outputdims[2];
        
        // Setup openMP for core inner loop
        // Have to ensure the pointers are private variables to each thread.
        int im,km,x,y,kx,ky,outx,outy = 0;
        int offsetx = (int) (kerneldims[0]-1)/2;
        int offsety = (int) (kerneldims[1]-1)/2;
        float inval = 0;
//#pragma omp parallel for num_threads(number_threads) shared(imagep_base, kernelp_base, outputp_base, output_size, kernel_size, image_size, outputdims) private(i, imagep, kernelp, outputp)
#pragma omp parallel for num_threads(number_threads) shared(outputp) private(km,im,x,y,kx,ky,outx,outy,inval)

// Main loop over all images in stack. Stuff inside this loop ...
// will be multi-threaded out to different cores.
for (im=0;im<num_images;im++){
//     for(km=0;km<num_kernels;km++){
    
    
        // Have to handle the boundary conditions separately. 
    // Do do the boundaries we will look over the 0 to kerneldims[0] first and last columns
    // and the 0 to kerneldims[1] first and last rows, do the multiplication with the filters
    // and this is the boundary pixel.
    // Actually a better way is to use the same logic, but don't check the boundary once you are far enough inwards.
    
    for(x=0;x<imagedims[0];x++){
        for(y=0;y<imagedims[1];y++){
            // Only plop down a filter if there is an activation in input map.
            inval = imagep[x+y*imagedims[0]+im*imagedims[0]*imagedims[1]];
//             mexPrintf("x,y = %d,%d; kx,ky=%d,%d; outx,outy=%d,%d,offsetx,offsety=%d,%d\n", x, y, kx, ky, outx, outy, offsetx, offsety);
            if(inval!=0){
                // Loop over the region in the output image where teh filter should be plopped down. 
                // This does automatic bound checking.
                for(kx=max(kerneldims[0]-x-1,0);kx<min(kerneldims[0],imagedims[0]-x);kx++){
                    for(ky=max(kerneldims[1]-y-1,0);ky<min(kerneldims[1],imagedims[0]-y);ky++){
//                         outputp_base[x+y*outputdims[0]] = 1;
                        outx = x+kx-kerneldims[0]+1;
                        outy = y+ky-kerneldims[1]+1;
//                                                     mexPrintf("kx,ky=%d,%d\n",kx,ky);
//                         if(outx<0 || outy<0 || outx>=outputdims[0] || outy>=outputdims[1]){
//                                                                                     mexPrintf("Inside: outx,outy=%d,%d\n",outx,outy);
//                                                      mexPrintf("No inside as outx,outy=%d,%d\n",outx,outy);
                        outputp[outx+outy*outputdims[0]+im*(outputdims[0]*outputdims[1])] += inval*kernelp[kx+ky*kerneldims[0]+km*kerneldims[0]*kerneldims[1]];
                    }
                }
            }           
        }       
    }    
    

//     
//     // Change pointers depending on single or multiple images and kernels.
//     switch(MIXING_TYPE){
//         case 0 : {// multiple images single kernel
//             // set pointer offset for input
//             imagep = imagep_base + (im*imagedims[0]*imagedims[1]);
//             kernelp = kernelp_base;
//             // set pointer offset for output
//             outputp = outputp_base + (im*outputdims[0]*outputdims[1]);
//         }    break;
//         case 1 : {// single images multiple kernels
//             imagep = imagep_base;
//             // set pointer offset for kernel
//             kernelp = kernelp_base + (im*kerneldims[0]*kerneldims[1]);
//             // set pointer offset for output
//             outputp = outputp_base + (im*outputdims[0]*outputdims[1]);
//         }   break;
//         case 2 : {// multiple images multiple kernels
//             // set pointer offset for input
//             imagep = imagep_base + (im*imagedims[0]*imagedims[1]);
//             // set pointer offset for kernel
//             kernelp = kernelp_base + (im*kerneldims[0]*kerneldims[1]);
//             // set pointer offset for output
//             outputp = outputp_base + (im*outputdims[0]*outputdims[1]);
//         }    break;
//         default : {// multiple multple
//             // set pointer offset for input
//             imagep = imagep_base + (im*imagedims[0]*imagedims[1]);
//             // set pointer offset for kernel
//             kernelp = kernelp_base + (im*kerneldims[0]*kerneldims[1]);
//             // set pointer offset for output
//             outputp = outputp_base + (im*outputdims[0]*outputdims[1]);
//         }    break;
//     }
    
    //mexPrintf("Image: %d imagep: %p %f outputp: %p %f kernelp: %p %f\n", i, imagep, *imagep, outputp, *outputp, kernelp, *kernelp);
    
    // call IPP valid convolution routine for 32-bit floating point matrices
//     retval = ippiConvValid_32f_C1R(imagep, sizeof(float)*imagedims[0], image_size, kernelp, sizeof(float)*kerneldims[0], kernel_size, outputp, sizeof(float)*outputdims[0]);
    
}

    }
    
    
//     // Parse any error (can't put inside inner loop as it will stop ...
//     //  multithreading)
//     if (retval!=ippStsNoErr){
//         mexPrintf("Error performing convolution\n");
//         
//         if (retval==ippStsNullPtrErr)
//             mexErrMsgTxt("Pointers are NULL\n");
//         
//         if (retval==ippStsSizeErr)
//             mexErrMsgTxt("Sizes negative or zero\n");
//         
//         if (retval==ippStsStepErr)
//             mexErrMsgTxt("Steps negative or zero\n");
//         
//         if (retval==ippStsMemAllocErr)
//             mexErrMsgTxt("Memory allocation error\n");
//         
//     }
    
    
    // Free up
    mxFree(modep);
    
    
    return;
    
}

