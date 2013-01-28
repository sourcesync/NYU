/*!
 * @brief MEX wrapper for IPP and OpenMP implementation of the valid_each3_each4 operation.
 * @copybrief valid_each3_each4.m
 * @see valid_each3_each4.m for full documentation and usage. 
 *
 * @file
 * @author Matthew Zeiler
 * @data Mar 11, 2011
 *
 * @conv_file @copybrief valid_each3_each4_ipp.cpp
 * @see valid_each3_each4_gpu.m valid_each3_each4_ipp.cpp valid_each3_each4_3d.m
 *
 * Requirements:
 *   \li A system with one or more multi-core Intel 64-bit CPU's
 *   \li Up to date installations of:
 *       1) Intel 64-bit C compiler (tested on version current)
 *       2) Intel Integrated Performance Primitive (IPP) libraries (tested on version 5.3)
 *   \li Matlab - to actually use the MEX file (tested on 7.5.0.338 (R2007b))
 *
 * Points to note:
 *
 * 1. These environment variables that need to be set in bash before running the mex file:
 * export PATH="/opt/intel/cce/current/bin:${PATH}";
 * export LD_LIBRARY_PATH="/opt/intel/cce/current/lib:${LD_LIBRARY_PATH}";
 *
 * 2.  The IPP libraries will automatically swap to using Fourier domain ...
 * multiplication once the size of the kernel is above 15x15 or so.
 *
 * 3. I've no idea why the valid convolution is so much faster than the full one.
 *
 *
 *
 *****************
 * How to compile:
 *
 * Type in Matlab to compile:<br>
 * >> mex -f ./Helper/ipp/mexopts.sh -I/opt/intel/ipp/current/em64t/include -L/opt/intel/ipp/current/em64t/lib  -L/opt/intel/cce/current/lib -lguide  -lippiemergedem64t -lippimergedem64t  -lippcoreem64t  -lippsemergedem64t   -lippsmergedem64t  -lstdc++ valid_each3_each4.cpp
 *
 * Normal output:<br>
 * valid_each3_each4_ipp.cpp(229): (col. 1) remark: OpenMP DEFINED LOOP WAS PARALLELIZED.<br>
 * >>
 *
 * Note that you will need to change the paths in the command to find (i) the IPP libraries (ii) Intel compiler on your system and (iii) the customized mexopts.sh file - see below.
 *****************
 *
 * You will need to alter the mexopts.sh file that Matlab uses. We have included one with
 * this package, but you may have to alter it. Then alter the -f option in the mex ...
 * command above to call it.
 *
 *
 */

#include <mex.h> // Mex header
#include <stdio.h>
#include <ipp.h> // Intel IPP header
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
#define IMAGES          prhs[1] // The kernel: i x j x c x num_images matrix of single precision floating point numbers.
#define CONMAT          prhs[2] // Connectivity Matrix
#define INPUT_THREADS   prhs[3] // User can optionally input the number of computation threads to parallelized over.

// Output Arguments
#define	OUTPUT   	plhs[0] // Convolved stack of images. If in valid mode, this will be (n-i+1) x (m-j+1) x c  matrix of single ...
//  precision floating point numbers. If in full mode, this will be  (n+i-1) x (m+j-1) x c  matrix of single ...
//  precision floating point numbers.

/*!
 * @copybrief valid_each3_each4.m
 *
 * @param MAPS (this is prhs[0]) n x m x num_feature_maps x num_images matrix of single precision floating point numbers.
 * @param IMAGES (this is prhs[1]) p x q x num_input_maps x num_images matrix of single precision floating point numbers.
 * @param CONMAT (this is prhs[2]) num_input_maps x num_feature_maps connectivity matrix.
 * @param INPUT_THEADS (this is prhs[3] and is optional) specifies number of computation threads to parallelize over. Defaults to 4.
 *
 * @retval out (this is plhs[0]) A 4D matrix of (n-p+1) x (m-q+1) x num_input_maps x num_feature_maps x num_images
 */
void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray*prhs[] )
        
{
    unsigned int image_x, image_y, kernel_x, kernel_y, output_x, output_y;
    int status, num_images, num_kernels, i, j, k, k_num_dims, num_dims, c_num_dims, number_threads, indi, indj;
    int num_input_maps, num_feature_maps;
    float *kernelp, *imagep, *outputp, *kernelp_base, *imagep_base, *outputp_base, *conmatp, *conmatp_base;
    int *number_threadsp;
    int image_cases, kernel_cases, im;
    
    IppStatus retval;
    IppiSize output_size, kernel_size, image_size, conmat_size;
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
    if (mxIsSingle(IMAGES) != 1)
        mexErrMsgTxt("Kernel must be a single precision (float), not double.");
    
    if (mxIsSingle(CONMAT) != 1)
        mexErrMsgTxt("Connectivity Matrix must be a single precision (float), not double.");
    
    // Get dimensions of image and kernel
    num_dims = mxGetNumberOfDimensions(MAPS);
    imagedims = (mwSize*) mxCalloc(num_dims, sizeof(mwSize));
    imagedims = mxGetDimensions(MAPS);
    
    image_size.width = imagedims[0];
    image_size.height = imagedims[1];
    if(num_dims==4){
        image_cases = imagedims[3];
    }else{
        image_cases = 1;
    }
    k_num_dims = mxGetNumberOfDimensions(IMAGES);
    kerneldims = (mwSize*) mxCalloc(k_num_dims, sizeof(mwSize));
    kerneldims = mxGetDimensions(IMAGES);
    if(k_num_dims==4){
        kernel_cases = kerneldims[3];
    }else{
        kernel_cases = 1;
    }
    kernel_size.width = kerneldims[0];
    kernel_size.height = kerneldims[1];
    
    c_num_dims = mxGetNumberOfDimensions(CONMAT);
    conmatdims = (mwSize*) mxCalloc(c_num_dims, sizeof(mwSize));
    conmatdims = mxGetDimensions(CONMAT);
    
    num_input_maps = conmatdims[0];
    num_feature_maps = conmatdims[1];
    conmat_size.width = conmatdims[0];
    conmat_size.height = conmatdims[1];
    
    
    
    if (num_dims == 2)
        num_images = 1;
    else
        num_images = imagedims[2];
    
    if (k_num_dims == 2)
        num_kernels = 1;
    else
        num_kernels = kerneldims[2];
    
    if(image_cases != kernel_cases){
        mexErrMsgTxt("The number of input maps is not the same as the number of input images.");
    }
    if(num_input_maps != num_kernels){
        mexErrMsgTxt("Connectivity Matrix first dimension doesn't match number of the second input's planes.");
    }
    if(num_feature_maps != num_images){
        mexErrMsgTxt("Connectivity Matrix second dimension doesn't match number of the first input's planes.");
    }
    
//     mexPrintf("%d images at %d by %d\n", num_images, image_size.width, image_size.height);
//     mexPrintf("%d kernels at %d by %d\n", num_kernels, kernel_size.width, kernel_size.height);
//     mexPrintf("Conmat is %d by %d\n",num_input_maps,num_feature_maps);
//mexPrintf("Mixing Type: %d\n", MIXING_TYPE);
    
// Get pointers to MAPS and IMAGES
    imagep_base = (float*) mxGetData(MAPS);
    kernelp_base = (float*) mxGetData(IMAGES);
    conmatp_base = (float*) mxGetData(CONMAT);
    
    
    
// *****************************************************************************************************
// Main part of code
// Always a valid convolution.
// Create output matrix of appropriate size
    output_size.width  = image_size.width  - kernel_size.width + 1;
    output_size.height = image_size.height - kernel_size.height + 1;
    outputdims[0] = output_size.width;    
    outputdims[1] = output_size.height;
    
// Arrange the outpu dimensions like F' (outsize,outsize,num_feature_maps,num_input_maps)
    outputdims[2] = num_input_maps;
    outputdims[3] = num_feature_maps; // This is what should be summed over outside with sum(...,4)
    outputdims[4] = image_cases;
    OUTPUT = mxCreateNumericArray(5, outputdims, mxSINGLE_CLASS, mxREAL);
    
// Can't modify kernel directly to multiply by C(j,k).
//     int tempkerndims[4];
//     tempkerndims[3] = num_feature_maps;
//     tempkerndims[2] = num_input_maps;
//     tempkerndims[1] = kernel_size.height;
//     tempkerndims[0] = kernel_size.width;
//     mxArray* TEMP_KERNEL = mxCreateNumericArray(4, tempkerndims, mxSINGLE_CLASS, mxREAL);
//     float* temp_kernelp_base = (float*) mxGetData(TEMP_KERNEL);
//     float* temp_kernelp = temp_kernelp_base;
//     float kernelValue;
//     float conmatValue;
    
    
// Check matrix generated OK
    if (OUTPUT==NULL)
        mexErrMsgTxt("Could not allocate output array");
    
// Get pointer to output matrix
    outputp_base = (float*) mxGetData(OUTPUT);
    
//     if (outputdims[2]<number_threads)
//         number_threads= outputdims[2];
    
    
    
// ///////////////////////////////////////////////////////////////////////
// // Apply the connectivity matrix first to each kernel.
// for (im=0;im<image_cases;im++){
// // Main loop over all images in stack. Stuff inside this loop ...
//     for (j=0;j<num_input_maps;j++){
//
//         // set pointer offset for kernel
//         kernelp = kernelp_base + (j*kernel_size.width*kernel_size.height) + (im*kernel_size.width*kernel_size.height*num_input_maps;
//
//         #pragma omp parallel for num_threads(number_threads) shared(im, j, kernelp) private(k, conmatp, temp_kernelp, kernelValue, conmatValue) schedule(static, 1)
//         for(k=0;k<num_feature_maps;k++){
//             // Update the pointer in the connectivity matrix.
//             conmatp = conmatp_base + j+k*conmat_size.width;
//
//             // Have to multiply the kernel by C(j,k) sol loop over all kernel locations.
//             for(indj=0;indj<kernel_size.height;indj++){
//                 for(indi=0;indi<kernel_size.width;indi++){
//                     temp_kernelp = temp_kernelp_base + (indi+indj*kernel_size.width) + (j*kernel_size.width*kernel_size.height) + (k*num_input_maps*kernel_size.width*kernel_size.height);
//                     kernelValue = *(kernelp+indi+indj*kernel_size.width);
//                     conmatValue = *conmatp;
//                     *temp_kernelp = kernelValue * conmatValue;
//                 }
//             }
//         }
//     }
// }
    
    
    
    
///////////////////////////////////////////////////////////////////////
// Setup openMP for core inner loop of convolutions.
// Have to ensure the pointers are private variables to each thread.
//#pragma omp parallel for num_threads(number_threads) shared(imagep_base, kernelp_base, outputp_base, output_size, kernel_size, image_size, outputdims) private(i, imagep, kernelp, outputp)
    #pragma omp parallel for num_threads(number_threads) private(k,im, j, kernelp, imagep, outputp, conmatp)
    for (im=0;im<image_cases;im++){
// Main loop over all images in stack. Stuff inside this loop ...
// will be multi-threaded out to different cores.
        for (j=0;j<num_input_maps;j++){
            kernelp = kernelp_base + (j*kernel_size.width*kernel_size.height)  + (im*num_input_maps*kernel_size.width*kernel_size.height);;

            for(k=0;k<num_feature_maps;k++){
                // Update the pointer in the connectivity matrix.
//         conmatp = conmatp_base + (k*conmat_size.width) + (j*conmat_size.width/num_input_maps);
                 conmatp = conmatp_base + j+k*conmat_size.width;
//         mexPrintf("Size of float: %d\n",sizeof(float));
//         mexPrintf("conbase: %p %f j: %d k: %d \n", conmatp, *conmatp, j, k);
                if(*conmatp != 0){
                
//                 temp_kernelp = temp_kernelp_base + (j*kernel_size.width*kernel_size.height) + (k*num_input_maps*kernel_size.width*kernel_size.height);;
                
                // Looping over the image planes with k now.
                imagep = imagep_base + (k*image_size.width*image_size.height) + (im*num_feature_maps*image_size.width*image_size.height);
                
//         mexPrintf("conbase: %p %f j: %d k: %d conmatp: %p %f\n",conmatp_base,*conmatp_base,j,k,conmatp,*conmatp);
                
                // set pointer offset for output in it's 4D array.
                outputp = outputp_base + (j*output_size.width*output_size.height) + (k*num_input_maps*output_size.width*output_size.height) + (im*num_input_maps*num_feature_maps*output_size.width*output_size.height);
                
//             mexPrintf("J: %d K: %d ip: %p %f op: %p %f kp: %p %f\n", j, k, imagep, *imagep, outputp, *outputp, kernelp, *kernelp);
                retval = ippiConvValid_32f_C1R(imagep, sizeof(float)*image_size.width, image_size, kernelp, sizeof(float)*kernel_size.width, kernel_size, outputp, sizeof(float)*output_size.width);
                  }
            }
        }
    }
    
    
    
// Parse any error (can't put inside inner loop as it will stop ...
//  multithreading)
    if (retval!=ippStsNoErr){
        mexPrintf("Error performing convolution\n");
        
        if (retval==ippStsNullPtrErr)
            mexErrMsgTxt("Pointers are NULL\n");
        
        if (retval==ippStsSizeErr)
            mexErrMsgTxt("Sizes negative or zero\n");
        
        if (retval==ippStsStepErr)
            mexErrMsgTxt("Steps negative or zero\n");
        
        if (retval==ippStsMemAllocErr)
            mexErrMsgTxt("Memory allocation error\n");
        
    }
    
    
// Free up
// mxFree(modep);
    
    
    return;
    
}

