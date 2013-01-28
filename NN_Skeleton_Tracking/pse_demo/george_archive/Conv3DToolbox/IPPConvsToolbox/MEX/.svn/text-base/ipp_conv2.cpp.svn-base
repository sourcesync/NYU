/*!
 * @brief MEX wrapper for 2-D image convolutions using Intel's Integrated 
 * Performance Primitive (IPP) Libraries and multi-threading.
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
 * @data Mar 11, 2010
 *
 * @ipp_file @copybrief ipp_conv2.cpp
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
 * >> mex -f ./Helper/ipp/mexopts.sh -I/opt/intel/ipp/current/em64t/include -L/opt/intel/ipp/current/em64t/lib  -L/opt/intel/cce/current/lib -lguide  -lippiemergedem64t -lippimergedem64t  -lippcoreem64t  -lippsemergedem64t   -lippsmergedem64t  -lstdc++ ipp_conv2.cpp
 *
 * Normal output:<br>
 * ipp_conv2.cpp(432): (col. 1) remark: OpenMP DEFINED LOOP WAS PARALLELIZED.<br>
 * ipp_conv2.cpp(522): (col. 1) remark: OpenMP DEFINED LOOP WAS PARALLELIZED.<br>
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

//
//
// 4. Typical speedup when running on stacks of images (using same machine as 3.) is at least 5x or so.
// On a machine with two dualcore Xeons, the relative performance is (using 'valid'):
//
// a. Large grayscale image - 2000x2000 pixels, kernel size = 5x5
// Results --- Matlab (conv2): 0.68 secs. IPP: 0.11 secs. Speedup: 6.2
//
// b. Large grayscale image - 2000x2000 pixels, kernel size = 30x30
// Results --- Matlab (fft2): 1.28 secs. IPP: 0.29 secs. Speedup: 4.5
//
// c. Large color image - 2000x2000x3 pixels, kernel size = 5x5
// Results --- Matlab (conv2): 3.91 secs. IPP: 0.14 secs. Speedup: 27.0
//
// d. Large color image - 2000x2000x3 pixels, kernel size = 30x30
// Results --- Matlab (fft2): 3.92 secs. IPP: 0.34 secs. Speedup: 11.6
//
// e. Multiple tiny images - 32 x 32 x 10000, kernel size = 5x5
// Results --- Matlab (conv2): 1.83 secs. IPP: 0.06 secs. Speedup: 32.6
//
//
//
// *************************
//
// This is an enhanced version of Matlab's conv2 command. It can be used in the same manner as conv2 but has additional features.
// Instead of convolving a single 2D image matrix with another 2D kernel matrix, ...
// the image can be a 3D matrix, i.e. a stack of images. For example ...
// if you have a 1024x768 color image, this is a 1028x768x3 matrix ...
// which can be directly passed to the routine. The multi-threading ...
// will ensure that a different processor core will run on each ...
// color channel. Note that the kernel is still 2D, so it is not ...
// doing a 3D convolution.

// But the stack can be any size, for example a pile ...
// of 1000 400x300 images can be convolved with kernel by passing ...
// them in a 400x300x1000 matrix.
//
//
// Example 1:
//
// a = single(rand(400,300));
// b = single(rand(5,8));
//
// out = ipp_conv2(a,b,'valid');
// out2 = conv2(a,b,'valid');
// % out and out2 are identical
//
//
// Example 2:
//
// a = single(rand(400,300,1000));
// b = single(rand(5,8));
//
// out = ipp_conv2(a,b,'full');
// % Now compare to Matlab's command
// for i=1:1000,
//   out2(:,:,i) = conv2(a(:,:,i),b,'full');
// end
// % out and out2 are identical. But Matlab's command will be
// % much slower
//
//
// *************************************************************************************************************************
// ********************************************************************************************************************************************************************
// More detailed output of compile command in case you are have ...
// difficulties compiling/linking:
//
// >> mex -v -f /home/fergus/Helper/ipp/mexopts.sh -I/opt/intel/ipp/current/em64t/include -L/opt/intel/ipp/current/em64t/lib  -L/opt/intel/cce/current/lib -lguide  -lippiemergedem64t -lippimergedem64t  -lippcoreem64t  -lippsemergedem64t   -lippsmergedem64t  -lstdc++ ipp_conv2.cpp
// ----------------------------------------------------------------
// -> options file specified on command line:
//    FILE = /home/fergus/matlab/mexopts.sh
// ----------------------------------------------------------------
// ->    MATLAB                = /misc/linux/64/opt/matlab/R2007b
// ->    CC                    = /opt/intel/cce/current/bin/icc
// ->    CC flags:
//          CFLAGS             = -ansi -D_GNU_SOURCE -fexceptions -fPIC -fno-omit-frame-pointer -openmp -pthread
//          CDEBUGFLAGS        = -g
//          COPTIMFLAGS        = -O -DNDEBUG
//          CLIBS              = -Wl,-rpath-link,/misc/linux/64/opt/matlab/R2007b/bin/glnxa64 -L/misc/linux/64/opt/matlab/R2007b/bin/glnxa64 -lmx -lmex -lmat -lm
//          arguments          =  -DMX_COMPAT_32
// ->    CXX                   = /opt/intel/cce/current/bin/icc
// ->    CXX flags:
//          CXXFLAGS           = -ansi -D_GNU_SOURCE -openmp -fPIC -fno-omit-frame-pointer -pthread
//          CXXDEBUGFLAGS      = -g
//          CXXOPTIMFLAGS      = -O3 -DNDEBUG
//          CXXLIBS            = -Wl,-rpath-link,/misc/linux/64/opt/matlab/R2007b/bin/glnxa64 -L/misc/linux/64/opt/matlab/R2007b/bin/glnxa64 -lmx -lmex -lmat -lm
//          arguments          =  -DMX_COMPAT_32
// ->    FC                    = g95
// ->    FC flags:
//          FFLAGS             = -fexceptions -fPIC -fno-omit-frame-pointer
//          FDEBUGFLAGS        = -g
//          FOPTIMFLAGS        = -O
//          FLIBS              = -Wl,-rpath-link,/misc/linux/64/opt/matlab/R2007b/bin/glnxa64 -L/misc/linux/64/opt/matlab/R2007b/bin/glnxa64 -lmx -lmex -lmat -lm
//          arguments          =  -DMX_COMPAT_32
// ->    LD                    = /opt/intel/cce/current/bin/icc
// ->    Link flags:
//          LDFLAGS            = -pthread -shared -Wl,--version-script,/misc/linux/64/opt/matlab/R2007b/extern/lib/glnxa64/mexFunction.map -Wl,--no-undefined
//          LDDEBUGFLAGS       = -g
//          LDOPTIMFLAGS       = -O
//          LDEXTENSION        = .mexa64
//          arguments          =  -L/opt/intel/ipp/current/em64t/lib -L/opt/intel/cce/current/lib -lguide -lippiemergedem64t -lippimergedem64t -lippcoreem64t -lippsemergedem64t -lippsmergedem64t -lstdc++
// ->    LDCXX                 =
// ->    Link flags:
//          LDCXXFLAGS         =
//          LDCXXDEBUGFLAGS    =
//          LDCXXOPTIMFLAGS    =
//          LDCXXEXTENSION     =
//          arguments          =  -L/opt/intel/ipp/current/em64t/lib -L/opt/intel/cce/current/lib -lguide -lippiemergedem64t -lippimergedem64t -lippcoreem64t -lippsemergedem64t -lippsmergedem64t -lstdc++
// ----------------------------------------------------------------
//
// -> /opt/intel/cce/current/bin/icc -c  -I/opt/intel/ipp/current/em64t/include -I/misc/linux/64/opt/matlab/R2007b/extern/include -I/misc/linux/64/opt/matlab/R2007b/simulink/include -DMATLAB_MEX_FILE -ansi -D_GNU_SOURCE -openmp -fPIC -fno-omit-frame-pointer -pthread  -DMX_COMPAT_32 -O3 -DNDEBUG ipp_conv2.cpp
//
// ipp_conv2.cpp(130): (col. 7) remark: OpenMP DEFINED LOOP WAS PARALLELIZED.
// ipp_conv2.cpp(169): (col. 7) remark: OpenMP DEFINED LOOP WAS PARALLELIZED.
// -> /opt/intel/cce/current/bin/icc -O -pthread -shared -Wl,--version-script,/misc/linux/64/opt/matlab/R2007b/extern/lib/glnxa64/mexFunction.map -Wl,--no-undefined -o ipp_conv2.mexa64  ipp_conv2.o mexversion.o  -L/opt/intel/ipp/current/em64t/lib -L/opt/intel/cce/current/lib -lguide -lippiemergedem64t -lippimergedem64t -lippcoreem64t -lippsemergedem64t -lippsmergedem64t -lstdc++ -Wl,-rpath-link,/misc/linux/64/opt/matlab/R2007b/bin/glnxa64 -L/misc/linux/64/opt/matlab/R2007b/bin/glnxa64 -lmx -lmex -lmat -lm
//
// >>
//
// Here is a copy of the glnxa64 portion of my mexopts.sh file:
//
// #----------------------------------------------------------------------------
//             ;;
//         glnxa64)
// #----------------------------------------------------------------------------
// # CC and CXX should be path to Intel's 64-bit compiler
// # note that cce is the 64-bit version and cc is the 32-bit version
//             RPATH="-Wl,-rpath-link,$TMW_ROOT/bin/$Arch"
//             CC='/opt/intel/cce/current/bin/icc'
//             CFLAGS='-ansi -D_GNU_SOURCE -fexceptions'
//             CFLAGS="$CFLAGS -fPIC -fno-omit-frame-pointer -openmp -pthread"
//             CLIBS="$RPATH $MLIBS -lm"
//             COPTIMFLAGS='-O -DNDEBUG'
//             CDEBUGFLAGS='-g'
// #
//             CXX='/opt/intel/cce/current/bin/icc'
// # Ensure we have OpenMP in here too
//             CXXFLAGS='-ansi -D_GNU_SOURCE -openmp'
//             CXXFLAGS="$CXXFLAGS -fPIC -fno-omit-frame-pointer -pthread"
//             CXXLIBS="$RPATH $MLIBS -lm"
// # Use -O3 (agressive) or -O (which is -O2) is more conservative
//             CXXOPTIMFLAGS='-O3 -DNDEBUG'
//             CXXDEBUGFLAGS='-g'
// #
// #
//             FC='g95'
//             FFLAGS='-fexceptions'
//             FFLAGS="$FFLAGS -fPIC -fno-omit-frame-pointer"
//             FLIBS="$RPATH $MLIBS -lm"
//             FOPTIMFLAGS='-O'
//             FDEBUGFLAGS='-g'
// #
//             LD="$COMPILER"
//             LDEXTENSION='.mexa64'
//             LDFLAGS="-pthread -shared -Wl,--version-script,$TMW_ROOT/extern/lib/$Arch/$MAPFILE -Wl,--no-undefined"
//             LDOPTIMFLAGS='-O'
//             LDDEBUGFLAGS='-g'
// #
//             POSTLINK_CMDS=':'
// #----------------------------------------------------------------------------
// **************************************************************************************************************************************************

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
#define	IMAGE   	    prhs[0] // The stack of images you want to convolve: n x m x c matrix of single precision floating point numbers.
#define KERNEL          prhs[1] // The kernel: i x j x c matrix of single precision floating point numbers.
#define MODE            prhs[2] // String: either "full" or "valid". Unless string is "full", code will default to "valid".
#define INPUT_THREADS   prhs[3] // User can optionally input the number of computation threads to parallelized over.

// Output Arguments
#define	OUTPUT   	plhs[0] // Convolved stack of images. If in valid mode, this will be (n-i+1) x (m-j+1) x c  matrix of single ...
//  precision floating point numbers. If in full mode, this will be  (n+i-1) x (m+j-1) x c  matrix of single ...
//  precision floating point numbers.

/*!
 * @copybrief ipp_conv2.cpp
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
    
    IppStatus retval;
    IppiSize output_size, kernel_size, image_size;
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
    imagedims = (mwSize*) mxCalloc(num_dims, sizeof(mwSize));
    imagedims = mxGetDimensions(IMAGE);
    
    image_size.width = imagedims[0];
    image_size.height = imagedims[1];
    
    k_num_dims = mxGetNumberOfDimensions(KERNEL);
    kerneldims = (mwSize*) mxCalloc(k_num_dims, sizeof(mwSize));
    kerneldims = mxGetDimensions(KERNEL);
    
    kernel_size.width = kerneldims[0];
    kernel_size.height = kerneldims[1];
    
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
    
    //mexPrintf("%d images at %d by %d\n", num_images, image_size.width, image_size.height);
    //mexPrintf("%d kernels at %d by %d\n", num_kernels, kernel_size.width, kernel_size.height);
    //mexPrintf("Mixing Type: %d\n", MIXING_TYPE);
    
    // Get pointers to IMAGE and KERNEL
    imagep_base = (float*) mxGetData(IMAGE);
    kernelp_base = (float*) mxGetData(KERNEL);
    
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
        output_size.width  = image_size.width  + kernel_size.width - 1;
        output_size.height = image_size.height + kernel_size.height - 1;
        outputdims[1] = output_size.height;
        outputdims[0] = output_size.width;
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
        outputp_base = (float*) mxGetData(OUTPUT);
        
        if (outputdims[2]<number_threads)
            number_threads= outputdims[2];
        
        // Setup openMP for core inner loop
        // Have to ensure the pointers are private variables to each thread.
//#pragma omp parallel for num_threads(number_threads) shared(imagep_base, kernelp_base, outputp_base, output_size, kernel_size, image_size, outputdims) private(i, imagep, kernelp, outputp)
#pragma omp parallel for num_threads(number_threads) private(i, imagep, kernelp, outputp)

// Main loop over all images in stack. Stuff inside this loop ...
// will be multi-threaded out to different cores.
for (i=0;i<outputdims[2];i++){
    
    // Don't put any Matlab functions (e.g. mx...) in here - I ...
    // think it kill the multithreading.
    
    // Change pointers depending on single or multiple images and kernels.
    switch(MIXING_TYPE){
        case 0 : {// multiple images single kernel
            // set pointer offset for input
            imagep = imagep_base + (i*image_size.width*image_size.height);
            kernelp = kernelp_base;
            // set pointer offset for output
            outputp = outputp_base + (i*output_size.width*output_size.height);
        }    break;
        case 1 : {// single images multiple kernels
            imagep = imagep_base;
            // set pointer offset for kernel
            kernelp = kernelp_base + (i*kernel_size.width*kernel_size.height);
            // set pointer offset for output
            outputp = outputp_base + (i*output_size.width*output_size.height);
        }    break;
        case 2 : {// multiple images multiple kernels
            // set pointer offset for input
            imagep = imagep_base + (i*image_size.width*image_size.height);
            // set pointer offset for kernel
            kernelp = kernelp_base + (i*kernel_size.width*kernel_size.height);
            // set pointer offset for output
            outputp = outputp_base + (i*output_size.width*output_size.height);
        }    break;
        default : {// multiple multple
            // set pointer offset for input
            imagep = imagep_base + (i*image_size.width*image_size.height);
            // set pointer offset for kernel
            kernelp = kernelp_base + (i*kernel_size.width*kernel_size.height);
            // set pointer offset for output
            outputp = outputp_base + (i*output_size.width*output_size.height);
        }    break;
    }
    
    //mexPrintf("Image: %d imagep: %p %f outputp: %p %f kernelp: %p %f\n", i, imagep, *imagep, outputp, *outputp, kernelp, *kernelp);
    
    // call IPP full convolution routine for 32-bit floating point matrices
    retval = ippiConvFull_32f_C1R(imagep, sizeof(float)*image_size.width, image_size, kernelp, sizeof(float)*kernel_size.width, kernel_size, outputp, sizeof(float)*output_size.width);
    
}

    }
    
    else{ //Valid convolution
        
        // Create output matrix of appropriate size
        output_size.width  = image_size.width  - kernel_size.width + 1;
        output_size.height = image_size.height - kernel_size.height + 1;
        outputdims[1] = output_size.height;
        outputdims[0] = output_size.width;
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
        outputp_base = (float*) mxGetData(OUTPUT);
        
        if (outputdims[2]<number_threads)
            number_threads= outputdims[2];
        
        // Setup openMP for core inner loop
        // Have to ensure the pointers are private variables to each thread.
//#pragma omp parallel for num_threads(number_threads) shared(imagep_base, kernelp_base, outputp_base, output_size, kernel_size, image_size, outputdims) private(i, imagep, kernelp, outputp)
#pragma omp parallel for num_threads(number_threads) private(i, imagep, kernelp, outputp)

// Main loop over all images in stack. Stuff inside this loop ...
// will be multi-threaded out to different cores.
for (i=0;i<outputdims[2];i++){
    
    
    // Change pointers depending on single or multiple images and kernels.
    switch(MIXING_TYPE){
        case 0 : {// multiple images single kernel
            // set pointer offset for input
            imagep = imagep_base + (i*image_size.width*image_size.height);
            kernelp = kernelp_base;
            // set pointer offset for output
            outputp = outputp_base + (i*output_size.width*output_size.height);
        }    break;
        case 1 : {// single images multiple kernels
            imagep = imagep_base;
            // set pointer offset for kernel
            kernelp = kernelp_base + (i*kernel_size.width*kernel_size.height);
            // set pointer offset for output
            outputp = outputp_base + (i*output_size.width*output_size.height);
        }   break;
        case 2 : {// multiple images multiple kernels
            // set pointer offset for input
            imagep = imagep_base + (i*image_size.width*image_size.height);
            // set pointer offset for kernel
            kernelp = kernelp_base + (i*kernel_size.width*kernel_size.height);
            // set pointer offset for output
            outputp = outputp_base + (i*output_size.width*output_size.height);
        }    break;
        default : {// multiple multple
            // set pointer offset for input
            imagep = imagep_base + (i*image_size.width*image_size.height);
            // set pointer offset for kernel
            kernelp = kernelp_base + (i*kernel_size.width*kernel_size.height);
            // set pointer offset for output
            outputp = outputp_base + (i*output_size.width*output_size.height);
        }    break;
    }
    
    //mexPrintf("Image: %d imagep: %p %f outputp: %p %f kernelp: %p %f\n", i, imagep, *imagep, outputp, *outputp, kernelp, *kernelp);
    
    // call IPP valid convolution routine for 32-bit floating point matrices
    retval = ippiConvValid_32f_C1R(imagep, sizeof(float)*image_size.width, image_size, kernelp, sizeof(float)*kernel_size.width, kernel_size, outputp, sizeof(float)*output_size.width);
    
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
    mxFree(modep);
    
    
    return;
    
}

