/*!
 * @brief MEX implementation of unpooling a max pooling over subregions of image planes.
 * The size of the subregions can be arbitrary, as can the number of planes.
 * If more planes are input for the provided indices than the images, only the 
 * first corresponding number of image planes will be used from the indices.
 * Note: input images must be single precision floats.
 *
 * @file
 * @author Matthew Zeiler (zeiler@cs.nyu.edu)
 * @data Aug 14, 2010
 *
 * @pooltoolbox_file @copybrief reverse_max_pool.cpp
 *
 *****************
 * How to compile:
 *
 * Type in Matlab to compile (using the included compilemex.m script):<br>
 * >> compilemex
 *
 */

#include <mex.h> // Mex header
#include <stdio.h>
#include <math.h>
#include <string.h>

#ifdef MULTITHREADING_OMP
#include <omp.h> // OpenMP header
#endif

#define MAX_NUM_THREADS      4 // Max number of parallel threads that the ...
//code will try to use (Integer). Set to 1 if you want to use a single core ...

// Input Arguments
#define	IMAGES   	    prhs[0] // The stack of images you want to unpool: n x m x c x d matrix of single precision floating point numbers.
#define INDICES         prhs[1] // Previous indices from a pooling operation (optional). They must be the same size of the resulting output of pooling the IMAGES by this POOL_SIZE.
#define POOL_SIZE       prhs[2] // The size of the subregions to pool over: 1x2  matrix indicating the p x q size of the region.
#define UNPOOLED_SIZE   prhs[3] // The unpooled size of the images which is typically n*p x m*q (defaults to this if this parameter is not passed in) (just cares abou the first two dimensions so only input those as a 1x2 matrix).
#define INPUT_THREADS   prhs[4] // User can optionally input the number of computation threads to parallelized over.


// Output Arguments
#define	UNPOOLED   	    plhs[0] // Unpooled stack of images. This will be roughly n/q x m/p x c x d

/*!
 * @copybrief reverse_max_pool.cpp
 *
 * @param IMAGES   	    prhs[0] // The stack of images you want to npool: n x m x c x d matrix
 * @param INDICES       prhs[1] // Previous indices from a pooling operation (optional). They must be the same size of the resulting output of pooling the IMAGES by this POOL_SIZE.
 * @param POOL_SIZE     prhs[2] // The size of the subregions to pool over: 1x2  matrix indicating the p x q size of the region.
 * @param UNPOOLED_SIZE prhs[3] // The unpooled size of the images which is typically n*p x m*q (just cares abou the first two dimensions so only input those as a 1x2 matrix).
 * @param INPUT_THREADS prhs[4] // User can optionally input the number of computation threads to parallelized over.
 *
 * @retval POOLED   	   plhs[0] // Unpooled stack of images. This will be roughly n*q x m*p x c x d
 */
void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray*prhs[] )
        
{
    int num_dims, num_inddims, number_threads;
    float *outputp, *imagesp;
    unsigned short int *indicesp;
    double *poolp; // For the pool_size array.
    double *unpooled_sizep; // For the unpooled_size array.
    
    int *number_threadsp;
    int blocksx, blocksy, remx, remy, bcoljump, plane3jump, plane4jump, top_left;
    int poolx, pooly, poolsizex, poolsizey;
    
    const mwSize *dims;
    const mwSize *inddims;
//    mxArray *TEMP_INDICES; // Only used if the number of indices is more than the input maps.
    int* outputdims = NULL;
    
    // Check for proper number of arguments
    if (nrhs < 3) {
        mexErrMsgTxt("Three input arguments required at minimum.");
    } else if (nlhs > 1) {
        mexErrMsgTxt("Too many output arguments.");
    }
    
    // If number of threads was not input, then use default set abvoe.
    if (nrhs < 5)
        number_threads = MAX_NUM_THREADS;
    else
        number_threads = (int)mxGetScalar(INPUT_THREADS);
    
    // Make sure it is single input.
    if (mxIsSingle(IMAGES) != 1){
        mexErrMsgTxt("Input Images must be a single precision (float).");
    }
    // Make sure it is single input.
    if (mxIsClass(INDICES,"uint16") != 1){
        mexErrMsgTxt("Input Indices must be a unsigned short int (uint16).");
    }
    
    // Get dimensions of image and kernel
    num_dims = mxGetNumberOfDimensions(IMAGES);
    dims = (mwSize*) mxCalloc(num_dims, sizeof(mwSize));
    dims = mxGetDimensions(IMAGES);
    
    
    // Make sure the indices are provided and the same number of dimensions as input.
    num_inddims = mxGetNumberOfDimensions(INDICES);
    inddims = (mwSize*) mxCalloc(num_dims, sizeof(mwSize));
    inddims = mxGetDimensions(INDICES);
    
    if(inddims[0]!=dims[0]){
        mexErrMsgTxt("reverse_max_pool.cpp: First dimension of indices doesn't match first dimension of input pooled images.");
    }
    if(inddims[1]!=dims[1]){
        mexErrMsgTxt("reverse_max_pool.cpp: Second dimension of indices doesn't match first dimension of input pooled images.");
    }
    
    // Get pointers to IMAGE
    imagesp = (float*) mxGetData(IMAGES);
    
    // Get pointers to previous indices if they were passed in.
    indicesp = (unsigned short int*) mxGetData(INDICES);
    
    // Get the pool size
    poolp = mxGetPr(POOL_SIZE);
    // Get the pool region size
    poolx = (int) poolp[0];
    pooly = (int) poolp[1];
    if(poolx*pooly>65536){
        mexErrMsgTxt("Cannot pool in 2D larger than 256x256 due to uint16 indices limitation.");
    }
    // Number of blocks in rows on each plane.
    blocksx = (int)dims[0];
    // Number of block in columns on each plane.
    blocksy = (int)dims[1];
    
    // Create the output dimensions
    outputdims = new int[num_dims];
    
    if(nrhs < 4 || mxIsEmpty(UNPOOLED_SIZE)==true){ // unpooled_size wasn't passed in, so create it
        outputdims[0] = (int)poolp[0]*dims[0];
        outputdims[1] = (int)poolp[1]*dims[1];
        // Remaining dimensions are same as input.
        for(int k=2;k<(int)num_dims;k++){
            outputdims[k] = dims[k];
        }
    }else{ // Get the unpooled_size from input arguments
        unpooled_sizep = mxGetPr(UNPOOLED_SIZE);
        outputdims[0] = (int)unpooled_sizep[0];
        outputdims[1] = (int)unpooled_sizep[1];
        // Remaining dimensions are same as input.
        for(int k=2;k<(int)num_dims;k++){
            outputdims[k] = dims[k];
        }
    }
    
    
    
    // Create the output arrays.
    UNPOOLED = mxCreateNumericArray(num_dims, (mwSize*) outputdims, mxSINGLE_CLASS, mxREAL);
    // Get pointer to output matrix
    outputp = (float*) mxGetData(UNPOOLED);
    
    
    
    
    
    
    // How much each bcol jumps in linear indices
   // bcoljump = pooly*dims[0];
   // plane3jump = dims[0]*dims[1];
   // plane4jump = dims[0]*dims[1]*dims[2];
    
//     mexPrintf("Number of input dimensions: %d %d %d %d, num_dims: %d", dims[0], dims[1], dims[2], dims[3], num_dims);
//     mexPrintf("Number of output dimensions: %d %d %d %d",outputdims[0],outputdims[1],outputdims[2],outputdims[3]);
    //  mexErrMsgTxt("Kernel must be a single precision (float), not double.");
    
    
    
    
    
    
    //
    //
    // Special Case where the number of indices in the 4th dimensions is greater than the number of input planes int he 4th dimension
    // This is takes the mean over the 4th dimension of the indices and then does the unpooled over the first 3 dimensions as normal.
    //
    //
//     if(num_dims==3 && num_inddims==4){ // More dimensions in the indices than the images input.
//         
//         // Create temporary indices (so don't overwrite the input ones).
//         TEMP_INDICES = mxCreateNumericArray(num_dims, (mwSize*) inddims, mxSINGLE_CLASS, mxREAL);
//         // Get pointer to output matrix
//         temp_indicesp = (float*) mxGetData(TEMP_INDICES);
//         
//         int brow, bcol, i, j, maxi, outputelem, plane3, plane4 = 0;
//         int mean=0;
//         
//         #pragma omp parallel for num_threads(number_threads) shared(indicesp,inddims) private(plane4, plane3, brow, bcol, top_left, mean)
//         for(plane3=0;plane3<inddims[2];plane3++){ // Loop over the planes int he 3rd dimension
//             for(bcol=0;bcol<blocksy;bcol++){ // Loop over blocks in columns
//                 for(brow=0;brow<blocksx;brow++){ // Loop of blocks in rows
//                     
//                     // Linear index into the top left element of the block (but just for the first 3 dimensions).
//                     top_left = plane3*(blocksx*blocksy)+bcol*(blocksx)+brow;
//                     // Set the mean index back to zero.
//                     mean=0;
//                     // Loop over the 4th dimension of indices and get the mean.
//                     for(plane4=0;plane4<inddims[3];plane4++){                        
//                         mean += indicesp[plane4*(blocksx*blocksy*inddims[2])+top_left];                              
//                     }
// //                     mexPrintf("Mean: %d at brow: %d, bcol: %d, plane3: %d\n",(int)(mean/inddims[3]),brow,bcol,plane3);
//                     // Change the indices to the mean indices
//                     temp_indicesp[top_left] = (int)(mean/inddims[3]);
//                 }
//             }
//         }
//         // Now use the temp indices below.
//         indicesp = (float *) temp_indicesp;
//     }else if(num_dims==2 && num_inddims==4){
//         
//         // Create temporary indices (so don't overwrite the input ones).
//         TEMP_INDICES = mxCreateNumericArray(num_dims, (mwSize*) inddims, mxSINGLE_CLASS, mxREAL);
//         // Get pointer to output matrix
//         temp_indicesp = (float*) mxGetData(TEMP_INDICES);
//         
//         int brow, bcol, i, j, maxi, outputelem, plane3, plane4 = 0;
//         int mean=0;
//         
//         #pragma omp parallel for num_threads(number_threads) shared(indicesp,inddims) private(plane4,plane3, brow, bcol, top_left, mean)
//         for(bcol=0;bcol<blocksy;bcol++){ // Loop over blocks in columns
//             for(brow=0;brow<blocksx;brow++){ // Loop of blocks in rows
//                     
//                 // Linear index into the top left element of the block (but just for the first 3 dimensions).
//                 top_left = plane3*(blocksx*blocksy)+bcol*(blocksx)+brow;
//                 // Set the mean index back to zero.
//                 mean=0;
//                 // Loop over the 4th dimension of indices and get the mean.
//                 for(plane4=0;plane4<inddims[3];plane4++){
//                     mean += indicesp[plane4*(blocksx*blocksy*inddims[2])+top_left];
//                 }
// //                 mexPrintf("Mean: %d at brow: %d, bcol: %d, plane3: %d\n", (int)(mean/inddims[3]), brow, bcol, plane3);
//                 // Change the indices to the mean indices
//                 temp_indicesp[top_left] = (int)(mean/inddims[3]);
//             }
//         }
//         // Now use the temp indices below.
//         indicesp = (float *) temp_indicesp;
//     }
    
    
//     mexPrintf("num_dims: %d\n",num_dims);
    
    //
    //
    //
    //Just select the max locations from IMAGES and return them in POOLED
    //
    //
    //
    //
    //
    if(num_dims==2){ // Just unpool the single image.
        int brow, bcol, i, j, outputelem = 0;
        unsigned short int maxi=0;
        #pragma omp parallel for num_threads(number_threads) shared(imagesp, outputp, indicesp) private(brow, bcol, i, j, top_left, maxi, outputelem)
        for(bcol=0;bcol<blocksy;bcol++){ // Loop over blocks in columns
            for(brow=0;brow<blocksx;brow++){ // Loop of blocks in rows
                
                // Linear index into the top left element of the block (note this uses the original block sizes always).
                top_left = bcol*(blocksx)+brow;
                
                // Get the index at the pooled location from the input indices.
                maxi = indicesp[top_left];
                maxi = maxi; // convert from MATLAB indexing
                // Get the row and column offsets.
                i = maxi%poolx;
                j = (int)(maxi/poolx);
                
                // Have to skip (brow*poolx)+i elements down, bcol*poolx+j elements over.
                outputelem = brow*poolx+i+(bcol*pooly+j)*outputdims[0];
//                     mexPrintf("Input Index: %d, i: %d, j: %dm top_left: %d, outputelem: %d, dims[0]: %d\n",maxi+1,i,j,top_left,outputelem,dims[0]);
                // Output at the pooled index is the selected element using the input indices.
                outputp[outputelem] = imagesp[top_left];
            }
        }
    }else if(num_dims==3){ // Unpool each plane independently
        int brow, bcol, i, j, outputelem, plane3 = 0;
        unsigned short int maxi=0;
        #pragma omp parallel for num_threads(number_threads) shared(imagesp, outputp, indicesp) private(plane3, brow, bcol, i, j, top_left, maxi, outputelem)
        for(plane3=0;plane3<outputdims[2];plane3++){ // Loop over the planes int he 3rd dimension
            for(bcol=0;bcol<blocksy;bcol++){ // Loop over blocks in columns
                for(brow=0;brow<blocksx;brow++){ // Loop of blocks in rows
                    
                    // Linear index into the top left element of the block (note this uses the original block sizes always).
                    top_left = plane3*(blocksx*blocksy)+bcol*(blocksx)+brow;
                    
                    // Get the index at the pooled location from the input indices.
                    maxi = indicesp[top_left];
                    maxi = maxi; // convert from MATLAB indexing
                    // Get the row and column offsets.
                    i = maxi%poolx;
                    j = (int)(maxi/poolx);
                    
                    // Have to skip (brow*poolx)+i elements down, bcol*poolx+j elements over.
                    outputelem = plane3*(outputdims[0]*outputdims[1])+brow*poolx+i+(bcol*pooly+j)*outputdims[0];
//                     mexPrintf("Input Index: %d, i: %d, j: %dm top_left: %d, inputelem: %d, dims[0]: %d\n",maxi+1,i,j,top_left,inputelem,dims[0]);
                    // Output at the pooled index is the selected element using the input indices.
                    outputp[outputelem] = imagesp[top_left];
                }
            }
        }
    }else if(num_dims==4){ // Unpool each plane independently
        int brow, bcol, i, j, outputelem, plane3, plane4 = 0;
        unsigned short int maxi=0;
        #pragma omp parallel for num_threads(number_threads) shared(imagesp, outputp, indicesp) private(plane4, plane3, brow, bcol, i, j, top_left, maxi, outputelem)
        for(plane4=0;plane4<outputdims[3];plane4++){
            for(plane3=0;plane3<outputdims[2];plane3++){ // Loop over the planes int he 3rd dimension
                for(bcol=0;bcol<blocksy;bcol++){ // Loop over blocks in columns
                    for(brow=0;brow<blocksx;brow++){ // Loop of blocks in rows
                        
                        // Linear index into the top left element of the block (note this uses the original block sizes always).
                        top_left = plane4*(blocksx*blocksy*outputdims[2])+plane3*(blocksx*blocksy)+bcol*(blocksx)+brow;
                        
                        // Get the index at the pooled location from the input indices.
                        maxi = indicesp[top_left];
                        maxi = maxi; // convert from MATLAB indexing
                        // Get the row and column offsets.
                        i = maxi%poolx;
                        j = (int)(maxi/poolx);
                        // Have to skip (brow*poolx)+i elements down, bcol*poolx+j elements over.
                        outputelem = plane4*(outputdims[0]*outputdims[1]*outputdims[2])+plane3*(outputdims[0]*outputdims[1])+brow*poolx+i+(bcol*pooly+j)*outputdims[0];
//                         mexPrintf("Input Index: %d, i: %d, j: %dm top_left: %d, inputelem: %f, dims[0]: %d\n",maxi+1,i,j,top_left,imagesp[top_left],dims[0]);
                        // Output at the pooled index is the selected element using the input indices.
                        outputp[outputelem] = imagesp[top_left];
                    }
                }
            }
        }
    }
    
//     mxFree(dims);
//     mxFree(inddims);
    
    return;
    
}

