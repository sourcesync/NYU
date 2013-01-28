/*!
 * @brief MEX implementation of unpooling an average pooling over subregions of image planes.
 * The size of the subregions can be arbitrary, as can the number of planes.
 * The indices input are ignored, they are just there so that each pooling
 * operation takes in the same format of parameters.
 *
 * @file
 * @author Matthew Zeiler (zeiler@cs.nyu.edu)
 * @data Aug 14, 2010
 *
 * @pooltoolbox_file @copybrief reverse_avg_pool.cpp
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
#define INDICES         prhs[1] // UNUSED
#define POOL_SIZE       prhs[2] // The size of the subregions to pool over: 1x2  matrix indicating the p x q size of the region.
#define UNPOOLED_SIZE   prhs[3] // The unpooled size of the images which is typically n*p x m*q (defaults to this if this parameter is not passed in) (just cares abou the first two dimensions so only input those as a 1x2 matrix).
#define INPUT_THREADS   prhs[4] // User can optionally input the number of computation threads to parallelized over.


// Output Arguments
#define	UNPOOLED   	    plhs[0] // Unpooled stack of images. This will be roughly n/q x m/p x c x d

/*!
 * @copybrief reverse_avg_pool.cpp
 *
 * @param IMAGES   	    prhs[0] // The stack of images you want to npool: n x m x c x d matrix
 * @param INDICES       prhs[1] // UNUSED
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
    double *poolp; // For the pool_size array.
    double *unpooled_sizep; // For the unpooled_size array.
    
    int *number_threadsp;
    int blocksx, blocksy, remx, remy, bcoljump, plane3jump, plane4jump, top_left;
    int poolx, pooly, poolsizex, poolsizey;
    
    const mwSize *dims;
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
    
    // Get dimensions of image and kernel
    num_dims = mxGetNumberOfDimensions(IMAGES);
    dims = (mwSize*) mxCalloc(num_dims, sizeof(mwSize));
    dims = mxGetDimensions(IMAGES);
    
    // Get pointers to IMAGE
    imagesp = (float*) mxGetData(IMAGES);
    
    // Get the pooling size.
    poolp = mxGetPr(POOL_SIZE);
    
    
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
    
    
    // Get the pool region size
    poolx = (int) poolp[0];
    pooly = (int) poolp[1];
    if(poolx*pooly>65536){
        mexErrMsgTxt("Cannot pool in 2D larger than 256x256 due to uint16 indices limitation.");
    }
    // Number of blocks in rows on each plane.
    blocksx = (int)outputdims[0]/poolx;
    // Remaining elements past blocksx
    remx = (int)(outputdims[0])%poolx;
    // Number of block in columns on each plane.
    blocksy = (int)outputdims[1]/pooly;
    // Remaining elements past blocksy
    remy = (int)(outputdims[1])%pooly;
    
    if(remx>0)
        blocksx = blocksx+1;
    if(remy>0)
        blocksy = blocksy+1;
    
    
    // Create the output arrays.
    UNPOOLED = mxCreateNumericArray(num_dims, (mwSize*) outputdims, mxSINGLE_CLASS, mxREAL);
    // Get pointer to output matrix
    outputp = (float*) mxGetData(UNPOOLED);
    
    //
    //
    //
    //Just select the locations from IMAGES and return them in all locations in regions of POOLED
    //
    //
    //
    //
    //
    if(num_dims==2){ // Just unpool the single image.
        int brow, bcol, i, j, outputelem = 0;
        
        #pragma omp parallel for num_threads(number_threads) shared(imagesp, outputp) private(brow, bcol, i, j, top_left, outputelem)
        for(bcol=0;bcol<blocksy;bcol++){ // Loop over blocks in columns
            for(brow=0;brow<blocksx;brow++){ // Loop of blocks in rows
                
                // Linear index into the top left element of the block (note this uses the original block sizes always).
                top_left = bcol*(blocksx)+brow;
                
                // Have to adjust poolsizes for non even block ratio
                if(remx>0 && brow==blocksx-1){poolsizex = remx;}else{poolsizex = poolx;}
                if(remy>0 && bcol==blocksy-1){poolsizey = remy;}else{poolsizey = pooly;}
                
                // Put the average into each location of the pool region.
                for(j=0;j<poolsizey;j++){
                    for(i=0;i<poolsizex;i++){ // Loop over rows of pool region
                        outputelem = brow*poolx+i+(bcol*pooly+j)*outputdims[0];
                        // Output at the pooled index is the selected element using the input indices.
                        outputp[outputelem] = imagesp[top_left];
                    }
                }
            }
        }
    }else if(num_dims==3){ // Pool each plane independently
        int brow, bcol, i, j, outputelem, plane3 = 0;
        
        #pragma omp parallel for num_threads(number_threads) shared(imagesp, outputp) private(plane3, brow, bcol, i, j, top_left, outputelem)
        for(plane3=0;plane3<outputdims[2];plane3++){ // Loop over the planes int he 3rd dimension
            for(bcol=0;bcol<blocksy;bcol++){ // Loop over blocks in columns
                for(brow=0;brow<blocksx;brow++){ // Loop of blocks in rows
                    
                    // Linear index into the top left element of the block (note this uses the original block sizes always).
                    top_left = plane3*(blocksx*blocksy)+bcol*(blocksx)+brow;
                    
                    // Have to adjust poolsizes for non even block ratio
                    if(remx>0 && brow==blocksx-1){poolsizex = remx;}else{poolsizex = poolx;}
                    if(remy>0 && bcol==blocksy-1){poolsizey = remy;}else{poolsizey = pooly;}
                    
                    // Put the average into each location of the pool region.
                    for(j=0;j<poolsizey;j++){
                        for(i=0;i<poolsizex;i++){ // Loop over rows of pool region
                            // Have to skip (brow*poolx)+i elements down, bcol*poolx+j elements over.
                            outputelem = plane3*(outputdims[0]*outputdims[1])+brow*poolx+i+(bcol*pooly+j)*outputdims[0];
                            // Output at the pooled index is the selected element using the input indices.
                            outputp[outputelem] = imagesp[top_left];
                        }
                    }
                }
            }
        }
    }else if(num_dims==4){ // Pool each plane independently
        int brow, bcol, i, j, outputelem, plane3, plane4 = 0;
        
        #pragma omp parallel for num_threads(number_threads) shared(imagesp, outputp) private(plane4, plane3, brow, bcol, i, j, top_left, outputelem)
        for(plane4=0;plane4<outputdims[3];plane4++){
            for(plane3=0;plane3<outputdims[2];plane3++){ // Loop over the planes int he 3rd dimension
                for(bcol=0;bcol<blocksy;bcol++){ // Loop over blocks in columns
                    for(brow=0;brow<blocksx;brow++){ // Loop of blocks in rows
                        
                        // Linear index into the top left element of the block (note this uses the original block sizes always).
                        top_left = plane4*(blocksx*blocksy*outputdims[2])+plane3*(blocksx*blocksy)+bcol*(blocksx)+brow;
                        
                                            
                        // Have to adjust poolsizes for non even block ratio
                        if(remx>0 && brow==blocksx-1){poolsizex = remx;}else{poolsizex = poolx;}
                        if(remy>0 && bcol==blocksy-1){poolsizey = remy;}else{poolsizey = pooly;}
                        
                        // Put the average into each location of the pool region.
                        for(j=0;j<poolsizey;j++){
                            for(i=0;i<poolsizex;i++){ // Loop over rows of pool region
                                // Have to skip (brow*poolx)+i elements down, bcol*poolx+j elements over.
                                outputelem = plane4*(outputdims[0]*outputdims[1]*outputdims[2])+plane3*(outputdims[0]*outputdims[1])+brow*poolx+i+(bcol*pooly+j)*outputdims[0];
                                // Output at the pooled index is the selected element using the input indices.
                                outputp[outputelem] = imagesp[top_left];
                            }
                        }                                                
                    }
                }
            }
        }
    }
    
    return;
    
}

