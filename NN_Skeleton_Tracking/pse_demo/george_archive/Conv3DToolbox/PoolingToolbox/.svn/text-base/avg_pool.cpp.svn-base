/*!
 * @brief MEX implementation of avg pooling over subregions of image planes.
 * The size of the subregions can be arbitrary, as can the number of planes.
 * If there are negative values in the input, this pooling takes the average
 * value (but keeps its sign as well). Note: input images must be
 * single precision floats.
 *
 *
 * @file
 * @author Matthew Zeiler (zeiler@cs.nyu.edu)
 * @data Aug 14, 2010
 *
 * @pooltoolbox_file @copybrief max_pool.cpp
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
#define	IMAGES   	    prhs[0] // The stack of images you want to pool: n x m x c x d matrix of single precision floating point numbers.
#define POOL_SIZE       prhs[1] // The size of the subregions to pool over: 1x2  matrix indicating the p x q size of the region.
#define INDICES         prhs[2] // NOT USED
#define INPUT_THREADS   prhs[3] // User can optionally input the number of computation threads to parallelized over.

// Output Arguments
#define	POOLED   	    plhs[0] // Pooled stack of images. This will be roughly n/q x m/p x c x d
#define POOLED_INDICES  plhs[1] // The indices of where the max within each subregion was located.

/*!
 * @copybrief max_pool.cpp
 *
 * @param IMAGES   	      prhs[0] // The stack of images you want to pool: n x m x c x dmatrix
 * @param POOL_SIZE       prhs[1] // The size of the subregions to pool over: 1x2  matrix indicating the p x q size of the regino.
 * @param INDICES         prhs[2] // Previous indices from a pooling operation (optional). They must be the same size of the resulting output of pooling the IMAGES by this POOL_SIZE. Pass in [] to ignore this and set number of compute threads as 4th input.
 * @param INPUT_THREADS   prhs[3] // User can optionally input the number of computation threads to parallelized over.
 *
 * @retval POOLED   	  plhs[0] // Pooled stack of images. This will be roughly n/q x m/p x c x d
 * @retval POOLED_INDICES plhs[1] // The indices of where the max within each subregion was located.
 */
void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray*prhs[] )
        
{
    int num_dims, number_threads;
    float *imagesp;
    unsigned short int *inputindsp; // Pointers to input.
    float *outputp;
//    float *indicesp;   // Pointers to outputs.
    double *poolp; // For the pool_size array.

    
    int *number_threadsp;
    int blocksx, blocksy, remx, remy, bcoljump, plane3jump, plane4jump, top_left;
    int poolx, pooly, poolsizex, poolsizey;
    
    const mwSize *dims;
    const mwSize *inddims;
    int* outputdims = NULL;
    
    // Check for proper number of arguments
    if (nrhs < 2) {
        mexErrMsgTxt("Two input arguments required at minimum.");
    } else if (nlhs > 2) {
        mexErrMsgTxt("Too many output arguments.");
    }
    
    // If number of threads was not input, then use default set abvoe.
    if (nrhs < 5)
        number_threads = MAX_NUM_THREADS;
    else
        number_threads = (int)mxGetScalar(INPUT_THREADS);
    
    
    
    
    // Get dimensions of image and kernel
    num_dims = mxGetNumberOfDimensions(IMAGES);
    dims = (mwSize*) mxCalloc(num_dims, sizeof(mwSize));
    dims = mxGetDimensions(IMAGES);
    
    // Make sure the indices are provided and the same number of dimensions as input.
//    if (nrhs >=3 && mxIsEmpty(INDICES)==false){
//        inddims = (mwSize*) mxCalloc(num_dims, sizeof(mwSize));
//        inddims = mxGetDimensions(INDICES);
//        INDICES_PROVIDED = true;
//        if(mxIsSingle(INDICES) == 1){
//            inputindsp = (float *) mxGetData(INDICES);
//        }else{
//            mexErrMsgTxt("Input Indices must be a single precision (float).");
//        }
//    }
    

//      // Passed in an non-empty array
//      if(inddims[0]>0){
//          INDICES_PROVIDED=true;
//          mexPrintf("number_threads: %d and %d Indices dims: %d x %d\n", number_threads,mxIsSingle(IMAGES),inddims[0],inddims[1]);
//         mexPrintf("indices(2,2): %f\n",inputindsp[6]);
//      }
    
    
    
    
    // Get pointers to IMAGE and POOL_SIZE
    if (mxIsSingle(IMAGES) == 1){
        imagesp = (float*) mxGetData(IMAGES);
    }else{
        mexErrMsgTxt("Input Images must be a single precision (float).");
    }
    poolp = mxGetPr(POOL_SIZE);

    
    // Get the pool region size
    poolx = (int) poolp[0];
    pooly = (int) poolp[1];
    if(poolx*pooly>65536){
        mexErrMsgTxt("Cannot pool in 2D larger than 256x256 due to uint16 indices limitation.");
    }
    // Number of blocks in rows on each plane.
    blocksx = (int)dims[0]/poolx;
    // Remaining elements past blocksx
    remx = (int)(dims[0])%poolx;
    // Number of block in columns on each plane.
    blocksy = (int)dims[1]/pooly;
    // Remaining elements past blocksy
    remy = (int)(dims[1])%pooly;
    
    if(remx>0)
        blocksx = blocksx+1;
    if(remy>0)
        blocksy = blocksy+1;
    
//     mexPrintf("imagesp: %f %f %f poolp: %p: %d x %d\n", imagesp[0], imagesp[1], imagesp[2], poolp,poolx, pooly);
//     mexPrintf("blocksx: %d blocksy: %d,   remx: %d, remy %d\n", blocksx, blocksy, remx, remy);
    
    
    outputdims = new int[4];
    // Create empty 1x1 matrix to return.
    outputdims[0] = 0;
    outputdims[1] = 0;
    POOLED_INDICES = mxCreateNumericArray(2, (mwSize*) outputdims, mxUINT16_CLASS, mxREAL);
    
    
    
    outputdims = new int[4];
    // Number of blocks is the output x and y dimensions.
    outputdims[0] = blocksx;
    outputdims[1] = blocksy;
    // Remaining dimensions are same as input.
    for(int k=2;k<(int)num_dims;k++){
        outputdims[k] = dims[k];
    }
    // Set dimensions like matlab.
    if(num_dims==2){
        outputdims[2] = 1;
        outputdims[3] = 1;
    }else if(num_dims==3){
        outputdims[3] = 1;
    }
    
    // Create the output arrays.
    POOLED = mxCreateNumericArray(num_dims, (mwSize*) outputdims, mxSINGLE_CLASS, mxREAL);
    
    // Get pointer to output matrix
    outputp = (float*) mxGetData(POOLED);
    // Create new indices array if none provided.
//    POOLED_INDICES = mxCreateNumericArray(num_dims, (mwSize*) outputdims, mxSINGLE_CLASS, mxREAL);
//    indicesp = (float*) mxGetData(POOLED_INDICES);
    
   
    
    
    // How much each bcol jumps in linear indices
    bcoljump = pooly*dims[0];
    plane3jump = dims[0]*dims[1];
    plane4jump = dims[0]*dims[1]*dims[2];
    
//     mexPrintf("Number of input dimensions: %d %d %d %d, num_dims: %d", dims[0], dims[1], dims[2], dims[3], num_dims);
//     mexPrintf("Number of output dimensions: %d %d %d %d, bcoljump %d plane3jump %d",outputdims[0],outputdims[1],outputdims[2],outputdims[3],bcoljump,plane3jump);
    //  mexErrMsgTxt("Kernel must be a single precision (float), not double.");
    
    

        if(num_dims==2){ // Just pool the single image.
//             mexPrintf("dims: %d x %d \n", dims[0], dims[1]);
            float avg, count = 0;
            int brow, bcol, i, j = 0;
            
            #pragma omp parallel for num_threads(number_threads) shared(imagesp, outputp) private(avg, count, brow, bcol, i, j, top_left, poolsizex, poolsizey)
            for(bcol=0;bcol<blocksy;bcol++){ // Loop over blocks in columns
                for(brow=0;brow<blocksx;brow++){ // Loop of blocks in rows
                    
                    // Have to adjust poolsizes for non even block ratio
                    if(remx>0 && brow==blocksx-1){poolsizex = remx;}else{poolsizex = poolx;}
                    if(remy>0 && bcol==blocksy-1){poolsizey = remy;}else{poolsizey = pooly;}
                    
                    // Linear index into the top left element of the block (note this uses the original block sizes always).
                    top_left = bcol*(bcoljump)+brow*poolx;
                    
                    avg = 0;
                    count = 0;
                    for(j=0;j<poolsizey;j++){
                        for(i=0;i<poolsizex;i++){ // Loop over rows of pool region
                            int elem = top_left+j*dims[0]+i;
                            avg = avg + imagesp[elem];
                            count = count + 1;
                        }
                    }
                    avg = avg/count;                                                                           
                    
                    outputp[bcol*blocksx+brow] = avg;                    
                }
            }
            
            
        }else if(num_dims==3){ // Pool each plane independently
//             mexPrintf("dims: %d x %d x %d \n", dims[0], dims[1], dims[2]);
            float avg, count = 0;
            int brow, bcol, i, j, plane3 = 0;
            
            #pragma omp parallel for num_threads(number_threads) shared(imagesp, outputp) private(avg, count, plane3, brow, bcol, i, j, top_left, poolsizex, poolsizey)
            for(plane3=0;plane3<outputdims[2];plane3++){ // Loop over the planes int he 3rd dimension
                for(bcol=0;bcol<blocksy;bcol++){ // Loop over blocks in columns
                    for(brow=0;brow<blocksx;brow++){ // Loop of blocks in rows
                        
                        // Have to adjust poolsizes for non even block ratio
                        if(remx>0 && brow==blocksx-1){
                            poolsizex = remx;
                        }else{
                            poolsizex = poolx;
                        }
                        if(remy>0 && bcol==blocksy-1){
                            poolsizey = remy;
                        }else{
                            poolsizey = pooly;
                        }
                        
                        // Linear index into the top left element of the block (note this uses the original block sizes always).
                        top_left = plane3*(plane3jump)+bcol*(bcoljump)+brow*poolx;
                        
//                     mexPrintf("poosizex: %d poolsizey: %d topleft: %d, brow %d, bcol %d\n",poolsizex,poolsizey,top_left,brow,bcol);
                            avg = 0;
                            count = 0;
                            for(j=0;j<poolsizey;j++){
                                for(i=0;i<poolsizex;i++){ // Loop over rows of pool region
                                    int elem = top_left+j*dims[0]+i;
                                    avg = avg + imagesp[elem];
                                    count = count + 1;
                                }
                            }
                            avg = avg/count;                            
                                                        
//                     mexPrintf("Max: %f Index: %d\n\n",max,maxi);
//                         mexPrintf("Index: %d\n",plane3*(blocksx*blocksy)+bcol*blocksx+brow);
                        int outputelem = plane3*(blocksx*blocksy)+bcol*blocksx+brow;
                        outputp[outputelem] = avg;                        
                    }
                }
            }
        }else if(num_dims==4){ // Pool each plane independently
//             mexPrintf("dims: %d x %d x %d \n", dims[0], dims[1], dims[2]);
            float avg, count = 0;
            int brow, bcol, i, j, plane3, plane4 = 0;
            
            #pragma omp parallel for num_threads(number_threads) shared(imagesp, outputp) private(avg, count, plane4, plane3, brow, bcol, i, j, top_left, poolsizex, poolsizey)
            for(plane4=0;plane4<outputdims[3];plane4++){
                for(plane3=0;plane3<outputdims[2];plane3++){ // Loop over the planes int he 3rd dimension
                    for(bcol=0;bcol<blocksy;bcol++){ // Loop over blocks in columns
                        for(brow=0;brow<blocksx;brow++){ // Loop of blocks in rows
                            
                            // Have to adjust poolsizes for non even block ratio
                            if(remx>0 && brow==blocksx-1){
                                poolsizex = remx;
                            }else{
                                poolsizex = poolx;
                            }
                            if(remy>0 && bcol==blocksy-1){
                                poolsizey = remy;
                            }else{
                                poolsizey = pooly;
                            }
                            
                            // Linear index into the top left element of the block (note this uses the original block sizes always).
                            top_left = plane4*(plane4jump)+plane3*(plane3jump)+bcol*(bcoljump)+brow*poolx;
                            
//                     mexPrintf("poosizex: %d poolsizey: %d topleft: %d, brow %d, bcol %d\n",poolsizex,poolsizey,top_left,brow,bcol);
                            avg = 0;
                            count = 0;
                            for(j=0;j<poolsizey;j++){
                                for(i=0;i<poolsizex;i++){ // Loop over rows of pool region
                                    int elem = top_left+j*dims[0]+i;
                                    avg = avg + imagesp[elem];
                                    count = count + 1;
                                }
                            }
                            avg = avg/count;                            
                            
//                     mexPrintf("Max: %f Index: %d\n\n",max,maxi);
//                         mexPrintf("Index: %d\n",plane3*(blocksx*blocksy)+bcol*blocksx+brow);
                            int outputelem = plane4*(blocksx*blocksy*outputdims[2])+plane3*(blocksx*blocksy)+bcol*blocksx+brow;
                            outputp[outputelem] = avg;                            
                        }
                    }
                }
            }
        }
    return;
    
}

