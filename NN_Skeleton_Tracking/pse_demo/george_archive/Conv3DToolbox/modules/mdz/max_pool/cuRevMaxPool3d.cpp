/*!
 * @brief GPUmat GPU implementation of unpooling a max pooling over subregions of image planes.
 * The size of the subregions can be arbitrary, as can the number of planes.
 * If more planes are input for the provided indices than the images, only the 
 * first corresponding number of image planes will be used from the indices.
 * Note: input images must be single precision floats.
 *
 * @file
 * @author Matthew Zeiler (zeiler@cs.nyu.edu)
 * @data Apr 14, 2011
 *
 * @pooltoolbox_file @copybrief cuRevMaxPool3d.cpp
 * @gpu_file @copybrief cuRevMaxPool3d.cpp
 *
 *****************
 * How to compile:
 *
 * In matlab go to the directory of this file and type:
 * >> make all
 *
 */

#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#ifdef UNIX
#include <stdint.h>
#endif

#include "mex.h"
#include "math.h"



// CUDA
#include "cuda.h"
#include "cuda_runtime.h"

// #include "cuMaxPool_kernels.cu"
// #include "GPUkernel.hh"
#include "GPUmat.hh"


/*!
 * @copybrief reverse_max_pool.cpp
 *
 * @param IMAGES   	    prhs[0] // The stack of images you want to npool: n x m x c x d matrix
 * @param INDICES       prhs[1] // Previous indices from a pooling operation (optional). They must be the same size of the resulting output of pooling the IMAGES by this POOL_SIZE.
 * @param POOL_SIZE     prhs[2] // The size of the subregions to pool over: 1x3  matrix indicating the p x q x s size of the region.
 * @param UNPOOLED_SIZE prhs[3] // The unpooled size of the images which is typically n*p x m*q x c*s (just cares abou the first three dimensions so only input those as a 1x2 matrix).
 * @param INPUT_THREADS prhs[4] // User can optionally input the number of computation threads to parallelized over.
 *
 * @retval POOLED   	   plhs[0] // Unpooled stack of images. This will be roughly n*q x m*p x c*s x d
 */
// static paramaters
static CUfunction drvfunfunp; // kernel for unpooling.

static int init = 0;

static GPUmat *gm;

void hostDriver(CUfunction drvfun, int N, int nrhs, hostdrv_pars_t *prhs, int imx, int imy, int imk, int outx, int outy, int outk, int poolx, int pooly, int poolk);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    CUresult cudastatus = CUDA_SUCCESS;
    
    if (nrhs < 3)
        mexErrMsgTxt("Wrong number of arguments, atleast two required.");
    
    
    if (init == 0) {
        // Initialize function
        //mexLock();
        
        // load GPUmat
        gm = gmGetGPUmat();
        
        // load module
        CUmodule *drvmod = gmGetModule("max_pool");
        
        // load float GPU function
//         CUresult status = cuModuleGetFunction(&drvfunf, *drvmod, "MAXPOOLF");
        CUresult status = cuModuleGetFunction(&drvfunfunp, *drvmod,  "REVMAXPOOLF3D");
        if (CUDA_SUCCESS != status) {
            mexErrMsgTxt("Unable to load user function.");
        }
        

        init = 1;
    }
    
        //IN1 is the input GPU array
    GPUtype IN1 = gm->gputype.getGPUtype(prhs[0]);
    
    // IND is the second input
    GPUtype  IND = gm->gputype.getGPUtype(prhs[1]);


    // number of elements
    int nin1 = gm->gputype.getNumel(IN1);
    int nin2 = gm->gputype.getNumel(IND);
    

    
    gpuTYPE_t tin1 = gm->gputype.getType(IN1);
    gpuTYPE_t tin2 = gm->gputype.getType(IND);
    
        // Get the number of dimensions.
    int ndims = gm->gputype.getNdims(IN1);
    
        // create GPUtype, with given dimensions (same size as input).
    int* im_size;
    int* out_size;
    im_size = (int*) mxCalloc(ndims, sizeof(mwSize));
    out_size = (int*) mxCalloc(ndims, sizeof(mwSize));
    im_size = (int*) gm->gputype.getSize(IN1);
    
//     mexPrintf("nrhs %d\n",nrhs);
    
    double *psize = (double *) mxGetData(prhs[2]); // For the pool_size array.
    int numpdims = mxGetNumberOfDimensions(prhs[2]);
    int* pdims = (int*) mxCalloc(numpdims, sizeof(int));
    pdims = (int*) mxGetDimensions(prhs[2]);
    if(pdims[1]<3){
        mexPrintf("Number of pooling dims: %d\n",pdims);
        mexErrMsgTxt("The pooling sizes must be an array of atleast three dimensions for 3D pooling.");
    }
    
    double *unpsize;
    int numunpdims;
    int* unpdims;
    // Unpooled size was passed in.
    if(nrhs>=4 || mxIsEmpty(prhs[3])==true){
//         mexPrintf("Unpooled size is passed in\n");
        unpsize = (double *) mxGetData(prhs[3]); // For the pool_size array.    
        numunpdims = mxGetNumberOfDimensions(prhs[3]);
        unpdims = (int*) mxCalloc(numunpdims,sizeof(int));
        unpdims = (int*) mxGetDimensions(prhs[3]);        
    }else{ // Predict the unpooled size by multiplying by pooling regions.
        numunpdims = 2;
        unpdims = (int*) mxCalloc(numunpdims,sizeof(int));        
        unpdims[1] = 3; // Just 3D pooling.
        unpsize = (double*) mxCalloc(3, sizeof(double));
    //mexPrintf("blah\n\n");
        unpsize[0] = (double) im_size[0]*psize[0];
        unpsize[1] = (double) im_size[1]*psize[1];
        unpsize[2] = (double) im_size[2]*psize[2];
    }
    // Make sure only use up to the planes dimension of the passed in dimensions.
    if(unpdims[1]>3){
        unpdims[1] = 3;
    }
        
    
    

    

    
    // Set the unpooled size based on unpooled_size passed in and number of images passed in. .
    for(int dim=0;dim<ndims;dim++){
        // If we have an unpooled size then use that one. 
        if(dim<unpdims[1]){
//             mexPrintf("dim %d, unpsize %f\n",dim,unpsize[dim]);
            out_size[dim] = (int) unpsize[dim];
        }else{ // Otherwise just use the image size for the number of planes or images.
            out_size[dim] = (int) im_size[dim];
        }
    }
    
//     mexErrMsgTxt("Right after out_size\n");
    GPUtype OUT = gm->gputype.create(tin1, ndims, out_size, NULL);
    // Make them all zeros.
    gm->gputype.zeros(OUT);
    
    // The kernel we ultimately call depends on if the indices were passed in.
    CUfunction drvfun = drvfunfunp;
       
    
    int nout = gm->gputype.getNumel(OUT);
    gpuTYPE_t tout = gm->gputype.getType(OUT);
    
//     mexPrintf("nout: %d, nin1: %d, out_size: %d x %d x %d x %d\n", nout, nin1, out_size[0], out_size[1],out_size[2],out_size[3]);
//     mexPrintf("unpdims: %d, nin1: %d, out_size: %d x %d x %d x %d\n", unpdims, nin1, unpsize[0], unpsize[1],unpsize[2],out_size[4]);

    
    if (tin1 !=gpuFLOAT)
        mexErrMsgTxt("Input must be GPUsingle");
    
    // check input/out size and type
    if (nin1!=nin2)
        mexErrMsgTxt("Input arguments must have the same number of elements.");
    
//       if (nin1!=nout)
//         mexErrMsgTxt("Input and output arguments must have the same number of elements.");
    
    if (tin1!=tin2)
        mexErrMsgTxt("Input arguments must be of the same type.");
    
    if (tin1!=tout)
        mexErrMsgTxt("Input and output arguments must be of the same type.");
    
    
    // I need the pointers to GPU memory
    CUdeviceptr d_IN1  = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(IN1));
    CUdeviceptr d_IND  = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(IND));
    CUdeviceptr d_OUT = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(OUT));
    
    // CUdeviceptr d_IN1  = (UINTPTR gm->gputype.getGPUptr(IN1));
    // CUdeviceptr d_IN2  = (UINTPTR gm->gputype.getGPUptr(IN2));
    // CUdeviceprt d_OUT = (UINTPTR gm->gputype.getGPUptr(OUT));
    
    
    hostdrv_pars_t gpuprhs[3];
    int gpunrhs = 3;
    gpuprhs[0] = hostdrv_pars(&d_IN1, sizeof(d_IN1), __alignof(d_IN1));
    gpuprhs[1] = hostdrv_pars(&d_OUT, sizeof(d_OUT), __alignof(d_OUT));
    gpuprhs[2] = hostdrv_pars(&d_IND, sizeof(d_IND), __alignof(d_IND));
//     GPUtype pool_size = gm->gputype.mxToGPUtype(prhs[2]);
//     gpuprhs[3] = hostdrv_pars(&psize, sizeof(psize), __alignof(psize));
    
    
    int N = nin1;
    

    
//      hostGPUDRV(drvfun, N, gpunrhs, gpuprhs);
    hostDriver(drvfun, N, gpunrhs, gpuprhs, (int)im_size[0], (int)im_size[1], (int)im_size[2], (int)out_size[0], (int)out_size[1], (int)out_size[2], (int)psize[0], (int)psize[1], (int)psize[2]);
    //   int numThreadsPerBlock = 256;
    //   const unsigned int numBlocks = (out_size[0] + numThreadsPerBlock - 1) / numThreadsPerBlock;
    
    //  PLUSF<<<numBlocks,numThreadsPerBlock>>>(N, 0, (float *)d_IN1,(float *)d_IN2,(float *)d_OUT);

    
    // Finally make the output available to MATLAB
    plhs[0] = gm->gputype.createMxArray(OUT);
//     plhs[1] = gm->gputype.createMxArray(IND);
    
    
//             mexPrintf("nout: %d, nin1: %d, nind: %d, out_size: %d x %d\n", nout, nin1, nin2, out_size[0], out_size[1]);

    
    
}




//host driver
//void hostDriver(CUfunction drvfun, dim3 grid, dim3 threads, int shmem, int imgSizeX, int imgSizeY, int shmemX, int nrhs, hostdrv_pars_t *prhs) {
void hostDriver(CUfunction drvfun, int N, int nrhs, hostdrv_pars_t *prhs, int imx, int imy, int imk, int outx, int outy, int outk, int poolx, int pooly, int poolk){
    //mexPrintf("threads.x: %d threads.y: %d threads.z %d\n",threads.x,threads.y,threads.z);
    
    
    unsigned int maxthreads = 65000;
    // Set threads per block here.
        unsigned int blocksdim1d = 256;
    dim3 threads(blocksdim1d, 1, 1);
    int nstreams = iDivUp(N, maxthreads*blocksdim1d);
    CUresult err = CUDA_SUCCESS;
    for (int str = 0; str < nstreams; str++) {
        int offset = str * maxthreads * blocksdim1d;
        int size = 0;
        if (str == (nstreams - 1))
            size = N - str * maxthreads * blocksdim1d;
        else
            size = maxthreads * blocksdim1d;
        
        
        int gridx = iDivUp(size, blocksdim1d); // number of x blocks
        
        // setup execution parameters
        
        if (CUDA_SUCCESS != (err = cuFuncSetBlockShape(drvfun, threads.x, threads.y, threads.y))) {
            mexErrMsgTxt("Error in cuFuncSetBlockShape");
        }
        
        if (CUDA_SUCCESS != cuFuncSetSharedSize(drvfun, 0)) {
            mexErrMsgTxt("Error in cuFuncSetSharedSize");
        }
        
        //mexPrintf("block shape ok\n");
        
        // add parameters
        int poffset = 0;
        
        // CUDA kernels interface
        // N: number of elements
        for (int p=0;p<nrhs;p++) {
            ALIGN_UP(poffset, prhs[p].align);
            if (CUDA_SUCCESS
                    != cuParamSetv(drvfun, poffset, prhs[p].par, prhs[p].psize)) {
                mexErrMsgTxt("Error in cuParamSetv");
            }
            poffset += prhs[p].psize;
        }
        
        ALIGN_UP(poffset, __alignof(size));
        if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, size)) {
            mexErrMsgTxt("Error in cuParamSeti");
        }
        poffset += sizeof(size);
        
        ALIGN_UP(poffset, __alignof(offset));
        if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, offset)) {
            mexErrMsgTxt("Error in cuParamSeti");
        }
        poffset += sizeof(offset);
        
        ALIGN_UP(poffset, __alignof(imx));
        if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, imx)) {
            mexErrMsgTxt("Error in cuParamSeti");
        }
        poffset += sizeof(imx);
        
        ALIGN_UP(poffset, __alignof(imy));
        if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, imy)) {
            mexErrMsgTxt("Error in cuParamSeti");
        }
        poffset += sizeof(imy);
        
        ALIGN_UP(poffset, __alignof(imk));
        if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, imk)) {
            mexErrMsgTxt("Error in cuParamSeti");
        }
        poffset += sizeof(imk);
        
        ALIGN_UP(poffset, __alignof(outx));
        if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, outx)) {
            mexErrMsgTxt("Error in cuParamSeti");
        }
        poffset += sizeof(outx);
        
        ALIGN_UP(poffset, __alignof(outy));
        if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, outy)) {
            mexErrMsgTxt("Error in cuParamSeti");
        }
        poffset += sizeof(outy);
        
         ALIGN_UP(poffset, __alignof(outk));
        if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, outk)) {
            mexErrMsgTxt("Error in cuParamSeti");
        }
        poffset += sizeof(outk);
        
        ALIGN_UP(poffset, __alignof(poolx));
        if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, poolx)) {
            mexErrMsgTxt("Error in cuParamSeti");
        }
        poffset += sizeof(poolx);
        
        ALIGN_UP(poffset, __alignof(pooly));
        if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, pooly)) {
            mexErrMsgTxt("Error in cuParamSeti");
        }
        poffset += sizeof(pooly);
        
         ALIGN_UP(poffset, __alignof(poolk));
        if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, poolk)) {
            mexErrMsgTxt("Error in cuParamSeti");
        }
        poffset += sizeof(poolk);
        
//   if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, shmemX)) {
//     mexErrMsgTxt("Error in cuParamSeti");
//   }
//   poffset += sizeof(shmemX);
        
        if (CUDA_SUCCESS != cuParamSetSize(drvfun, poffset)) {
            mexErrMsgTxt("Error in cuParamSetSize");
        }
        
        err = cuLaunchGridAsync(drvfun, gridx, 1, 0);
        if (CUDA_SUCCESS != err) {
            mexErrMsgTxt("Error running kernel");
        }
        
    }
}

