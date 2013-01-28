/*!
 * @brief  This GPUmat GPU implementation does the L1 shrinkage function max(z-beta,0).*sign(z);
 * This does it in place so the input matrix is also the result.
 *
 * @file
 * @author Matthew Zeiler (zeiler@cs.nyu.edu)
 * @data Apr 28, 2011
 *
 * @gpu_file @copybrief cuShrink.cpp
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
 * @copybrief cuMaxPool3d.cpp
 *
 * @param INPUT   	      prhs[0] // The stack of variable you want to shrink
 * @param beta            prhs[1] // The shrinkage threshold.
 */

// static paramaters
static CUfunction drvfun;

static int init = 0;

static GPUmat *gm;

void hostDriver(CUfunction drvfun, int N, int nrhs, hostdrv_pars_t *prhs, float beta);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    CUresult cudastatus = CUDA_SUCCESS;
    
    if (nrhs < 2)
        mexErrMsgTxt("Wrong number of arguments, two required.");
    
    
    if (init == 0) {
        // Initialize function
        //mexLock();
        
        // load GPUmat
        gm = gmGetGPUmat();
        
        // load module
        CUmodule *drvmod = gmGetModule("max_pool");
        
        // load float GPU function
//         CUresult status = cuModuleGetFunction(&drvfunf, *drvmod, "MAXPOOLF");
        CUresult status = cuModuleGetFunction(&drvfun, *drvmod,  "L1SHRINK");
        if (CUDA_SUCCESS != status) {
            mexErrMsgTxt("Unable to load user function.");
        }
        init = 1;
    }
    
//     double *tempbeta = mxGetPr(prhs[1]); // For the pool_size array.
//     float *beta = (float) mxCalloc(1,sizeof(float));
//     beta[0] = (float)tempbeta[0];
       float beta = (float)mxGetScalar(prhs[1]);
    
//     GPUtype beta = gm->gputype.mxToGPUtype(prhs[1]);
    
    
//     mexPrintf("beta[i] is: %f\n",beta);
//     mexErrMsgTxt("");
    // mex parameters are:
    
    // 1. IN1
    // 2. IN2
    // 3. OUT
    
    //IN1 is the input GPU array
    GPUtype IN1 = gm->gputype.getGPUtype(prhs[0]);
    
    
    
    
    
    //OUT is the output GPU array (result)
    //   GPUtype OUT = gm->gputype.getGPUtype(prhs[2]);
    
    // number of elements
    int nin1 = gm->gputype.getNumel(IN1);
//     int nin2 = gm->gputype.getNumel(IN2);
    
    // Get the number of dimensions.
    int ndims = gm->gputype.getNdims(IN1);
    
    gpuTYPE_t tin1 = gm->gputype.getType(IN1);
//     gpuTYPE_t tin2 = gm->gputype.getType(IN2);
    
    // create GPUtype, with given dimensions (same size as input).
    int* im_size;
    im_size = (int*) mxCalloc(ndims, sizeof(mwSize));
    im_size = (int*) gm->gputype.getSize(IN1);
    
//     mexErrMsgTxt("Right after out_size\n");
//     GPUtype OUT = gm->gputype.create(tin1, ndims, im_size, NULL);
    
    // The kernel we ultimately call depends on if the indices were passed in.
//     CUfunction drvfun;
    
//     GPUtype IND;
    
//     int nout = gm->gputype.getNumel(OUT);
    
    if (tin1 !=gpuFLOAT)
        mexErrMsgTxt("Input must be GPUsingle");
    
    
    // I need the pointers to GPU memory
    CUdeviceptr d_IN1  = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(IN1));
//     CUdeviceptr d_beta = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(beta));
    
    
    hostdrv_pars_t gpuprhs[1];
    int gpunrhs = 1;
    gpuprhs[0] = hostdrv_pars(&d_IN1, sizeof(d_IN1), __alignof(d_IN1));
//     gpuprhs[1] = hostdrv_pars(&d_beta, sizeof(d_beta), __alignof(d_beta));
    
    
    int N = nin1;
    
//      hostGPUDRV(drvfun, N, gpunrhs, gpuprhs);
    hostDriver(drvfun, N, gpunrhs, gpuprhs,beta);
    //   int numThreadsPerBlock = 256;
    //   const unsigned int numBlocks = (out_size[0] + numThreadsPerBlock - 1) / numThreadsPerBlock;
    
    //  PLUSF<<<numBlocks,numThreadsPerBlock>>>(N, 0, (float *)d_IN1,(float *)d_IN2,(float *)d_OUT);
    
    
    // Finally make the output available to MATLAB
//     plhs[0] = gm->gputype.createMxArray(IN1);
//     plhs[1] = gm->gputype.createMxArray(IND);
}




//host driver
//void hostDriver(CUfunction drvfun, dim3 grid, dim3 threads, int shmem, int imgSizeX, int imgSizeY, int shmemX, int nrhs, hostdrv_pars_t *prhs) {
void hostDriver(CUfunction drvfun, int N, int nrhs, hostdrv_pars_t *prhs, float beta){
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
        
        ALIGN_UP(poffset, __alignof(beta));
        if (CUDA_SUCCESS != cuParamSetv(drvfun, poffset, &beta, sizeof(beta))) {
            mexErrMsgTxt("Error in cuParamSeti");
        }
        poffset += sizeof(beta);
        
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


















