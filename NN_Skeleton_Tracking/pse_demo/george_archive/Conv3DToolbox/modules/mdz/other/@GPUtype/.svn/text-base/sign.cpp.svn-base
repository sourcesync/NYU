/*!
 * @brief GPUmat GPU implementation of the matlab max function (see implementation
 * for max).
 *
 * @file
 * @author Matthew Zeiler (zeiler@cs.nyu.edu)
 * @data Apr 29, 2011
 *
 * @pooltoolbox_file @copybrief max.cpp
 * @gpu_file @copybrief max.cpp
 *
 *****************
 * How to compile:
 *
 * In matlab go to the directory of this file and type:
 * >> make all
 *
 * Installs in GPUmat's @GPUtype folder so you can call max directly.
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
 * @copybrief max.cpp
 *
 * @param INPUT   	      prhs[0] // The input you want max of. Does max along first non-singleton if no other parameters are passed in.
 * @param INPUT2          prhs[1] // The same size as input or a scalara to compare to. Pass [] to do max over dimensions.
 * @param DIM             prhs[2] // If [] is arg 2 then this is the dimension to take max over.
 *
 * @retval MAX   	      plhs[0] // Corresponding maxes.
 * @retval IND            plhs[1] // Locations of the maxes along the dimension specified (only when specifying a dimension).
 */

// static paramaters
static CUfunction drvfun;

static int init = 0;

static GPUmat *gm;

void hostDriver(CUfunction drvfun, int N, int nrhs, hostdrv_pars_t *prhs);


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    CUresult cudastatus = CUDA_SUCCESS;
    
    if (nrhs != 1)
        mexErrMsgTxt("Wrong number of arguments, ONE required.");
    
    
    if (init == 0) {
        // Initialize function
        //mexLock();
        
        // load GPUmat
        gm = gmGetGPUmat();
        
        // load module
        CUmodule *drvmod = gmGetModule("other");
        
        // load float GPU function
//         CUresult status = cuModuleGetFunction(&drvfunf, *drvmod, "MAXPOOLF");
        CUresult status = cuModuleGetFunction(&drvfun, *drvmod,  "SIGN");
        if (CUDA_SUCCESS != status) {
            mexErrMsgTxt("Unable to load user function.");
        }
        init = 1;
    }
    
    
    //IN1 is the input GPU array
    GPUtype IN1 = gm->gputype.getGPUtype(prhs[0]);
    
    
    int nin1 = gm->gputype.getNumel(IN1);
    int ndims = gm->gputype.getNdims(IN1);
    gpuTYPE_t tin1 = gm->gputype.getType(IN1);
    
    
    // create GPUtype, with given dimensions (same size as input).
    int* im_size;
    im_size = (int*) mxCalloc(ndims, sizeof(int));
    im_size = (int*) gm->gputype.getSize(IN1);
    
    // If we are just comparing two matrices or a matrix and scalar.
    if(nlhs>1){
        mexErrMsgTxt("Only one output arguement supported.");
    }
    
    // Make new output array.
    GPUtype OUT = gm->gputype.create(tin1, ndims, im_size, NULL);
    
    // check input/out size and type
    int nout = gm->gputype.getNumel(OUT);
    
    if (tin1 !=gpuFLOAT)
        mexErrMsgTxt("Input must be GPUsingle");
    
    // I need the pointers to GPU memory
    CUdeviceptr d_IN1  = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(IN1));
    CUdeviceptr d_OUT = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(OUT));
    
    hostdrv_pars_t gpuprhs[2];
    int gpunrhs = 2;
    gpuprhs[0] = hostdrv_pars(&d_IN1, sizeof(d_IN1), __alignof(d_IN1));
    gpuprhs[1] = hostdrv_pars(&d_OUT, sizeof(d_OUT), __alignof(d_OUT));
    
    int N = nin1;
    
//     mexErrMsgTxt("Right before driver");
    
    hostDriver(drvfun, N, gpunrhs, gpuprhs);
    
    // Finally make the output available to MATLAB
    plhs[0] = gm->gputype.createMxArray(OUT);
    
}




//host driver
//void hostDriver(CUfunction drvfun, dim3 grid, dim3 threads, int shmem, int imgSizeX, int imgSizeY, int shmemX, int nrhs, hostdrv_pars_t *prhs) {
void hostDriver(CUfunction drvfun, int N, int nrhs, hostdrv_pars_t *prhs){
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
        
//         printf("size: %d, str: %d\n",size,str);
        
        
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
        
//         ALIGN_UP(poffset, __alignof(beta));
//         if (CUDA_SUCCESS != cuParamSetv(drvfun, poffset, &beta, sizeof(beta))) {
//             mexErrMsgTxt("Error in cuParamSeti");
//         }
//         poffset += sizeof(beta);
        
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














