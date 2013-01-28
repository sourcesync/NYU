#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#ifdef UNIX
#include <stdint.h>
#endif

#include "mex.h"

// CUDA
#include "cuda.h"
#include "cuda_runtime.h"


#include "GPUmat.hh"

#include "math.h"
// #include "misc.cuh"

// static paramaters
static CUfunction drvfunf; // float
/*static CUfunction drvfunc; // complex
 * static CUfunction drvfund; // double
 * static CUfunction drvfuncd;//double complex
 */

static int init = 0;

static GPUmat *gm;



#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

//host driver
void hostDriver(CUfunction drvfun, int imgSize, int paddingSize, int numImages, int nrhs, hostdrv_pars_t *prhs) {
    
    dim3 threads(16, 16, 1);
    dim3 blocks(numImages, 1, 1);
    
    while(blocks.x > 65535) {
        blocks.x = DIVUP(blocks.x, 2);
        blocks.y *= 2;
    }
    //mexPrintf("threads.x: %d threads.y: %d threads.z %d\n",threads.x,threads.y,threads.z);
    
    CUresult err = CUDA_SUCCESS;
    
    // setup execution parameters
    if (CUDA_SUCCESS != (err = cuFuncSetBlockShape(drvfun, threads.x, threads.y, threads.z))) {
        mexErrMsgTxt("Error in cuFuncSetBlockShape");
    }
    
    //mexPrintf("block shape ok\n");
    
    // add parameters
    int poffset = 0;
    
    // CUDA kernels interface
    // N: number of elements
    for (int p=0;p<nrhs;p++) {
        if (CUDA_SUCCESS
                != cuParamSetv(drvfun, poffset, prhs[p].par, prhs[p].psize)) {
            mexErrMsgTxt("Error in cuParamSetv");
        }
        poffset += prhs[p].psize;
    }
    
    if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, imgSize)) {
        mexErrMsgTxt("Error in cuParamSeti");
    }
    poffset += sizeof(imgSize);
    
    if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, paddingSize)) {
        mexErrMsgTxt("Error in cuParamSeti");
    }
    poffset += sizeof(paddingSize);
    
    if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, numImages)) {
        mexErrMsgTxt("Error in cuParamSeti");
    }
    poffset += sizeof(numImages);
    
    if (CUDA_SUCCESS != cuParamSetSize(drvfun, poffset)) {
        mexErrMsgTxt("Error in cuParamSetSize");
    }
    
    err = cuLaunchGridAsync(drvfun, blocks.x, blocks.y, 0);
    if (CUDA_SUCCESS != err) {
        mexErrMsgTxt("Error running kernel");
    }
    
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    CUresult cudastatus = CUDA_SUCCESS;
    
    if (nrhs != 2)
        mexErrMsgTxt("Wrong number of arguments (2 expected)");
    
    if (init == 0) {
        // Initialize function
        //mexLock();
        
        // load GPUmat
        gm = gmGetGPUmat();
        
        // load module
        CUmodule *drvmod = gmGetModule("other");
        
        
        // load float GPU function
        CUresult status = cuModuleGetFunction(&drvfunf, *drvmod, "PADARRAY");
        if (CUDA_SUCCESS != status) {
            mexErrMsgTxt("Unable to load user function.");
        }
        
        init = 1;
    }
    
    // mex parameters are:
    
    // 1. IN1
    // 2. OUT
    // 3. paddingSize
    
    
    //IN1 is the input GPU array
    GPUtype IN1 = gm->gputype.getGPUtype(prhs[0]);
    
    //OUT is the output GPU array (result)
//   GPUtype OUT = gm->gputype.getGPUtype(prhs[1]);
    
    //last parameter is the filterSize (int)
    int paddingSize = (int) mxGetScalar(prhs[1]);
    
    // number of elements
    int nin1 = gm->gputype.getNumel(IN1);
    int ndims1 = gm->gputype.getNdims(IN1);
    
    //dimensions
    const int * sin1 = gm->gputype.getSize(IN1);
    
    
    if (sin1[0] != sin1[1])
        mexErrMsgTxt("Images not square");
    
    int imgSize = sin1[0];
    
//     if (nout != numImages * (imgSize + 2*paddingSize)*(imgSize + 2*paddingSize))
//         mexErrMsgTxt("Target dimensions not consistent");
    
    gpuTYPE_t tin1 = gm->gputype.getType(IN1);
//   gpuTYPE_t tout = gm->gputype.getType(OUT);
    
    // check input/out size and type
//     if (tin1!=tout)
//         mexErrMsgTxt("Input and output arguments must be of the same type.");
    
    int numOutputsX = sin1[0]+paddingSize*2;
    
    int outsize[6];
    outsize[0] = numOutputsX;
    outsize[1] = numOutputsX;
    if(ndims1>2){
        outsize[2] = sin1[2]; // num_input_maps
    }else{
        outsize[2] = 1;
    }
    if(ndims1>3){
        outsize[3] = sin1[3]; // num_feature_maps
    }else{
        outsize[3] = 1;
    }
    if(ndims1>4){
        outsize[4] = sin1[4]; // num_cases
    }else{
        outsize[4] = 1;
    }
    if(ndims1>5){
        outsize[5] = sin1[5];
    }else{
        outsize[5] = 1;
    }
    
    // Make new output array.
    GPUtype OUT = gm->gputype.create(tin1, ndims1, outsize, NULL);
//     gm->gputype.zeros(OUT);
    int nout = gm->gputype.getNumel(OUT);
    const int * sout = gm->gputype.getSize(OUT);
    
    // Get the total number of images needed.
    int numImages = 1;
    for(int i=3;i<=ndims1;i++){
        numImages *= outsize[i-1];
    }
    
    
    // I need the pointers to GPU memory
    CUdeviceptr d_IN1  = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(IN1));
    CUdeviceptr d_OUT = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(OUT));
    
    // The GPU kernel depends on the type of input/output
    CUfunction drvfun;
    if (tin1 == gpuFLOAT) {
        drvfun = drvfunf;
    } else
        mexErrMsgTxt("Currently only single types supported.");
    
    hostdrv_pars_t gpuprhs[2];
    int gpunrhs = 2;
    gpuprhs[0] = hostdrv_pars(&d_IN1, sizeof(d_IN1));
    gpuprhs[1] = hostdrv_pars(&d_OUT, sizeof(d_OUT));
    
    int N = nin1;
    
    //hostDriver(drvfun, N, gpunrhs, gpuprhs);
    hostDriver(drvfun, imgSize, paddingSize, numImages, gpunrhs, gpuprhs);
    
    // Finally make the output available to MATLAB
    plhs[0] = gm->gputype.createMxArray(OUT);
    
}
