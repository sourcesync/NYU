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

#include "misc.cuh"

#include<algorithm>

// static paramaters
static CUfunction drvfunf; // float
//static CUfunction drvfunc; // complex
//static CUfunction drvfund; // double
//static CUfunction drvfuncd;//double complex

static int init = 0;

static GPUmat *gm;

/* Driver */
void hostGPUmisc(int nrhs, hostdrv_pars_t *prhs, int numA, int numB, int numC, int numCases) {


  int N = numA*numB*numC; //total # elements
  
  /* Each thread block computes a linear block of the target */
  //int blocksX = (N + BLOCK_DIM1D - 1) / BLOCK_DIM1D; //BLOCK_DIM1D set in GPUkernel.hh
  
  CUresult err = CUDA_SUCCESS;

    // setup execution parameters

    if (CUDA_SUCCESS != (err = cuFuncSetBlockShape(drvfunf, NUM_APPLY_THREADS_PER_BLOCK, 1, 1))) {
      mexErrMsgTxt("Error in cuFuncSetBlockShape");
    }

    if (CUDA_SUCCESS != cuFuncSetSharedSize(drvfunf, 0)) {
      mexErrMsgTxt("Error in cuFuncSetSharedSize");
    }

    /*    if (CUDA_SUCCESS != cuFuncSetSharedSize(drvfunf, 0)) {
      mexErrMsgTxt("Error in cuFuncSetSharedSize");
      }*/

    // add parameters
    int poffset = 0;

    // CUDA kernels interface
    // N: number of elements
    // offset: used for streams

    ALIGN_UP(poffset, __alignof(N));
    if (CUDA_SUCCESS != cuParamSeti(drvfunf, poffset, N)) {
      mexErrMsgTxt("Error in cuParamSeti");
    }
    poffset += sizeof(N);

    ALIGN_UP(poffset, __alignof(numA));
    if (CUDA_SUCCESS != cuParamSeti(drvfunf, poffset, numA)) {
      mexErrMsgTxt("Error in cuParamSeti");
    }
    poffset += sizeof(numA);

    ALIGN_UP(poffset, __alignof(numB));
    if (CUDA_SUCCESS != cuParamSeti(drvfunf, poffset, numB)) {
      mexErrMsgTxt("Error in cuParamSeti");
    }
    poffset += sizeof(numB);

    ALIGN_UP(poffset, __alignof(numC));
    if (CUDA_SUCCESS != cuParamSeti(drvfunf, poffset, numC)) {
      mexErrMsgTxt("Error in cuParamSeti");
    }
    poffset += sizeof(numC);

    ALIGN_UP(poffset, __alignof(numCases));
    if (CUDA_SUCCESS != cuParamSeti(drvfunf, poffset, numCases)) {
      mexErrMsgTxt("Error in cuParamSeti");
    }
    poffset += sizeof(numCases);

    for (int p=0;p<nrhs;p++) {
      ALIGN_UP(poffset, prhs[p].align);
      if (CUDA_SUCCESS
          != cuParamSetv(drvfunf, poffset, prhs[p].par, prhs[p].psize)) {
        mexErrMsgTxt("Error in cuParamSetv");
      }
      poffset += prhs[p].psize;
    }

    if (CUDA_SUCCESS != cuParamSetSize(drvfunf, poffset)) {
      mexErrMsgTxt("Error in cuParamSetSize");
    }

    const int numBlocks = std::min(NUM_BLOCKS_MAX,DIVUP(N, NUM_APPLY_THREADS_PER_BLOCK));
    mexPrintf("N: %d numThreadsPerBlock: %d numBlocks: %d\n",N,NUM_APPLY_THREADS_PER_BLOCK,numBlocks);

    err = cuLaunchGridAsync(drvfunf, numBlocks, 1, 0);
    if (CUDA_SUCCESS != err) {
      mexErrMsgTxt("Error running kernel");
    }
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  CUresult cudastatus = CUDA_SUCCESS;

  if (nrhs != 4)
    mexErrMsgTxt("Wrong number of arguments");

  if (init == 0) {
    // Initialize function
    //mexLock();

    // load GPUmat
    gm = gmGetGPUmat();

    // load module
    CUmodule *drvmod = gmGetModule("misc");

    // load float GPU function
    CUresult status = cuModuleGetFunction(&drvfunf, *drvmod, "THREEWAYPROD");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }

    /*    // load complex GPU function
    status = cuModuleGetFunction(&drvfunc, *drvmod, "TIMESC");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }

    // load double GPU function
    status = cuModuleGetFunction(&drvfund, *drvmod, "TIMESD");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }

    // load complex GPU function
    status = cuModuleGetFunction(&drvfuncd, *drvmod, "TIMESCD");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    */
    init = 1;
  }

  // mex parameters are:

  // 1. IN1
  // 2. IN2
  // 3. OUT

  //A is the input GPU array
  GPUtype A = gm->gputype.getGPUtype(prhs[0]);

  //B is the input GPU array
  GPUtype B = gm->gputype.getGPUtype(prhs[1]);

  //B is the input GPU array
  GPUtype C = gm->gputype.getGPUtype(prhs[2]);

  //OUT is the output GPU array (result)
  GPUtype OUT = gm->gputype.getGPUtype(prhs[3]);

  /*  // number of elements
  int ninA = gm->gputype.getNumel(A);
  int ninB = gm->gputype.getNumel(B);
  int ninC = gm->gputype.getNumel(C);
  int nout = gm->gputype.getNumel(OUT);*/

  //dimensions
  const int * sA = gm->gputype.getSize(A);
  const int * sB = gm->gputype.getSize(B);
  const int * sC = gm->gputype.getSize(C);
  const int * sout = gm->gputype.getSize(OUT);

  int numA = sA[0];
  int numB = sB[0];
  int numC = sC[0];
  //mexPrintf("numdims A: %d B: %d C: %d\n",numA,numB,numC);

  int ncA = sA[1];
  int ncB = sB[1];
  int ncC = sC[1];

  int noutA = sout[0];
  int noutB = sout[1];
  int noutC = sout[2];
  //mexPrintf("dim of target %d x %d x %d\n",noutA,noutB,noutC);
  
  gpuTYPE_t tA = gm->gputype.getType(A);
  gpuTYPE_t tB = gm->gputype.getType(B);
  gpuTYPE_t tC = gm->gputype.getType(C);
  gpuTYPE_t tout = gm->gputype.getType(OUT);

  // check input/out size and type
  if (ncA!=ncB)
    mexErrMsgTxt("Number of cases must be the same for all inputs.");

  if (ncB!=ncC)
    mexErrMsgTxt("Number of cases must be the same for all inputs.");
  
  if (noutA!=numA || noutB!=numB || noutC!=numC)
    mexErrMsgTxt("Dimensions of output must match the first dimension of each input.");

  if (tA!=tB)
    mexErrMsgTxt("Input arguments must be of the same type.");

  if (tB!=tC)
    mexErrMsgTxt("Input arguments must be of the same type.");

  if (tA!=tout)
    mexErrMsgTxt("Input and output arguments must be of the same type.");

  // I need the pointers to GPU memory
  CUdeviceptr d_A  = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(A));
  CUdeviceptr d_B  = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(B));
  CUdeviceptr d_C  = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(C));
  CUdeviceptr d_OUT = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(OUT));
  
  // The GPU kernel depends on the type of input/output
  CUfunction drvfun;
  if (tA == gpuFLOAT) {
    drvfun = drvfunf;
  } else 
    mexErrMsgTxt("Doubles and complex types not supported yet.");
    /*else if (tin1 == gpuCFLOAT) {
    drvfun = drvfunc;
  } else if (tin1 == gpuDOUBLE) {
    drvfun = drvfund;
  } else if (tin1 == gpuCDOUBLE) {
    drvfun = drvfuncd;
    } */
 
  hostdrv_pars_t gpuprhs[4];
  int gpunrhs = 4;
  gpuprhs[0] = hostdrv_pars(&d_A,sizeof(d_A),__alignof(d_A));
  gpuprhs[1] = hostdrv_pars(&d_B,sizeof(d_B),__alignof(d_B));
  gpuprhs[2] = hostdrv_pars(&d_C,sizeof(d_C),__alignof(d_C));
  gpuprhs[3] = hostdrv_pars(&d_OUT,sizeof(d_OUT),__alignof(d_OUT));

  //hostGPUDRV(drvfun, N, gpunrhs, gpuprhs);
  hostGPUmisc(gpunrhs, gpuprhs, numA, numB, numC, ncA);

}
