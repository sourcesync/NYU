/*
  Mimics Alex's eltWiseDivideByVector2 function in nvmatrix.cu
  Divide a matrix (elementwise) by a row or column vector


  INPUT1: in (mxn) data matrix 
  INPUT2: vec (mx1) or (1xn) vector
  vector dimensions must match input matrix based on row OR column                
  OUTPUT: out (mxn) output matrix
  cuEltWiseDivideByVector2(in,vec,out)

 */
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
#include "GPUkernel.hh"

#include "misc.cuh"

// static paramaters
static CUfunction drvfunf; // float
static CUfunction drvfunf2; // float

//static CUfunction drvfunc; // complex
//static CUfunction drvfund; // double
//static CUfunction drvfuncd;//double complex

static int init = 0;

static GPUmat *gm;

/* Driver */
void hostDivideOp(CUfunction drvfun, int nrhs, hostdrv_pars_t *prhs, int width, int height) {

  CUresult err = CUDA_SUCCESS;

  // setup execution parameters
  // block dimensions
  if (CUDA_SUCCESS != (err = cuFuncSetBlockShape(drvfun, NUM_VECTOR_OP_THREADS_PER_BLOCK, 1, 1))) {
    mexErrMsgTxt("Error in cuFuncSetBlockShape");
  }

  // if (CUDA_SUCCESS != cuFuncSetSharedSize(drvfun, m*sizeof(float))) {
  //   mexErrMsgTxt("Error in cuFuncSetSharedSize");
  // }

  // add parameters
  int poffset = 0;

  //input,vec,target
  for (int p=0;p<nrhs;p++) {
    if (CUDA_SUCCESS
	!= cuParamSetv(drvfun, poffset, prhs[p].par, prhs[p].psize)) {
      mexErrMsgTxt("Error in cuParamSetv");
    }
    poffset += prhs[p].psize;
  }

  //width
  if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, width)) {
    mexErrMsgTxt("Error in cuParamSeti");
  }
  poffset += sizeof(width);

  //height
  if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, height)) {
    mexErrMsgTxt("Error in cuParamSeti");
  }
  poffset += sizeof(height);


  if (CUDA_SUCCESS != cuParamSetSize(drvfun, poffset)) {
    mexErrMsgTxt("Error in cuParamSetSize");
  }

  //grid dimensions
  err = cuLaunchGridAsync(drvfun, NUM_VECTOR_OP_BLOCKS, 1, 0);
  if (CUDA_SUCCESS != err) {
    mexErrMsgTxt("Error running kernel");
  }
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  CUresult cudastatus = CUDA_SUCCESS;

  if (nrhs != 3) 
    mexErrMsgTxt("Wrong number of arguments");

  if (init == 0) {
    // Initialize function
    //mexLock();

    // load GPUmat
    gm = gmGetGPUmat();

    // load module
    CUmodule *drvmod = gmGetModule("misc");

    // load float GPU functions
    CUresult status;
    status = cuModuleGetFunction(&drvfunf, *drvmod, "kDivideByColVector");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }

    // load float GPU functions
    status = cuModuleGetFunction(&drvfunf2, *drvmod, "kDivideByRowVector");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }

    init = 1;
  }

  // mex parameters are:

  // 1. IN1 (in)
  // 2. IN2 (vec)
  // 3. OUT (out)

  //IN1 is an input GPU array
  GPUtype IN1 = gm->gputype.getGPUtype(prhs[0]);
  gpuTYPE_t tIn1 = gm->gputype.getType(IN1);

  //IN2 is an input GPU array
  GPUtype IN2 = gm->gputype.getGPUtype(prhs[1]);
  gpuTYPE_t tIn2 = gm->gputype.getType(IN2);

  //OUT is the output GPU array
  GPUtype OUT = gm->gputype.getGPUtype(prhs[2]);
  gpuTYPE_t tOut = gm->gputype.getType(OUT);

  //dimensions
  const int * sIn1 = gm->gputype.getSize(IN1);
  const int * sIn2 = gm->gputype.getSize(IN2);
  const int * sOut = gm->gputype.getSize(OUT);


  if (&prhs[1] == &prhs[2])
    mexErrMsgTxt("vec and target cannot be the same");

  if (sIn2[0] !=1 && sIn2[1] !=1)
    mexErrMsgTxt("second argument is not a vector");

  if (sIn2[0] != sIn1[0] && sIn2[1] != sIn1[1])
    mexErrMsgTxt("vec must match input in EITHER rows OR cols");

  if (sOut[0] != sIn1[0] || sOut[1] != sIn1[1])
    mexErrMsgTxt("target dims must match input dims");


  /* 
     Modified Alex's kernels to support col-major data
     So these are now defined properly
  */
  const unsigned int height = sIn1[0];
  const unsigned int width = sIn1[1];

  // Output for debugging
  //mexPrintf("input rows: %d cols: %d\n",sIn1[0],sIn1[1]);
  //mexPrintf("vec rows: %d cols: %d\n",sIn2[0],sIn2[1]);
  //mexPrintf("output: %d x %d\n",sOut[0],sOut[1]);
  //mexPrintf("wxh: %d x %d\n",width,height);

  CUfunction drvfun;
  CUfunction drvfun2;
  if ((tIn1 == gpuFLOAT) && (tIn2 == gpuFLOAT)) {
    drvfun = drvfunf;
    drvfun2 = drvfunf2;
  }
  else {
    mexErrMsgTxt("Only singles are supported at present.");
  }

  // I need the pointers to GPU memory
  CUdeviceptr d_IN1  = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(IN1));
  CUdeviceptr d_IN2  = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(IN2));
  CUdeviceptr d_OUT = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(OUT));
  
  hostdrv_pars_t gpuprhs[3];
  int gpunrhs = 3;
  gpuprhs[0] = hostdrv_pars(&d_IN1,sizeof(d_IN1));
  gpuprhs[1] = hostdrv_pars(&d_IN2,sizeof(d_IN2));
  gpuprhs[2] = hostdrv_pars(&d_OUT,sizeof(d_OUT));

  if (sIn2[0] == sIn1[0]) {
    //row match
    //mexPrintf("Trying col vector divide\n");
    hostDivideOp(drvfun, gpunrhs, gpuprhs, width, height); 
  }
  else {
    //col match
    //mexPrintf("Trying row vector divide\n");
    hostDivideOp(drvfun2, gpunrhs, gpuprhs, width, height); 
  }

  //hostGPUDRV(drvfun, N, gpunrhs, gpuprhs);
  //hostGPUPdist(drvfun, gpunrhs, gpuprhs, n1, m1); 
  
}
