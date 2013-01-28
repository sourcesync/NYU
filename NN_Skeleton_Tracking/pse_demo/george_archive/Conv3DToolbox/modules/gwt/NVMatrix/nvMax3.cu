/*

  wrapper to Alex's NVMatrix::max
  similar to nvMax but also returns argmax

  nvMax2(in,axis,out,argmax)

  in is a mxn matrix
  out is a mx1 or 1xn vector
  argmax is a mx1 or 1xn vector
  axes is 1,2 depending on which dimension the max is taken over

  Note that I am not compiling through "make"
  I have to change the name of all C code with CUDA to ".cu" (instead of cpp)
  I am manually compiling using nvmex which is a wrapper for mex that can handle .cu


MATLAB=/opt/pkg/matlab/current ./nvmex -f nvopts.sh  -DUNIX -outdir . nvMax2.cu nvmatrix.cu nvmatrix_kernel.cu ../../common/GPUmat.cpp  -I"/usr/local/pkg/cuda/3.1/cuda/include" -I"../../include"  -L"/usr/local/pkg/cuda/3.1/cuda/lib64" -lcuda -lcudart -lcufft -lcublas

Note that I am passing in both this file and the .cu code for the kernel (nvmatrix_kernel.cu) -- otherwise I will get undefined symbols

This document was useful:
http://faculty.washington.edu/dushaw/epubs/Matlab_CUDA_Tutorial_2_10.pdf

nvmex, nvopts.sh came from here: 
http://developer.nvidia.com/object/matlab_cuda.html
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

//#include "misc.cuh"
#include "nvmatrix.cuh"

//#include "nvmatrix_kernel.cuh"

// static paramaters
//static CUfunction drvfunf; // float
//static CUfunction drvfunf2; // float

//static CUfunction drvfunc; // complex
//static CUfunction drvfund; // double
//static CUfunction drvfuncd;//double complex


static int init = 0;

static GPUmat *gm;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  //CUresult cudastatus = CUDA_SUCCESS;

  if (nrhs != 4) 
    mexErrMsgTxt("Wrong number of arguments");

  if (init == 0) {
    // Initialize function
    //mexLock();

    // load GPUmat
    gm = gmGetGPUmat();

    init = 1;
  }

  // mex parameters are:

  // 1. IN1 (in)
  // 2. IN2 (axis) - {1,2}
  // 2. OUT (out)

  //IN1 is an input GPU array
  GPUtype IN1 = gm->gputype.getGPUtype(prhs[0]);
  gpuTYPE_t tIn1 = gm->gputype.getType(IN1);

  //IN2 is an axis {1,2}
  //Note it is converted to C-style indexing {0,1}
  int axis = (int) mxGetScalar(prhs[1]) - 1;

  //OUT1 is the output GPU array (max)
  GPUtype OUT1 = gm->gputype.getGPUtype(prhs[2]);
  gpuTYPE_t tOut1 = gm->gputype.getType(OUT1);

  //OUT2 is the output GPU array (argmax)
  GPUtype OUT2 = gm->gputype.getGPUtype(prhs[3]);
  gpuTYPE_t tOut2 = gm->gputype.getType(OUT2);

  //dimensions
  const int * sIn1 = gm->gputype.getSize(IN1);
  const int * sOut1 = gm->gputype.getSize(OUT1);
  const int * sOut2 = gm->gputype.getSize(OUT2);

  // if (&prhs[1] == &prhs[2])
  //   mexErrMsgTxt("vec and target cannot be the same");

  // if (sIn2[0] !=1 && sIn2[1] !=1)
  //   mexErrMsgTxt("second argument is not a vector");

  // if (sIn2[0] != sIn1[0] && sIn2[1] != sIn1[1])
  //   mexErrMsgTxt("vec must match input in EITHER rows OR cols");

  // if (sOut[0] != sIn1[0] || sOut[1] != sIn1[1])
  //   mexErrMsgTxt("target dims must match input dims");

  if (sOut1[0] != sOut2[0] || sOut1[1] != sOut2[1])
    mexErrMsgTxt("output (max) and argmax dimensions must be the same");

  // // Output for debugging
  // //mexPrintf("input rows: %d cols: %d\n",sIn1[0],sIn1[1]);
  // //mexPrintf("vec rows: %d cols: %d\n",sIn2[0],sIn2[1]);
  // //mexPrintf("output: %d x %d\n",sOut[0],sOut[1]);
  // //mexPrintf("wxh: %d x %d\n",width,height);

  // CUfunction drvfun;
  // CUfunction drvfun2;
  // if ((tIn1 == gpuFLOAT) && (tIn2 == gpuFLOAT)) {
  //   drvfun = drvfunf;
  //   drvfun2 = drvfunf2;
  // }
  // else {
  //   mexErrMsgTxt("Only singles are supported at present.");
  // }

  // I need the pointers to GPU memory
  CUdeviceptr d_IN1  = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(IN1));
  // CUdeviceptr d_IN2  = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(IN2));
  CUdeviceptr d_OUT1 = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(OUT1));
  CUdeviceptr d_OUT2 = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(OUT2));

  /* Create NVMatrix by initializing to device data in (in)
     I think the right thing to do is pass in true for the isTrans parameter
     Since GPUsingle data is in col-major rather than row-major order */
  NVMatrix nvIn((float*) d_IN1,sIn1[0],sIn1[1],true);
  NVMatrix nvOut1((float*) d_OUT1,sOut1[0],sOut1[1],true);
  NVMatrix nvOut2((float*) d_OUT2,sOut2[0],sOut2[1],true);

  nvIn.max3(axis,nvOut1,nvOut2);
  
}
