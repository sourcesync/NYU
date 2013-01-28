/*

  INPUT: either one or two mxn matrices where m are the dim and n is the number of cases
  OUTPUT: D the (nxn) or (n2xn1) SQUARED distance matrix
  (plus whatever was in D before)
  cuDist(A,D) 
  cuDist(A,B,D)

  Two differences between cuSquaredDist and cuDist
  1) We do not square the distances
  2) The output, D, is not overwritten but is added to
  These changes are useful in a high-dimensional setting, where we can call cuSquaredDist many times for different blocks of dimensions

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

// static paramaters
static CUfunction drvfunf; // float
static CUfunction drvfunf2; // float

//static CUfunction drvfunc; // complex
//static CUfunction drvfund; // double
//static CUfunction drvfuncd;//double complex

static int init = 0;

static GPUmat *gm;

/* Driver */
void hostGPUPdist(CUfunction drvfun, int nrhs, hostdrv_pars_t *prhs, int n, int m) {

  /* Each thread block computes a linear block of the target */
  int gridx = (n + BLOCK_DIM1D - 1) / BLOCK_DIM1D; //BLOCK_DIM1D set in GPUkernel.hh
  
  CUresult err = CUDA_SUCCESS;

    // setup execution parameters

    if (CUDA_SUCCESS != (err = cuFuncSetBlockShape(drvfun, BLOCK_DIM1D, 1, 1))) {
      mexErrMsgTxt("Error in cuFuncSetBlockShape");
    }

    if (CUDA_SUCCESS != cuFuncSetSharedSize(drvfun, m*sizeof(float))) {
      mexErrMsgTxt("Error in cuFuncSetSharedSize");
      }

    // add parameters
    int poffset = 0;

    // CUDA kernels interface
    // N: number of elements
    // offset: used for streams

    for (int p=0;p<nrhs;p++) {
      if (CUDA_SUCCESS
          != cuParamSetv(drvfun, poffset, prhs[p].par, prhs[p].psize)) {
        mexErrMsgTxt("Error in cuParamSetv");
      }
      poffset += prhs[p].psize;
    }

    if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, n)) {
      mexErrMsgTxt("Error in cuParamSeti");
    }
    poffset += sizeof(n);

    if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, m)) {
      mexErrMsgTxt("Error in cuParamSeti");
    }
    poffset += sizeof(m);


    if (CUDA_SUCCESS != cuParamSetSize(drvfun, poffset)) {
      mexErrMsgTxt("Error in cuParamSetSize");
    }

    err = cuLaunchGridAsync(drvfun, gridx, 1, 0);
    if (CUDA_SUCCESS != err) {
      mexErrMsgTxt("Error running kernel");
    }
}

void hostGPUPdist2(CUfunction drvfun, int nrhs, hostdrv_pars_t *prhs, int n1, int n2, int m) {

  /* Each thread block computes a linear block of the target */
  int gridx = (n2 + BLOCK_DIM1D - 1) / BLOCK_DIM1D; //BLOCK_DIM1D set in GPUkernel.hh
  
  CUresult err = CUDA_SUCCESS;

    // setup execution parameters

    if (CUDA_SUCCESS != (err = cuFuncSetBlockShape(drvfun, BLOCK_DIM1D, 1, 1))) {
      mexErrMsgTxt("Error in cuFuncSetBlockShape");
    }

    if (CUDA_SUCCESS != cuFuncSetSharedSize(drvfun, m*sizeof(float))) {
      mexErrMsgTxt("Error in cuFuncSetSharedSize");
      }

    // add parameters
    int poffset = 0;

    // CUDA kernels interface
    // N: number of elements
    // offset: used for streams

    for (int p=0;p<nrhs;p++) {
      if (CUDA_SUCCESS
          != cuParamSetv(drvfun, poffset, prhs[p].par, prhs[p].psize)) {
        mexErrMsgTxt("Error in cuParamSetv");
      }
      poffset += prhs[p].psize;
    }

    if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, n1)) {
      mexErrMsgTxt("Error in cuParamSeti");
    }
    poffset += sizeof(n1);

    if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, n2)) {
      mexErrMsgTxt("Error in cuParamSeti");
    }
    poffset += sizeof(n2);

    if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, m)) {
      mexErrMsgTxt("Error in cuParamSeti");
    }
    poffset += sizeof(m);


    if (CUDA_SUCCESS != cuParamSetSize(drvfun, poffset)) {
      mexErrMsgTxt("Error in cuParamSetSize");
    }

    err = cuLaunchGridAsync(drvfun, gridx, 1, 0);
    if (CUDA_SUCCESS != err) {
      mexErrMsgTxt("Error running kernel");
    }
}



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  CUresult cudastatus = CUDA_SUCCESS;

  if ((nrhs < 2) || (nrhs > 3)) 
    mexErrMsgTxt("Wrong number of arguments");

  if (init == 0) {
    // Initialize function
    //mexLock();

    // load GPUmat
    gm = gmGetGPUmat();

    // load module
    CUmodule *drvmod = gmGetModule("misc");

    // load float GPU function
    // One input
    CUresult status = cuModuleGetFunction(&drvfunf, *drvmod, "gpusqPdist");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    // Two inputs
    status = cuModuleGetFunction(&drvfunf2, *drvmod, "gpusqPdist2");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }

    init = 1;
  }

  // mex parameters are:

  // 1. IN1
  // 2. IN2 (optional)
  // 3. OUT


  if (nrhs<3) { //IN1,OUT

    //IN is the input GPU array
    GPUtype IN = gm->gputype.getGPUtype(prhs[0]);
    gpuTYPE_t tIn = gm->gputype.getType(IN);

    //IN is the input GPU array
    GPUtype OUT = gm->gputype.getGPUtype(prhs[1]);
    gpuTYPE_t tOut = gm->gputype.getType(OUT);

    //dimensions
    const int * sIn = gm->gputype.getSize(IN);
    const int * sOut = gm->gputype.getSize(OUT);

    int m = sIn[0];
    int n = sIn[1];

    int sout1 = sOut[0];
    int sout2 = sOut[1];

    /* Output for debugging
    mexPrintf("numcases: %d numdims: %d\n",n,m);
    mexPrintf("output: %d x %d\n",sout1,sout2);
    */

    CUfunction drvfun;
    if (tIn == gpuFLOAT) 
      drvfun = drvfunf;
    else {
      mexErrMsgTxt("Only singles are supported at present.");
    }

    /*

    // create a GPUtype just like the input
    // nrhs-1 because last argument is a GPUtype
    // r is the returned output
    // tin - same as input
    // 2 dimensions
    // size of each dimension
    const mxArray *d;
    d = mxCreateDoubleMatrix(1,2,mxREAL);
    double* dd=mxGetPr(d);

    *(dd)=n;
    *(dd+1)=n;
    GPUtype r = gm->gputype.createMx(tIn, 2, &d);

    // number of elements
    int N = gm->gputype.getNumel(r);

    plhs[0] = gm->gputype.createMxArray(r);
    */
    // I need the pointers to GPU memory
    CUdeviceptr d_IN  = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(IN));
    CUdeviceptr d_OUT = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(OUT));
  
    hostdrv_pars_t gpuprhs[2];
    int gpunrhs = 2;
    gpuprhs[0] = hostdrv_pars(&d_IN,sizeof(d_IN));
    gpuprhs[1] = hostdrv_pars(&d_OUT,sizeof(d_OUT));

    //hostGPUDRV(drvfun, N, gpunrhs, gpuprhs);
    hostGPUPdist(drvfun, gpunrhs, gpuprhs, n, m); 
  } else { //IN1,IN2,OUT
    
    //IN is the input GPU array
    GPUtype IN1 = gm->gputype.getGPUtype(prhs[0]);
    gpuTYPE_t tIn1 = gm->gputype.getType(IN1);

    //IN is the input GPU array
    GPUtype IN2 = gm->gputype.getGPUtype(prhs[1]);
    gpuTYPE_t tIn2 = gm->gputype.getType(IN2);

    //IN is the input GPU array
    GPUtype OUT = gm->gputype.getGPUtype(prhs[2]);
    gpuTYPE_t tOut = gm->gputype.getType(OUT);

    //dimensions
    const int * sIn1 = gm->gputype.getSize(IN1);
    const int * sIn2 = gm->gputype.getSize(IN2);
    const int * sOut = gm->gputype.getSize(OUT);

    int m1 = sIn1[0];
    int n1 = sIn1[1];

    int m2 = sIn2[0];
    int n2 = sIn2[1];

    if (m1!=m2)
      mexErrMsgTxt("row dimensions must match");
    int m = m1;

    int sout2 = sOut[0];
    int sout1 = sOut[1];

    /* Output for debugging

    mexPrintf("numcases 1: %d numdims: %d\n",n1,m);
    mexPrintf("numcases 2: %d numdims: %d\n",n2,m);

    mexPrintf("output: %d x %d\n",sout1,sout2);
    */

    if (sout1!=n1)
      mexErrMsgTxt("output dim 1 must match input 1 dim 2");
    if (sout2!=n2)
      mexErrMsgTxt("output dim 2 must match input 2 dim 2");

    CUfunction drvfun;
    if ((tIn1 == gpuFLOAT) && (tIn2 == gpuFLOAT))
      drvfun = drvfunf2;
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

    //hostGPUDRV(drvfun, N, gpunrhs, gpuprhs);
    hostGPUPdist2(drvfun, gpunrhs, gpuprhs, n1, n2, m); 
    
  }
  
}
