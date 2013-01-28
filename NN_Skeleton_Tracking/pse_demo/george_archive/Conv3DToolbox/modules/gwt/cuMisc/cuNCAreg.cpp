/*
  Gradients for NCA regression
  Computes:
    for kk=1:n
      for jj=1:n
        out(kk,:) = out(kk,:) + (in(:,kk)-in(:,jj)) * ...
            w(jj,kk);
      end
    end
  Note that out has cases first, dims last
  So that writes are coalesced
  INPUT1: in (mxn) data matrix where m are the dim and n is the number of cases
  INPUT2: w  (nxn) weight matrix: columns correspond to n in the output
                rows correspond to the cases that are summed over
  OUTPUT: out the (nxm) gradient matrix
  cuNCAreg(in,w,out)

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

    // load float GPU function
    CUresult status = cuModuleGetFunction(&drvfunf, *drvmod, "gpuNCAreg");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }

    init = 1;
  }

  // mex parameters are:

  // 1. IN1 (in)
  // 2. IN2 (w)
  // 3. OUT (out)

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

  int m1 = sIn1[0]; //# of dimensions
  int n1 = sIn1[1]; //# of cases

  int n2a = sIn2[0]; //# of cases
  int n2b = sIn2[1]; //# of cases

  int sout1 = sOut[0]; //# of cases
  int sout2 = sOut[1]; //# of dimensions

  if ( (n1!=n2a) || (n2a!=n2b) || (n1!=sout1) )
    mexErrMsgTxt("Number of cases must be consistent");

  if (m1!=sout2)
    mexErrMsgTxt("Number of dimensions must be consistent");

  /* Output for debugging
     mexPrintf("numcases: %d numdims: %d\n",n,m);
     mexPrintf("output: %d x %d\n",sout1,sout2);
  */

  CUfunction drvfun;
  if ((tIn1 == gpuFLOAT) && (tIn2 == gpuFLOAT))
    drvfun = drvfunf;
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
  hostGPUPdist(drvfun, gpunrhs, gpuprhs, n1, m1); 
  
}
