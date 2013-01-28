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
#include "misc.cuh"

// static paramaters
static CUfunction drvfunf; // float
/*static CUfunction drvfunc; // complex
static CUfunction drvfund; // double
static CUfunction drvfuncd;//double complex
*/

static int init = 0;

static GPUmat *gm;

//host driver
void hostDriver(CUfunction drvfun, int filterSize, int numFilters, int nrhs, hostdrv_pars_t *prhs) {

  dim3 threads(16, 16, 1);
  dim3 blocks(numFilters, 1, 1);

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

  if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, filterSize)) {
    mexErrMsgTxt("Error in cuParamSeti");
  }
  poffset += sizeof(filterSize);

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
    mexErrMsgTxt("Wrong number of arguments");

  if (init == 0) {
    // Initialize function
    //mexLock();

    // load GPUmat
    gm = gmGetGPUmat();

    // load module
    CUmodule *drvmod = gmGetModule("misc");


    // load float GPU function
    CUresult status = cuModuleGetFunction(&drvfunf, *drvmod, "kRotate180");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }

    init = 1;
  }

  // mex parameters are:

  // 1. IN1
  // 2. OUT

  bool color = false; //hard-coded

  //IN1 is the input GPU array
  GPUtype IN1 = gm->gputype.getGPUtype(prhs[0]);

  //OUT is the output GPU array (result)
  GPUtype OUT = gm->gputype.getGPUtype(prhs[1]);

  // number of elements
  int nin1 = gm->gputype.getNumel(IN1);
  int nout = gm->gputype.getNumel(OUT);

  //dimensions 
  const int * sin1 = gm->gputype.getSize(IN1);
  const int * sout = gm->gputype.getSize(OUT);

  if ( color && sin1[0] %3 != 0)
    mexErrMsgTxt("Color image specified but rows not divisible by 3");
  if ( !color && floor(sqrt(float(sin1[0]))) != sqrt(float(sin1[0])) )
    mexErrMsgTxt("Images not square");
  if ( color && floor(sqrt(float(sin1[0] / 3))) != sqrt(float(sin1[0] / 3)) )
    mexErrMsgTxt("Images not square");
  if ( nin1 != nout )
    mexErrMsgTxt("Filters and targets must have the same number of elements");

  int filterSize = color ? int(sqrt(sin1[0] / 3)) : int(sqrt(sin1[0]));
  int numFilters =  (color ? 3 : 1) * sin1[1]; //image per column

  gpuTYPE_t tin1 = gm->gputype.getType(IN1);
  gpuTYPE_t tout = gm->gputype.getType(OUT);

  // check input/out size and type
  if (tin1!=tout)
    mexErrMsgTxt("Input and output arguments must be of the same type.");

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
  gpuprhs[0] = hostdrv_pars(&d_IN1,sizeof(d_IN1));
  gpuprhs[1] = hostdrv_pars(&d_OUT,sizeof(d_OUT));

  //hostDriver(drvfun, N, gpunrhs, gpuprhs);
  hostDriver(drvfun, filterSize, numFilters, gpunrhs, gpuprhs);

}
