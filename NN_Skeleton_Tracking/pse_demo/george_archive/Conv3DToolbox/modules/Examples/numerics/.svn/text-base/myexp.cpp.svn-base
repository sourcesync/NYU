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



// static paramaters

static CUfunction drvfuns[4];
static int init = 0;
static GPUmat *gm;

/*
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  CUresult cudastatus = CUDA_SUCCESS;

  // At least 2 arguments expected
  // Input and result
  if (nrhs!=2)
     mexErrMsgTxt("Wrong number of arguments");

  if (init == 0) {
    // Initialize function
    //mexLock();

    // load GPUmat
    gm = gmGetGPUmat();

    // load module
    // NOT REQUIRED

    // load float GPU function
    // NOT REQUIRED

    init = 1;
  }



  // mex parameters are:
  // IN
  // OUT

  GPUtype IN  = gm->gputype.getGPUtype(prhs[0]);
  GPUtype OUT = gm->gputype.getGPUtype(prhs[1]);


  gm->numerics.Exp(IN,OUT);

}
