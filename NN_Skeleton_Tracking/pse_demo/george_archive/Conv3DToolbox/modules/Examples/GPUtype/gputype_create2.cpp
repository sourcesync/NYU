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



// static parameters
static int init = 0;
static GPUmat *gm;

/*
 * This function creates a GPUtype from a Matlab array, using GPUmat
 * function mxToGPUtype. Return the GPUtype to Matlab as a GPUmat object (GPUsingle, GPUdouble)
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {



  // At least 1 argument expected
  if (nrhs!=1)
     mexErrMsgTxt("Wrong number of arguments.");

  if (init == 0) {
    // Initialize function

    // load GPUmat
    gm = gmGetGPUmat();

    // load module
    // NOT REQUIRED

    // load float GPU function
    // NOT REQUIRED

    init = 1;
  }



  // mex parameters are:
  // IN: Matlab array

  GPUtype IN = gm->gputype.mxToGPUtype(prhs[0]);
  plhs[0] = gm->gputype.createMxArray(IN);

}
