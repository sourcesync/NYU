/*
 * Samples from a bunch of multinomial distributions, where each COLUMN of the "multi" matrix
 * is a different distribution. Of course, each COLUMN of the "multi" matrix must sum to 1.
 *
 * It's optimized for the case when you want to sample from lots (hundreds of thousands)
 * of fairly small multinomial distributions.
 *
 * The case when the multinomials are in ROWS is much easier and faster.
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

#include "math.h"
#include "misc.cuh"

// static paramaters
static CUfunction _sampleMultinomial_32;
static CUfunction _sampleMultinomial_64;
static CUfunction _sampleMultinomial_128;
static CUfunction _sampleMultinomial_256;
static CUfunction _sampleMultinomial_512;

static CUfunction _sampleSmallMultinomial_1_4;
static CUfunction _sampleSmallMultinomial_1_8;
static CUfunction _sampleSmallMultinomial_1_12;
static CUfunction _sampleSmallMultinomial_1_16;

static CUfunction _sampleSmallMultinomial_2_SSM_THREADS_X;
static CUfunction _sampleSmallMultinomial_3_SSM_THREADS_X;
static CUfunction _sampleSmallMultinomial_4_SSM_THREADS_X;
static CUfunction _sampleSmallMultinomial_5_SSM_THREADS_X;
static CUfunction _sampleSmallMultinomial_6_SSM_THREADS_X;
static CUfunction _sampleSmallMultinomial_7_SSM_THREADS_X;
static CUfunction _sampleSmallMultinomial_8_SSM_THREADS_X;
static CUfunction _sampleSmallMultinomial_9_SSM_THREADS_X;
static CUfunction _sampleSmallMultinomial_10_SSM_THREADS_X;
static CUfunction _sampleSmallMultinomial_11_SSM_THREADS_X;
static CUfunction _sampleSmallMultinomial_12_SSM_THREADS_X;
static CUfunction _sampleSmallMultinomial_13_SSM_THREADS_X;
static CUfunction _sampleSmallMultinomial_14_SSM_THREADS_X;
static CUfunction _sampleSmallMultinomial_15_SSM_THREADS_X;
static CUfunction _sampleSmallMultinomial_16_SSM_THREADS_X;

static int init = 0;

static GPUmat *gm;

//host driver
void hostDriver(CUfunction drvfun, dim3 grid, dim3 threads, int multiSize, int numMulti, int nrhs, hostdrv_pars_t *prhs) {
  //mexPrintf("threads.x: %d threads.y: %d threads.z %d\n",threads.x,threads.y,threads.z);

  //mexPrintf("multinomials: %d nomials: %d\n",numMulti,multiSize);

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

  if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, multiSize)) {
    mexErrMsgTxt("Error in cuParamSeti");
  }
  poffset += sizeof(multiSize);

  if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, numMulti)) {
    mexErrMsgTxt("Error in cuParamSeti");
  }
  poffset += sizeof(numMulti);

  if (CUDA_SUCCESS != cuParamSetSize(drvfun, poffset)) {
    mexErrMsgTxt("Error in cuParamSetSize");
  }

  err = cuLaunchGridAsync(drvfun, grid.x, grid.y, 0);
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
    CUmodule *drvmod = gmGetModule("subsample");

    //load appropriate GPU kernel (mangled name)
    CUresult status;

    status = cuModuleGetFunction(&_sampleMultinomial_32, *drvmod, "_Z18kSampleMultinomialILi32EEvPfS0_S0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function 1.");
    }
    status = cuModuleGetFunction(&_sampleMultinomial_64, *drvmod, "_Z18kSampleMultinomialILi64EEvPfS0_S0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function 2.");
    }
    status = cuModuleGetFunction(&_sampleMultinomial_128, *drvmod, "_Z18kSampleMultinomialILi128EEvPfS0_S0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function 3.");
    }
    status = cuModuleGetFunction(&_sampleMultinomial_256, *drvmod, "_Z18kSampleMultinomialILi256EEvPfS0_S0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function 4.");
    }
    status = cuModuleGetFunction(&_sampleMultinomial_512, *drvmod, "_Z18kSampleMultinomialILi512EEvPfS0_S0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function 5.");
    }
    status = cuModuleGetFunction(&_sampleSmallMultinomial_1_4, *drvmod, "_Z23kSampleSmallMultinomialILi1ELi4EEvPfS0_S0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function i.");
    }
    status = cuModuleGetFunction(&_sampleSmallMultinomial_1_8, *drvmod, "_Z23kSampleSmallMultinomialILi1ELi8EEvPfS0_S0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function ii.");
    }
    status = cuModuleGetFunction(&_sampleSmallMultinomial_1_12, *drvmod, "_Z23kSampleSmallMultinomialILi1ELi12EEvPfS0_S0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function iii.");
    }
    status = cuModuleGetFunction(&_sampleSmallMultinomial_1_16, *drvmod, "_Z23kSampleSmallMultinomialILi1ELi16EEvPfS0_S0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function iv.");
    }
    /*Note that mangled name is based on define SSM_THREADS_X
      So we build the mangled name dynamically for these using sprintf*/
    char buffer[50];

    sprintf(buffer,"_Z23kSampleSmallMultinomialILi2ELi%dEEvPfS0_S0_ii",SSM_THREADS_X);
    status = cuModuleGetFunction(&_sampleSmallMultinomial_2_SSM_THREADS_X, *drvmod, buffer);
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function 2.");
    }
    sprintf(buffer,"_Z23kSampleSmallMultinomialILi3ELi%dEEvPfS0_S0_ii",SSM_THREADS_X);
    status = cuModuleGetFunction(&_sampleSmallMultinomial_3_SSM_THREADS_X, *drvmod, buffer);
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function 3.");
    }
    sprintf(buffer,"_Z23kSampleSmallMultinomialILi4ELi%dEEvPfS0_S0_ii",SSM_THREADS_X);
    status = cuModuleGetFunction(&_sampleSmallMultinomial_4_SSM_THREADS_X, *drvmod, buffer);
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function 4.");
    }
    sprintf(buffer,"_Z23kSampleSmallMultinomialILi5ELi%dEEvPfS0_S0_ii",SSM_THREADS_X);    
    status = cuModuleGetFunction(&_sampleSmallMultinomial_5_SSM_THREADS_X, *drvmod, buffer);
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function 5.");
    }
    sprintf(buffer,"_Z23kSampleSmallMultinomialILi6ELi%dEEvPfS0_S0_ii",SSM_THREADS_X);    
    status = cuModuleGetFunction(&_sampleSmallMultinomial_6_SSM_THREADS_X, *drvmod, buffer);
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function 6.");
    }
    sprintf(buffer,"_Z23kSampleSmallMultinomialILi7ELi%dEEvPfS0_S0_ii",SSM_THREADS_X);    
    status = cuModuleGetFunction(&_sampleSmallMultinomial_7_SSM_THREADS_X, *drvmod, buffer);
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function 7.");
    }
    sprintf(buffer,"_Z23kSampleSmallMultinomialILi8ELi%dEEvPfS0_S0_ii",SSM_THREADS_X);    
    status = cuModuleGetFunction(&_sampleSmallMultinomial_8_SSM_THREADS_X, *drvmod, buffer);
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function 8.");
    }
    sprintf(buffer,"_Z23kSampleSmallMultinomialILi9ELi%dEEvPfS0_S0_ii",SSM_THREADS_X);    
    status = cuModuleGetFunction(&_sampleSmallMultinomial_9_SSM_THREADS_X, *drvmod, buffer);
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function 9.");
    }
    sprintf(buffer,"_Z23kSampleSmallMultinomialILi10ELi%dEEvPfS0_S0_ii",SSM_THREADS_X);    
    status = cuModuleGetFunction(&_sampleSmallMultinomial_10_SSM_THREADS_X, *drvmod, buffer);
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function 10.");
    }
    sprintf(buffer,"_Z23kSampleSmallMultinomialILi11ELi%dEEvPfS0_S0_ii",SSM_THREADS_X);    
    status = cuModuleGetFunction(&_sampleSmallMultinomial_11_SSM_THREADS_X, *drvmod, buffer);
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function 11.");
    }
    sprintf(buffer,"_Z23kSampleSmallMultinomialILi12ELi%dEEvPfS0_S0_ii",SSM_THREADS_X);    
    status = cuModuleGetFunction(&_sampleSmallMultinomial_12_SSM_THREADS_X, *drvmod, buffer);
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function 12.");
    }
    sprintf(buffer,"_Z23kSampleSmallMultinomialILi13ELi%dEEvPfS0_S0_ii",SSM_THREADS_X);    
    status = cuModuleGetFunction(&_sampleSmallMultinomial_13_SSM_THREADS_X, *drvmod, buffer);
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function 13.");
    }
    sprintf(buffer,"_Z23kSampleSmallMultinomialILi14ELi%dEEvPfS0_S0_ii",SSM_THREADS_X);    
    status = cuModuleGetFunction(&_sampleSmallMultinomial_14_SSM_THREADS_X, *drvmod, buffer);
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function 14.");
    }
    sprintf(buffer,"_Z23kSampleSmallMultinomialILi15ELi%dEEvPfS0_S0_ii",SSM_THREADS_X);    
    status = cuModuleGetFunction(&_sampleSmallMultinomial_15_SSM_THREADS_X, *drvmod, buffer);
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function 15.");
    }
    sprintf(buffer,"_Z23kSampleSmallMultinomialILi16ELi%dEEvPfS0_S0_ii",SSM_THREADS_X);    
    status = cuModuleGetFunction(&_sampleSmallMultinomial_16_SSM_THREADS_X, *drvmod, buffer);
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function 16.");
    }

    init = 1;
  }

  // mex parameters are:

  // 1. multi
  // 2. randoms
  // 3. targets (OUT)

  //IN1 is the input GPU array (multi)
  GPUtype IN1 = gm->gputype.getGPUtype(prhs[0]);

  //IN2 is the input GPU array (randoms)
  GPUtype IN2 = gm->gputype.getGPUtype(prhs[1]);

  //OUT is the output GPU array (result)
  GPUtype OUT = gm->gputype.getGPUtype(prhs[2]);

  // number of elements
  int nin1 = gm->gputype.getNumel(IN1);
  int nin2 = gm->gputype.getNumel(IN2);
  int nout = gm->gputype.getNumel(OUT);

  //dimensions 
  const int * sin1 = gm->gputype.getSize(IN1);
  const int * sin2 = gm->gputype.getSize(IN2);
  const int * sout = gm->gputype.getSize(OUT);


  if (sin1[0] != sout[0] || sin1[1] != sout[1])
    mexErrMsgTxt("multi and targets must have same dimensions");
  if (sin1[0] > 1024)
    mexErrMsgTxt("K must < 1024");
  if (nin2 != sin1[1])
    mexErrMsgTxt("Number of randoms must = number of multinomials (cols of multi)");
  int nomials = sin1[0]; // K
  int multinomials = sin1[1];

  gpuTYPE_t tin1 = gm->gputype.getType(IN1);
  gpuTYPE_t tin2 = gm->gputype.getType(IN2);
  gpuTYPE_t tout = gm->gputype.getType(OUT);

  // check input/out size and type
  if (tin1!=tout || tin2!=tout)
    mexErrMsgTxt("Input and output arguments must be of the same type.");

  // I need the pointers to GPU memory
  CUdeviceptr d_IN1  = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(IN1));
  CUdeviceptr d_IN2  = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(IN2));
  CUdeviceptr d_OUT = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(OUT));

  // // The GPU kernel depends on the type of input/output
  // CUfunction drvfun;
  // if (tin1 == gpuFLOAT) {
  //   drvfun = drvfunf;
  // } else 
  //   mexErrMsgTxt("Currently only single types supported.");

  hostdrv_pars_t gpuprhs[3];
  int gpunrhs = 3;
  gpuprhs[0] = hostdrv_pars(&d_IN1,sizeof(d_IN1));
  gpuprhs[1] = hostdrv_pars(&d_IN2,sizeof(d_IN2));
  gpuprhs[2] = hostdrv_pars(&d_OUT,sizeof(d_OUT));

  //mexPrintf("multinomials: %d nomials: %d\n",multinomials,nomials);

  if(nomials > 256 || multinomials < 8192) {
    /*
     * I'm really not sure about the merits of this tree-based function. I may
     * remove it in the future. It's faster in some cases (e.g. when the number of
     * multinomials is small and the multinomials are very large), but you can get
     * similar performance from the non-tree-based one by reducing the number of
     * y-loops.
     */
    dim3 grid(1, DIVUP(multinomials, 1));
    while (grid.y > NUM_BLOCKS_MAX) {
      grid.y = DIVUP(grid.y, 2);
      grid.x *= 2;
    }
    //    printf("grid: %dx%d\n", grid.x, grid.y);
    if(nomials <= 64) { // yes i know this can't happen under current conditions
      dim3 threads(32, 1);
	hostDriver(_sampleMultinomial_32, grid, threads, nomials, multinomials, gpunrhs, gpuprhs);
      //kSampleMultinomial<32><<<grid, threads>>>(multi->getDevData(), randoms->getDevData(), targets->getDevData(), nomials, multinomials);
    } else if(nomials <= 128) {
      dim3 threads(64, 1);
      hostDriver(_sampleMultinomial_64, grid, threads, nomials, multinomials, gpunrhs, gpuprhs);
    } else if(nomials <= 256) {
      dim3 threads(128, 1);
      hostDriver(_sampleMultinomial_128, grid, threads, nomials, multinomials, gpunrhs, gpuprhs);
    } else if(nomials <= 512) {
      dim3 threads(256, 1);
      hostDriver(_sampleMultinomial_256, grid, threads, nomials, multinomials, gpunrhs, gpuprhs);
    } else {
      dim3 threads(512, 1);
      hostDriver(_sampleMultinomial_512, grid, threads, nomials, multinomials, gpunrhs, gpuprhs);
    }
  } else {
    dim3 grid(1,DIVUP(multinomials, SSM_THREADS_Y*SSM_LOOPS_Y));
    dim3 threads(SSM_THREADS_X, SSM_THREADS_Y);

    while (grid.y > NUM_BLOCKS_MAX) {
      grid.y = DIVUP(grid.y, 2);
      grid.x *= 2;
    }
    if(nomials <= 16) {
      if(nomials <= 4) {
	hostDriver(_sampleSmallMultinomial_1_4, grid, threads, nomials, multinomials, gpunrhs, gpuprhs);
	//kSampleSmallMultinomial<1, 4><<<grid, threads>>>(multi->getDevData(), randoms->getDevData(), targets->getDevData(),nomials, multinomials);
      } else if(nomials <= 8) {
	hostDriver(_sampleSmallMultinomial_1_8, grid, threads, nomials, multinomials, gpunrhs, gpuprhs);
      } else if(nomials <= 12) {
	hostDriver(_sampleSmallMultinomial_1_12, grid, threads, nomials, multinomials, gpunrhs, gpuprhs);
      } else {
	hostDriver(_sampleSmallMultinomial_1_16, grid, threads, nomials, multinomials, gpunrhs, gpuprhs);
      }
    } else if(nomials <= 32) {
      hostDriver(_sampleSmallMultinomial_2_SSM_THREADS_X, grid, threads, nomials, multinomials, gpunrhs, gpuprhs);
    } else if(nomials <= 48){
      hostDriver(_sampleSmallMultinomial_3_SSM_THREADS_X, grid, threads, nomials, multinomials, gpunrhs, gpuprhs);
    } else if(nomials <= 64){
      hostDriver(_sampleSmallMultinomial_4_SSM_THREADS_X, grid, threads, nomials, multinomials, gpunrhs, gpuprhs);
    } else if(nomials <= 80){
      hostDriver(_sampleSmallMultinomial_5_SSM_THREADS_X, grid, threads, nomials, multinomials, gpunrhs, gpuprhs);
    } else if(nomials <= 96){
      hostDriver(_sampleSmallMultinomial_6_SSM_THREADS_X, grid, threads, nomials, multinomials, gpunrhs, gpuprhs);
    } else if(nomials <= 112){
      hostDriver(_sampleSmallMultinomial_7_SSM_THREADS_X, grid, threads, nomials, multinomials, gpunrhs, gpuprhs);
    } else if(nomials <= 128){
      hostDriver(_sampleSmallMultinomial_8_SSM_THREADS_X, grid, threads, nomials, multinomials, gpunrhs, gpuprhs);
    } else if(nomials <= 144){
      hostDriver(_sampleSmallMultinomial_9_SSM_THREADS_X, grid, threads, nomials, multinomials, gpunrhs, gpuprhs);
    } else if(nomials <= 160){
      hostDriver(_sampleSmallMultinomial_10_SSM_THREADS_X, grid, threads, nomials, multinomials, gpunrhs, gpuprhs);
    } else if(nomials <= 176){
      hostDriver(_sampleSmallMultinomial_11_SSM_THREADS_X, grid, threads, nomials, multinomials, gpunrhs, gpuprhs);
    } else if(nomials <= 192){
      hostDriver(_sampleSmallMultinomial_12_SSM_THREADS_X, grid, threads, nomials, multinomials, gpunrhs, gpuprhs);
    } else if(nomials <= 208){
      hostDriver(_sampleSmallMultinomial_13_SSM_THREADS_X, grid, threads, nomials, multinomials, gpunrhs, gpuprhs);
    } else if(nomials <= 224){
      hostDriver(_sampleSmallMultinomial_14_SSM_THREADS_X, grid, threads, nomials, multinomials, gpunrhs, gpuprhs);
    } else if(nomials <= 240){
      hostDriver(_sampleSmallMultinomial_15_SSM_THREADS_X, grid, threads, nomials, multinomials, gpunrhs, gpuprhs);
    } else if(nomials <= 256){
      hostDriver(_sampleSmallMultinomial_16_SSM_THREADS_X, grid, threads, nomials, multinomials, gpunrhs, gpuprhs);
    }
  }


}
