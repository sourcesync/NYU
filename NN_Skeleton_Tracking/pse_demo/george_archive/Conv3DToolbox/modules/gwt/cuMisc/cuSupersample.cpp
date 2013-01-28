/*
 * This is kind of a mess...could use some cleanup.
 * Blows up a bunch of mxm images to (mf)x(mf)
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
//first, different kernels for different factors
static CUfunction _supersampleMedium_2;
static CUfunction _supersampleMediumLoopy_2;
static CUfunction _supersampleMedium_3;
static CUfunction _supersampleMediumLoopy_3;
static CUfunction _supersampleMedium_4;
static CUfunction _supersampleMediumLoopy_4;
static CUfunction _supersampleMedium_5;
static CUfunction _supersampleMediumLoopy_5;
static CUfunction _supersampleMedium_6;
static CUfunction _supersampleMediumLoopy_6;
static CUfunction _supersampleMedium_7;
static CUfunction _supersampleMediumLoopy_7;
static CUfunction _supersampleMedium_8;
static CUfunction _supersampleMediumLoopy_8;
static CUfunction _supersampleMedium_9;
static CUfunction _supersampleMediumLoopy_9;
static CUfunction _supersampleMedium_10;
static CUfunction _supersampleMediumLoopy_10;
static CUfunction _supersampleMedium_11;
static CUfunction _supersampleMediumLoopy_11;
static CUfunction _supersampleMedium_12;
static CUfunction _supersampleMediumLoopy_12;
static CUfunction _supersampleMedium_13;
static CUfunction _supersampleMediumLoopy_13;
static CUfunction _supersampleMedium_14;
static CUfunction _supersampleMediumLoopy_14;
static CUfunction _supersampleMedium_15;
static CUfunction _supersampleMediumLoopy_15;
static CUfunction _supersampleMedium_16;
static CUfunction _supersampleMediumLoopy_16;

static int init = 0;

static GPUmat *gm;

//host driver
void hostDriver(CUfunction drvfun, dim3 grid, dim3 threads, int shmem, int imgSizeX, int imgSizeY, int nrhs, hostdrv_pars_t *prhs) {
  //mexPrintf("threads.x: %d threads.y: %d threads.z %d\n",threads.x,threads.y,threads.z);

  CUresult err = CUDA_SUCCESS;

  // setup execution parameters
  if (CUDA_SUCCESS != (err = cuFuncSetBlockShape(drvfun, threads.x, threads.y, threads.z))) {
    mexErrMsgTxt("Error in cuFuncSetBlockShape");
  }

  if (CUDA_SUCCESS != cuFuncSetSharedSize(drvfun, shmem)) {
    mexErrMsgTxt("Error in cuFuncSetSharedSize");
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

  if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, imgSizeX)) {
    mexErrMsgTxt("Error in cuParamSeti");
  }
  poffset += sizeof(imgSizeX);

  if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, imgSizeY)) {
    mexErrMsgTxt("Error in cuParamSeti");
  }
  poffset += sizeof(imgSizeY);

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

    status = cuModuleGetFunction(&_supersampleMedium_2, *drvmod, "_Z18kSupersampleMediumILi2EEvPfS0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_supersampleMedium_3, *drvmod, "_Z18kSupersampleMediumILi3EEvPfS0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_supersampleMedium_4, *drvmod, "_Z18kSupersampleMediumILi4EEvPfS0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_supersampleMedium_5, *drvmod, "_Z18kSupersampleMediumILi5EEvPfS0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_supersampleMedium_6, *drvmod, "_Z18kSupersampleMediumILi6EEvPfS0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_supersampleMedium_7, *drvmod, "_Z18kSupersampleMediumILi7EEvPfS0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_supersampleMedium_8, *drvmod, "_Z18kSupersampleMediumILi8EEvPfS0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_supersampleMedium_9, *drvmod, "_Z18kSupersampleMediumILi9EEvPfS0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_supersampleMedium_10, *drvmod, "_Z18kSupersampleMediumILi10EEvPfS0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_supersampleMedium_11, *drvmod, "_Z18kSupersampleMediumILi11EEvPfS0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_supersampleMedium_12, *drvmod, "_Z18kSupersampleMediumILi12EEvPfS0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_supersampleMedium_13, *drvmod, "_Z18kSupersampleMediumILi13EEvPfS0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_supersampleMedium_14, *drvmod, "_Z18kSupersampleMediumILi14EEvPfS0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_supersampleMedium_15, *drvmod, "_Z18kSupersampleMediumILi15EEvPfS0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_supersampleMedium_16, *drvmod, "_Z18kSupersampleMediumILi16EEvPfS0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_supersampleMediumLoopy_2, *drvmod, "_Z23kSupersampleMediumLoopyILi2EEvPfS0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_supersampleMediumLoopy_3, *drvmod, "_Z23kSupersampleMediumLoopyILi3EEvPfS0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_supersampleMediumLoopy_4, *drvmod, "_Z23kSupersampleMediumLoopyILi4EEvPfS0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_supersampleMediumLoopy_5, *drvmod, "_Z23kSupersampleMediumLoopyILi5EEvPfS0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_supersampleMediumLoopy_6, *drvmod, "_Z23kSupersampleMediumLoopyILi6EEvPfS0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_supersampleMediumLoopy_7, *drvmod, "_Z23kSupersampleMediumLoopyILi7EEvPfS0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_supersampleMediumLoopy_8, *drvmod, "_Z23kSupersampleMediumLoopyILi8EEvPfS0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_supersampleMediumLoopy_9, *drvmod, "_Z23kSupersampleMediumLoopyILi9EEvPfS0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_supersampleMediumLoopy_10, *drvmod, "_Z23kSupersampleMediumLoopyILi10EEvPfS0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_supersampleMediumLoopy_11, *drvmod, "_Z23kSupersampleMediumLoopyILi11EEvPfS0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_supersampleMediumLoopy_12, *drvmod, "_Z23kSupersampleMediumLoopyILi12EEvPfS0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_supersampleMediumLoopy_13, *drvmod, "_Z23kSupersampleMediumLoopyILi13EEvPfS0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_supersampleMediumLoopy_14, *drvmod, "_Z23kSupersampleMediumLoopyILi14EEvPfS0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_supersampleMediumLoopy_15, *drvmod, "_Z23kSupersampleMediumLoopyILi15EEvPfS0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_supersampleMediumLoopy_16, *drvmod, "_Z23kSupersampleMediumLoopyILi16EEvPfS0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }

    init = 1;
  }

  // mex parameters are:

  // 1. IN1
  // 2. OUT
  // 3. supersampling factor

  bool avoidBankConflicts = true; //hard-coded
  bool trans = false; //hard-coded

  //IN1 is the input GPU array
  GPUtype IN1 = gm->gputype.getGPUtype(prhs[0]);

  //OUT is the output GPU array (result)
  GPUtype OUT = gm->gputype.getGPUtype(prhs[1]);

  //last parameter is the filterSize (int)
  int factor = (int) mxGetScalar(prhs[2]);

  // number of elements
  int nin1 = gm->gputype.getNumel(IN1);
  int nout = gm->gputype.getNumel(OUT);

  //dimensions 
  const int * sin1 = gm->gputype.getSize(IN1);
  const int * sout = gm->gputype.getSize(OUT);

  int imgPixels = sin1[0];
  if ( floor(sqrt(float(imgPixels))) != sqrt(float(imgPixels)) )
    mexErrMsgTxt("Images not square");
  int imgSize = int(sqrt(imgPixels));
  int numImages = sin1[1];
  

  /* if (imgSize <= factor)
     mexErrMsgTxt("imgSize must be > factor"); */
  if (factor > 16)
    mexErrMsgTxt("factor > 16");
  if (factor < 2)
    mexErrMsgTxt("factor < 2");
  if (imgSize > 512)
    mexErrMsgTxt("max imgSize: 512");
  if (imgSize < 1)
    mexErrMsgTxt("min imgSize: 1");

  int targetPixels = sout[0];
  if ( floor(sqrt(float(targetPixels))) != sqrt(float(targetPixels)) )
    mexErrMsgTxt("Targets not square");
  int targetSize = int(sqrt(targetPixels));
  if (targetSize % factor !=0)
    mexErrMsgTxt("imgSize must be evenly divisible by factor");
  if (targetSize / factor != imgSize)
    mexErrMsgTxt("targetSize/ factor must = imgSize");
  if (nout != nin1 * factor*factor)
    mexErrMsgTxt("Target dimensions not consistent");

  int threadsX, threadsY;
  int SHMEM_MAX = 8192; // don't use more than this much shmem
  int shmemX, shmemY, blocksX, blocksY;
  bool useLoopy = false;
  int THREADS_MAX_LOOPY = 512, THREADS_MAX = trans ? 256 : 512;
  if (!trans) {
    threadsX = imgSize;
    threadsY = factor * MIN(THREADS_MAX / (factor*threadsX), SHMEM_MAX / (4*threadsX*factor)); // to avoid running out of shmem

    if(threadsY == 0) {
      if (factor > 32)
	mexErrMsgTxt("factor can't be > 32");
      //assert(factor <= 32); // yes this is covered by assert above but in case i ever remove that
      THREADS_MAX = 512;
      useLoopy = true;
      threadsX = MIN(16, imgSize); // not that imgsize can be < 16 here under current conditions
      threadsY = factor * MIN(THREADS_MAX_LOOPY / (factor*threadsX), SHMEM_MAX / (4*threadsX*factor)); // to avoid running out of shmem
    }

    shmemY = threadsY;
    shmemX = threadsX;
    blocksX = imgSize;
    blocksY = DIVUP(numImages, threadsY);
    //        printf("boundary problems: %u\n", numImages % threadsY != 0);
  } else {

    threadsY = imgSize;
    threadsX = factor * MIN(THREADS_MAX / (factor*threadsY), SHMEM_MAX / (4*threadsY*factor)); // to avoid running out of shmem

    if(threadsX < 8) {
      useLoopy = true;
      int xFactorMult = DIVUP(16, factor);
      threadsX = xFactorMult * factor;
      threadsY = THREADS_MAX / threadsX;
      int newThreadsX = threadsX, newThreadsY = threadsY;
      while (newThreadsY > 0 && imgSize % newThreadsY != 0) { // let's see if we can make threadsY divide imgSize
	newThreadsX += factor;
	newThreadsY = THREADS_MAX / newThreadsX;
      }
      if (newThreadsY > 0) {
	threadsY = newThreadsY;
	threadsX = newThreadsX;
      }

      if (threadsY <=0)
	mexErrMsgTxt("threadsY <=0; not expected");
      //assert(threadsY > 0);
    }

    shmemY = threadsX;
    shmemX = threadsY + (1 - (threadsY % 2));
    blocksX = DIVUP(numImages, threadsX);
    blocksY = imgSize;
    //        printf("boundary problems: %u\n", numImages % threadsX != 0);
  }
  int shmem = 4 * shmemX * shmemY;
  if (shmem == 0 || shmem > 16300) {
    // this really shouldn't happen and i've only put this here as a precautionary measure
    // to avoid getting mysteriously wrong results.
    mexErrMsgTxt("supersample: not enough shared memory!");
    //exit(EXIT_FAILURE);
  }

  dim3 grid(blocksX, blocksY);
  dim3 threads(threadsX, threadsY);
  //mexPrintf("blocks: %dx%d, threads: %dx%d\n", blocksY, blocksX, threadsY, threadsX);
  //mexPrintf("using %dx%d = %d bytes of shmem\n", shmemY, shmemX, shmem);


  gpuTYPE_t tin1 = gm->gputype.getType(IN1);
  gpuTYPE_t tout = gm->gputype.getType(OUT);

  // check input/out size and type
  if (tin1!=tout)
    mexErrMsgTxt("Input and output arguments must be of the same type.");

  // I need the pointers to GPU memory
  CUdeviceptr d_IN1  = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(IN1));
  CUdeviceptr d_OUT = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(OUT));

  // // The GPU kernel depends on the type of input/output
  // CUfunction drvfun;
  // if (tin1 == gpuFLOAT) {
  //   drvfun = drvfunf;
  // } else 
  //   mexErrMsgTxt("Currently only single types supported.");

  hostdrv_pars_t gpuprhs[2];
  int gpunrhs = 2;
  gpuprhs[0] = hostdrv_pars(&d_IN1,sizeof(d_IN1));
  gpuprhs[1] = hostdrv_pars(&d_OUT,sizeof(d_OUT));


  /* trans not implemented; so always !trans    if(!trans) { */
        if(!useLoopy) {
            if(factor == 2) {
	      hostDriver(_supersampleMedium_2, grid, threads, shmem, imgSize, numImages*imgSize, gpunrhs, gpuprhs);
	      //kSupersampleMedium<2><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize);
            } else if(factor == 3) {
                hostDriver(_supersampleMedium_3, grid, threads, shmem, imgSize, numImages*imgSize, gpunrhs, gpuprhs);
            } else if(factor == 4) {
                hostDriver(_supersampleMedium_4, grid, threads, shmem, imgSize, numImages*imgSize, gpunrhs, gpuprhs);
            } else if(factor == 5) {
                hostDriver(_supersampleMedium_5, grid, threads, shmem, imgSize, numImages*imgSize, gpunrhs, gpuprhs);
            } else if(factor == 6) {
                hostDriver(_supersampleMedium_6, grid, threads, shmem, imgSize, numImages*imgSize, gpunrhs, gpuprhs);
            } else if(factor == 7) {
                hostDriver(_supersampleMedium_7, grid, threads, shmem, imgSize, numImages*imgSize, gpunrhs, gpuprhs);
            } else if(factor == 8) {
                hostDriver(_supersampleMedium_8, grid, threads, shmem, imgSize, numImages*imgSize, gpunrhs, gpuprhs);
            } else if(factor == 9) {
                hostDriver(_supersampleMedium_9, grid, threads, shmem, imgSize, numImages*imgSize, gpunrhs, gpuprhs);
            } else if(factor == 10) {
                hostDriver(_supersampleMedium_10, grid, threads, shmem, imgSize, numImages*imgSize, gpunrhs, gpuprhs);
            } else if(factor == 11) {
                hostDriver(_supersampleMedium_11, grid, threads, shmem, imgSize, numImages*imgSize, gpunrhs, gpuprhs);
            } else if(factor == 12) {
                hostDriver(_supersampleMedium_12, grid, threads, shmem, imgSize, numImages*imgSize, gpunrhs, gpuprhs);
            } else if(factor == 13) {
                hostDriver(_supersampleMedium_13, grid, threads, shmem, imgSize, numImages*imgSize, gpunrhs, gpuprhs);
            } else if(factor == 14) {
                hostDriver(_supersampleMedium_14, grid, threads, shmem, imgSize, numImages*imgSize, gpunrhs, gpuprhs);
            } else if(factor == 15) {
                hostDriver(_supersampleMedium_15, grid, threads, shmem, imgSize, numImages*imgSize, gpunrhs, gpuprhs);
            } else if(factor == 16) {
                hostDriver(_supersampleMedium_16, grid, threads, shmem, imgSize, numImages*imgSize, gpunrhs, gpuprhs);
            }
        } else {
            if(factor == 2) {
                hostDriver(_supersampleMediumLoopy_2, grid, threads, shmem, imgSize, numImages*imgSize, gpunrhs, gpuprhs);
            } else if(factor == 3) {
                hostDriver(_supersampleMediumLoopy_3, grid, threads, shmem, imgSize, numImages*imgSize, gpunrhs, gpuprhs);
            } else if(factor == 4) {
                hostDriver(_supersampleMediumLoopy_4, grid, threads, shmem, imgSize, numImages*imgSize, gpunrhs, gpuprhs);
            } else if(factor == 5) {
                hostDriver(_supersampleMediumLoopy_5, grid, threads, shmem, imgSize, numImages*imgSize, gpunrhs, gpuprhs);
            } else if(factor == 6) {
                hostDriver(_supersampleMediumLoopy_6, grid, threads, shmem, imgSize, numImages*imgSize, gpunrhs, gpuprhs);
            } else if(factor == 7) {
                hostDriver(_supersampleMediumLoopy_7, grid, threads, shmem, imgSize, numImages*imgSize, gpunrhs, gpuprhs);
            } else if(factor == 8) {
                hostDriver(_supersampleMediumLoopy_8, grid, threads, shmem, imgSize, numImages*imgSize, gpunrhs, gpuprhs);
            } else if(factor == 9) {
                hostDriver(_supersampleMediumLoopy_9, grid, threads, shmem, imgSize, numImages*imgSize, gpunrhs, gpuprhs);
            } else if(factor == 10) {
                hostDriver(_supersampleMediumLoopy_10, grid, threads, shmem, imgSize, numImages*imgSize, gpunrhs, gpuprhs);
            } else if(factor == 11) {
                hostDriver(_supersampleMediumLoopy_11, grid, threads, shmem, imgSize, numImages*imgSize, gpunrhs, gpuprhs);
            } else if(factor == 12) {
                hostDriver(_supersampleMediumLoopy_12, grid, threads, shmem, imgSize, numImages*imgSize, gpunrhs, gpuprhs);
            } else if(factor == 13) {
                hostDriver(_supersampleMediumLoopy_13, grid, threads, shmem, imgSize, numImages*imgSize, gpunrhs, gpuprhs);
            } else if(factor == 14) {
                hostDriver(_supersampleMediumLoopy_14, grid, threads, shmem, imgSize, numImages*imgSize, gpunrhs, gpuprhs);
            } else if(factor == 15) {
                hostDriver(_supersampleMediumLoopy_15, grid, threads, shmem, imgSize, numImages*imgSize, gpunrhs, gpuprhs);
            } else if(factor == 16) {
                hostDriver(_supersampleMediumLoopy_16, grid, threads, shmem, imgSize, numImages*imgSize, gpunrhs, gpuprhs);
            }
        }
	/* } else {
        if(!useLoopy) {
            if(factor == 2) {
                kSupersampleMediumTrans<2><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 3) {
                kSupersampleMediumTrans<3><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 4) {
                kSupersampleMediumTrans<4><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 5) {
                kSupersampleMediumTrans<5><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 6) {
                kSupersampleMediumTrans<6><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 7) {
                kSupersampleMediumTrans<7><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 8) {
                kSupersampleMediumTrans<8><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 9) {
                kSupersampleMediumTrans<9><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 10) {
                kSupersampleMediumTrans<10><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 11) {
                kSupersampleMediumTrans<11><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 12) {
                kSupersampleMediumTrans<12><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 13) {
                kSupersampleMediumTrans<13><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 14) {
                kSupersampleMediumTrans<14><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 15) {
                kSupersampleMediumTrans<15><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 16) {
                kSupersampleMediumTrans<16><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            }
        } else {
            if(factor == 2) {
                kSupersampleMediumTransLoopy<2><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 3) {
                kSupersampleMediumTransLoopy<3><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 4) {
                kSupersampleMediumTransLoopy<4><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 5) {
                kSupersampleMediumTransLoopy<5><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 6) {
                kSupersampleMediumTransLoopy<6><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 7) {
                kSupersampleMediumTransLoopy<7><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 8) {
                kSupersampleMediumTransLoopy<8><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 9) {
                kSupersampleMediumTransLoopy<9><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 10) {
                kSupersampleMediumTransLoopy<10><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 11) {
                kSupersampleMediumTransLoopy<11><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 12) {
                kSupersampleMediumTransLoopy<12><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 13) {
                kSupersampleMediumTransLoopy<13><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 14) {
                kSupersampleMediumTransLoopy<14><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 15) {
                kSupersampleMediumTransLoopy<15><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 16) {
                kSupersampleMediumTransLoopy<16><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            }
        }
    }
	*/

}
