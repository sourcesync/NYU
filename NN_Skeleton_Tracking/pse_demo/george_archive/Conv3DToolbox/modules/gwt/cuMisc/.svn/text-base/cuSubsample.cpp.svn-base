/*
 * f = factor, m = image size
 * Converts a bunch of mxm images to (m/f)x(m/f) images by averaging non-overlapping fxf regions.
 *
 * The avoidBankConflicts option causes this function to use extra shared memory to avoid all
 * bank conflicts. Most bank conflicts are avoided regardless of the setting of this parameter,
 * and so setting this parameter to true will have minimal impact on performance (I noticed
 * a 5% improvement). (stil can get 2-way conflicts if factor doesn't divide 16)
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
static CUfunction subsample_noreduc_2T;
static CUfunction subsample_noreduc_2F;
static CUfunction subsample_noreduc_3T;
static CUfunction subsample_noreduc_3F;
static CUfunction subsample_noreduc_4T;
static CUfunction subsample_noreduc_4F;
static CUfunction subsample_noreduc_5T;
static CUfunction subsample_noreduc_5F;
static CUfunction subsample_noreduc_6T;
static CUfunction subsample_noreduc_6F;
static CUfunction subsample_noreduc_7T;
static CUfunction subsample_noreduc_7F;
static CUfunction subsample_noreduc_8T;
static CUfunction subsample_noreduc_8F;
static CUfunction subsample_noreduc_9T;
static CUfunction subsample_noreduc_9F;
static CUfunction subsample_noreduc_10T;
static CUfunction subsample_noreduc_10F;
static CUfunction subsample_noreduc_11T;
static CUfunction subsample_noreduc_11F;
static CUfunction subsample_noreduc_12T;
static CUfunction subsample_noreduc_12F;
static CUfunction subsample_noreduc_13T;
static CUfunction subsample_noreduc_13F;
static CUfunction subsample_noreduc_14T;
static CUfunction subsample_noreduc_14F;
static CUfunction subsample_noreduc_15T;
static CUfunction subsample_noreduc_15F;
static CUfunction subsample_noreduc_16T;
static CUfunction subsample_noreduc_16F;

static int init = 0;

static GPUmat *gm;

//host driver
void hostDriver(CUfunction drvfun, dim3 grid, dim3 threads, int imgSize, int numRegionsY, int shmemX, int shmem, int nrhs, hostdrv_pars_t *prhs) {

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

  if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, imgSize)) {
    mexErrMsgTxt("Error in cuParamSeti");
  }
  poffset += sizeof(imgSize);

  if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, numRegionsY)) {
    mexErrMsgTxt("Error in cuParamSeti");
  }
  poffset += sizeof(numRegionsY);

  if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, shmemX)) {
    mexErrMsgTxt("Error in cuParamSeti");
  }
  poffset += sizeof(shmemX);

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

    status = cuModuleGetFunction(&subsample_noreduc_2T, *drvmod, "_Z18kSubsample_noreducILi2ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&subsample_noreduc_2F, *drvmod, "_Z18kSubsample_noreducILi2ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&subsample_noreduc_3T, *drvmod, "_Z18kSubsample_noreducILi3ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&subsample_noreduc_3F, *drvmod, "_Z18kSubsample_noreducILi3ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&subsample_noreduc_4T, *drvmod, "_Z18kSubsample_noreducILi4ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&subsample_noreduc_4F, *drvmod, "_Z18kSubsample_noreducILi4ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&subsample_noreduc_5T, *drvmod, "_Z18kSubsample_noreducILi5ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&subsample_noreduc_5F, *drvmod, "_Z18kSubsample_noreducILi5ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&subsample_noreduc_6T, *drvmod, "_Z18kSubsample_noreducILi6ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&subsample_noreduc_6F, *drvmod, "_Z18kSubsample_noreducILi6ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&subsample_noreduc_7T, *drvmod, "_Z18kSubsample_noreducILi7ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&subsample_noreduc_7F, *drvmod, "_Z18kSubsample_noreducILi7ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&subsample_noreduc_8T, *drvmod, "_Z18kSubsample_noreducILi8ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&subsample_noreduc_8F, *drvmod, "_Z18kSubsample_noreducILi8ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&subsample_noreduc_9T, *drvmod, "_Z18kSubsample_noreducILi9ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&subsample_noreduc_9F, *drvmod, "_Z18kSubsample_noreducILi9ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&subsample_noreduc_10T, *drvmod, "_Z18kSubsample_noreducILi10ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&subsample_noreduc_10F, *drvmod, "_Z18kSubsample_noreducILi10ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&subsample_noreduc_11T, *drvmod, "_Z18kSubsample_noreducILi11ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&subsample_noreduc_11F, *drvmod, "_Z18kSubsample_noreducILi11ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&subsample_noreduc_12T, *drvmod, "_Z18kSubsample_noreducILi12ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&subsample_noreduc_12F, *drvmod, "_Z18kSubsample_noreducILi12ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&subsample_noreduc_13T, *drvmod, "_Z18kSubsample_noreducILi13ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&subsample_noreduc_13F, *drvmod, "_Z18kSubsample_noreducILi13ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&subsample_noreduc_14T, *drvmod, "_Z18kSubsample_noreducILi14ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&subsample_noreduc_14F, *drvmod, "_Z18kSubsample_noreducILi14ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&subsample_noreduc_15T, *drvmod, "_Z18kSubsample_noreducILi15ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&subsample_noreduc_15F, *drvmod, "_Z18kSubsample_noreducILi15ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&subsample_noreduc_16T, *drvmod, "_Z18kSubsample_noreducILi16ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&subsample_noreduc_16F, *drvmod, "_Z18kSubsample_noreducILi16ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }

    init = 1;
  }

  // mex parameters are:

  // 1. IN1
  // 2. OUT
  // 3. subsampling factor

  bool avoidBankConflicts = true; //hard-coded

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

  if (imgSize <= factor)
    mexErrMsgTxt("imgSize must be > factor");
  if (imgSize % factor !=0)
    mexErrMsgTxt("imgSize must be evenly divisible by factor");
  if (factor > 16)
    mexErrMsgTxt("factor > 16");
  if (factor < 2)
    mexErrMsgTxt("factor < 2");
  if (imgSize > 512)
    mexErrMsgTxt("max imgSize: 512");

  int numRegions = nin1 / (factor*factor);
  int numRegionsY = (imgSize / factor) * sin1[1];
  if (nout != numRegions)
    mexErrMsgTxt("Target dimensions not consistent");


  int regionsXPerBlock = imgSize / factor;
  int numThreadsX = imgSize;
  int SHMEM_MAX = 8192; // don't use more than this much shmem
  int regionsYPerBlock = MIN(512 / numThreadsX, SHMEM_MAX / (4*imgSize)); // to avoid running out of shmem
//    regionsYPerBlock--;
  int regionsPerBlock = regionsYPerBlock * regionsXPerBlock;

  // this will avoid all bank conflicts but may (?) use up too much shmem
  int shmemPadX = avoidBankConflicts * (DIVUP(16,factor) + (regionsPerBlock % 16 == 0 ? 0 : 16 - regionsPerBlock % 16));
  //    shmemPadX = 0;
  int shmemY = factor, shmemX = regionsPerBlock + shmemPadX;
  int shmem = 4 * shmemX * shmemY;
  if (shmem == 0 || shmem > 16300) {
    // this really shouldn't happen and i've only put this here as a precautionary measure
    // to avoid getting mysteriously wrong results.
    mexErrMsgTxt("subsample: not enough shared memory!");
  }

  int numThreadsY = regionsYPerBlock;
  //    int blocks = numRegionsY / regionsYPerBlock;
  int blocksX = imgSize / factor, blocksY = DIVUP(sin1[1], regionsYPerBlock);
  if (blocksX >=65535 || blocksY >= 65535)
    mexErrMsgTxt("Exceeded max block size");

  //    assert(numRegionsY % regionsYPerBlock == 0);
  bool checkThreadBounds = numRegionsY % regionsYPerBlock != 0;
  //    printf("num regions y: %d, regions y per block: %d\n", numRegionsY, regionsYPerBlock);
  dim3 grid(blocksX, blocksY);
  dim3 threads(numThreadsX, numThreadsY);
  /*
  mexPrintf("grid: %ux%u, threads: %ux%u\n", grid.y, grid.x, threads.y, threads.x);
  mexPrintf("check bounds: %u\n", checkThreadBounds);
  mexPrintf("using %u bytes of shmem\n", shmem);
  */

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

  //int N = nin1;

  if (factor == 2) {
    if (checkThreadBounds) {
      hostDriver(subsample_noreduc_2T, grid, threads, imgSize, numRegionsY, shmemX, shmem, gpunrhs, gpuprhs);
    } else {
      hostDriver(subsample_noreduc_2F, grid, threads, imgSize, numRegionsY, shmemX, shmem, gpunrhs, gpuprhs);
    }
  } else if (factor == 3) {
    if (checkThreadBounds) {
      hostDriver(subsample_noreduc_3T, grid, threads, imgSize, numRegionsY, shmemX, shmem, gpunrhs, gpuprhs);
    } else {
      hostDriver(subsample_noreduc_3F, grid, threads, imgSize, numRegionsY, shmemX, shmem, gpunrhs, gpuprhs);
    }
  } else if (factor == 4) {
    if (checkThreadBounds) {
      hostDriver(subsample_noreduc_4T, grid, threads, imgSize, numRegionsY, shmemX, shmem, gpunrhs, gpuprhs);
    } else {
      hostDriver(subsample_noreduc_4F, grid, threads, imgSize, numRegionsY, shmemX, shmem, gpunrhs, gpuprhs);
    }
  } else if (factor == 5) {
    if (checkThreadBounds) {
      hostDriver(subsample_noreduc_5T, grid, threads, imgSize, numRegionsY, shmemX, shmem, gpunrhs, gpuprhs);
    } else {
      hostDriver(subsample_noreduc_5F, grid, threads, imgSize, numRegionsY, shmemX, shmem, gpunrhs, gpuprhs);
    }
  } else if (factor == 6) {
    if (checkThreadBounds) {
      hostDriver(subsample_noreduc_6T, grid, threads, imgSize, numRegionsY, shmemX, shmem, gpunrhs, gpuprhs);
    } else {
      hostDriver(subsample_noreduc_6F, grid, threads, imgSize, numRegionsY, shmemX, shmem, gpunrhs, gpuprhs);
    }
  } else if (factor == 7) {
    if (checkThreadBounds) {
      hostDriver(subsample_noreduc_7T, grid, threads, imgSize, numRegionsY, shmemX, shmem, gpunrhs, gpuprhs);
    } else {
      hostDriver(subsample_noreduc_7F, grid, threads, imgSize, numRegionsY, shmemX, shmem, gpunrhs, gpuprhs);
    }
  } else if (factor == 8) {
    if (checkThreadBounds) {
      hostDriver(subsample_noreduc_8T, grid, threads, imgSize, numRegionsY, shmemX, shmem, gpunrhs, gpuprhs);
    } else {
      hostDriver(subsample_noreduc_8F, grid, threads, imgSize, numRegionsY, shmemX, shmem, gpunrhs, gpuprhs);
    }
  } else if (factor == 9) {
    if (checkThreadBounds) {
      hostDriver(subsample_noreduc_9T, grid, threads, imgSize, numRegionsY, shmemX, shmem, gpunrhs, gpuprhs);
    } else {
      hostDriver(subsample_noreduc_9F, grid, threads, imgSize, numRegionsY, shmemX, shmem, gpunrhs, gpuprhs);
    }
  } else if (factor == 10) {
    if (checkThreadBounds) {
      hostDriver(subsample_noreduc_10T, grid, threads, imgSize, numRegionsY, shmemX, shmem, gpunrhs, gpuprhs);
    } else {
      hostDriver(subsample_noreduc_10F, grid, threads, imgSize, numRegionsY, shmemX, shmem, gpunrhs, gpuprhs);
    }
  } else if (factor == 11) {
    if (checkThreadBounds) {
      hostDriver(subsample_noreduc_11T, grid, threads, imgSize, numRegionsY, shmemX, shmem, gpunrhs, gpuprhs);
    } else {
      hostDriver(subsample_noreduc_11F, grid, threads, imgSize, numRegionsY, shmemX, shmem, gpunrhs, gpuprhs);
    }
  } else if (factor == 12) {
    if (checkThreadBounds) {
      hostDriver(subsample_noreduc_12T, grid, threads, imgSize, numRegionsY, shmemX, shmem, gpunrhs, gpuprhs);
    } else {
      hostDriver(subsample_noreduc_12F, grid, threads, imgSize, numRegionsY, shmemX, shmem, gpunrhs, gpuprhs);
    }
  } else if (factor == 13) {
    if (checkThreadBounds) {
      hostDriver(subsample_noreduc_13T, grid, threads, imgSize, numRegionsY, shmemX, shmem, gpunrhs, gpuprhs);
    } else {
      hostDriver(subsample_noreduc_13F, grid, threads, imgSize, numRegionsY, shmemX, shmem, gpunrhs, gpuprhs);
    }
  } else if (factor == 14) {
    if (checkThreadBounds) {
      hostDriver(subsample_noreduc_14T, grid, threads, imgSize, numRegionsY, shmemX, shmem, gpunrhs, gpuprhs);
    } else {
      hostDriver(subsample_noreduc_14F, grid, threads, imgSize, numRegionsY, shmemX, shmem, gpunrhs, gpuprhs);
    }
  } else if (factor == 15) {
    if (checkThreadBounds) {
      hostDriver(subsample_noreduc_15T, grid, threads, imgSize, numRegionsY, shmemX, shmem, gpunrhs, gpuprhs);
    } else {
      hostDriver(subsample_noreduc_15F, grid, threads, imgSize, numRegionsY, shmemX, shmem, gpunrhs, gpuprhs);
    }
  } else if (factor == 16) {
    if (checkThreadBounds) {
      hostDriver(subsample_noreduc_16T, grid, threads, imgSize, numRegionsY, shmemX, shmem, gpunrhs, gpuprhs);
    } else {
      hostDriver(subsample_noreduc_16F, grid, threads, imgSize, numRegionsY, shmemX, shmem, gpunrhs, gpuprhs);
    }
  }
    //cutilCheckMsg("kernel execution failed");
    
    //    if(factor == 4) {
    ////        kSubsample_reduc<4><<<grid, threads,4*numThreadsX*numThreadsY>>>(images->getDevData(), targets->getDevData(), imgSize, numRegionsY);
    //    }

}
