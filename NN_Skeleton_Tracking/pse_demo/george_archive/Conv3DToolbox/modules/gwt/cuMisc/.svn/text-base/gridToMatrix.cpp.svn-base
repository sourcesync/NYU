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
#include "gridToMatrix.h"


// static paramaters
//first, different kernels for different factors
static CUfunction _gridToMatrix_2T;
static CUfunction _gridToMatrix_2F;
static CUfunction _gridToMatrix_3T;
static CUfunction _gridToMatrix_3F;
static CUfunction _gridToMatrix_4T;
static CUfunction _gridToMatrix_4F;
static CUfunction _gridToMatrix_5T;
static CUfunction _gridToMatrix_5F;
static CUfunction _gridToMatrix_6T;
static CUfunction _gridToMatrix_6F;
static CUfunction _gridToMatrix_7T;
static CUfunction _gridToMatrix_7F;
static CUfunction _gridToMatrix_8T;
static CUfunction _gridToMatrix_8F;
static CUfunction _gridToMatrix_9T;
static CUfunction _gridToMatrix_9F;
static CUfunction _gridToMatrix_10T;
static CUfunction _gridToMatrix_10F;
static CUfunction _gridToMatrix_11T;
static CUfunction _gridToMatrix_11F;
static CUfunction _gridToMatrix_12T;
static CUfunction _gridToMatrix_12F;
static CUfunction _gridToMatrix_13T;
static CUfunction _gridToMatrix_13F;
static CUfunction _gridToMatrix_14T;
static CUfunction _gridToMatrix_14F;
static CUfunction _gridToMatrix_15T;
static CUfunction _gridToMatrix_15F;
static CUfunction _gridToMatrix_16T;
static CUfunction _gridToMatrix_16F;
static CUfunction _gridToMatrixLoopy_2T;
static CUfunction _gridToMatrixLoopy_2F;
static CUfunction _gridToMatrixLoopy_3T;
static CUfunction _gridToMatrixLoopy_3F;
static CUfunction _gridToMatrixLoopy_4T;
static CUfunction _gridToMatrixLoopy_4F;
static CUfunction _gridToMatrixLoopy_5T;
static CUfunction _gridToMatrixLoopy_5F;
static CUfunction _gridToMatrixLoopy_6T;
static CUfunction _gridToMatrixLoopy_6F;
static CUfunction _gridToMatrixLoopy_7T;
static CUfunction _gridToMatrixLoopy_7F;
static CUfunction _gridToMatrixLoopy_8T;
static CUfunction _gridToMatrixLoopy_8F;
static CUfunction _gridToMatrixLoopy_9T;
static CUfunction _gridToMatrixLoopy_9F;
static CUfunction _gridToMatrixLoopy_10T;
static CUfunction _gridToMatrixLoopy_10F;
static CUfunction _gridToMatrixLoopy_11T;
static CUfunction _gridToMatrixLoopy_11F;
static CUfunction _gridToMatrixLoopy_12T;
static CUfunction _gridToMatrixLoopy_12F;
static CUfunction _gridToMatrixLoopy_13T;
static CUfunction _gridToMatrixLoopy_13F;
static CUfunction _gridToMatrixLoopy_14T;
static CUfunction _gridToMatrixLoopy_14F;
static CUfunction _gridToMatrixLoopy_15T;
static CUfunction _gridToMatrixLoopy_15F;
static CUfunction _gridToMatrixLoopy_16T;
static CUfunction _gridToMatrixLoopy_16F;

static int init = 0;

static GPUmat *gm;

//host driver
void hostDriver(CUfunction drvfun, dim3 grid, dim3 threads, int shmem, int imgSizeX, int imgSizeY, int shmemX, int nrhs, hostdrv_pars_t *prhs) {

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

void _gtm(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], bool reverse) {
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

    status = cuModuleGetFunction(&_gridToMatrix_2T, *drvmod, "_Z13kGridToMatrixILi2ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrix_2F, *drvmod, "_Z13kGridToMatrixILi2ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrix_3T, *drvmod, "_Z13kGridToMatrixILi3ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrix_3F, *drvmod, "_Z13kGridToMatrixILi3ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrix_4T, *drvmod, "_Z13kGridToMatrixILi4ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrix_4F, *drvmod, "_Z13kGridToMatrixILi4ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrix_5T, *drvmod, "_Z13kGridToMatrixILi5ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrix_5F, *drvmod, "_Z13kGridToMatrixILi5ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrix_6T, *drvmod, "_Z13kGridToMatrixILi6ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrix_6F, *drvmod, "_Z13kGridToMatrixILi6ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrix_7T, *drvmod, "_Z13kGridToMatrixILi7ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrix_7F, *drvmod, "_Z13kGridToMatrixILi7ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrix_8T, *drvmod, "_Z13kGridToMatrixILi8ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrix_8F, *drvmod, "_Z13kGridToMatrixILi8ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrix_9T, *drvmod, "_Z13kGridToMatrixILi9ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrix_9F, *drvmod, "_Z13kGridToMatrixILi9ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrix_10T, *drvmod, "_Z13kGridToMatrixILi10ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrix_10F, *drvmod, "_Z13kGridToMatrixILi10ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrix_11T, *drvmod, "_Z13kGridToMatrixILi11ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrix_11F, *drvmod, "_Z13kGridToMatrixILi11ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrix_12T, *drvmod, "_Z13kGridToMatrixILi12ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrix_12F, *drvmod, "_Z13kGridToMatrixILi12ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrix_13T, *drvmod, "_Z13kGridToMatrixILi13ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrix_13F, *drvmod, "_Z13kGridToMatrixILi13ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrix_14T, *drvmod, "_Z13kGridToMatrixILi14ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrix_14F, *drvmod, "_Z13kGridToMatrixILi14ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrix_15T, *drvmod, "_Z13kGridToMatrixILi15ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrix_15F, *drvmod, "_Z13kGridToMatrixILi15ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrix_16T, *drvmod, "_Z13kGridToMatrixILi16ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrix_16F, *drvmod, "_Z13kGridToMatrixILi16ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
status = cuModuleGetFunction(&_gridToMatrixLoopy_2T, *drvmod, "_Z18kGridToMatrixLoopyILi2ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrixLoopy_2F, *drvmod, "_Z18kGridToMatrixLoopyILi2ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrixLoopy_3T, *drvmod, "_Z18kGridToMatrixLoopyILi3ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrixLoopy_3F, *drvmod, "_Z18kGridToMatrixLoopyILi3ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrixLoopy_4T, *drvmod, "_Z18kGridToMatrixLoopyILi4ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrixLoopy_4F, *drvmod, "_Z18kGridToMatrixLoopyILi4ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrixLoopy_5T, *drvmod, "_Z18kGridToMatrixLoopyILi5ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrixLoopy_5F, *drvmod, "_Z18kGridToMatrixLoopyILi5ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrixLoopy_6T, *drvmod, "_Z18kGridToMatrixLoopyILi6ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrixLoopy_6F, *drvmod, "_Z18kGridToMatrixLoopyILi6ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrixLoopy_7T, *drvmod, "_Z18kGridToMatrixLoopyILi7ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrixLoopy_7F, *drvmod, "_Z18kGridToMatrixLoopyILi7ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrixLoopy_8T, *drvmod, "_Z18kGridToMatrixLoopyILi8ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrixLoopy_8F, *drvmod, "_Z18kGridToMatrixLoopyILi8ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrixLoopy_9T, *drvmod, "_Z18kGridToMatrixLoopyILi9ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrixLoopy_9F, *drvmod, "_Z18kGridToMatrixLoopyILi9ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrixLoopy_10T, *drvmod, "_Z18kGridToMatrixLoopyILi10ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrixLoopy_10F, *drvmod, "_Z18kGridToMatrixLoopyILi10ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrixLoopy_11T, *drvmod, "_Z18kGridToMatrixLoopyILi11ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrixLoopy_11F, *drvmod, "_Z18kGridToMatrixLoopyILi11ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrixLoopy_12T, *drvmod, "_Z18kGridToMatrixLoopyILi12ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrixLoopy_12F, *drvmod, "_Z18kGridToMatrixLoopyILi12ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrixLoopy_13T, *drvmod, "_Z18kGridToMatrixLoopyILi13ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrixLoopy_13F, *drvmod, "_Z18kGridToMatrixLoopyILi13ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrixLoopy_14T, *drvmod, "_Z18kGridToMatrixLoopyILi14ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrixLoopy_14F, *drvmod, "_Z18kGridToMatrixLoopyILi14ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrixLoopy_15T, *drvmod, "_Z18kGridToMatrixLoopyILi15ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrixLoopy_15F, *drvmod, "_Z18kGridToMatrixLoopyILi15ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrixLoopy_16T, *drvmod, "_Z18kGridToMatrixLoopyILi16ELb1EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&_gridToMatrixLoopy_16F, *drvmod, "_Z18kGridToMatrixLoopyILi16ELb0EEvPfS0_iii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    init = 1;
  }

  // mex parameters are:

  // 1. IN1
  // 2. OUT
  // 3. squareSize

  bool avoidBankConflicts = true; //hard-coded

  //IN1 is the input GPU array
  GPUtype IN1 = gm->gputype.getGPUtype(prhs[0]);

  //OUT is the output GPU array (result)
  GPUtype OUT = gm->gputype.getGPUtype(prhs[1]);

  //integer parameters
  int squareSize = (int) mxGetScalar(prhs[2]);
  //bool reverse = (bool) mxGetScalar(prhs[3]); // if 1 then MatrixToGrid

  // number of elements
  int nin1 = gm->gputype.getNumel(IN1);
  int nout = gm->gputype.getNumel(OUT);

  //dimensions 
  const int * sin1 = gm->gputype.getSize(IN1);
  const int * sout = gm->gputype.getSize(OUT);

  int imgPixels = reverse ? sout[0] : sin1[0]; //GridToMatrix -> input is images
  int numImages = reverse ? sout[1] : sin1[1]; //GridToMatrix -> input is images

  /*
  mexPrintf("images: %dx%d\n", sin1[1], sin1[0]);
  mexPrintf("targets: %dx%d\n", sout[1], sout[0]);
  mexPrintf("imgPixels: %d\n", imgPixels);
  */

  if ( floor(sqrt(float(imgPixels))) != sqrt(float(imgPixels)) )
    mexErrMsgTxt("Images not square");
  int imgSize = int(sqrt(imgPixels));

  if (squareSize > 16)
    mexErrMsgTxt("squareSize > 16");
  if (squareSize < 2)
    mexErrMsgTxt("squareSize < 2");
  if (imgSize < 1)
    mexErrMsgTxt("min imgSize: 1");
  if (imgSize > 512)
    mexErrMsgTxt("max imgSize: 512");
  if (imgSize <= squareSize)
    mexErrMsgTxt("imgSize must be > squareSize");
  if (imgSize % squareSize !=0)
    mexErrMsgTxt("imgSize must be evenly divisible by squareSize");
  if (nout != nin1)
    mexErrMsgTxt("Number of target elements must equal number of input elements");

  bool useLoopy = false;
  int SHMEM_MAX = 8192; // don't use more than this much shmem
  int THREADS_MAX = 512;

  int threadsX = imgSize;
  int threadsY = squareSize * MIN(THREADS_MAX / (squareSize*threadsX), SHMEM_MAX / (4*threadsX*squareSize)); // to avoid running out of shmem
  if (threadsY == 0) {
    threadsX = 16;
    threadsY = squareSize * MIN(THREADS_MAX / (squareSize*threadsX), SHMEM_MAX / (4*threadsX*squareSize));
    useLoopy = true;
    //mexPrintf("using loopy\n");
  }

  int shmemX = squareSize;
  int shmemPadX = avoidBankConflicts * (1 - (shmemX % 2));
  shmemX += shmemPadX;
  int shmemY = threadsX * (threadsY / squareSize);

  int loopsYPerBlock = useLoopy ? GTM_LOOPY_BLOCK_LOOPS_Y : GTM_BLOCK_LOOPS_Y;
  int blocksX = imgSize;
  int blocksY = DIVUP(numImages, loopsYPerBlock * threadsY);
  //    printf("boundary problems: %u\n", numImages % (loopsYPerBlock*threadsY) != 0);

  int shmem = 4 * shmemX * shmemY;
  if (shmem == 0 || shmem > 16300) {
    // this really shouldn't happen and i've only put this here as a precautionary measure
    // to avoid getting mysteriously wrong results.
    mexErrMsgTxt("_gtm: not enough shared memory!");
  }

  dim3 grid(blocksX, blocksY);
  dim3 threads(threadsX, threadsY);
  /*
  mexPrintf("blocks: %dx%d, threads: %dx%d\n", blocksY, blocksX, threadsY, threadsX);
  mexPrintf("using %dx%d = %d bytes of shmem\n", shmemY, shmemX, shmem);
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
  if(reverse) {
    if(!useLoopy) {
      if(squareSize == 2) {
	hostDriver(_gridToMatrix_2T, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
	//kGridToMatrix<<<true, grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
      } else if(squareSize == 3) {
	hostDriver(_gridToMatrix_3T, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 4) {
	hostDriver(_gridToMatrix_4T, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 5) {
	hostDriver(_gridToMatrix_5T, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 6) {
	hostDriver(_gridToMatrix_6T, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 7) {
	hostDriver(_gridToMatrix_7T, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 8) {
	hostDriver(_gridToMatrix_8T, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 9) {
	hostDriver(_gridToMatrix_9T, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 10) {
	hostDriver(_gridToMatrix_10T, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 11) {
	hostDriver(_gridToMatrix_11T, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 12) {
	hostDriver(_gridToMatrix_12T, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 13) {
	hostDriver(_gridToMatrix_13T, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 14) {
	hostDriver(_gridToMatrix_14T, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 15) {
	hostDriver(_gridToMatrix_15T, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 16) {
	hostDriver(_gridToMatrix_16T, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      }
    } else {
      if(squareSize == 2) {
	hostDriver(_gridToMatrixLoopy_2T, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 3) {
	hostDriver(_gridToMatrixLoopy_3T, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 4) {
	hostDriver(_gridToMatrixLoopy_4T, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 5) {
	hostDriver(_gridToMatrixLoopy_5T, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 6) {
	hostDriver(_gridToMatrixLoopy_6T, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 7) {
	hostDriver(_gridToMatrixLoopy_7T, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 8) {
	hostDriver(_gridToMatrixLoopy_8T, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 9) {
	hostDriver(_gridToMatrixLoopy_9T, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 10) {
	hostDriver(_gridToMatrixLoopy_10T, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 11) {
	hostDriver(_gridToMatrixLoopy_11T, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 12) {
	hostDriver(_gridToMatrixLoopy_12T, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 13) {
	hostDriver(_gridToMatrixLoopy_13T, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 14) {
	hostDriver(_gridToMatrixLoopy_14T, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 15) {
	hostDriver(_gridToMatrixLoopy_15T, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 16) {
	hostDriver(_gridToMatrixLoopy_16T, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      }
    }
  } else {
    if(!useLoopy) {
      if(squareSize == 2) {
	hostDriver(_gridToMatrix_2F, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 3) {
	hostDriver(_gridToMatrix_3F, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 4) {
	hostDriver(_gridToMatrix_4F, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 5) {
	hostDriver(_gridToMatrix_5F, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 6) {
	hostDriver(_gridToMatrix_6F, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 7) {
	hostDriver(_gridToMatrix_7F, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 8) {
	hostDriver(_gridToMatrix_8F, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 9) {
	hostDriver(_gridToMatrix_9F, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 10) {
	hostDriver(_gridToMatrix_10F, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 11) {
	hostDriver(_gridToMatrix_11F, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 12) {
	hostDriver(_gridToMatrix_12F, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 13) {
	hostDriver(_gridToMatrix_13F, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 14) {
	hostDriver(_gridToMatrix_14F, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 15) {
	hostDriver(_gridToMatrix_15F, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 16) {
	hostDriver(_gridToMatrix_16F, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      }
    } else {
      if(squareSize == 2) {
	hostDriver(_gridToMatrixLoopy_2F, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 3) {
	hostDriver(_gridToMatrixLoopy_3F, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 4) {
	hostDriver(_gridToMatrixLoopy_4F, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 5) {
	hostDriver(_gridToMatrixLoopy_5F, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 6) {
	hostDriver(_gridToMatrixLoopy_6F, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 7) {
	hostDriver(_gridToMatrixLoopy_7F, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 8) {
	hostDriver(_gridToMatrixLoopy_8F, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 9) {
	hostDriver(_gridToMatrixLoopy_9F, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 10) {
	hostDriver(_gridToMatrixLoopy_10F, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 11) {
	hostDriver(_gridToMatrixLoopy_11F, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 12) {
	hostDriver(_gridToMatrixLoopy_12F, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 13) {
	hostDriver(_gridToMatrixLoopy_13F, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 14) {
	hostDriver(_gridToMatrixLoopy_14F, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 15) {
	hostDriver(_gridToMatrixLoopy_15F, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      } else if(squareSize == 16) {
	hostDriver(_gridToMatrixLoopy_16F, grid, threads, shmem, imgSize, numImages*imgSize,shmemX, gpunrhs, gpuprhs);
      }
    }
  }
}

