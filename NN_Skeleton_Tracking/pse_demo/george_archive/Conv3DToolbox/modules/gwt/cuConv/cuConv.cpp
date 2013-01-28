#include <stdio.h>
#include <string.h>
#include <stdarg.h>

//added to support floor, sqrt
#include <math.h>
//added to support assert
#include <assert.h>

#ifdef UNIX
#include <stdint.h>
#endif

#include "mex.h"

// CUDA
#include "cuda.h"
#include "cuda_runtime.h"

#include "GPUmat.hh"

/* Static function definitions for each instantiation of kernel */
static CUfunction conv_bw_nofit_4x16_2per_11;
static CUfunction conv_bw_nofit_4x16_2per_13;
static CUfunction conv_bw_nofit_4x16_2per_01;
static CUfunction conv_bw_nofit_4x16_2per_03;

static CUfunction conv_bw_fit_4x16_1per_1511;
static CUfunction conv_bw_fit_4x16_1per_1513;
static CUfunction conv_bw_fit_4x16_1per_1501;
static CUfunction conv_bw_fit_4x16_1per_1503;

static CUfunction conv_bw_fit_4x16_1per_1611;
static CUfunction conv_bw_fit_4x16_1per_1613;
static CUfunction conv_bw_fit_4x16_1per_1601;
static CUfunction conv_bw_fit_4x16_1per_1603;

static CUfunction conv_bw_fit_4x16_1per_1711;
static CUfunction conv_bw_fit_4x16_1per_1713;
static CUfunction conv_bw_fit_4x16_1per_1701;
static CUfunction conv_bw_fit_4x16_1per_1703;

static CUfunction conv_bw_fit_4x16_1per_1811;
static CUfunction conv_bw_fit_4x16_1per_1813;
static CUfunction conv_bw_fit_4x16_1per_1801;
static CUfunction conv_bw_fit_4x16_1per_1803;

static CUfunction conv_bw_fit_4x16_1per_1911;
static CUfunction conv_bw_fit_4x16_1per_1913;
static CUfunction conv_bw_fit_4x16_1per_1901;
static CUfunction conv_bw_fit_4x16_1per_1903;

static CUfunction conv_bw_fit_4x16_1per_2011;
static CUfunction conv_bw_fit_4x16_1per_2013;
static CUfunction conv_bw_fit_4x16_1per_2001;
static CUfunction conv_bw_fit_4x16_1per_2003;

static CUfunction conv_bw_fit_4x16_2per_211;
static CUfunction conv_bw_fit_4x16_2per_213;
static CUfunction conv_bw_fit_4x16_2per_201;
static CUfunction conv_bw_fit_4x16_2per_203;

static CUfunction conv_bw_fit_4x16_2per_311;
static CUfunction conv_bw_fit_4x16_2per_313;
static CUfunction conv_bw_fit_4x16_2per_301;
static CUfunction conv_bw_fit_4x16_2per_303;

static CUfunction conv_bw_fit_4x16_2per_411;
static CUfunction conv_bw_fit_4x16_2per_413;
static CUfunction conv_bw_fit_4x16_2per_401;
static CUfunction conv_bw_fit_4x16_2per_403;

static CUfunction conv_bw_fit_4x16_2per_511;
static CUfunction conv_bw_fit_4x16_2per_513;
static CUfunction conv_bw_fit_4x16_2per_501;
static CUfunction conv_bw_fit_4x16_2per_503;

static CUfunction conv_bw_fit_4x16_2per_611;
static CUfunction conv_bw_fit_4x16_2per_613;
static CUfunction conv_bw_fit_4x16_2per_601;
static CUfunction conv_bw_fit_4x16_2per_603;

static CUfunction conv_bw_fit_4x16_2per_711;
static CUfunction conv_bw_fit_4x16_2per_713;
static CUfunction conv_bw_fit_4x16_2per_701;
static CUfunction conv_bw_fit_4x16_2per_703;

static CUfunction conv_bw_fit_4x16_2per_811;
static CUfunction conv_bw_fit_4x16_2per_813;
static CUfunction conv_bw_fit_4x16_2per_801;
static CUfunction conv_bw_fit_4x16_2per_803;

static CUfunction conv_bw_fit_4x16_2per_911;
static CUfunction conv_bw_fit_4x16_2per_913;
static CUfunction conv_bw_fit_4x16_2per_901;
static CUfunction conv_bw_fit_4x16_2per_903;

static CUfunction conv_bw_fit_4x16_2per_1011;
static CUfunction conv_bw_fit_4x16_2per_1013;
static CUfunction conv_bw_fit_4x16_2per_1001;
static CUfunction conv_bw_fit_4x16_2per_1003;

static CUfunction conv_bw_fit_4x16_2per_1111;
static CUfunction conv_bw_fit_4x16_2per_1113;
static CUfunction conv_bw_fit_4x16_2per_1101;
static CUfunction conv_bw_fit_4x16_2per_1103;

static CUfunction conv_bw_fit_4x16_2per_1211;
static CUfunction conv_bw_fit_4x16_2per_1213;
static CUfunction conv_bw_fit_4x16_2per_1201;
static CUfunction conv_bw_fit_4x16_2per_1203;

static CUfunction conv_bw_fit_4x16_2per_1311;
static CUfunction conv_bw_fit_4x16_2per_1313;
static CUfunction conv_bw_fit_4x16_2per_1301;
static CUfunction conv_bw_fit_4x16_2per_1303;

static CUfunction conv_bw_fit_4x16_2per_1411;
static CUfunction conv_bw_fit_4x16_2per_1413;
static CUfunction conv_bw_fit_4x16_2per_1401;
static CUfunction conv_bw_fit_4x16_2per_1403;

static int init = 0;

static GPUmat *gm;

/* Define wrappers for each of Alex's convolution kernels
   Since GPUmat uses the driver API it's easier to write these wrappers and put the driver API-related calls in here */
void _conv_bw_fit_4x16_1per(CUfunction drvfun, void* images, unsigned int images_s, void* filters, unsigned int filters_s, void* targets, unsigned int targets_s, int imgSize, dim3 grid, dim3 threads) {

  //mexPrintf("Setting up kernel\n");
  CUresult err = CUDA_SUCCESS;

  if (CUDA_SUCCESS != (err = cuFuncSetBlockShape(drvfun, threads.x, threads.y, threads.z))) {
    mexErrMsgTxt("Error in cuFuncSetBlockShape");
  }

  // add parameters
  int poffset = 0;

  if (CUDA_SUCCESS
      != cuParamSetv(drvfun, poffset, images, images_s)) {
    mexErrMsgTxt("Error in cuParamSetv");
  }
  poffset += images_s;

  if (CUDA_SUCCESS
      != cuParamSetv(drvfun, poffset, filters, filters_s)) {
    mexErrMsgTxt("Error in cuParamSetv");
  }
  poffset += filters_s;

  if (CUDA_SUCCESS
      != cuParamSetv(drvfun, poffset, targets, targets_s)) {
    mexErrMsgTxt("Error in cuParamSetv");
  }
  poffset += targets_s;

  //Next, the int imgSize
  if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, imgSize)) {
    mexErrMsgTxt("Error in cuParamSeti");
  }
  poffset += sizeof(imgSize);

  //total parameter size
  if (CUDA_SUCCESS != cuParamSetSize(drvfun, poffset)) {
    mexErrMsgTxt("Error in cuParamSetSize");
  }

  err = cuLaunchGridAsync(drvfun, grid.x, grid.y, 0);
  if (CUDA_SUCCESS != err) {
    mexErrMsgTxt("Error running kernel");
  }

}

void _conv_bw_fit_4x16_2per(CUfunction drvfun, void* images, unsigned int images_s, void* filters, unsigned int filters_s, void* targets, unsigned int targets_s, int imgSize, dim3 grid, dim3 threads) {

  //mexPrintf("Setting up kernel\n");
  CUresult err = CUDA_SUCCESS;

  if (CUDA_SUCCESS != (err = cuFuncSetBlockShape(drvfun, threads.x, threads.y, threads.z))) {
    mexErrMsgTxt("Error in cuFuncSetBlockShape");
  }

  // add parameters
  int poffset = 0;

  if (CUDA_SUCCESS
      != cuParamSetv(drvfun, poffset, images, images_s)) {
    mexErrMsgTxt("Error in cuParamSetv");
  }
  poffset += images_s;

  if (CUDA_SUCCESS
      != cuParamSetv(drvfun, poffset, filters, filters_s)) {
    mexErrMsgTxt("Error in cuParamSetv");
  }
  poffset += filters_s;

  if (CUDA_SUCCESS
      != cuParamSetv(drvfun, poffset, targets, targets_s)) {
    mexErrMsgTxt("Error in cuParamSetv");
  }
  poffset += targets_s;

  //Next, the int imgSize
  if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, imgSize)) {
    mexErrMsgTxt("Error in cuParamSeti");
  }
  poffset += sizeof(imgSize);

  //total parameter size
  if (CUDA_SUCCESS != cuParamSetSize(drvfun, poffset)) {
    mexErrMsgTxt("Error in cuParamSetSize");
  }

  err = cuLaunchGridAsync(drvfun, grid.x, grid.y, 0);
  if (CUDA_SUCCESS != err) {
    mexErrMsgTxt("Error running kernel");
  }

}

void _conv_bw_nofit_4x16_2per(CUfunction drvfun, void* images, unsigned int images_s, void* filters, unsigned int filters_s, void* targets, unsigned int targets_s, int imgSize, int filterSize, dim3 grid, dim3 threads) {

  //mexPrintf("Setting up kernel\n");
  CUresult err = CUDA_SUCCESS;

  if (CUDA_SUCCESS != (err = cuFuncSetBlockShape(drvfun, threads.x, threads.y, threads.z))) {
    mexErrMsgTxt("Error in cuFuncSetBlockShape");
  }

  // add parameters
  int poffset = 0;


  if (CUDA_SUCCESS
      != cuParamSetv(drvfun, poffset, images, images_s)) {
    mexErrMsgTxt("Error in cuParamSetv");
  }
  poffset += images_s;

  if (CUDA_SUCCESS
      != cuParamSetv(drvfun, poffset, filters, filters_s)) {
    mexErrMsgTxt("Error in cuParamSetv");
  }
  poffset += filters_s;

  if (CUDA_SUCCESS
      != cuParamSetv(drvfun, poffset, targets, targets_s)) {
    mexErrMsgTxt("Error in cuParamSetv");
  }
  poffset += targets_s;

  //Next, the int imgSize
  if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, imgSize)) {
    mexErrMsgTxt("Error in cuParamSeti");
  }
  poffset += sizeof(imgSize);

  //Next, the int filterSize
  if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, filterSize)) {
    mexErrMsgTxt("Error in cuParamSeti");
  }
  poffset += sizeof(filterSize);

  //total parameter size
  if (CUDA_SUCCESS != cuParamSetSize(drvfun, poffset)) {
    mexErrMsgTxt("Error in cuParamSetSize");
  }

  err = cuLaunchGridAsync(drvfun, grid.x, grid.y, 0);
  if (CUDA_SUCCESS != err) {
    mexErrMsgTxt("Error running kernel");
  }

}

void _convolve_bw(void *images, unsigned int images_s, void *filters, unsigned int filters_s, void *targets, unsigned int targets_s, int numCases, int numFilters, int imgSize, int filterSize, int stride, bool useDynamics = false) {

  if (stride != 1 && stride != 3)
    mexErrMsgTxt("Incorrect stride; must be 1 or 3");
  //assert(stride == 1 || stride == 3);

  int numOutputsX = imgSize - filterSize + 1;
  //    int numOutputs = numOutputsX*numOutputsX;
  bool checkOutputBounds = numOutputsX % 16 != 0;

  //static CUfunction drvfun; // float
    
  if(filterSize > 20) {
    bool checkFilterBounds = filterSize % 16 != 0;
    int blocksY = numFilters / 16, blocksX = numCases;
    dim3 grid(blocksX, blocksY);
    dim3 threads(16, 4, 8);

    if(checkFilterBounds) {
      if(stride == 1) {
	_conv_bw_nofit_4x16_2per(conv_bw_nofit_4x16_2per_11, images, images_s, filters, filters_s, targets, targets_s, imgSize, filterSize, grid, threads);

      } else {
	_conv_bw_nofit_4x16_2per(conv_bw_nofit_4x16_2per_13, images, images_s, filters, filters_s, targets, targets_s, imgSize, filterSize, grid, threads);
      } 
    } 
    else {
      if(stride ==1) {
	_conv_bw_nofit_4x16_2per(conv_bw_nofit_4x16_2per_01, images, images_s, filters, filters_s, targets, targets_s, imgSize, filterSize, grid, threads);
      } else{
	_conv_bw_nofit_4x16_2per(conv_bw_nofit_4x16_2per_03, images, images_s, filters, filters_s, targets, targets_s, imgSize, filterSize, grid, threads);
      }
    }
  } else if(filterSize > 14) {

    int blocksY = numFilters / 8, blocksX = numCases;
    dim3 grid(blocksX, blocksY);
    dim3 threads(16, 4, 8);

    /*
    mexPrintf("grid.x: %d\n",grid.x);
    mexPrintf("grid.y: %d\n",grid.y);
    mexPrintf("threads.x: %d\n",threads.x);
    mexPrintf("threads.y: %d\n",threads.y);
    mexPrintf("threads.z: %d\n",threads.z);
    */

    if(filterSize == 15) {
      if(checkOutputBounds) {
	if (stride == 1) {
	  _conv_bw_fit_4x16_1per(conv_bw_fit_4x16_1per_1511, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	} else {
	  _conv_bw_fit_4x16_1per(conv_bw_fit_4x16_1per_1513, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	}
      } else {
	if (stride == 1) {
	  _conv_bw_fit_4x16_1per(conv_bw_fit_4x16_1per_1501, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	} else {
	  _conv_bw_fit_4x16_1per(conv_bw_fit_4x16_1per_1503, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	}
      }
    } else if(filterSize == 16) {
      if(checkOutputBounds) {
	if (stride == 1) {
	  _conv_bw_fit_4x16_1per(conv_bw_fit_4x16_1per_1611, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	} else {
	  _conv_bw_fit_4x16_1per(conv_bw_fit_4x16_1per_1613, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	}
      } else {
	if (stride == 1) {
	  _conv_bw_fit_4x16_1per(conv_bw_fit_4x16_1per_1601, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	} else {
	  _conv_bw_fit_4x16_1per(conv_bw_fit_4x16_1per_1603, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	}
      }
    }
    else if(filterSize == 17) {
      if(checkOutputBounds) {
	if (stride == 1) {
	  _conv_bw_fit_4x16_1per(conv_bw_fit_4x16_1per_1711, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	} else {
	  _conv_bw_fit_4x16_1per(conv_bw_fit_4x16_1per_1713, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	}
      } else {
	if (stride == 1) {
	  _conv_bw_fit_4x16_1per(conv_bw_fit_4x16_1per_1701, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	} else {
	  _conv_bw_fit_4x16_1per(conv_bw_fit_4x16_1per_1703, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	}
      }
    }
    else if(filterSize == 18) {
      if(checkOutputBounds) {
	if (stride == 1) {
	  _conv_bw_fit_4x16_1per(conv_bw_fit_4x16_1per_1811, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	} else {
	  _conv_bw_fit_4x16_1per(conv_bw_fit_4x16_1per_1813, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	}
      } else {
	if (stride == 1) {
	  _conv_bw_fit_4x16_1per(conv_bw_fit_4x16_1per_1801, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	} else {
	  _conv_bw_fit_4x16_1per(conv_bw_fit_4x16_1per_1803, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	}
      }
	  
    }
    else if(filterSize == 19) {
      if(checkOutputBounds) {
	if (stride == 1) {
	  _conv_bw_fit_4x16_1per(conv_bw_fit_4x16_1per_1911, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	} else {
	  _conv_bw_fit_4x16_1per(conv_bw_fit_4x16_1per_1913, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	}
      } else {
	if (stride == 1) {
	  _conv_bw_fit_4x16_1per(conv_bw_fit_4x16_1per_1901, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	} else {
	  _conv_bw_fit_4x16_1per(conv_bw_fit_4x16_1per_1903, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	}
      }
    }
    else if(filterSize == 20) {
      if(checkOutputBounds) {
	if (stride == 1) {
	  _conv_bw_fit_4x16_1per(conv_bw_fit_4x16_1per_2011, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	} else {
	  _conv_bw_fit_4x16_1per(conv_bw_fit_4x16_1per_2013, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	}
      } else {
	if (stride == 1) {
	  _conv_bw_fit_4x16_1per(conv_bw_fit_4x16_1per_2001, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	} else {
	  _conv_bw_fit_4x16_1per(conv_bw_fit_4x16_1per_2003, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	}
      }
    } 
  }  else {
    if (useDynamics) {
      //later
      //mexErrMsgTxt("filterSize not supported");
    } else {
      int blocksY = numFilters / 16, blocksX = numCases;
      dim3 grid(blocksX, blocksY);
      dim3 threads(16, 4, 8);
	
      if (filterSize == 1) {
	throw "try multByScalar";
      } else if(filterSize == 2) {
	if(checkOutputBounds) {
	  if (stride == 1) {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_211, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  } else {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_213, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  }
	} else {
	  if (stride == 1) {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_201, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  } else {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_203, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  }
	}
      } else if(filterSize == 3) {
	if(checkOutputBounds) {
	  if (stride == 1) {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_311, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  } else {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_313, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  }
	} else {
	  if (stride == 1) {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_301, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  } else {
	    
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_303, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  }
	}
      }else if(filterSize == 4) {
	if(checkOutputBounds) {
	  if (stride == 1) {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_411, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  } else {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_413, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  }
	} else {
	  if (stride == 1) {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_401, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  } else {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_403, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  }
	}
      }else if(filterSize == 5) {
	if(checkOutputBounds) {
	  if (stride == 1) {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_511, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  } else {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_513, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  }
	} else {
	  if (stride == 1) {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_501, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  } else {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_503, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  }
	}
      }else if(filterSize == 6) {
	if(checkOutputBounds) {
	  if (stride == 1) {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_611, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  } else {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_613, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  }
	} else {
	  if (stride == 1) {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_601, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  } else {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_603, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  }
	}
      }else if(filterSize == 7) {
	if(checkOutputBounds) {
	  if (stride == 1) {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_711, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  } else {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_713, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  }
	} else {
	  if (stride == 1) {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_701, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  } else {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_703, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  }
	}
      }else if(filterSize == 8) {
	if(checkOutputBounds) {
	  if (stride == 1) {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_811, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  } else {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_813, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  }
	} else {
	  if (stride == 1) {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_801, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  } else {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_803, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  }
	}
      }else if(filterSize == 9) {
	if(checkOutputBounds) {
	  if (stride == 1) {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_911, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  } else {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_913, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  }
	} else {
	  if (stride == 1) {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_901, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  } else {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_903, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  }
	}
      }else if(filterSize == 10) {
	if(checkOutputBounds) {
	  if (stride == 1) {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_1011, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  } else {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_1013, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  }
	} else {
	  if (stride == 1) {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_1001, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  } else {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_1003, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  }
	}
      }else if(filterSize == 11) {
	if(checkOutputBounds) {
	  if (stride == 1) {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_1111, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  } else {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_1113, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  }
	} else {
	  if (stride == 1) {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_1101, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  } else {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_1103, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  }
	}
      }else if(filterSize == 12) {
	if(checkOutputBounds) {
	  if (stride == 1) {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_1211, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  } else {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_1213, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  }
	} else {
	  if (stride == 1) {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_1201, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  } else {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_1203, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  }
	}
      } else if(filterSize == 13) {
	if(checkOutputBounds) {
	  if (stride == 1) {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_1311, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  } else {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_1313, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  }
	} else {
	  if (stride == 1) {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_1301, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  } else {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_1303, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  }
	}
      } else if(filterSize == 14) {
	if(checkOutputBounds) {
	  if (stride == 1) {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_1411, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  } else {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_1413, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  }
	} else {
	  if (stride == 1) {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_1401, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  } else {
	    _conv_bw_fit_4x16_2per(conv_bw_fit_4x16_2per_1403, images, images_s, filters, filters_s, targets, targets_s, imgSize, grid, threads);
	  }
	}
      }
    }
  } 
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  CUresult cudastatus = CUDA_SUCCESS;

  if (nrhs != 3)
    //Should pass in images, filters, targets
    mexErrMsgTxt("Wrong number of arguments");

  if (init == 0) {
    // Initialize function
    mexLock();

    // load GPUmat
    gm = gmGetGPUmat();

    /* Set up modules here, so they only need to be loaded the first time the function is called */
    CUmodule *drvmod = gmGetModule("conv");

    //load appropriate GPU kernel (mangled name)
    CUresult status;
    status = cuModuleGetFunction(&conv_bw_nofit_4x16_2per_11, *drvmod, "_Z23conv_bw_nofit_4x16_2perILb1ELi1EEvPfS0_S0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_nofit_4x16_2per_13, *drvmod, "_Z23conv_bw_nofit_4x16_2perILb1ELi3EEvPfS0_S0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_nofit_4x16_2per_01, *drvmod, "_Z23conv_bw_nofit_4x16_2perILb0ELi1EEvPfS0_S0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_nofit_4x16_2per_03, *drvmod, "_Z23conv_bw_nofit_4x16_2perILb0ELi3EEvPfS0_S0_ii");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }

    status = cuModuleGetFunction(&conv_bw_fit_4x16_1per_1511, *drvmod, "_Z21conv_bw_fit_4x16_1perILi15ELb1ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_1per_1513, *drvmod, "_Z21conv_bw_fit_4x16_1perILi15ELb1ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_1per_1501, *drvmod, "_Z21conv_bw_fit_4x16_1perILi15ELb0ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_1per_1503, *drvmod, "_Z21conv_bw_fit_4x16_1perILi15ELb0ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_1per_1611, *drvmod, "_Z21conv_bw_fit_4x16_1perILi16ELb1ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_1per_1613, *drvmod, "_Z21conv_bw_fit_4x16_1perILi16ELb1ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_1per_1601, *drvmod, "_Z21conv_bw_fit_4x16_1perILi16ELb0ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_1per_1603, *drvmod, "_Z21conv_bw_fit_4x16_1perILi16ELb0ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_1per_1711, *drvmod, "_Z21conv_bw_fit_4x16_1perILi17ELb1ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_1per_1713, *drvmod, "_Z21conv_bw_fit_4x16_1perILi17ELb1ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_1per_1701, *drvmod, "_Z21conv_bw_fit_4x16_1perILi17ELb0ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_1per_1703, *drvmod, "_Z21conv_bw_fit_4x16_1perILi17ELb0ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_1per_1811, *drvmod, "_Z21conv_bw_fit_4x16_1perILi18ELb1ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_1per_1813, *drvmod, "_Z21conv_bw_fit_4x16_1perILi18ELb1ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_1per_1801, *drvmod, "_Z21conv_bw_fit_4x16_1perILi18ELb0ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_1per_1803, *drvmod, "_Z21conv_bw_fit_4x16_1perILi18ELb0ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_1per_1911, *drvmod, "_Z21conv_bw_fit_4x16_1perILi19ELb1ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_1per_1913, *drvmod, "_Z21conv_bw_fit_4x16_1perILi19ELb1ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_1per_1901, *drvmod, "_Z21conv_bw_fit_4x16_1perILi19ELb0ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_1per_1903, *drvmod, "_Z21conv_bw_fit_4x16_1perILi19ELb0ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_1per_2011, *drvmod, "_Z21conv_bw_fit_4x16_1perILi20ELb1ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_1per_2013, *drvmod, "_Z21conv_bw_fit_4x16_1perILi20ELb1ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_1per_2001, *drvmod, "_Z21conv_bw_fit_4x16_1perILi20ELb0ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_1per_2003, *drvmod, "_Z21conv_bw_fit_4x16_1perILi20ELb0ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_211, *drvmod, "_Z21conv_bw_fit_4x16_2perILi2ELb1ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_213, *drvmod, "_Z21conv_bw_fit_4x16_2perILi2ELb1ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_201, *drvmod, "_Z21conv_bw_fit_4x16_2perILi2ELb0ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_203, *drvmod, "_Z21conv_bw_fit_4x16_2perILi2ELb0ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_311, *drvmod, "_Z21conv_bw_fit_4x16_2perILi3ELb1ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_313, *drvmod, "_Z21conv_bw_fit_4x16_2perILi3ELb1ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_301, *drvmod, "_Z21conv_bw_fit_4x16_2perILi3ELb0ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_303, *drvmod, "_Z21conv_bw_fit_4x16_2perILi3ELb0ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_411, *drvmod, "_Z21conv_bw_fit_4x16_2perILi4ELb1ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_413, *drvmod, "_Z21conv_bw_fit_4x16_2perILi4ELb1ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_401, *drvmod, "_Z21conv_bw_fit_4x16_2perILi4ELb0ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_403, *drvmod, "_Z21conv_bw_fit_4x16_2perILi4ELb0ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_511, *drvmod, "_Z21conv_bw_fit_4x16_2perILi5ELb1ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_513, *drvmod, "_Z21conv_bw_fit_4x16_2perILi5ELb1ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_501, *drvmod, "_Z21conv_bw_fit_4x16_2perILi5ELb0ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_503, *drvmod, "_Z21conv_bw_fit_4x16_2perILi5ELb0ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_611, *drvmod, "_Z21conv_bw_fit_4x16_2perILi6ELb1ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_613, *drvmod, "_Z21conv_bw_fit_4x16_2perILi6ELb1ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_601, *drvmod, "_Z21conv_bw_fit_4x16_2perILi6ELb0ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_603, *drvmod, "_Z21conv_bw_fit_4x16_2perILi6ELb0ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_711, *drvmod, "_Z21conv_bw_fit_4x16_2perILi7ELb1ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_713, *drvmod, "_Z21conv_bw_fit_4x16_2perILi7ELb1ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_701, *drvmod, "_Z21conv_bw_fit_4x16_2perILi7ELb0ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_703, *drvmod, "_Z21conv_bw_fit_4x16_2perILi7ELb0ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_811, *drvmod, "_Z21conv_bw_fit_4x16_2perILi8ELb1ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_813, *drvmod, "_Z21conv_bw_fit_4x16_2perILi8ELb1ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_801, *drvmod, "_Z21conv_bw_fit_4x16_2perILi8ELb0ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_803, *drvmod, "_Z21conv_bw_fit_4x16_2perILi8ELb0ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_911, *drvmod, "_Z21conv_bw_fit_4x16_2perILi9ELb1ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_913, *drvmod, "_Z21conv_bw_fit_4x16_2perILi9ELb1ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_901, *drvmod, "_Z21conv_bw_fit_4x16_2perILi9ELb0ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_903, *drvmod, "_Z21conv_bw_fit_4x16_2perILi9ELb0ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_1011, *drvmod, "_Z21conv_bw_fit_4x16_2perILi10ELb1ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_1013, *drvmod, "_Z21conv_bw_fit_4x16_2perILi10ELb1ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_1001, *drvmod, "_Z21conv_bw_fit_4x16_2perILi10ELb0ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_1003, *drvmod, "_Z21conv_bw_fit_4x16_2perILi10ELb0ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_1111, *drvmod, "_Z21conv_bw_fit_4x16_2perILi11ELb1ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_1113, *drvmod, "_Z21conv_bw_fit_4x16_2perILi11ELb1ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_1101, *drvmod, "_Z21conv_bw_fit_4x16_2perILi11ELb0ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_1103, *drvmod, "_Z21conv_bw_fit_4x16_2perILi11ELb0ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_1211, *drvmod, "_Z21conv_bw_fit_4x16_2perILi12ELb1ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_1213, *drvmod, "_Z21conv_bw_fit_4x16_2perILi12ELb1ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_1201, *drvmod, "_Z21conv_bw_fit_4x16_2perILi12ELb0ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_1203, *drvmod, "_Z21conv_bw_fit_4x16_2perILi12ELb0ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_1311, *drvmod, "_Z21conv_bw_fit_4x16_2perILi13ELb1ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_1313, *drvmod, "_Z21conv_bw_fit_4x16_2perILi13ELb1ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_1301, *drvmod, "_Z21conv_bw_fit_4x16_2perILi13ELb0ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_1303, *drvmod, "_Z21conv_bw_fit_4x16_2perILi13ELb0ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_1411, *drvmod, "_Z21conv_bw_fit_4x16_2perILi14ELb1ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_1413, *drvmod, "_Z21conv_bw_fit_4x16_2perILi14ELb1ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_1401, *drvmod, "_Z21conv_bw_fit_4x16_2perILi14ELb0ELi1EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }
    status = cuModuleGetFunction(&conv_bw_fit_4x16_2per_1403, *drvmod, "_Z21conv_bw_fit_4x16_2perILi14ELb0ELi3EEvPfS0_S0_i");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }

    init = 1;
  }

  // mex parameters are:

  // 1. images
  // 2. filters
  // 3. targets

  //IN1 is the input GPU array
  GPUtype IN1 = gm->gputype.getGPUtype(prhs[0]);

  //IN2 is the input GPU array
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

  //double dImgSize = sqrt(sin1[1]);
  //double dFilterSize = sqrt(sin2[1]);
  double dImgSize = sqrt(sin1[0]);
  double dFilterSize = sqrt(sin2[0]);

  //make sure images, filters are square

  if (dImgSize != floor(dImgSize))
        mexErrMsgTxt("Images are not square");
  if (dFilterSize != floor(dFilterSize))
        mexErrMsgTxt("Images are not square");
  //assert(dImgSize == floor(dImgSize));
  //assert(dFilterSize == floor(dFilterSize));
  int imgSize = int(dImgSize);
  int filterSize = int(dFilterSize);
  //int numCases = sin1[0];
  //int numFilters = sin2[0];
  int numCases = sin1[1];
  int numFilters = sin2[1];

  int numOutputsX = imgSize - filterSize + 1;
  int numOutputs = numOutputsX * numOutputsX;

  //some checks
  if (numFilters % 16 != 0)
    mexErrMsgTxt("Number of filters must be a multiple of 16");
  //assert(numFilters % 16 == 0);
  if (nout != numOutputs * numFilters * numCases)
    mexErrMsgTxt("Output dimensions not consistent");
  //assert(nout == numOutputs * numFilters * numCases);

  /*
  mexPrintf("imgSize: %d\n",imgSize);
  mexPrintf("filterSize: %d\n",filterSize);
  mexPrintf("numCases: %d\n",numCases);
  mexPrintf("numFilters: %d\n",numFilters);
  mexPrintf("numOutputsX: %d\n",numOutputsX);
  */

  gpuTYPE_t tin1 = gm->gputype.getType(IN1);
  gpuTYPE_t tin2 = gm->gputype.getType(IN2);
  gpuTYPE_t tout = gm->gputype.getType(OUT);

  
  // check input/out types
  if (tin1 != gpuFLOAT ) 
    mexErrMsgTxt("Currently only gpuFLOAT type supported.");

  if (tin1!=tin2)
    mexErrMsgTxt("Input arguments must be of the same type.");

  if (tin1!=tout)
    mexErrMsgTxt("Input and output arguments must be of the same type.");

  // I need the pointers to GPU memory
  CUdeviceptr d_IN1  = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(IN1));
  CUdeviceptr d_IN2  = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(IN2));
  CUdeviceptr d_OUT = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(OUT));

  //last argument is stride (1 for bw images)
  _convolve_bw(&d_IN1, sizeof(d_IN1), &d_IN2, sizeof(d_IN2), &d_OUT, sizeof(d_OUT), numCases, numFilters, imgSize, filterSize, 1);  



}
