#include <assert.h>
//#include <cublas.h>
//#include <cutil_inline.h>
//#include <stdlib.h>
#include <stdio.h>
//#include <fstream>
//#include <iostream>
#include <algorithm>
#include "nvmatrix.cuh"


void NVMatrix::_init(unsigned int numRows, unsigned int numCols) {
    _numRows = numRows;
    _numCols = numCols;
    _numElements = numRows * numCols;
    _ownsData = true;
    /*
     * By default, new matrices are in column-major order because that's how CUBLAS likes it.
     */
    _isTrans = true;
    _devData = NULL;
    if (_numElements > 0) {
        cublasAlloc(_numElements, sizeof(float), (void**) &_devData);
        checkCublasError("!!!! device memory allocation error\n");
    }
}

NVMatrix::NVMatrix() {
    _init(0, 0);
}

NVMatrix::NVMatrix(bool isTrans) {
    _init(0, 0);
    setTrans(isTrans);
}

NVMatrix::NVMatrix(int numRows, int numCols, bool isTrans) {
    _init(numRows, numCols);
    setTrans(isTrans);
}

NVMatrix::NVMatrix(const NVMatrix& like, bool copy) {
    _init(like.getNumRows(), like.getNumCols());
    _isTrans = like.isTrans();
    if(copy) {
        copyFromDevice(like);
    }
}

/*
 * Initializes NVMatrix with same dimensions as given matrix but
 * does not copy any data.
 */
NVMatrix::NVMatrix(const NVMatrix& like) {
    _init(like.getNumRows(), like.getNumCols());
    _isTrans = like.isTrans();
}


/* Alex's constructor to make NVMatrix from data alreay on device 
   (i.e. from GPUsingle) */
NVMatrix::NVMatrix(float* devData, int numRows, int numCols, bool isTrans) {
    _numRows = numRows;
    _numCols = numCols;
    _numElements = numRows * numCols;
    _ownsData = false;
    _devData = devData;
    _isTrans = isTrans;
}

/* Note that first and last line have been commented out for now */
void NVMatrix::addScalar(float scalar, NVMatrix& target) {
  //target.resize(*this);
    kAddScalar<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(_devData, scalar, target._devData,_numElements);
    //cutilCheckMsg("Kernel execution failed");
}

NVMatrix::~NVMatrix() {
    if(_ownsData && _numElements > 0) {
        cublasStatus status = cublasFree(_devData);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "!!!! memory free error\n");
            exit(EXIT_FAILURE);
        }
    }
}


void NVMatrix::copyFromDevice(const NVMatrix& devMatrix) {
    assert(isSameDims(devMatrix));
    cublasScopy(_numElements,devMatrix._devData, 1, _devData,1);
    checkCublasError("cublasScopy failed");
    _isTrans = devMatrix.isTrans();
}

void NVMatrix::copyFromDevice(const NVMatrix& devMatrix, bool resizeTarget) {
    if (resizeTarget) {
        resize(devMatrix);
    }
    copyFromDevice(devMatrix);
}

/*
 * Guaranteed to not change the data if the number of elements doesn't change.
 * So you can use this to "reshape" a matrix.
 */
bool NVMatrix::resize(int numRows, int numCols) {
    bool reallocated = false;
    if (numRows != _numRows || numCols != _numCols) {
        assert(_ownsData);
        if (_numElements != numRows * numCols) {
            cublasStatus status = cublasFree(_devData);
            if (status != CUBLAS_STATUS_SUCCESS) {
                fprintf(stderr, "!!!! memory free error\n");
                exit(EXIT_FAILURE);
            }
            status = cublasAlloc(numCols * numRows, sizeof(float), (void**) &_devData);
            if (status != CUBLAS_STATUS_SUCCESS) {
                fprintf(stderr, "!!!! device memory allocation error\n");
                exit(EXIT_FAILURE);
            }
            reallocated = true;
        }
        _numRows = numRows;
        _numCols = numCols;
        _numElements = numRows * numCols;
    }
    return reallocated;
}

bool NVMatrix::resize(const NVMatrix& like) {
    bool r = resize(like.getNumRows(), like.getNumCols());
    _isTrans = like._isTrans;
    return r;
}


/*
 * num threads per block is ignored when summing rows (axis=1) because
 * it has to be a power of 2.
 */
void NVMatrix::aggregate(int axis, NVMatrix& target, int numThreadsPerBlock, NVMatrix::AGGREGATIONS agg) {
    assert(&target != this);
    unsigned int width = _isTrans ? _numRows : _numCols;
    const int height = _isTrans ? _numCols : _numRows;

    target.setTrans(_isTrans);
    assert(width > 0);
    assert(height > 0);
    if(axis == 0 && !_isTrans || axis == 1 && _isTrans) { //col sum
        target.resize(!_isTrans ? 1 : _numRows, !_isTrans ? _numCols : 1);
        const unsigned int numBlocks = (width + numThreadsPerBlock - 1) / numThreadsPerBlock;
        assert(numBlocks * numThreadsPerBlock >= width);
        assert(numBlocks < NUM_BLOCKS_MAX);
//        target.resize(1, width);
        if(agg == NVMatrix::MAX) {
            kDumbMaxCols<<<numBlocks,numThreadsPerBlock>>>(_devData, target._devData, width, height);
        } else if(agg == NVMatrix::SUM) {
            kDumbSumCols<<<numBlocks,numThreadsPerBlock>>>(_devData, target._devData, width, height);
        }
        //cutilCheckMsg("Kernel execution failed");
    } else { // row sum
        target.resize(_isTrans ? 1 : _numRows, _isTrans ? _numCols : 1);
        if (width > 1) {
            NVMatrix *prevSum = this;

            while (prevSum->getLeadingDim()  > 1) {
                int numBlocksX, numBlocksY, numThreadsX, numThreadsY;
                bool doLinearAgg = height >= 16384;
//                doQuickAgg = !doQuickAgg;

                if(doLinearAgg) { // call the special short aggregation functions
                    numBlocksX = 1;
                    numBlocksY = DIVUP(height, AGG_SHORT_ROWS_THREADS_Y*AGG_SHORT_ROWS_LOOPS_Y);
                    numThreadsX = AGG_SHORT_ROWS_THREADS_X;
                    numThreadsY = AGG_SHORT_ROWS_THREADS_Y;
                    while(numBlocksY > NUM_BLOCKS_MAX) {
                        numBlocksY = DIVUP(numBlocksY,2);
                        numBlocksX *= 2;
                    }
                } else {
                    numThreadsX = width <= 64 ? 32 : (width <= 128 ? 64 : (width <= 256 ? 128 : (width <= 512 ? 256 : 512)));
                    numThreadsY = 1;
                    numBlocksX = DIVUP(width, 2*numThreadsX);
                    numBlocksY = std::min(height, NUM_BLOCKS_MAX);
                }

                dim3 grid(numBlocksX, numBlocksY), threads(numThreadsX, numThreadsY);
                assert(numBlocksX <= NUM_BLOCKS_MAX);
                assert(numBlocksY <= NUM_BLOCKS_MAX);
//                printf("%d %d %d %d %d \n", numThreadsX, numThreadsY, numBlocksX, numBlocksY, numBlocksZ);

                NVMatrix *nvSumAccum = target.getFollowingDim() == height && target.getLeadingDim() == numBlocksX ? &target : new NVMatrix(height, numBlocksX, false);
//                printf("target size: %dx%d\n", target.getNumRows(), target.getNumCols());
//                printf("liear agg: %d, width: %d, height: %d\n", doLinearAgg, width, height);
//                printf("accum is target: %d\n", nvSumAccum == &target);
                if(agg == NVMatrix::MAX) {
                    if(doLinearAgg) {
                        if(width <= 16) {
                            if(width <= 4) {
                                kAggShortRows<AGG_MAX, 1, 4><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                            } else if(width <= 8) {
                                kAggShortRows<AGG_MAX, 1, 8><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                            } else if(width <= 12) {
                                kAggShortRows<AGG_MAX, 1, 12><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                            } else {
                                kAggShortRows<AGG_MAX, 1, 16><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                            }

                        } else if(width <= 32) {
                            kAggShortRows<AGG_MAX, 2, AGG_SHORT_ROWS_THREADS_X><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                        } else if(width <= 48){
                            kAggShortRows<AGG_MAX, 3, AGG_SHORT_ROWS_THREADS_X><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                        } else if(width <= 64){
                            kAggShortRows<AGG_MAX, 4, AGG_SHORT_ROWS_THREADS_X><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                        } else {
                            kAggShortRows2<AGG_MAX><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                        }
                    } else if(width <= 64) {
                        kMaxRows<32><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                                   width, height, nvSumAccum->getLeadingDim());
                    } else if(width <= 128) {
                        kMaxRows<64><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                                   width, height, nvSumAccum->getLeadingDim());
                    } else if(width <= 256) {
                        kMaxRows<128><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                                   width, height, nvSumAccum->getLeadingDim());
                    } else if(width <= 512) {
                        kMaxRows<256><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                                   width, height, nvSumAccum->getLeadingDim());
                    } else {
                        kMaxRows<512><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                                   width, height, nvSumAccum->getLeadingDim());
                    }
                } else if(agg == NVMatrix::SUM) {
                    if(doLinearAgg) {
                        if(width <= 16) {
                            if(width <= 4) {
                                kAggShortRows<AGG_SUM, 1, 4><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                            } else if(width <= 8) {
                                kAggShortRows<AGG_SUM, 1, 8><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                            } else if(width <= 12) {
                                kAggShortRows<AGG_SUM, 1, 12><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                            } else {
                                kAggShortRows<AGG_SUM, 1, 16><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                            }
                        } else if(width <= 32) {
                            kAggShortRows<AGG_SUM, 2, AGG_SHORT_ROWS_THREADS_X><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                        } else if(width <= 48) {
                            kAggShortRows<AGG_SUM, 3, AGG_SHORT_ROWS_THREADS_X><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                        } else if(width <= 64){
                            kAggShortRows<AGG_SUM, 4, AGG_SHORT_ROWS_THREADS_X><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                        } else {
                            kAggShortRows2<AGG_SUM><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                        }
                    } else if (width <= 64) {
                        kSumRows<32><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                width, height, nvSumAccum->getLeadingDim());
                    } else if (width <= 128) {
                        kSumRows<64><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                width, height, nvSumAccum->getLeadingDim());
                    } else if (width <= 256) {
                        kSumRows<128><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                width, height, nvSumAccum->getLeadingDim());
                    } else if (width <= 512) {
                        kSumRows<256><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                width, height, nvSumAccum->getLeadingDim());
                    } else {
                        kSumRows<512><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                width, height, nvSumAccum->getLeadingDim());
                    }
                }
                //cutilCheckMsg("Kernel execution failed");
                cudaThreadSynchronize();
                width = numBlocksX;

                if (prevSum != this) {
                    delete prevSum;
                }
                prevSum = nvSumAccum;
            }
//            if (_isTrans) {
//                prevSum->_numCols = prevSum->_numRows;
//                prevSum->_numRows = 1;
//            }
//            target.copyFromDevice(*prevSum);
//            delete prevSum;
        } else {
            target.resize(*this);
            target.copyFromDevice(*this);
        }
    }
}

/*
 * num threads per block is ignored when summing rows (axis=1) because
 * it has to be a power of 2.
 % Like aggregate but does argmax as well as max
 */
void NVMatrix::aggregate2(int axis, NVMatrix& target, NVMatrix& argmax, int numThreadsPerBlock, NVMatrix::AGGREGATIONS agg) {
    assert(&target != this);
    unsigned int width = _isTrans ? _numRows : _numCols;
    const int height = _isTrans ? _numCols : _numRows;

    target.setTrans(_isTrans);
    assert(width > 0);
    assert(height > 0);
    if(axis == 0 && !_isTrans || axis == 1 && _isTrans) { //col sum
        target.resize(!_isTrans ? 1 : _numRows, !_isTrans ? _numCols : 1);
        const unsigned int numBlocks = (width + numThreadsPerBlock - 1) / numThreadsPerBlock;
        assert(numBlocks * numThreadsPerBlock >= width);
        assert(numBlocks < NUM_BLOCKS_MAX);
//        target.resize(1, width);
        if(agg == NVMatrix::MAX) {
	  kDumbMaxCols2<<<numBlocks,numThreadsPerBlock>>>(_devData, target._devData, argmax._devData, width, height);
        } 
        //cutilCheckMsg("Kernel execution failed");
    } else { // row sum
        target.resize(_isTrans ? 1 : _numRows, _isTrans ? _numCols : 1);
        if (width > 1) {
            NVMatrix *prevSum = this;

            while (prevSum->getLeadingDim()  > 1) {
                int numBlocksX, numBlocksY, numThreadsX, numThreadsY;
                bool doLinearAgg = height >= 16384;
//                doQuickAgg = !doQuickAgg;

                if(doLinearAgg) { // call the special short aggregation functions
                    numBlocksX = 1;
                    numBlocksY = DIVUP(height, AGG_SHORT_ROWS_THREADS_Y*AGG_SHORT_ROWS_LOOPS_Y);
                    numThreadsX = AGG_SHORT_ROWS_THREADS_X;
                    numThreadsY = AGG_SHORT_ROWS_THREADS_Y;
                    while(numBlocksY > NUM_BLOCKS_MAX) {
                        numBlocksY = DIVUP(numBlocksY,2);
                        numBlocksX *= 2;
                    }
                } else {
                    numThreadsX = width <= 64 ? 32 : (width <= 128 ? 64 : (width <= 256 ? 128 : (width <= 512 ? 256 : 512)));
                    numThreadsY = 1;
                    numBlocksX = DIVUP(width, 2*numThreadsX);
                    numBlocksY = std::min(height, NUM_BLOCKS_MAX);
                }

                dim3 grid(numBlocksX, numBlocksY), threads(numThreadsX, numThreadsY);
                assert(numBlocksX <= NUM_BLOCKS_MAX);
                assert(numBlocksY <= NUM_BLOCKS_MAX);
//                printf("%d %d %d %d %d \n", numThreadsX, numThreadsY, numBlocksX, numBlocksY, numBlocksZ);

                NVMatrix *nvSumAccum = target.getFollowingDim() == height && target.getLeadingDim() == numBlocksX ? &target : new NVMatrix(height, numBlocksX, false);
		//holds argmax
                NVMatrix *nvIdxAccum = target.getFollowingDim() == height && target.getLeadingDim() == numBlocksX ? &target : new NVMatrix(height, numBlocksX, false);

//                printf("target size: %dx%d\n", target.getNumRows(), target.getNumCols());
//                printf("liear agg: %d, width: %d, height: %d\n", doLinearAgg, width, height);
//                printf("accum is target: %d\n", nvSumAccum == &target);
                if(agg == NVMatrix::MAX) {
                    if(doLinearAgg) {
                        if(width <= 16) {
                            if(width <= 4) {
                                kAggShortRows<AGG_MAX, 1, 4><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                            } else if(width <= 8) {
                                kAggShortRows<AGG_MAX, 1, 8><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                            } else if(width <= 12) {
                                kAggShortRows<AGG_MAX, 1, 12><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                            } else {
                                kAggShortRows<AGG_MAX, 1, 16><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                            }

                        } else if(width <= 32) {
                            kAggShortRows<AGG_MAX, 2, AGG_SHORT_ROWS_THREADS_X><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                        } else if(width <= 48){
                            kAggShortRows<AGG_MAX, 3, AGG_SHORT_ROWS_THREADS_X><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                        } else if(width <= 64){
                            kAggShortRows<AGG_MAX, 4, AGG_SHORT_ROWS_THREADS_X><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                        } else {
                            kAggShortRows2<AGG_MAX><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                        }
                    } else if(width <= 64) {
                        kMaxRows<32><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                                   width, height, nvSumAccum->getLeadingDim());
                    } else if(width <= 128) {
                        kMaxRows<64><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                                   width, height, nvSumAccum->getLeadingDim());
                    } else if(width <= 256) {
                        kMaxRows<128><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                                   width, height, nvSumAccum->getLeadingDim());
                    } else if(width <= 512) {
                        kMaxRows<256><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                                   width, height, nvSumAccum->getLeadingDim());
                    } else {
                        kMaxRows<512><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                                   width, height, nvSumAccum->getLeadingDim());
                    }
                } 
                //cutilCheckMsg("Kernel execution failed");
                cudaThreadSynchronize();
                width = numBlocksX;

                if (prevSum != this) {
                    delete prevSum;
                }
                prevSum = nvSumAccum;
            }
//            if (_isTrans) {
//                prevSum->_numCols = prevSum->_numRows;
//                prevSum->_numRows = 1;
//            }
//            target.copyFromDevice(*prevSum);
//            delete prevSum;
        } else {
            target.resize(*this);
            target.copyFromDevice(*this);
        }
    }
}

/*
 * num threads per block is ignored when summing rows (axis=1) because
 * it has to be a power of 2.
 % Like aggregate2 but returns sign(x)*abs(max(x)) and arg of this
 */
void NVMatrix::aggregate3(int axis, NVMatrix& target, NVMatrix& argmax, int numThreadsPerBlock, NVMatrix::AGGREGATIONS agg) {
    assert(&target != this);
    unsigned int width = _isTrans ? _numRows : _numCols;
    const int height = _isTrans ? _numCols : _numRows;

    target.setTrans(_isTrans);
    assert(width > 0);
    assert(height > 0);
    if(axis == 0 && !_isTrans || axis == 1 && _isTrans) { //col sum
        target.resize(!_isTrans ? 1 : _numRows, !_isTrans ? _numCols : 1);
        const unsigned int numBlocks = (width + numThreadsPerBlock - 1) / numThreadsPerBlock;
        assert(numBlocks * numThreadsPerBlock >= width);
        assert(numBlocks < NUM_BLOCKS_MAX);
//        target.resize(1, width);
        if(agg == NVMatrix::MAX) {
	  kDumbMaxCols3<<<numBlocks,numThreadsPerBlock>>>(_devData, target._devData, argmax._devData, width, height);
        } 
        //cutilCheckMsg("Kernel execution failed");
    } else { // row sum
        target.resize(_isTrans ? 1 : _numRows, _isTrans ? _numCols : 1);
        if (width > 1) {
            NVMatrix *prevSum = this;

            while (prevSum->getLeadingDim()  > 1) {
                int numBlocksX, numBlocksY, numThreadsX, numThreadsY;
                bool doLinearAgg = height >= 16384;
//                doQuickAgg = !doQuickAgg;

                if(doLinearAgg) { // call the special short aggregation functions
                    numBlocksX = 1;
                    numBlocksY = DIVUP(height, AGG_SHORT_ROWS_THREADS_Y*AGG_SHORT_ROWS_LOOPS_Y);
                    numThreadsX = AGG_SHORT_ROWS_THREADS_X;
                    numThreadsY = AGG_SHORT_ROWS_THREADS_Y;
                    while(numBlocksY > NUM_BLOCKS_MAX) {
                        numBlocksY = DIVUP(numBlocksY,2);
                        numBlocksX *= 2;
                    }
                } else {
                    numThreadsX = width <= 64 ? 32 : (width <= 128 ? 64 : (width <= 256 ? 128 : (width <= 512 ? 256 : 512)));
                    numThreadsY = 1;
                    numBlocksX = DIVUP(width, 2*numThreadsX);
                    numBlocksY = std::min(height, NUM_BLOCKS_MAX);
                }

                dim3 grid(numBlocksX, numBlocksY), threads(numThreadsX, numThreadsY);
                assert(numBlocksX <= NUM_BLOCKS_MAX);
                assert(numBlocksY <= NUM_BLOCKS_MAX);
//                printf("%d %d %d %d %d \n", numThreadsX, numThreadsY, numBlocksX, numBlocksY, numBlocksZ);

                NVMatrix *nvSumAccum = target.getFollowingDim() == height && target.getLeadingDim() == numBlocksX ? &target : new NVMatrix(height, numBlocksX, false);
		//holds argmax
                NVMatrix *nvIdxAccum = target.getFollowingDim() == height && target.getLeadingDim() == numBlocksX ? &target : new NVMatrix(height, numBlocksX, false);

//                printf("target size: %dx%d\n", target.getNumRows(), target.getNumCols());
//                printf("liear agg: %d, width: %d, height: %d\n", doLinearAgg, width, height);
//                printf("accum is target: %d\n", nvSumAccum == &target);
                if(agg == NVMatrix::MAX) {
                    if(doLinearAgg) {
                        if(width <= 16) {
                            if(width <= 4) {
                                kAggShortRows<AGG_MAX, 1, 4><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                            } else if(width <= 8) {
                                kAggShortRows<AGG_MAX, 1, 8><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                            } else if(width <= 12) {
                                kAggShortRows<AGG_MAX, 1, 12><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                            } else {
                                kAggShortRows<AGG_MAX, 1, 16><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                            }

                        } else if(width <= 32) {
                            kAggShortRows<AGG_MAX, 2, AGG_SHORT_ROWS_THREADS_X><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                        } else if(width <= 48){
                            kAggShortRows<AGG_MAX, 3, AGG_SHORT_ROWS_THREADS_X><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                        } else if(width <= 64){
                            kAggShortRows<AGG_MAX, 4, AGG_SHORT_ROWS_THREADS_X><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                        } else {
                            kAggShortRows2<AGG_MAX><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                        }
                    } else if(width <= 64) {
                        kMaxRows<32><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                                   width, height, nvSumAccum->getLeadingDim());
                    } else if(width <= 128) {
                        kMaxRows<64><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                                   width, height, nvSumAccum->getLeadingDim());
                    } else if(width <= 256) {
                        kMaxRows<128><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                                   width, height, nvSumAccum->getLeadingDim());
                    } else if(width <= 512) {
                        kMaxRows<256><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                                   width, height, nvSumAccum->getLeadingDim());
                    } else {
                        kMaxRows<512><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                                   width, height, nvSumAccum->getLeadingDim());
                    }
                } 
                //cutilCheckMsg("Kernel execution failed");
                cudaThreadSynchronize();
                width = numBlocksX;

                if (prevSum != this) {
                    delete prevSum;
                }
                prevSum = nvSumAccum;
            }
//            if (_isTrans) {
//                prevSum->_numCols = prevSum->_numRows;
//                prevSum->_numRows = 1;
//            }
//            target.copyFromDevice(*prevSum);
//            delete prevSum;
        } else {
            target.resize(*this);
            target.copyFromDevice(*this);
        }
    }
}

void NVMatrix::max(int axis, NVMatrix& target) {
    if(axis == 0 && !_isTrans || axis == 1 && _isTrans) {
        aggregate(axis, target, NUM_SUM_COLS_THREADS_PER_BLOCK, NVMatrix::MAX);
    } else {
        aggregate(axis, target, NUM_SUM_ROWS_THREADS_PER_BLOCK, NVMatrix::MAX);
    }
}

void NVMatrix::max2(int axis, NVMatrix& target, NVMatrix& argmax) {
    if(axis == 0 && !_isTrans || axis == 1 && _isTrans) {
      aggregate2(axis, target, argmax, NUM_SUM_COLS_THREADS_PER_BLOCK, NVMatrix::MAX);
    } else {
      aggregate2(axis, target, argmax, NUM_SUM_ROWS_THREADS_PER_BLOCK, NVMatrix::MAX);
    }
}

void NVMatrix::max3(int axis, NVMatrix& target, NVMatrix& argmax) {
    if(axis == 0 && !_isTrans || axis == 1 && _isTrans) {
      aggregate3(axis, target, argmax, NUM_SUM_COLS_THREADS_PER_BLOCK, NVMatrix::MAX);
    } else {
      aggregate3(axis, target, argmax, NUM_SUM_ROWS_THREADS_PER_BLOCK, NVMatrix::MAX);
    }
}
