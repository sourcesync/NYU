#include "GPUkernel.hh"
#include "misc.cuh"

//To use templates we must have C++ kernels
//extern "C" {

/*
 * Factor must divide imgSize.
 * This routine is good when the subsampling region (factor) is pretty small (4x4 works well).
 * For large factors, it's inefficient but works. But my bet here is that for convolutional nets,
 * typically we won't want to subsample by a factor of more than 8 or so.
 *
 * Each sum of f^2 elements is computed by f cooperating threads. It works better than using
 * reduction in most (or all? I don't remember) the cases I've tried. One of the problems with reductions
 * is that if the number of elements you want to sum is not a power of 2, you have to do
 * a lot of bounds checking.
 *
 * Block size (y,x) = (regionsYPerBlock, imgSize).
 */
template<int factor, bool checkThreadBounds>
__global__ void kSubsample_noreduc(float* images, float* targets, const int imgSize, const int numRegionsY, const int shmemX) {
    extern __shared__ float shImg[];

    const int regionsYPerBlock = blockDim.y;
    const int bidx = MUL24(blockIdx.y, gridDim.x) + blockIdx.x;
    const int blockRegionIdxY = MUL24(regionsYPerBlock, bidx);

    if (blockRegionIdxY >= numRegionsY) {
        return;
    }

    const int tidx = MUL24(threadIdx.y, blockDim.x) + threadIdx.x;
    const int numRegionsX = imgSize / factor;
    const int regionPixels = factor * factor;
    const int regionsPerBlock = MUL24(numRegionsX, regionsYPerBlock);
    const int blockRegionIdx = MUL24(regionsPerBlock, bidx);
    const int threadRegionIdxY = blockRegionIdxY + threadIdx.y;
    const int regionsInThisBlock = numRegionsY - blockRegionIdxY < regionsYPerBlock
                                    ? MUL24(numRegionsX, numRegionsY - blockRegionIdxY) : regionsPerBlock;

    float* myShImg = shImg + MUL24((threadIdx.x % factor), shmemX) + (threadIdx.x / factor) + MUL24(threadIdx.y, numRegionsX);
    if (!checkThreadBounds || threadRegionIdxY < numRegionsY) {

        images += MUL24(MUL24(threadIdx.y, factor), imgSize)
                + MUL24(blockRegionIdx, regionPixels)
                + threadIdx.x;

        float mySum = 0;
        for (int d = 0; d < factor; d++) {
            mySum += images[0];
            images += imgSize;
        }
        myShImg[0] = mySum; // conflicts perhaps
    }

    __syncthreads();
    // Now sum out cols of shImg
    if (tidx < regionsInThisBlock) { // no bank conflicts
        float mySum = 0;
        myShImg = shImg + tidx;
        for (int d = 0; d < factor; d++) {
            mySum += myShImg[0];
            myShImg += shmemX;
        }
        targets[blockRegionIdx + tidx] = mySum / regionPixels;
    }
}

/*
 * Block size (y, x) = (nf, I) where I = img size, f = grid square size
 * This outputs the squares in row-major order. This requires 3 more ops per thread
 * than outputting in column-major order.
 */
//#define GTM_BLOCK_LOOPS_Y 32
template<int factor, bool reverse>
__global__ void kGridToMatrix(float* images, float* targets, const int imgSizeX, const int imgSizeY, const int shmemX) {
    extern __shared__ float shImg[];
    const int bidx = MUL24(blockIdx.y, gridDim.x) + blockIdx.x;

    int blockRowIdx = GTM_BLOCK_LOOPS_Y * bidx * blockDim.y;
    int tidx = MUL24(blockDim.x, threadIdx.y) + threadIdx.x;

    const int imgsOffset = imgSizeX * blockRowIdx + tidx;
    images += imgsOffset;
    targets += imgsOffset;

    const int shY = tidx / factor;
    const int shX = tidx % factor;
    float* shImgWrite = reverse ? &shImg[MUL24(shY, shmemX) + shX]
                                : &shImg[MUL24(MUL24(imgSizeX, shmemX), (threadIdx.y / factor))
                                        + MUL24(threadIdx.x, shmemX)
                                        + threadIdx.y % factor];
    float* shImgRead = reverse ? &shImg[MUL24(MUL24(imgSizeX, shmemX), (threadIdx.y / factor))
                                        + MUL24(threadIdx.x, shmemX)
                                        + threadIdx.y % factor]
                               : &shImg[MUL24(shY, shmemX) + shX];

    const int imgsInc = MUL24(blockDim.y, imgSizeX);
    for (int i = 0; i < GTM_BLOCK_LOOPS_Y; i++) {
        if (blockRowIdx >= imgSizeY) { // extra block
            return;
        }
        __syncthreads();
        if (threadIdx.y + blockRowIdx < imgSizeY) {
            shImgWrite[0] = images[0];
        }
        __syncthreads();

        if (threadIdx.y + blockRowIdx < imgSizeY) {
            targets[0] = shImgRead[0];
        }

        blockRowIdx += blockDim.y;
        images += imgsInc;
        targets += imgsInc;
    }
}

//#define GTM_LOOPY_BLOCK_LOOPS_Y 16
/*
 * Uses 14 registers
 */
template<int factor, bool reverse>
__global__ void kGridToMatrixLoopy(float* images, float* targets, const int imgSizeX, const int imgSizeY, const int shmemX) {
    extern __shared__ float shImg[];
    const int bidx = MUL24(blockIdx.y, gridDim.x) + blockIdx.x;

    int blockRowIdx = GTM_LOOPY_BLOCK_LOOPS_Y * bidx * blockDim.y;
    if (blockRowIdx >= imgSizeY) { // extra block
        return;
    }

    int tidx = MUL24(blockDim.x, threadIdx.y) + threadIdx.x;
    const int imgsOffset = imgSizeX * blockRowIdx;
    if (reverse) {
        targets += imgsOffset
                + MUL24(threadIdx.y, imgSizeX)
                + threadIdx.x;
        images += imgsOffset
                + MUL24(MUL24(tidx / (factor * blockDim.x), factor), imgSizeX)
                + tidx % (factor * blockDim.x);
    } else {
        images += imgsOffset
                + MUL24(threadIdx.y, imgSizeX)
                + threadIdx.x;
        targets += imgsOffset
                + MUL24(MUL24(tidx / (factor * blockDim.x), factor), imgSizeX)
                + tidx % (factor * blockDim.x);
    }

    const int shY = tidx / factor;
    const int shX = tidx % factor;
    float* shImgWrite = reverse ? &shImg[MUL24(shY, shmemX) + shX]
                                : &shImg[MUL24(MUL24(blockDim.x, shmemX), (threadIdx.y / factor))
                                        + MUL24(threadIdx.x, shmemX)
                                        + threadIdx.y % factor];
    float* shImgRead = reverse ? &shImg[MUL24(MUL24(blockDim.x, shmemX), (threadIdx.y / factor))
                                        + MUL24(threadIdx.x, shmemX)
                                        + threadIdx.y % factor]
                               : &shImg[MUL24(shY, shmemX) + shX];

    const int imgsInc = MUL24(blockDim.y, imgSizeX);
    for (int x = 0; x < imgSizeX; x += blockDim.x) {
        if (x + blockDim.x > imgSizeX) {
            x = imgSizeX - blockDim.x; // yea
        }

        float* targetsWrite = targets + (reverse ? x : MUL24(x, factor));
        float* imagesRead = images + (reverse ? MUL24(x, factor) : x);

        blockRowIdx = GTM_LOOPY_BLOCK_LOOPS_Y * bidx * blockDim.y;

        for (int y = 0; y < GTM_LOOPY_BLOCK_LOOPS_Y; y++) {
            if (blockRowIdx >= imgSizeY) { // extra block
                break;
            }
            __syncthreads();
            if (threadIdx.y + blockRowIdx < imgSizeY) {
                shImgWrite[0] = imagesRead[0];
            }
            __syncthreads();

            if (threadIdx.y + blockRowIdx < imgSizeY) {
                targetsWrite[0] = shImgRead[0];
            }

            blockRowIdx += blockDim.y;
            imagesRead += imgsInc;
            targetsWrite += imgsInc;
        }
    }
}

/*
 * Block size (y, x) = (n, imgSize). This one requires that n be divisible by
 * f to allow for easy indexing into the target and source matrices. Slightly slower (~8%)
 * than kSupersampleFast, but less restrictive on block dimensions so I use this.
 *
 * TODO: there's someting strange about this function. It seems like it should go faster
 * than it does. It has f times fewer shmem accesses than kSupersampleFast and yet that seems
 * to count for nothing...
 */
template<int factor>
__global__ void kSupersampleMedium(float* images, float* targets, const int imgSizeX, const int imgSizeY) {
    extern __shared__ float shImg[];
    const int bidx = blockIdx.y * gridDim.x + blockIdx.x;
    const int targetImgSizeX = MUL24(imgSizeX, factor);
    const int numThreads = MUL24(blockDim.x, blockDim.y);
    int tidx = MUL24(blockDim.x, threadIdx.y) + threadIdx.x;

    const int blockRowIdx = bidx * blockDim.y;
    if(blockRowIdx >= imgSizeY) { // extra block
        return;
    }

    if (threadIdx.y + blockRowIdx < imgSizeY) {
        images += MUL24(MUL24(imgSizeX, bidx), blockDim.y) + tidx;
        shImg[tidx] = images[0];
    }
    __syncthreads();

    const bool lastBlock = blockRowIdx + blockDim.y > imgSizeY;
    targets += MUL24(MUL24(MUL24(targetImgSizeX, bidx), blockDim.y), factor)
            + MUL24(MUL24(targetImgSizeX, factor), tidx / targetImgSizeX)
            + tidx % targetImgSizeX;

    float* myShImg = shImg + MUL24(imgSizeX, tidx / targetImgSizeX) + (tidx % targetImgSizeX) / factor;
    const int shImgInc = numThreads / factor;
    const int targetsInc = (numThreads - targetImgSizeX) * factor;
    if (!lastBlock) {
//        #pragma unroll
        for (int i = 0; i < factor; i++) {
            float value = myShImg[0];
            for (int j = 0; j < factor; j++) {
                targets[0] = value;
                targets += targetImgSizeX;
            }

            myShImg += shImgInc;
            targets += targetsInc;
        }
    } else {
        const int rowsPerIter = blockDim.y / factor;
        const int rowIdx = blockRowIdx + tidx / targetImgSizeX;
        for (int row = rowIdx; row < imgSizeY; row += rowsPerIter) {
            float value = myShImg[0];
            for (int j = 0; j < factor; j++) {
                targets[0] = value;
                targets += targetImgSizeX;
            }

            myShImg += shImgInc;
            targets += targetsInc;
        }
    }
}

/*
 * This version is like kSupersampleMedium but the number of threads in the y dimension doesn't
 * have to be equal to the image size, so it will work for any image/filter sizes.
 */
template<int factor>
__global__ void kSupersampleMediumLoopy(float* images, float* targets, const int imgSizeX, const int imgSizeY) {
    extern __shared__ float shImg[];
    const int bidx = blockIdx.y * gridDim.x + blockIdx.x;
    const int targetImgSizeX = MUL24(imgSizeX, factor);
    const int numThreads = MUL24(blockDim.x, blockDim.y);
    int tidx = MUL24(blockDim.x, threadIdx.y) + threadIdx.x;

    const int blockRowIdx = bidx * blockDim.y;
    if (blockRowIdx >= imgSizeY) { // extra block
        return;
    }
    const int targetViewSizeX = MUL24(blockDim.x, factor);
    const int targetY = tidx / targetViewSizeX;
    const int targetX = tidx % targetViewSizeX;

    const int targetImgSizeXTimesFactor = MUL24(targetImgSizeX, factor);
    // hahahh these indices are so big that you have to use 32-bit multiplication
    targets += targetImgSizeXTimesFactor * bidx * blockDim.y
            + MUL24(targetImgSizeXTimesFactor, targetY)
            + targetX;
    images += MUL24(MUL24(imgSizeX, bidx), blockDim.y)
            + MUL24(threadIdx.y, imgSizeX)
            + threadIdx.x;

    const int rowsPerIter = blockDim.y / factor;
    const int shImgInc = numThreads / factor;
    const int targetsInc = MUL24(rowsPerIter - 1, targetImgSizeXTimesFactor);

    const int iters = MIN(factor, DIVUP(imgSizeY - (blockRowIdx + targetY), rowsPerIter));
    float* shImgLoad = &shImg[tidx];
    float* myShImg2 = shImg + MUL24(blockDim.x, targetY) + targetX / factor;
    const bool load = threadIdx.y + blockRowIdx < imgSizeY;
    for (int c = 0; c < imgSizeX; c += blockDim.x) {
        if (c + blockDim.x > imgSizeX) {
            c = imgSizeX - blockDim.x; // oh what a wacky hack
        }

        __syncthreads();
        if (load) {
            shImgLoad[0] = images[c];
        }
        __syncthreads();

        float* targetWrite = targets + MUL24(c, factor);
        float* myShImg = myShImg2;
        for (int i = 0; i < iters; i++) {
            for (int j = 0; j < factor; j++) {
                targetWrite[0] = myShImg[0];
                targetWrite += targetImgSizeX;
            }

            myShImg += shImgInc;
            targetWrite += targetsInc;
        }
    }
}

/*
 * Samples from a bunch of multinomial distributions, where each row of data
 * is a different distribution.
 *
 * Uses the scan algorithm from these slides http://www.eecg.toronto.edu/~moshovos/CUDA08/slides/007%20-%20Scans.ppt
 * to compute prefix-sums.
 */
template<int bX>
__global__ void kSampleMultinomial(float* data, float* randoms, float* targets, const int multiSize, const int numMulti)  {
    const int bidx = blockIdx.y * gridDim.x + blockIdx.x;
    data += multiSize * bidx  + threadIdx.x;
    targets += multiSize * bidx  + threadIdx.x;
    __shared__ float shmem[IDX(bX * 2 + 1)];
    __shared__ float rand;

    if (bidx >= numMulti)
        return;

    shmem[IDX(threadIdx.x)] = 0;
    shmem[IDX(threadIdx.x + bX)] = 0;
    if (threadIdx.x < multiSize) {
        shmem[IDX(threadIdx.x)] = data[0]; // load input into shared memory
        if (threadIdx.x + bX < multiSize) {
            shmem[IDX(threadIdx.x + bX)] = data[bX];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        rand = randoms[bidx];
    }
    /*=============================================================
     * Reduction
     */
    int ai = 2 * threadIdx.x;
    int bi = ai + 1;
    if (bX >= 512) {
        __syncthreads();

        shmem[IDX(bi)] += shmem[IDX(ai)];

        ai = (ai << 1) + 1;
        bi = (bi << 1) + 1;
    }

    if (bX >= 256) {
        __syncthreads();
        if (threadIdx.x < 256) {
            shmem[IDX(bi)] += shmem[IDX(ai)];
        }
        ai = (ai << 1) + 1;
        bi = (bi << 1) + 1;
    }

    if (bX >= 128) {
        __syncthreads();
        if (threadIdx.x < 128) {
            shmem[IDX(bi)] += shmem[IDX(ai)];
        }
        ai = (ai << 1) + 1;
        bi = (bi << 1) + 1;
    }

    if (bX >= 64) {
        __syncthreads();
        if (threadIdx.x < 64) {
            shmem[IDX(bi)] += shmem[IDX(ai)];
        }
        ai = (ai << 1) + 1;
        bi = (bi << 1) + 1;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        shmem[IDX(bi)] += shmem[IDX(ai)];
    }
    ai = (ai << 1) + 1;
    bi = (bi << 1) + 1;

    if (threadIdx.x < 16) {
        shmem[IDX(bi)] += shmem[IDX(ai)];
    }
    ai = (ai << 1) + 1;
    bi = (bi << 1) + 1;

    if (threadIdx.x < 8) {
        shmem[IDX(bi)] += shmem[IDX(ai)];
    }
    ai = (ai << 1) + 1;
    bi = (bi << 1) + 1;

    if (threadIdx.x < 4) {
        shmem[IDX(bi)] += shmem[IDX(ai)];
    }
    ai = (ai << 1) + 1;
    bi = (bi << 1) + 1;

    if (threadIdx.x < 2) {
        shmem[IDX(bi)] += shmem[IDX(ai)];
    }
    ai = (ai << 1) + 1;
    bi = (bi << 1) + 1;

    if (threadIdx.x < 1) {
        shmem[IDX(bi)] += shmem[IDX(ai)];

        /*=============================================================
         * Scan
         */
        shmem[IDX(bX * 2 - 1)] = 0;

        const float t = shmem[IDX(ai)];
        shmem[IDX(ai)] = shmem[IDX(bi)];
        shmem[IDX(bi)] += t;
    }
    ai >>= 1;
    bi >>= 1;

    if (threadIdx.x < 2) {
        const float t = shmem[IDX(ai)];
        shmem[IDX(ai)] = shmem[IDX(bi)];
        shmem[IDX(bi)] += t;
    }
    ai >>= 1;
    bi >>= 1;

    if (threadIdx.x < 4) {
        const float t = shmem[IDX(ai)];
        shmem[IDX(ai)] = shmem[IDX(bi)];
        shmem[IDX(bi)] += t;
    }
    ai >>= 1;
    bi >>= 1;

    if (threadIdx.x < 8) {
        const float t = shmem[IDX(ai)];
        shmem[IDX(ai)] = shmem[IDX(bi)];
        shmem[IDX(bi)] += t;
    }
    ai >>= 1;
    bi >>= 1;

    if (threadIdx.x < 16) {
        const float t = shmem[IDX(ai)];
        shmem[IDX(ai)] = shmem[IDX(bi)];
        shmem[IDX(bi)] += t;
    }
    ai >>= 1;
    bi >>= 1;

    if (threadIdx.x < 32) {
        const float t = shmem[IDX(ai)];
        shmem[IDX(ai)] = shmem[IDX(bi)];
        shmem[IDX(bi)] += t;
    }

    if (bX >= 64) {
        __syncthreads();
        ai >>= 1;
        bi >>= 1;
        if (threadIdx.x < 64) {
            const float t = shmem[IDX(ai)];
            shmem[IDX(ai)] = shmem[IDX(bi)];
            shmem[IDX(bi)] += t;
        }
    }

    if (bX >= 128) {
        __syncthreads();
        ai >>= 1;
        bi >>= 1;
        if (threadIdx.x < 128) {
            const float t = shmem[IDX(ai)];
            shmem[IDX(ai)] = shmem[IDX(bi)];
            shmem[IDX(bi)] += t;
        }
    }

    if (bX >= 256) {
        __syncthreads();
        ai >>= 1;
        bi >>= 1;
        if (threadIdx.x < 256) {
            const float t = shmem[IDX(ai)];
            shmem[IDX(ai)] = shmem[IDX(bi)];
            shmem[IDX(bi)] += t;
        }
    }

    if (bX >= 512) {
        __syncthreads();
        ai >>= 1;
        bi >>= 1;
        const float t = shmem[IDX(ai)];
        shmem[IDX(ai)] = shmem[IDX(bi)];
        shmem[IDX(bi)] += t;
    }

    __syncthreads();
    if (threadIdx.x < multiSize) {
        shmem[IDX(threadIdx.x)] += data[0]; // load input into shared memory
        if (threadIdx.x + bX < multiSize) {
            shmem[IDX(threadIdx.x + bX)] += data[bX];
        }
    }
    __syncthreads();
    if (threadIdx.x < multiSize) {
        const float prev = threadIdx.x == 0 ? 0 : shmem[IDX(threadIdx.x - 1)];
        targets[0] = rand >= prev && rand < shmem[IDX(threadIdx.x)];
//        targets[0] = shmem[IDX(threadIdx.x)];

        if (threadIdx.x + bX < multiSize) {
            targets[bX] = rand >= shmem[IDX(threadIdx.x - 1 + bX)] && rand < shmem[IDX(threadIdx.x + bX)];
//            targets[bX] = shmem[IDX(threadIdx.x + bX)];
        }
    }
}
//#define SSM_THREADS_X   16
//#define SSM_THREADS_Y   32
//#define SSM_LOOPS_Y     16
/*
 * This routine is just always faster than the fancy tree-based one above...
 * Oh ok, not in all cases. In the cases when the number of distributions
 * that you want to sample from (height) is fairly large.
 *
 * TODO: revisit this routine cause that doWrite statement is too long
 * and it all can probably be simplified if i control the block size at run-time
 */
template <int LOOPS_X, int SUM_WIDTH_UPPERBOUND>
__global__ void kSampleSmallMultinomial(float* multi, float* randoms, float* targets, const int width, const int height) {
    const int shmemX = SSM_THREADS_X + 1;
    __shared__ float shmem[SSM_THREADS_Y*shmemX];

//    const int LOOPS_X = DIVUP(width, AGG_SHORT_ROWS_THREADS_X);

    const int bidx = blockIdx.y * gridDim.x + blockIdx.x;
    const int blockRowIdx = bidx * SSM_LOOPS_Y * SSM_THREADS_Y;

    if(blockRowIdx < height) {
        const int tidx = threadIdx.y * SSM_THREADS_X + threadIdx.x;
        int ty = LOOPS_X == 1 ? tidx / width : threadIdx.y;
        const int tx = LOOPS_X == 1 ? tidx % width : threadIdx.x;
        float* shmemWrite = shmem + MUL24(ty, shmemX) + tx;
        //    targets += blockIdx.y * AGG_SHORT_ROWS_LOOPS_Y * AGG_SHORT_ROWS_THREADS_Y + tidx;
        const int dataOffset = width * blockRowIdx + MUL24(ty, width) + tx;
        multi += dataOffset;
        targets += dataOffset;

        float* shmemWriteZeros = &shmem[MUL24(threadIdx.y,shmemX) + threadIdx.x];
//        ty += blockRowIdx;
//#pragma unroll
        for (int y = 0; y < SSM_LOOPS_Y*SSM_THREADS_Y; y += SSM_THREADS_Y) {
//            if (y * AGG_SHORT_ROWS_THREADS_Y + idxY >= height) {
//                return; // we're done here
//            }
            const bool doSum = tidx < SSM_THREADS_Y && tidx + y + blockRowIdx < height;
            float rnd;
            if (doSum) {
                rnd = randoms[tidx + y + blockRowIdx];
            }
            float accum = 0, accumPrev = 0;
//#pragma unroll // this causes > 16 registers to be used in some cases, avoid
            for(int x = 0; x < LOOPS_X * SSM_THREADS_X; x+= SSM_THREADS_X) {
                __syncthreads();
                shmemWriteZeros[0] = 0;
                if (LOOPS_X == 1) { // because the part we zeroed might not be same as one we're writing to
                    __syncthreads();
                }
                const bool doWrite = ty + blockRowIdx + y < height && (LOOPS_X > 1 || ty < SSM_THREADS_Y) && x + tx < width;
                if (doWrite) {
                    shmemWrite[0] = multi[y * width + x];
                }
                __syncthreads();

                if (doSum) {
                    float* shmemRead = shmem + MUL24(tidx, shmemX);

                    // this loops too much if the rows are really short :(
                    for (int i = 0; i < SUM_WIDTH_UPPERBOUND; i++) {
                        accumPrev = accum;
                        accum += shmemRead[0];
                        shmemRead[0] = rnd >= accumPrev && rnd < accum;
                        shmemRead++;
                    }
                }
                __syncthreads();
                if (doWrite) {
                    targets[y * width + x] = shmemWrite[0];
                }
            }
//            multi += width * SSM_THREADS_Y;
//            targets += width * SSM_THREADS_Y;
//            ty += SSM_THREADS_Y;
        }
    }
}


/* Templates need to be instantiated
   See: http://forums.nvidia.com/index.php?showtopic=31953&pid=178825&mode=threaded&start=0#entry178825
 */
__host__ void dummyTemplateInstantiator(void){

  kSubsample_noreduc<2,true><<<0,0,0>>>(0,0,0,0,0);
  kSubsample_noreduc<2,false><<<0,0,0>>>(0,0,0,0,0);

  kSubsample_noreduc<3,true><<<0,0,0>>>(0,0,0,0,0);
  kSubsample_noreduc<3,false><<<0,0,0>>>(0,0,0,0,0);

  kSubsample_noreduc<4,true><<<0,0,0>>>(0,0,0,0,0);
  kSubsample_noreduc<4,false><<<0,0,0>>>(0,0,0,0,0);

  kSubsample_noreduc<5,true><<<0,0,0>>>(0,0,0,0,0);
  kSubsample_noreduc<5,false><<<0,0,0>>>(0,0,0,0,0);

  kSubsample_noreduc<6,true><<<0,0,0>>>(0,0,0,0,0);
  kSubsample_noreduc<6,false><<<0,0,0>>>(0,0,0,0,0);

  kSubsample_noreduc<7,true><<<0,0,0>>>(0,0,0,0,0);
  kSubsample_noreduc<7,false><<<0,0,0>>>(0,0,0,0,0);

  kSubsample_noreduc<8,true><<<0,0,0>>>(0,0,0,0,0);
  kSubsample_noreduc<8,false><<<0,0,0>>>(0,0,0,0,0);

  kSubsample_noreduc<9,true><<<0,0,0>>>(0,0,0,0,0);
  kSubsample_noreduc<9,false><<<0,0,0>>>(0,0,0,0,0);

  kSubsample_noreduc<10,true><<<0,0,0>>>(0,0,0,0,0);
  kSubsample_noreduc<10,false><<<0,0,0>>>(0,0,0,0,0);

  kSubsample_noreduc<11,true><<<0,0,0>>>(0,0,0,0,0);
  kSubsample_noreduc<11,false><<<0,0,0>>>(0,0,0,0,0);

  kSubsample_noreduc<12,true><<<0,0,0>>>(0,0,0,0,0);
  kSubsample_noreduc<12,false><<<0,0,0>>>(0,0,0,0,0);

  kSubsample_noreduc<13,true><<<0,0,0>>>(0,0,0,0,0);
  kSubsample_noreduc<13,false><<<0,0,0>>>(0,0,0,0,0);

  kSubsample_noreduc<14,true><<<0,0,0>>>(0,0,0,0,0);
  kSubsample_noreduc<14,false><<<0,0,0>>>(0,0,0,0,0);

  kSubsample_noreduc<15,true><<<0,0,0>>>(0,0,0,0,0);
  kSubsample_noreduc<15,false><<<0,0,0>>>(0,0,0,0,0);

  kSubsample_noreduc<16,true><<<0,0,0>>>(0,0,0,0,0);
  kSubsample_noreduc<16,false><<<0,0,0>>>(0,0,0,0,0);

  kGridToMatrix<2,true><<<0,0,0>>>(0,0,0,0,0);
  kGridToMatrix<2,false><<<0,0,0>>>(0,0,0,0,0);

  kGridToMatrix<3,true><<<0,0,0>>>(0,0,0,0,0);
  kGridToMatrix<3,false><<<0,0,0>>>(0,0,0,0,0);

  kGridToMatrix<4,true><<<0,0,0>>>(0,0,0,0,0);
  kGridToMatrix<4,false><<<0,0,0>>>(0,0,0,0,0);

  kGridToMatrix<5,true><<<0,0,0>>>(0,0,0,0,0);
  kGridToMatrix<5,false><<<0,0,0>>>(0,0,0,0,0);

  kGridToMatrix<6,true><<<0,0,0>>>(0,0,0,0,0);
  kGridToMatrix<6,false><<<0,0,0>>>(0,0,0,0,0);

  kGridToMatrix<7,true><<<0,0,0>>>(0,0,0,0,0);
  kGridToMatrix<7,false><<<0,0,0>>>(0,0,0,0,0);

  kGridToMatrix<8,true><<<0,0,0>>>(0,0,0,0,0);
  kGridToMatrix<8,false><<<0,0,0>>>(0,0,0,0,0);

  kGridToMatrix<9,true><<<0,0,0>>>(0,0,0,0,0);
  kGridToMatrix<9,false><<<0,0,0>>>(0,0,0,0,0);

  kGridToMatrix<10,true><<<0,0,0>>>(0,0,0,0,0);
  kGridToMatrix<10,false><<<0,0,0>>>(0,0,0,0,0);

  kGridToMatrix<11,true><<<0,0,0>>>(0,0,0,0,0);
  kGridToMatrix<11,false><<<0,0,0>>>(0,0,0,0,0);

  kGridToMatrix<12,true><<<0,0,0>>>(0,0,0,0,0);
  kGridToMatrix<12,false><<<0,0,0>>>(0,0,0,0,0);

  kGridToMatrix<13,true><<<0,0,0>>>(0,0,0,0,0);
  kGridToMatrix<13,false><<<0,0,0>>>(0,0,0,0,0);

  kGridToMatrix<14,true><<<0,0,0>>>(0,0,0,0,0);
  kGridToMatrix<14,false><<<0,0,0>>>(0,0,0,0,0);

  kGridToMatrix<15,true><<<0,0,0>>>(0,0,0,0,0);
  kGridToMatrix<15,false><<<0,0,0>>>(0,0,0,0,0);

  kGridToMatrix<16,true><<<0,0,0>>>(0,0,0,0,0);
  kGridToMatrix<16,false><<<0,0,0>>>(0,0,0,0,0);

  kGridToMatrixLoopy<2,true><<<0,0,0>>>(0,0,0,0,0);
  kGridToMatrixLoopy<2,false><<<0,0,0>>>(0,0,0,0,0);

  kGridToMatrixLoopy<3,true><<<0,0,0>>>(0,0,0,0,0);
  kGridToMatrixLoopy<3,false><<<0,0,0>>>(0,0,0,0,0);

  kGridToMatrixLoopy<4,true><<<0,0,0>>>(0,0,0,0,0);
  kGridToMatrixLoopy<4,false><<<0,0,0>>>(0,0,0,0,0);

  kGridToMatrixLoopy<5,true><<<0,0,0>>>(0,0,0,0,0);
  kGridToMatrixLoopy<5,false><<<0,0,0>>>(0,0,0,0,0);

  kGridToMatrixLoopy<6,true><<<0,0,0>>>(0,0,0,0,0);
  kGridToMatrixLoopy<6,false><<<0,0,0>>>(0,0,0,0,0);

  kGridToMatrixLoopy<7,true><<<0,0,0>>>(0,0,0,0,0);
  kGridToMatrixLoopy<7,false><<<0,0,0>>>(0,0,0,0,0);

  kGridToMatrixLoopy<8,true><<<0,0,0>>>(0,0,0,0,0);
  kGridToMatrixLoopy<8,false><<<0,0,0>>>(0,0,0,0,0);

  kGridToMatrixLoopy<9,true><<<0,0,0>>>(0,0,0,0,0);
  kGridToMatrixLoopy<9,false><<<0,0,0>>>(0,0,0,0,0);

  kGridToMatrixLoopy<10,true><<<0,0,0>>>(0,0,0,0,0);
  kGridToMatrixLoopy<10,false><<<0,0,0>>>(0,0,0,0,0);

  kGridToMatrixLoopy<11,true><<<0,0,0>>>(0,0,0,0,0);
  kGridToMatrixLoopy<11,false><<<0,0,0>>>(0,0,0,0,0);

  kGridToMatrixLoopy<12,true><<<0,0,0>>>(0,0,0,0,0);
  kGridToMatrixLoopy<12,false><<<0,0,0>>>(0,0,0,0,0);

  kGridToMatrixLoopy<13,true><<<0,0,0>>>(0,0,0,0,0);
  kGridToMatrixLoopy<13,false><<<0,0,0>>>(0,0,0,0,0);

  kGridToMatrixLoopy<14,true><<<0,0,0>>>(0,0,0,0,0);
  kGridToMatrixLoopy<14,false><<<0,0,0>>>(0,0,0,0,0);

  kGridToMatrixLoopy<15,true><<<0,0,0>>>(0,0,0,0,0);
  kGridToMatrixLoopy<15,false><<<0,0,0>>>(0,0,0,0,0);

  kGridToMatrixLoopy<16,true><<<0,0,0>>>(0,0,0,0,0);
  kGridToMatrixLoopy<16,false><<<0,0,0>>>(0,0,0,0,0);

  kSupersampleMedium<2><<<0,0,0>>>(0,0,0,0);
  kSupersampleMedium<3><<<0,0,0>>>(0,0,0,0);
  kSupersampleMedium<4><<<0,0,0>>>(0,0,0,0);
  kSupersampleMedium<5><<<0,0,0>>>(0,0,0,0);
  kSupersampleMedium<6><<<0,0,0>>>(0,0,0,0);
  kSupersampleMedium<7><<<0,0,0>>>(0,0,0,0);
  kSupersampleMedium<8><<<0,0,0>>>(0,0,0,0);
  kSupersampleMedium<9><<<0,0,0>>>(0,0,0,0);
  kSupersampleMedium<10><<<0,0,0>>>(0,0,0,0);
  kSupersampleMedium<11><<<0,0,0>>>(0,0,0,0);
  kSupersampleMedium<12><<<0,0,0>>>(0,0,0,0);
  kSupersampleMedium<13><<<0,0,0>>>(0,0,0,0);
  kSupersampleMedium<14><<<0,0,0>>>(0,0,0,0);
  kSupersampleMedium<15><<<0,0,0>>>(0,0,0,0);
  kSupersampleMedium<16><<<0,0,0>>>(0,0,0,0);

  kSupersampleMediumLoopy<2><<<0,0,0>>>(0,0,0,0);
  kSupersampleMediumLoopy<3><<<0,0,0>>>(0,0,0,0);
  kSupersampleMediumLoopy<4><<<0,0,0>>>(0,0,0,0);
  kSupersampleMediumLoopy<5><<<0,0,0>>>(0,0,0,0);
  kSupersampleMediumLoopy<6><<<0,0,0>>>(0,0,0,0);
  kSupersampleMediumLoopy<7><<<0,0,0>>>(0,0,0,0);
  kSupersampleMediumLoopy<8><<<0,0,0>>>(0,0,0,0);
  kSupersampleMediumLoopy<9><<<0,0,0>>>(0,0,0,0);
  kSupersampleMediumLoopy<10><<<0,0,0>>>(0,0,0,0);
  kSupersampleMediumLoopy<11><<<0,0,0>>>(0,0,0,0);
  kSupersampleMediumLoopy<12><<<0,0,0>>>(0,0,0,0);
  kSupersampleMediumLoopy<13><<<0,0,0>>>(0,0,0,0);
  kSupersampleMediumLoopy<14><<<0,0,0>>>(0,0,0,0);
  kSupersampleMediumLoopy<15><<<0,0,0>>>(0,0,0,0);
  kSupersampleMediumLoopy<16><<<0,0,0>>>(0,0,0,0);

  kSampleMultinomial<32><<<0,0,0>>>(0,0,0,0,0);
  kSampleMultinomial<64><<<0,0,0>>>(0,0,0,0,0);
  kSampleMultinomial<128><<<0,0,0>>>(0,0,0,0,0);
  kSampleMultinomial<256><<<0,0,0>>>(0,0,0,0,0);
  kSampleMultinomial<512><<<0,0,0>>>(0,0,0,0,0);

  kSampleSmallMultinomial<1,4><<<0,0,0>>>(0,0,0,0,0);
  kSampleSmallMultinomial<1,8><<<0,0,0>>>(0,0,0,0,0);
  kSampleSmallMultinomial<1,12><<<0,0,0>>>(0,0,0,0,0);
  kSampleSmallMultinomial<1,16><<<0,0,0>>>(0,0,0,0,0);

  kSampleSmallMultinomial<2,SSM_THREADS_X><<<0,0,0>>>(0,0,0,0,0);
  kSampleSmallMultinomial<3,SSM_THREADS_X><<<0,0,0>>>(0,0,0,0,0);
  kSampleSmallMultinomial<4,SSM_THREADS_X><<<0,0,0>>>(0,0,0,0,0);
  kSampleSmallMultinomial<5,SSM_THREADS_X><<<0,0,0>>>(0,0,0,0,0);
  kSampleSmallMultinomial<6,SSM_THREADS_X><<<0,0,0>>>(0,0,0,0,0);
  kSampleSmallMultinomial<7,SSM_THREADS_X><<<0,0,0>>>(0,0,0,0,0);
  kSampleSmallMultinomial<8,SSM_THREADS_X><<<0,0,0>>>(0,0,0,0,0);
  kSampleSmallMultinomial<9,SSM_THREADS_X><<<0,0,0>>>(0,0,0,0,0);
  kSampleSmallMultinomial<10,SSM_THREADS_X><<<0,0,0>>>(0,0,0,0,0);
  kSampleSmallMultinomial<11,SSM_THREADS_X><<<0,0,0>>>(0,0,0,0,0);
  kSampleSmallMultinomial<12,SSM_THREADS_X><<<0,0,0>>>(0,0,0,0,0);
  kSampleSmallMultinomial<13,SSM_THREADS_X><<<0,0,0>>>(0,0,0,0,0);
  kSampleSmallMultinomial<14,SSM_THREADS_X><<<0,0,0>>>(0,0,0,0,0);
  kSampleSmallMultinomial<15,SSM_THREADS_X><<<0,0,0>>>(0,0,0,0,0);
  kSampleSmallMultinomial<16,SSM_THREADS_X><<<0,0,0>>>(0,0,0,0,0);

}


//}
