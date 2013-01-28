#include "GPUkernel.hh"
#include "misc.cuh"

extern "C" {

  /*
  __global__ void THREEWAYPROD(int N, int numA, int numB, int numC, int numCases, float *A, float *B, float *C, float *target) {

    //target index
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x;

    if (xIndex < N) {
      //indices to inputs
      const unsigned int a_idx = xIndex % numA;
      const unsigned int b_idx = ( xIndex % (numA*numB) ) / numA;
      const unsigned int c_idx = xIndex / (numA*numB);

      float p = 0;

      for (unsigned int t=0; t<numCases; t++) {
	p += A[t*numA + a_idx]*B[t*numB + b_idx]*C[t*numC + c_idx];
      }

      //Write out result to device mem
      target[xIndex] = p;
    }

  } */

  /* Modified to support much larger data
     Note that each thread may process multiple output elements (i.e. the loop function)
     Uses fixed 1-D block and grid dimension 
  */
  
  __global__ void THREEWAYPROD(int N, int numA, int numB, int numC, int numCases, float *A, float *B, float *C, float *target) {

    //target index (linear index to output)
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i=xIndex; i < N; i+= blockDim.x * gridDim.x) {
      //Now that we have the linear index to the output
      //We can obtain the linear index to the corresponding dimensions of A,B,C
      const unsigned int a_idx = i % numA; // linear index to A dimension
      const unsigned int b_idx = ( i % (numA*numB) ) / numA; //linear index to B dimension
      const unsigned int c_idx = i / (numA*numB); // linear index to C dimension

      float p = 0;

      //Given the correct dimensions, now we sum over cases
      //Note that dimension (not case) is changing quickest
      for (unsigned int t=0; t<numCases; t++) {
	p += A[t*numA + a_idx]*B[t*numB + b_idx]*C[t*numC + c_idx];
      }

      //Write out result to device mem
      target[i] = p;
    }

  } 

  /*
    __global__ void FFF(int N, int numA, int numB, int numC, int numCases, float *A, float *B, float *C, float *target) {

    //target index
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x;

    if (xIndex < N) {

    target[xIndex]=(float)xIndex;

    }



    }


    __global__ void GGG(int N, int numA, int numB, int numC, int numCases, float *A, float *B, float *C, float *target) {

    //target index
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x;

    if (xIndex < N) {

    const unsigned int a_idx = xIndex % numA;
    const unsigned int b_idx = ( xIndex % (numA*numB) ) / numA;
    const unsigned int c_idx = xIndex / (numA*numB);

    target[xIndex]=(float)c_idx;

    }

    }
  */

  /*
  __global__ void kRandomUniform(unsigned int* rndMults, unsigned long long* rndWords, float* gData, unsigned int numElements) {

    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(unsigned int i = idx; i < numElements; i += NUM_RND_STREAMS) {
      gData[i] = (float)rndWords[i];
    }

}
  */
  __global__ void kRandomUniform(unsigned int* rndMults, unsigned long long* rndWords, float* gData, unsigned int numElements) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long rndWord = rndWords[idx];
    const unsigned int rndMult = rndMults[idx];

    for(unsigned int i = idx; i < numElements; i += NUM_RND_STREAMS) {
        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
        gData[i] = (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;

    }
    rndWords[idx] = rndWord;
}
  
__global__ void kBinarizeProbs(unsigned int* rndMults, unsigned long long* rndWords, float *gData, unsigned int numElements) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long rndWord = rndWords[idx];
    const unsigned int rndMult = rndMults[idx];

    for(unsigned int i = idx; i < numElements; i += NUM_RND_STREAMS) {
        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
        gData[i] = gData[i] > (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;
    }
    rndWords[idx] = rndWord;
}

#define PI 3.1415926535897932f

/*
 * TODO: modify to take mean/stdev
 */
__global__ void kAddGaussianNoise(unsigned int* rndMults, unsigned long long* rndWords, float* gData, float stdev, unsigned int numElements) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long rndWord = rndWords[idx];
    const unsigned int rndMult = rndMults[idx];

    float rnd1, rnd2, R, T;
    for(unsigned int i = idx; i < numElements; i += 2*NUM_RND_STREAMS) {
        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
        rnd1 = (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;
        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
        rnd2 = (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;
        T = 2 * PI * rnd2;
        R = sqrtf(-2 * __logf(rnd1));
        gData[i] += stdev * R * __cosf(T);
        if (i + NUM_RND_STREAMS < numElements)
            gData[i + NUM_RND_STREAMS] += stdev * R * __sinf(T);
    }
    rndWords[idx] = rndWord;
}

/*
 * TODO: modify to take mean/stdev
 */
  __global__ void kRandomGaussian(unsigned int* rndMults, unsigned long long* rndWords, float* gData, const float stdev, unsigned int numElements) {
    //const float stdev = 1.0f;
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long rndWord = rndWords[idx];
    const unsigned int rndMult = rndMults[idx];

    float rnd1, rnd2, R, T;
    for(unsigned int i = idx; i < numElements; i += 2*NUM_RND_STREAMS) {
        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
        rnd1 = (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;
        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
        rnd2 = (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;
        T = 2 * PI * rnd2;
        R = sqrtf(-2 * __logf(rnd1));
        gData[i] = stdev * R * __cosf(T);
        if (i + NUM_RND_STREAMS < numElements)
            gData[i + NUM_RND_STREAMS] = stdev * R * __sinf(T);
    }
    rndWords[idx] = rndWord;
}

__global__ void kSeedRandom(unsigned int* rndMults, unsigned long long* rndWords, unsigned int seed) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // The initial x is the seed and the initial carry is 1
    unsigned long long rndWord = ((unsigned long long)seed << 32) + 1;
    const unsigned int rndMult = rndMults[idx];
    /*
     * Run the chain for a few steps so that all the streams have a chance
     * to differentiate. They start out generating similar random numbers
     * because all the multipliers are similar.
     */
    for(unsigned int i = 0; i < NUM_RND_BURNIN; i++) {
        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
    }
    rndWords[idx] = rndWord;
}

__global__ void kLogistic2(float* gData, float* target, unsigned int numElements) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = idx; i < numElements; i += blockDim.x * gridDim.x)
        target[i] = 1 / (1 + expf(-gData[i]));
}


  /*Accepts a single matrix
    Want to find the distance between every row vector in the matrix and every other row vector */
  __global__ void gpuPdistOrig(float* in, float* out, int n, int m) {
    extern __shared__ float Rs[];
    float tmp, s;
    int myRow = blockIdx.x*BLOCK_DIM1D + threadIdx.x;

    for(int r=0; r<n;r++){ //outer loop
      s = 0;
      for(int i=0; i<=m/BLOCK_DIM1D;i++) {
	if (i*BLOCK_DIM1D+threadIdx.x < m)
	  Rs[i*BLOCK_DIM1D+threadIdx.x] = in[r*m + i*BLOCK_DIM1D+threadIdx.x];
      }
      __syncthreads();

      for(int i=0; i<m && myRow<n; i++) {
	tmp = Rs[i] - in[myRow*m + i];
	s += tmp*tmp;
      }
      if (myRow<n)
	out[myRow*n+r] = sqrtf(s);
      __syncthreads();
    }
  } 

  __global__ void gpuPdist2Orig(float* in1, float* in2, float* out, int n1, int n2, int m) {
    extern __shared__ float Rs[];
    float tmp, s;
    int myRow = blockIdx.x*BLOCK_DIM1D + threadIdx.x;

    for(int r=0; r<n2;r++){ //outer loop
      s = 0;
      for(int i=0; i<=m/BLOCK_DIM1D;i++) {
	if (i*BLOCK_DIM1D+threadIdx.x < m)
	  Rs[i*BLOCK_DIM1D+threadIdx.x] = in2[r*m + i*BLOCK_DIM1D+threadIdx.x];
      }
      __syncthreads();

      for(int i=0; i<m && myRow<n1; i++) {
	tmp = Rs[i] - in1[myRow*m + i];
	s += tmp*tmp;
      }
      if (myRow<n1)
      out[myRow*n2+r] = sqrtf(s); //not coalesced
      __syncthreads();
    }
  } 

  /*Each thread computes a COLUMN of the n1xn2 distance matrix
    So writes are coalesced
    This is considerably faster than the original 
  Note that we need to change the driver code: n2 replaces n1
  int gridx = (n2 + BLOCK_DIM1D - 1) / BLOCK_DIM1D;
*/
  __global__ void gpuPdist2(float* in1, float* in2, float* out, int n1, int n2, int m) {
    extern __shared__ float Rs[];
    float tmp, s;
    int myCol = blockIdx.x*BLOCK_DIM1D + threadIdx.x;

    for(int r=0; r<n1;r++){ //outer loop
      s = 0;
      for(int i=0; i<=m/BLOCK_DIM1D;i++) {
	if (i*BLOCK_DIM1D+threadIdx.x < m)
	  Rs[i*BLOCK_DIM1D+threadIdx.x] = in1[r*m + i*BLOCK_DIM1D+threadIdx.x];
      }
      __syncthreads();

      for(int i=0; i<m && myCol<n2; i++) {
	tmp = Rs[i] - in2[myCol*m + i];
	s += tmp*tmp;
      }
      if (myCol<n2)
      out[r*n2+myCol] = sqrtf(s); //coalesced
      __syncthreads();
    }
  } 


  /*Each thread computes a COLUMN of the nxn distance matrix
    So writes are coalesced
    This is slightly faster than the original */
  __global__ void gpuPdist(float* in, float* out, int n, int m) {
    extern __shared__ float Rs[];
    float tmp, s;
    int myCol = blockIdx.x*BLOCK_DIM1D + threadIdx.x;

    for(int r=0; r<n;r++){ //outer loop
      s = 0;
      for(int i=0; i<=m/BLOCK_DIM1D;i++) {
	if (i*BLOCK_DIM1D+threadIdx.x < m)
	  Rs[i*BLOCK_DIM1D+threadIdx.x] = in[r*m + i*BLOCK_DIM1D+threadIdx.x]; //these reads are coalesced
      }
      __syncthreads();

      for(int i=0; i<m && myCol<n; i++) {
	tmp = Rs[i] - in[myCol*m + i]; //these reads from in are not coalesced since each thread looks at a different vector
	s += tmp*tmp;
      }
      if (myCol<n)
	out[r*n+myCol] = sqrtf(s);
      __syncthreads();
    }
  } 

  /*Like gpuPdist2 but it doesn't square the output and it doesn't overwrite
    the output, it simply accumulates
    Each thread computes a COLUMN of the n1xn2 distance matrix
    So writes are coalesced
    This is considerably faster than the original 
  Note that we need to change the driver code: n2 replaces n1
  int gridx = (n2 + BLOCK_DIM1D - 1) / BLOCK_DIM1D;
*/
  __global__ void gpusqPdist2(float* in1, float* in2, float* out, int n1, int n2, int m) {
    extern __shared__ float Rs[];
    float tmp, s;
    int myCol = blockIdx.x*BLOCK_DIM1D + threadIdx.x;

    for(int r=0; r<n1;r++){ //outer loop
      s = 0;
      for(int i=0; i<=m/BLOCK_DIM1D;i++) {
	if (i*BLOCK_DIM1D+threadIdx.x < m)
	  Rs[i*BLOCK_DIM1D+threadIdx.x] = in1[r*m + i*BLOCK_DIM1D+threadIdx.x];
      }
      __syncthreads();

      for(int i=0; i<m && myCol<n2; i++) {
	tmp = Rs[i] - in2[myCol*m + i];
	s += tmp*tmp;
      }
      if (myCol<n2)
      out[r*n2+myCol] += s; //coalesced
      __syncthreads();
    }
  } 

  /*Like gpuPdist but it doesn't square the output and it doesn't overwrite
    the output, it simply accumulates
    Each thread computes a COLUMN of the nxn distance matrix
    So writes are coalesced
    This is slightly faster than the original */
  __global__ void gpusqPdist(float* in, float* out, int n, int m) {
    extern __shared__ float Rs[];
    float tmp, s;
    int myCol = blockIdx.x*BLOCK_DIM1D + threadIdx.x;

    for(int r=0; r<n;r++){ //outer loop
      s = 0;
      for(int i=0; i<=m/BLOCK_DIM1D;i++) {
	if (i*BLOCK_DIM1D+threadIdx.x < m)
	  Rs[i*BLOCK_DIM1D+threadIdx.x] = in[r*m + i*BLOCK_DIM1D+threadIdx.x]; //these reads are coalesced
      }
      __syncthreads();

      for(int i=0; i<m && myCol<n; i++) {
	tmp = Rs[i] - in[myCol*m + i]; //these reads from in are not coalesced since each thread looks at a different vector
	s += tmp*tmp;
      }
      if (myCol<n)
	out[r*n+myCol] += s;
      __syncthreads();
    }
  } 

/*
 * Block size 16x16.
 * Probably a better idea to allocate multiple blocks per image so you don't have
 * to loop inside the block.
 */
__global__ void kCopyInto(float* images, float* targets, const int imgSize, const int paddingSize, const int numImages) {
    const int imgIdx = blockIdx.y * gridDim.x + blockIdx.x;
    if (imgIdx < numImages) {
        const int targetSize = imgSize + 2 * paddingSize;
        images += imgIdx * imgSize * imgSize;
        targets += imgIdx * targetSize * targetSize + MUL24(paddingSize, targetSize) + paddingSize;
        for (int y = threadIdx.y; y < imgSize; y += 16) {
            for (int x = threadIdx.x; x < imgSize; x += 16) {
                targets[MUL24(y, targetSize) + x] = images[MUL24(y, imgSize) + x];
            }
        }
    }
}

/*
 * Block size 16x16
 * Don't need shared memory on devices with compute capability 1.3 because memory
 * doesn't have to be accessed sequentially by threads.
 *
 * This is far from perfect, and in many cases is actually slwoer than doing it on the
 * CPU but still this takes so little time that it doesn't matter.
 */
__global__ void kRotate180(float* filters, float* targets, const int filterSize) {
//   __shared__ float shFilter[16][16];

    const int filtIdx = blockIdx.x;
    const int readStart = MUL24(MUL24(filterSize, filterSize), filtIdx);
    filters += readStart;
    targets += readStart;

    for(int y = threadIdx.y; y < filterSize; y += 16) {
        for(int x = threadIdx.x; x < filterSize; x += 16) {
            const int writeX = filterSize - 1 - x;
            const int writeY = filterSize - 1 - y;

            targets[MUL24(writeY, filterSize) + writeX] = filters[MUL24(y, filterSize) + x];
        }
    }
}



  // /*
  //  * Matrix in ROW-MAJOR order!
  //  */
  // __global__ void kDivideByRowVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height) {
  //   const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  //   const unsigned int numThreads = blockDim.x * gridDim.x;

  //   for (unsigned int i = idx; i < width * height; i += numThreads) {
  //     tgtMat[i] = __fdividef(mat[i], vec[i % width]);
  //   }
  // }

  // /*
  //  * Matrix in ROW-MAJOR order!
  //  */
  // __global__ void kDivideByColVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height) {
  //   const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  //   const unsigned int numThreads = blockDim.x * gridDim.x;

  //   for (unsigned int i = idx; i < width * height; i += numThreads) {
  //     tgtMat[i] = __fdividef(mat[i], vec[i / width]);
  //   }
  // }

  /*
   * Matrix in COL-MAJOR order!
   */
  __global__ void kDivideByColVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < width * height; i += numThreads) {
      tgtMat[i] = __fdividef(mat[i], vec[i % height]);
    }
  }

  /*
   * Matrix in COL-MAJOR order!
   */
  __global__ void kDivideByRowVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < width * height; i += numThreads) {
      tgtMat[i] = __fdividef(mat[i], vec[i / height]);
    }
  }

  

  /*Gradients for NCA regression
    Computes:
      for kk=1:n
       for jj=1:n
         out(kk,:) = out(kk,:) + (in(:,kk)-in(:,jj)) * ...
             w(jj,kk);
       end
     end
    Note that out has cases first, dims last
    So that writes are coalesced
    Each thread computes a COLUMN of the mxn gradient matrix
    So writes are coalesced
    n - number of cases (columns of in)
    m - number of dimensions (rows of in)
    This code is inspired by the distance matrix code
    */
  __global__ void gpuNCAreg(float* in, float* w, float*out, int n, int m) {
    extern __shared__ float Rs[];
    float tmp,s;//, s;
    int myCol = blockIdx.x*BLOCK_DIM1D + threadIdx.x;

    for(int r=0; r<n;r++){ //outer loop over columns
      s=w[myCol*n + r]; //read weight; not coalesced
      //s=w[r*n+myCol]; //it's coalesced if we pass in transposed w and use this
      for(int i=0; i<=m/BLOCK_DIM1D;i++) {
	if (i*BLOCK_DIM1D+threadIdx.x < m)
	  //these reads are coalesced
	  Rs[i*BLOCK_DIM1D+threadIdx.x] = in[r*m + i*BLOCK_DIM1D+threadIdx.x]; 
      }
      __syncthreads();

      for(int i=0; i<m && myCol<n; i++) {
	//these reads from in are not coalesced since each thread
	//looks at a different vector
	tmp = in[myCol*m + i] - Rs[i]; 

	tmp*=s; //multiply by weight

	//out[myCol*m + i] += tmp; //these writes are not coalesced
	out[i*n + myCol] += tmp; //these writes are coalesced

      }
      __syncthreads(); //sometimes wrong output if we don't do this
    }
  } 


    
}
