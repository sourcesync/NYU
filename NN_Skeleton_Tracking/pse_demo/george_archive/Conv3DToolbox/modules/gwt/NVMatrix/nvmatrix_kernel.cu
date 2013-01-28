#include "nvmatrix_kernel.cuh"

//extern "C" {

__global__ void kAddScalar(float* gData, float scalar, float* target, unsigned int numElements) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = idx; i < numElements; i += blockDim.x * gridDim.x)
        target[i] = scalar + gData[i];
}

/*
 * Bad when there are few columns. But if there are a few thousand columns, you can't really
 * go any faster than this because all the reads are coalesced and processor utilization is maximal.
 */
__global__ void kDumbSumCols(float* mat, float* vec, unsigned int width, unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    mat += idx;
    if (idx < width) {
        float sum = 0;
        for (int j = 0; j < height; j++) {
            sum += *mat;
            mat += width;
        }
        vec[idx] = sum;
    }
}

__global__ void kDumbMaxCols(float* mat, float* vec, unsigned int width, unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    mat += idx;
    if (idx < width) {
        float mx = *mat;
        mat += width;
        for (int j = 1; j < height; j++) {
            mx = myMax(*mat, mx);
            mat += width;
        }
        vec[idx] = mx;
    }
}

__global__ void kDumbMaxCols2(float* mat, float* vec, float* argmax, unsigned int width, unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    mat += idx;
    if (idx < width) {
        float mx = *mat;
	argmax[idx] = 1; // 1 for Matlab-style indexing
        mat += width;
        for (int j = 1; j < height; j++) {
	  if (*mat>mx){
	    mx=*mat;
	    argmax[idx] = j+1; //Add 1 for Matlab-style indexing
	  } 	    
	  //mx = myMax(*mat, mx);
            mat += width;
        }
        vec[idx] = mx;
    }
}

__global__ void kDumbMaxCols3(float* mat, float* vec, float* argmax, unsigned int width, unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    mat += idx;
    if (idx < width) {
        float mx = *mat;
	float mx1 = fabs(mx);
	argmax[idx] = 1; // 1 for Matlab-style indexing
        mat += width;
        for (int j = 1; j < height; j++) {
	  if (fabs(*mat)>mx1){
	    mx = *mat;
	    mx1 = fabs(mx);
	    argmax[idx] = j+1; //Add 1 for Matlab-style indexing
	  } 	    
	  //mx = myMax(*mat, mx);
            mat += width;
        }
        vec[idx] = mx;
    }
}


//}

