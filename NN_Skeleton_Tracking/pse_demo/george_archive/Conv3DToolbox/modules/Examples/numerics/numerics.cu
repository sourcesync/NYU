#include "GPUkernel.hh"

typedef float2  Complex;
typedef double2 DoubleComplex;

__device__ inline float times(float data1, float data2) {
  return data1*data2;
}
__device__ inline Complex times(Complex data1, Complex data2) {
  return make_float2(data1.x * data2.x - data1.y * data2.y, data1.x * data2.y + data1.y * data2.x);
}
__device__ inline double times(double data1, double data2) {
  return data1*data2;
}
__device__ inline DoubleComplex times(DoubleComplex data1, DoubleComplex data2) {
  return make_double2(data1.x * data2.x - data1.y * data2.y, data1.x * data2.y + data1.y * data2.x);
}

extern "C" {

/* PLUS FLOAT */
__global__ void PLUSF(int n, 
                      int offset, 
                      float * idata1, 
                      float * idata2, 
                      float * odata)
{  
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    if ((xIndex - offset) < n) 
        odata[xIndex] = idata1[xIndex] + idata2[xIndex];
}

/* PLUS COMPLEX FLOAT */
__global__ void PLUSC(int n, 
                      int offset, 
                      Complex * idata1, 
                      Complex * idata2, 
                      Complex * odata)
{  
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    if ((xIndex - offset) < n) { 
        Complex tmp = make_float2(0.0,0.0);
        Complex in1 = idata1[xIndex];
        Complex in2 = idata2[xIndex];
        tmp.x = in1.x + in2.x;
        tmp.y = in1.y + in2.y;
        odata[xIndex] = tmp;
    }
}

/* PLUS DOUBLE */
__global__ void PLUSD(int n, 
                      int offset, 
                      double * idata1, 
                      double * idata2, 
                      double * odata)
{  
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    if ((xIndex - offset) < n) 
        odata[xIndex] = idata1[xIndex] + idata2[xIndex];
}

/* PLUS COMPLEX DOUBLE */
__global__ void PLUSCD(int n, 
                      int offset, 
                      DoubleComplex * idata1, 
                      DoubleComplex * idata2, 
                      DoubleComplex * odata)
{  
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    if ((xIndex - offset) < n) { 
        DoubleComplex tmp = make_double2(0.0,0.0);
        DoubleComplex in1 = idata1[xIndex];
        DoubleComplex in2 = idata2[xIndex];
        tmp.x = in1.x + in2.x;
        tmp.y = in1.y + in2.y;
        odata[xIndex] = tmp;
    }
}

/******************************************************************************/

/* TIMES FLOAT */
__global__ void TIMESF(int n, 
                      int offset, 
                      float * idata1, 
                      float * idata2, 
                      float * odata)
{  
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    if ((xIndex - offset) < n) 
        odata[xIndex] = times(idata1[xIndex] , idata2[xIndex]);
}

/* TIMES COMPLEX FLOAT */
__global__ void TIMESC(int n, 
                      int offset, 
                      Complex * idata1, 
                      Complex * idata2, 
                      Complex * odata)
{  
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    if ((xIndex - offset) < n) { 
        odata[xIndex] = times(idata1[xIndex] , idata2[xIndex]);
    }
}

/* TIMES DOUBLE */


__global__ void TIMESD(int n, 
                      int offset, 
                      double * idata1, 
                      double * idata2, 
                      double * odata)
{  
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    if ((xIndex - offset) < n) 
        odata[xIndex] = times(idata1[xIndex], idata2[xIndex]);
}

/* TIMES COMPLEX DOUBLE */
__global__ void TIMESCD(int n, 
                      int offset, 
                      DoubleComplex * idata1, 
                      DoubleComplex * idata2, 
                      DoubleComplex * odata)
{  
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    if ((xIndex - offset) < n) { 
        odata[xIndex] = times(idata1[xIndex] , idata2[xIndex]);
    }
}


}
