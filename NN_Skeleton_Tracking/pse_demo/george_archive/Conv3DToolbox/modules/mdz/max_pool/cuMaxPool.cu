/*#include "GPUkernel.hh"*/

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

/* MAX POOLING when no indices are passed in. */
__global__ void MAXPOOLFNOIND(                     
                      float * idata,                       
                      float * omaxes,
                      float * oinds,
                      int n,
                      int offset, 
                      int imx,
                      int imy,
                      int outx,
                      int outy,
                      int poolx,
                      int pooly)
{  
    // Get the linear intext into odata (do not touch this).
    unsigned int outIndex = blockIdx.x * blockDim.x + threadIdx.x+offset;

    // Make sure the index to the last elemnt of the pool region
    if ((outIndex-offset) < n){

        // Based on outindex and outx (size of output) we can get the column we are in (in terms of pooling regions). 
        unsigned int outCol = outIndex/outx;
        // Plane we are currently on.
        unsigned int plane = outCol/outy;
        // Have to mod the outCol at this point in case we are not on the first plane.
        outCol = outCol % outy;
        // Have to also get the output row as well.
        unsigned int outRow = outIndex % outx;

        // top left element of pooling region (in the input image).
        unsigned int poolIndex = outRow*poolx+outCol*(imx*pooly)+plane*imx*imy;
        // Also need an index to the current element of the inptu pool region.
        unsigned int inIndex = 0;

        float mx = 0;
        float mind = 0;
        // Loop over the pooling region in the image. 
        for(int y=0;y<pooly;y++){    
            // Check we are in boundary in y dimension.
            if(outCol*pooly+y<imy){
                for(int x=0;x<poolx;x++){  
                    if(outRow*poolx+x<imx){
                        // Get current image element's linear index in the input.
                        inIndex = poolIndex+y*imx+x;
                        if(fabs(idata[inIndex])>fabs(mx)){
                            mx = idata[inIndex];
                            mind = x+y*poolx;
                        }
                    }
                }
            }
        }
        /*curInd = xIndex + x;*/
        omaxes[outIndex] = mx;
        oinds[outIndex] = mind;
    }
}


/* MAX POOLING when the indices are passed in*/
__global__ void MAXPOOLFIND(                     
                      float * idata,                       
                      float * omaxes,
                      float * oinds,
                      int n,
                      int offset, 
                      int imx,
                      int imy,
                      int outx,
                      int outy,
                      int poolx,
                      int pooly)
{  
    // Get the linear intext into odata (do not touch this).
    unsigned int outIndex = blockIdx.x * blockDim.x + threadIdx.x+offset;

    // Make sure the index to the last elemnt of the pool region
    if ((outIndex-offset) < n){

        // Based on outindex and outx (size of output) we can get the column we are in (in terms of pooling regions). 
        unsigned int outCol = outIndex/outx;
        // Plane we are currently on.
        unsigned int plane = outCol/outy;
        // Have to mod the outCol at this point in case we are not on the first plane.
        outCol = outCol % outy;
        // Have to also get the output row as well.
        unsigned int outRow = outIndex % outx;

        // top left element of pooling region (in the input image).
        unsigned int poolIndex = outRow*poolx+outCol*(imx*pooly)+plane*imx*imy;

        // Get current image element's linear index in the input.
        unsigned int inIndex = (int)oinds[outIndex];
        unsigned int x =  inIndex % poolx;
        unsigned int y = inIndex / poolx; 
        inIndex = poolIndex+y*imx+x;

        omaxes[outIndex] = idata[inIndex];
    }
}


/* REVERSE MAX POOLING when the indices are passed in*/
__global__ void REVMAXPOOLF(                     
                      float * idata,                       
                      float * omaxes,
                      float * iinds,
                      int n,
                      int offset, 
                      int imx,
                      int imy,
                      int outx,
                      int outy,
                      int poolx,
                      int pooly)
{  
    // OPPOSITE TO THE ABOVE KERNELS BECAUSE WE LOOP OVER THE POOLING INPUT MAPS NOW.
    // So imxy are the pooled sizes, outxy are the unpooled sizes.
    // Get the linear intext into idata (do not touch this).
    unsigned int inIndex = blockIdx.x * blockDim.x + threadIdx.x+offset;

    // Make sure the index to the last elemnt of the pool region
    if ((inIndex-offset) < n){

        // Based on outindex and outx (size of output) we can get the column we are in (in terms of pooling regions). 
        unsigned int inCol = inIndex/imx;
        // Plane we are currently on.
        unsigned int plane = inCol/imy;
        // Have to mod the outCol at this point in case we are not on the first plane.
        inCol = inCol % imy;
        // Have to also get the output row as well.
        unsigned int inRow = inIndex % imx;

        // top left element of pooling region (in the OUTPUT image).
        unsigned int poolIndex = inRow*poolx+inCol*(outx*pooly)+plane*outx*outy;

        // Get current image element's linear index in the input.
        unsigned int outIndex = (int)iinds[inIndex];
        unsigned int x =  outIndex % poolx;
        unsigned int y = outIndex / poolx; 
        outIndex = poolIndex+y*outx+x;

        omaxes[outIndex] = idata[inIndex];
    }
}












/**********************3D pooling**********************************/


/* MAX POOLING when no indices are passed in. */
__global__ void MAXPOOLFNOIND3D(                     
                      float * idata,                       
                      float * omaxes,
                      float * oinds,
                      int n,
                      int offset, 
                      int imx,
                      int imy,
                      int imk,
                      int outx,
                      int outy,
                      int outk,
                      int poolx,
                      int pooly,
                      int poolk)
{  
    // Get the linear intext into odata (do not touch this).
    unsigned int outIndex = blockIdx.x * blockDim.x + threadIdx.x+offset;

    // Make sure the index to the last elemnt of the pool region
    if ((outIndex-offset) < n){

        // Based on outindex and outx (size of output) we can get the column we are in (in terms of pooling regions). 
        unsigned int outCol = outIndex/outx;
        // Plane we are currently on.
        unsigned int plane = outCol/outy;
        // Have to mod the outCol at this point in case we are not on the first plane.
        outCol = outCol % outy;
        // Have to also get the output row as well.
        unsigned int outRow = outIndex % outx;
        // Have to get which image we're on too.
        unsigned int image = plane/outk;
        // Have to mod plane as well by poolk.
        plane = plane % outk;

        // top left element of pooling region (in the input image).
        unsigned int poolIndex = outRow*poolx+outCol*(imx*pooly)+plane*(imx*imy*poolk)+image*(imx*imy*imk);
        // Also need an index to the current element of the inptu pool region.
        unsigned int inIndex = 0;

        float mx = 0;
        float mind = 0;
        // Now also loop of the k dimension.
        for(int k=0;k<poolk;k++){
            // Make sure we don't go over the number of image planes.
            if(plane*poolk+k<imk){
                // Loop over the pooling region in the image. 
                for(int y=0;y<pooly;y++){    
                    // Check we are in boundary in y dimension.
                    if(outCol*pooly+y<imy){
                        for(int x=0;x<poolx;x++){  
                            if(outRow*poolx+x<imx){
                                // Get current image element's linear index in the input.
                                inIndex = poolIndex+k*imx*imy+y*imx+x;
                                if(fabs(idata[inIndex])>fabs(mx)){
                                    mx = idata[inIndex];
                                    mind = x+y*poolx+k*poolx*pooly;
                                }
                            }
                        }
                    }
                }
            }
        }
        /*curInd = xIndex + x;*/
        omaxes[outIndex] = mx;
        //omaxes[outIndex] = (float)plane;
        oinds[outIndex] = mind;
    }
}


/* MAX POOLING when the indices are passed in*/
__global__ void MAXPOOLFIND3D(                     
                      float * idata,                       
                      float * omaxes,
                      float * oinds,
                      int n,
                      int offset, 
                      int imx,
                      int imy,
                      int imk,
                      int outx,
                      int outy,
                      int outk,
                      int poolx,
                      int pooly,
                      int poolk)
{  
    // Get the linear intext into odata (do not touch this).
    unsigned int outIndex = blockIdx.x * blockDim.x + threadIdx.x+offset;

    // Make sure the index to the last elemnt of the pool region
    if ((outIndex-offset) < n){        

        // Based on outindex and outx (size of output) we can get the column we are in (in terms of pooling regions). 
        unsigned int outCol = outIndex/outx;
        // Plane we are currently on.
        unsigned int plane = outCol/outy;
        // Have to mod the outCol at this point in case we are not on the first plane.
        outCol = outCol % outy;
        // Have to also get the output row as well.
        unsigned int outRow = outIndex % outx;
        // Have to get which image we're on too.
        unsigned int image = plane/outk;
        // Have to mod plane by pook.
        plane = plane % outk;

        // top left element of pooling region (in the input image).
        unsigned int poolIndex = outRow*poolx+outCol*(imx*pooly)+plane*(imx*imy*poolk)+image*(imx*imy*imk);

        // Get current image element's linear index in the input.
        unsigned int inIndex = (int)oinds[outIndex];
        unsigned int k = inIndex / (poolx*pooly);
        unsigned int x =  inIndex % poolx;
        unsigned int y = (inIndex / poolx) % pooly; 
        inIndex = poolIndex+k*imx*imy+y*imx+x;

        omaxes[outIndex] = idata[inIndex];
    }
}


/* REVERSE MAX POOLING when the indices are passed in*/
__global__ void REVMAXPOOLF3D(                     
                      float * idata,                       
                      float * omaxes,
                      float * iinds,
                      int n,
                      int offset, 
                      int imx,
                      int imy,
                      int imk,
                      int outx,
                      int outy,
                      int outk,
                      int poolx,
                      int pooly,
                      int poolk)
{  
    // OPPOSITE TO THE ABOVE KERNELS BECAUSE WE LOOP OVER THE POOLING INPUT MAPS NOW.
    // So imxy are the pooled sizes, outxy are the unpooled sizes.
    // Get the linear intext into idata (do not touch this).
    unsigned int inIndex = blockIdx.x * blockDim.x + threadIdx.x+offset;

    // Make sure the index to the last elemnt of the pool region
    if ((inIndex-offset) < n){

        // Based on outindex and outx (size of output) we can get the column we are in (in terms of pooling regions). 
        unsigned int inCol = inIndex/imx;
        // Plane we are currently on.
        unsigned int plane = inCol/imy;
        // Have to mod the outCol at this point in case we are not on the first plane.
        inCol = inCol % imy;
        // Have to also get the output row as well.
        unsigned int inRow = inIndex % imx;
        // Have to get which image we're on too.
        unsigned int image = plane/imk;
        // Have to mod plane by poolk as well.
        plane = plane % imk;

        // top left element of pooling region (in the OUTPUT image).
        unsigned int poolIndex = inRow*poolx+inCol*(outx*pooly)+plane*(outx*outy*poolk)+image*(outx*outy*outk);

        // Get current image element's linear index in the input.
        unsigned int outIndex = (int)iinds[inIndex];
        unsigned int k = outIndex / (poolx*pooly);
        unsigned int x =  outIndex % poolx;
        unsigned int y = (outIndex / poolx) % pooly;
        outIndex = poolIndex+k*outx*outy+y*outx+x;

        omaxes[outIndex] = idata[inIndex];
    }
}











}
