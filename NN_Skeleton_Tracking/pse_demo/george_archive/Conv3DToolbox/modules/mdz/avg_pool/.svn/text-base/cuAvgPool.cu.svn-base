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

/* AVG POOLING when no indices are passed in. */
__global__ void AVGPOOLFNOIND(                     
                      float * idata,                       
                      float * oavges,
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
                        //if(fabs(idata[inIndex])>fabs(mx)){
                        //    mx = idata[inIndex];
                        //    mind = x+y*poolx;
                        //}
                        mx += idata[inIndex];
                        mind += 1;
                    }
                }
            }
        }
        /*curInd = xIndex + x;*/
        oavges[outIndex] = mx/mind;
        //oinds[outIndex] = mind;
    }
}

/* AVG POOLING when no indices are passed in. */
__global__ void REVAVGPOOLF(                     
                      float * idata,                       
                      float * oavges,
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
    unsigned int inIndex = blockIdx.x * blockDim.x + threadIdx.x+offset;

    // Make sure the index to the last elemnt of the pool region
    if ((inIndex-offset) < n){

        // Based on inIndex and imx (size of image) we can get the column we are in (in terms of pooling regions). 
        unsigned int inCol = inIndex/imx;
        // Plane we are currently on.
        unsigned int plane = inCol/imy;
        // Have to mod the inCol at this point in case we are not on the first plane.
        inCol = inCol % imy;
        // Have to also get the image row as well.
        unsigned int inRow = inIndex % imx;

        // top left element of pooling region (in the input image).
        unsigned int poolIndex = inRow*poolx+inCol*(outx*pooly)+plane*outx*outy;
        // Also need an index to the current element of the inptu pool region.
        unsigned int outIndex = 0;

        float avg = idata[inIndex]/((float)(poolx*pooly));
        
        // Loop over the pooling region in the image. 
        for(int y=0;y<pooly;y++){    
            // Check we are in boundary in y dimension.
            if(inCol*pooly+y<outy){
                for(int x=0;x<poolx;x++){  
                    if(inRow*poolx+x<outx){
                        // Get current image element's linear index in the input.
                        outIndex = poolIndex+y*outx+x;
                        // Copy same input into each pooling region.
                        oavges[outIndex] = avg;
                    }
                }
            }
        }        
    }
}











/**********************3D pooling**********************************/


/* AVG POOLING when no indices are passed in. */
__global__ void AVGPOOLFNOIND3D(                     
                      float * idata,                       
                      float * oavges,
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
                                //if(fabs(idata[inIndex])>fabs(mx)){
                                    //mx = idata[inIndex];
                                    //mind = x+y*poolx+k*poolx*pooly;
                                //}
                                mx += idata[inIndex];
                                mind += 1;
                            }
                        }
                    }
                }
            }
        }

        /*curInd = xIndex + x;*/
        oavges[outIndex] = mx/mind;
    }
}








/* AVG POOLING when no indices are passed in. */
__global__ void REVAVGPOOLF3D(                     
                      float * idata,                       
                      float * oavges,
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
    unsigned int inIndex = blockIdx.x * blockDim.x + threadIdx.x+offset;

    // Make sure the index to the last elemnt of the pool region
    if ((inIndex-offset) < n){

        // Based on inIndex and imx (size of image) we can get the column we are in (in terms of pooling regions). 
        unsigned int inCol = inIndex/imx;
        // Plane we are currently on.
        unsigned int plane = inCol/imy;
        // Have to mod the inCol at this point in case we are not on the first plane.
        inCol = inCol % imy;
        // Have to also get the output row as well.
        unsigned int inRow = inIndex % imx;
        // Have to get which image we're on too.
        unsigned int image = plane/imk;
        // Have to mod plane as well by poolk.
        plane = plane % imk;

        // top left element of pooling region (in the input image).
        unsigned int poolIndex = inRow*poolx+inCol*(outx*pooly)+plane*(outx*outy*poolk)+image*(outx*outy*outk);
        // Also need an index to the current element of the inptu pool region.
        unsigned int outIndex = 0;

        float avg = idata[inIndex]/((float)(poolx*pooly*poolk));
      
        // Now also loop of the k dimension.
        for(int k=0;k<poolk;k++){
            // Make sure we don't go over the number of image planes.
            if(plane*poolk+k<outk){
                // Loop over the pooling region in the image. 
                for(int y=0;y<pooly;y++){    
                    // Check we are in boundary in y dimension.
                    if(inCol*pooly+y<outy){
                        for(int x=0;x<poolx;x++){  
                            if(inRow*poolx+x<outx){
                                // Get current image element's linear index in the input.
                                outIndex = poolIndex+k*outx*outy+y*outx+x;

                                oavges[outIndex] = avg;                                
                            }
                        }
                    }
                }
            }
        }
    }
}





}
