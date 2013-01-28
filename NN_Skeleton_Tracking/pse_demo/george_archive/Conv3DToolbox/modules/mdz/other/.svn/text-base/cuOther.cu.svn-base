/*#include "GPUkernel.hh"*/

typedef float2  Complex;
typedef double2 DoubleComplex;

extern "C" {

/***********************L1 SHRINKAGE FUNCTION*****************************/
__global__ void L1SHRINK(                     
                      float * idata,                      
                      int n,
                      int offset,
                      float beta)
{  
    // OPPOSITE TO THE ABOVE KERNELS BECAUSE WE LOOP OVER THE POOLING INPUT MAPS NOW.
    // So imxy are the pooled sizes, outxy are the unpooled sizes.
    // Get the linear intext into idata (do not touch this).
    unsigned int inIndex = blockIdx.x * blockDim.x + threadIdx.x+offset;

    // Make sure the index to the last elemnt of the pool region
    if ((inIndex-offset) < n){
        float z = idata[inIndex];
        if(z<-beta){
            idata[inIndex] = z+beta;
        }else if(z>beta){
            idata[inIndex] = z-beta;
        }else{
            idata[inIndex] = 0.0;
        }
        //idata[inIndex] = (float)beta;
    }
}



/****************** Get sign of elements in A**********************************/
__global__ void SIGN(                     
                      float * idata,   
                      float * odata,
                      int n,
                      int offset)
{  
    // OPPOSITE TO THE ABOVE KERNELS BECAUSE WE LOOP OVER THE POOLING INPUT MAPS NOW.
    // So imxy are the pooled sizes, outxy are the unpooled sizes.
    // Get the linear intext into idata (do not touch this).
    unsigned int inIndex = blockIdx.x * blockDim.x + threadIdx.x+offset;

    // Make sure the index to the last elemnt of the pool region
    if ((inIndex-offset) < n){
        float z = idata[inIndex];
        if(z<0){
            odata[inIndex] = -1.0;
        }else if(z>0){
            odata[inIndex] = 1.0;
        }else{
            odata[inIndex] = 0.0;
        }
        //idata[inIndex] = (float)beta;
    }
}






/*
 * Block size 16x16.
 * Probably a better idea to allocate multiple blocks per image so you don't have
 * to loop inside the block.
 */
__global__ void PADARRAY(float* images, float* targets, const int imgSize, const int paddingSize, const int numImages) {
    const int imgIdx = blockIdx.y * gridDim.x + blockIdx.x;
    if (imgIdx < numImages) {
        const int targetSize = imgSize + 2 * paddingSize;
        images += imgIdx * imgSize * imgSize;
        //targets += imgIdx * targetSize * targetSize + (paddingSize*targetSize) + paddingSize;
        targets += imgIdx * targetSize * targetSize;
        for (int y = threadIdx.y; y < targetSize; y += 16) {
            for (int x = threadIdx.x; x < targetSize; x += 16) {
                if(x<paddingSize){
                    targets[(y*targetSize) + x] = 0;
                }else if(x>=imgSize+paddingSize){
                    targets[(y*targetSize) + x] = 0;
                }else if(y<paddingSize){
                    targets[(y*targetSize) + x] = 0;
                }else if(y>=imgSize+paddingSize){
                    targets[(y*targetSize) + x] = 0;
                }else{
                    targets[(y*targetSize) + x] = images[((y-paddingSize)*imgSize) + x - paddingSize];
                }   
            }
        }
    }
}





/***********************max(idata,idata2) function*****************************/
__global__ void MAXSAMESIZE(                     
                      float * idata, 
                      float * idata2,
                      float * out,
                      int n,
                      int offset)
{  
    // OPPOSITE TO THE ABOVE KERNELS BECAUSE WE LOOP OVER THE POOLING INPUT MAPS NOW.
    // So imxy are the pooled sizes, outxy are the unpooled sizes.
    // Get the linear intext into idata (do not touch this).
    unsigned int inIndex = blockIdx.x * blockDim.x + threadIdx.x+offset;

    // Make sure the index to the last elemnt of the pool region
    if ((inIndex-offset) < n){

        if(idata[inIndex]>=idata2[inIndex]){
            out[inIndex] = idata[inIndex];
        }else{
            out[inIndex] = idata2[inIndex];
        }
        //out[inIndex] = (float)n;
    }
}

/***********************max(idata,idata2) function*****************************/
__global__ void MAXSCALAR(                     
                      float * idata, 
                      float * idata2,
                      float * out,
                      int n,
                      int offset)
{  
    // OPPOSITE TO THE ABOVE KERNELS BECAUSE WE LOOP OVER THE POOLING INPUT MAPS NOW.
    // So imxy are the pooled sizes, outxy are the unpooled sizes.
    // Get the linear intext into idata (do not touch this).
    unsigned int inIndex = blockIdx.x * blockDim.x + threadIdx.x + offset;

    // Make sure the index to the last elemnt of the pool region
    if ((inIndex-offset) < n){

        if(idata[inIndex]>=idata2[0]){
            out[inIndex] = idata[inIndex];
        }else{
            out[inIndex] = idata2[0];
        }
    }
}

/***********************min(idata,idata2) function*****************************/
__global__ void MINSAMESIZE(                     
                      float * idata, 
                      float * idata2,
                      float * out,
                      int n,
                      int offset)
{  
    // OPPOSITE TO THE ABOVE KERNELS BECAUSE WE LOOP OVER THE POOLING INPUT MAPS NOW.
    // So imxy are the pooled sizes, outxy are the unpooled sizes.
    // Get the linear intext into idata (do not touch this).
    unsigned int inIndex = blockIdx.x * blockDim.x + threadIdx.x+offset;

    // Make sure the index to the last elemnt of the pool region
    if ((inIndex-offset) < n){

        if(idata[inIndex]<=idata2[inIndex]){
            out[inIndex] = idata[inIndex];
        }else{
            out[inIndex] = idata2[inIndex];
        }
        //out[inIndex] = (float)n;
    }
}

/***********************min(idata,idata2) function*****************************/
__global__ void MINSCALAR(                     
                      float * idata, 
                      float * idata2,
                      float * out,
                      int n,
                      int offset)
{  
    // OPPOSITE TO THE ABOVE KERNELS BECAUSE WE LOOP OVER THE POOLING INPUT MAPS NOW.
    // So imxy are the pooled sizes, outxy are the unpooled sizes.
    // Get the linear intext into idata (do not touch this).
    unsigned int inIndex = blockIdx.x * blockDim.x + threadIdx.x + offset;

    // Make sure the index to the last elemnt of the pool region
    if ((inIndex-offset) < n){

        if(idata[inIndex]<=idata2[0]){
            out[inIndex] = idata[inIndex];
        }else{
            out[inIndex] = idata2[0];
        }
    }
}







/***********************max(idata,[],1) function*****************************/
__global__ void MAXDIM1(                     
                      float * idata, 
                      float * out,
                      float * oind,
                      int n,
                      int offset,
                      int imx,
                      int imy)
{  
    // So imxy are the pooled sizes, outxy are the unpooled sizes.
    // Get the linear intext into idata (do not touch this).
    unsigned int outIndex = blockIdx.x * blockDim.x + threadIdx.x + offset;
    unsigned int inIndex = 0;

    // Make sure the index to the last elemnt of the pool region
    if ((outIndex-offset) < n){
        // Get the column to start from.
        unsigned int startIndex = outIndex*imx;
        float mx=idata[startIndex];
        float mind = 0;
        // Loop over the first dimension.
        for(unsigned int i=0;i<imx;i++){
            // Get the index into the input array.
            inIndex = startIndex + i;
            // Compute max.
            if(idata[inIndex]>mx){
                mx = idata[inIndex];
                mind = (float)i;
            }
        }
        out[outIndex] = mx;
        oind[outIndex] = (float)mind+1; // Matlab index
    }
}




/***********************max(idata,[],last_dimension_of_idata) function*****************************/
__global__ void MAXDIMLAST(                     
                      float * idata, 
                      float * out,
                      float * oind,
                      int n,
                      int offset,
                      int imx,
                      int imy)
{  
    // So imxy are the pooled sizes, outxy are the unpooled sizes.
    // Get the linear intext into idata (do not touch this).
    unsigned int outIndex = blockIdx.x * blockDim.x + threadIdx.x + offset;
    unsigned int inIndex = 0;

    // Make sure the index to the last elemnt of the pool region
    if ((outIndex-offset) < n){
        // Get the column to start from.
        unsigned int startIndex = outIndex;
        float mx=idata[startIndex];
        float mind = 0;
        // Loop over the second dimension.
        for(unsigned int j=0;j<imy;j++){
            // Get the index into the input array.
            inIndex = startIndex + j*imx;
            // Compute max.
            if(idata[inIndex]>mx){
                mx = idata[inIndex];
                mind = (float)j;
            }
        }
        out[outIndex] = mx;
        oind[outIndex] = (float)mind+1; // Matlab index.
    }
}




/***********************max(idata,[],last_dimension_of_idata) function*****************************/
__global__ void MAXDIMMIDDLE(                     
                      float * idata, 
                      float * out,
                      float * oind,
                      int n,
                      int offset,
                      int imx,
                      int imy)
{  
    // So imxy are the pooled sizes, outxy are the unpooled sizes.
    // Get the linear intext into idata (do not touch this).
    unsigned int outIndex = blockIdx.x * blockDim.x + threadIdx.x + offset;
    unsigned int inIndex = 0;
    // Now also need to know what 3rd dimension we're on.
    // Since the 2nd dimension will always be 1, we just divide by imx.
    unsigned int d3 = outIndex/imx;
    unsigned int d1 = outIndex % imx;

    // Make sure the index to the last elemnt of the pool region
    if ((outIndex-offset) < n){
        // Get the 
        unsigned int startIndex = d1 + d3*imx*imy;
        float mx=idata[startIndex];
        float mind = 0;
        // Loop over the second dimension.
        for(unsigned int j=0;j<imy;j++){
            // Get the index into the input array.
            inIndex = startIndex + j*imx;
            // Compute max.
            if(idata[inIndex]>mx){
                mx = idata[inIndex];
                mind = (float)j;
            }
        }
        out[outIndex] = mx;
        oind[outIndex] = (float)mind+1; // Matlab index
    }
}








/***********************min(idata,[],1) function*****************************/
__global__ void MINDIM1(                     
                      float * idata, 
                      float * out,
                      float * oind,
                      int n,
                      int offset,
                      int imx,
                      int imy)
{  
    // So imxy are the pooled sizes, outxy are the unpooled sizes.
    // Get the linear intext into idata (do not touch this).
    unsigned int outIndex = blockIdx.x * blockDim.x + threadIdx.x + offset;
    unsigned int inIndex = 0;

    // Make sure the index to the last elemnt of the pool region
    if ((outIndex-offset) < n){
        // Get the column to start from.
        unsigned int startIndex = outIndex*imx;
        float mx=idata[startIndex];
        float mind = 0;
        // Loop over the first dimension.
        for(unsigned int i=0;i<imx;i++){
            // Get the index into the input array.
            inIndex = startIndex + i;
            // Compute max.
            if(idata[inIndex]<mx){
                mx = idata[inIndex];
                mind = (float)i;
            }
        }
        out[outIndex] = mx;
        oind[outIndex] = (float)mind+1; // Matlab index
    }
}




/***********************min(idata,[],last_dimension_of_idata) function*****************************/
__global__ void MINDIMLAST(                     
                      float * idata, 
                      float * out,
                      float * oind,
                      int n,
                      int offset,
                      int imx,
                      int imy)
{  
    // So imxy are the pooled sizes, outxy are the unpooled sizes.
    // Get the linear intext into idata (do not touch this).
    unsigned int outIndex = blockIdx.x * blockDim.x + threadIdx.x + offset;
    unsigned int inIndex = 0;

    // Make sure the index to the last elemnt of the pool region
    if ((outIndex-offset) < n){
        // Get the column to start from.
        unsigned int startIndex = outIndex;
        float mx=idata[startIndex];
        float mind = 0;
        // Loop over the second dimension.
        for(unsigned int j=0;j<imy;j++){
            // Get the index into the input array.
            inIndex = startIndex + j*imx;
            // Compute max.
            if(idata[inIndex]<mx){
                mx = idata[inIndex];
                mind = (float)j;
            }
        }
        out[outIndex] = mx;
        oind[outIndex] = (float)mind+1; // Matlab index.
    }
}


/***********************min(idata,[],last_dimension_of_idata) function*****************************/
__global__ void MINDIMMIDDLE(                     
                      float * idata, 
                      float * out,
                      float * oind,
                      int n,
                      int offset,
                      int imx,
                      int imy)
{  
    // So imxy are the pooled sizes, outxy are the unpooled sizes.
    // Get the linear intext into idata (do not touch this).
    unsigned int outIndex = blockIdx.x * blockDim.x + threadIdx.x + offset;
    unsigned int inIndex = 0;
    // Now also need to know what 3rd dimension we're on.
    // Since the 2nd dimension will always be 1, we just divide by imx.
    unsigned int d3 = outIndex/imx;
    unsigned int d1 = outIndex % imx;

    // Make sure the index to the last elemnt of the pool region
    if ((outIndex-offset) < n){
        // Get the 
        unsigned int startIndex = d1 + d3*imx*imy;
        float mx=idata[startIndex];
        float mind = 0;
        // Loop over the second dimension.
        for(unsigned int j=0;j<imy;j++){
            // Get the index into the input array.
            inIndex = startIndex + j*imx;
            // Compute max.
            if(idata[inIndex]<mx){
                mx = idata[inIndex];
                mind = (float)j;
            }
        }
        out[outIndex] = mx;
        oind[outIndex] = (float)mind+1; // Matlab index
    }
}












}
