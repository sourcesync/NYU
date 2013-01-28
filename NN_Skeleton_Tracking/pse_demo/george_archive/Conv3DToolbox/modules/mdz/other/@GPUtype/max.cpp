/*!
 * @brief GPUmat GPU implementation of the matlab max function (see implementation
 * for max).
 * 
 * @file
 * @author Matthew Zeiler (zeiler@cs.nyu.edu)
 * @data Apr 29, 2011
 *
 * @pooltoolbox_file @copybrief max.cpp
 * @gpu_file @copybrief max.cpp
 *
 *****************
 * How to compile:
 *
 * In matlab go to the directory of this file and type:
 * >> make all
 *
 * Installs in GPUmat's @GPUtype folder so you can call max directly. 
 */
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#ifdef UNIX
#include <stdint.h>
#endif

#include "mex.h"
#include "math.h"

// CUDA
#include "cuda.h"
#include "cuda_runtime.h"

// #include "cuMaxPool_kernels.cu"
// #include "GPUkernel.hh"
#include "GPUmat.hh"

/*!
 * @copybrief max.cpp
 *
 * @param INPUT   	      prhs[0] // The input you want max of. Does max along first non-singleton if no other parameters are passed in.
 * @param INPUT2          prhs[1] // The same size as input or a scalara to compare to. Pass [] to do max over dimensions.
 * @param DIM             prhs[2] // If [] is arg 2 then this is the dimension to take max over.
 *
 * @retval MAX   	      plhs[0] // Corresponding maxes.
 * @retval IND            plhs[1] // Locations of the maxes along the dimension specified (only when specifying a dimension).
 */

// static paramaters
static CUfunction drvfunScalar;
static CUfunction drvfunMatrices;
static CUfunction drvfunDim1;
static CUfunction drvfunDimMiddle;
static CUfunction drvfunDimLast;

static int init = 0;

static GPUmat *gm;

void hostDriver(CUfunction drvfun, int N, int nrhs, hostdrv_pars_t *prhs);
void hostDriver2(CUfunction drvfun, int N, int nrhs, hostdrv_pars_t *prhs, int imx, int imy);


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    CUresult cudastatus = CUDA_SUCCESS;
    
    if (nrhs != 1 && nrhs != 2 && nrhs!=3)
        mexErrMsgTxt("Wrong number of arguments, two or three required.");
    
    
    if (init == 0) {
        // Initialize function
        //mexLock();
        
        // load GPUmat
        gm = gmGetGPUmat();
        
        // load module
        CUmodule *drvmod = gmGetModule("other");
        
        // load float GPU function
//         CUresult status = cuModuleGetFunction(&drvfunf, *drvmod, "MAXPOOLF");
        CUresult status = cuModuleGetFunction(&drvfunMatrices, *drvmod,  "MAXSAMESIZE");
        if (CUDA_SUCCESS != status) {
            mexErrMsgTxt("Unable to load user function.");
        }
        status = cuModuleGetFunction(&drvfunScalar, *drvmod,  "MAXSCALAR");
        if (CUDA_SUCCESS != status) {
            mexErrMsgTxt("Unable to load user function.");
        }
        status = cuModuleGetFunction(&drvfunDim1, *drvmod,  "MAXDIM1");
        if (CUDA_SUCCESS != status) {
            mexErrMsgTxt("Unable to load user function.");
        }
        status = cuModuleGetFunction(&drvfunDimLast, *drvmod,  "MAXDIMLAST");
        if (CUDA_SUCCESS != status) {
            mexErrMsgTxt("Unable to load user function.");
        }
        status = cuModuleGetFunction(&drvfunDimMiddle, *drvmod,  "MAXDIMMIDDLE");
        if (CUDA_SUCCESS != status) {
            mexErrMsgTxt("Unable to load user function.");
        }
        init = 1;
    }
    
//     double *tempbeta = mxGetPr(prhs[1]); // For the pool_size array.
//     float *beta = (float) mxCalloc(1,sizeof(float));
//     beta[0] = (float)tempbeta[0];
//        float beta = (float)mxGetScalar(prhs[1]);
    
//     GPUtype beta = gm->gputype.mxToGPUtype(prhs[1]);
    
    
//     mexPrintf("beta[i] is: %f\n",beta);
//     mexErrMsgTxt("");
    // mex parameters are:
    
    // 1. IN1
    // 2. IN2
    // 3. OUT
    
    //IN1 is the input GPU array
    GPUtype IN1 = gm->gputype.getGPUtype(prhs[0]);
    
    GPUtype IN2;
    
    int maxdim;
    bool MaxOverDimensions = false;
    
        int nin1 = gm->gputype.getNumel(IN1);
    int ndims = gm->gputype.getNdims(IN1);
    gpuTYPE_t tin1 = gm->gputype.getType(IN1);
    
    
    // create GPUtype, with given dimensions (same size as input).
    int* im_size;
    im_size = (int*) mxCalloc(ndims, sizeof(int));
    im_size = (int*) gm->gputype.getSize(IN1);
    
    
    if(nrhs==1){ // Max over dimension 1.
        MaxOverDimensions = true;
        maxdim = 1;
        for(int i = 0;i<ndims;i++){
            if(im_size[i]>1){
                maxdim = i+1;
                break;
            }
        }
    }else if(mxIsEmpty(prhs[1])==false){
        try{
            IN2 = gm->gputype.mxToGPUtype(prhs[1]);
            IN2 = gm->gputype.doubleToFloat(IN2);
        }catch (...){
            IN2 = gm->gputype.getGPUtype(prhs[1]);
        }
    }else{ // Do max over a dimension.
        if(mxIsEmpty(prhs[2])==true){
            mexErrMsgTxt("need to pass in dimension as 3rd arguement if second arguement is []");
        }else{
            MaxOverDimensions = true;
            maxdim = (int) mxGetScalar(prhs[2]);
        }
    }
    

            // The kernel we ultimately call depends on if the indices were passed in.
        CUfunction drvfun;
    
    
    // If we are just comparing two matrices or a matrix and scalar.
    if(MaxOverDimensions==false){                
        if(nlhs>1){
            mexErrMsgTxt("Max between two indices should only have one output arguement");
        }
        
//         if(gm->gputype.isFloat(IN2))
//             mexPrintf("Second input is a gpuSINGLE\n");
                
        // number of elements
        int nin2 = gm->gputype.getNumel(IN2);
        
//         mexPrintf("nin2: %d\n", nin2);
        
        // Get the number of dimensions.
        
        gpuTYPE_t tin2 = gm->gputype.getType(IN2);
        
        if (tin2 !=gpuFLOAT)
            mexErrMsgTxt("Second Input must be GPUsingle matrix or a single scalar");
                
//     mexErrMsgTxt("Right after out_size\n");
        GPUtype OUT = gm->gputype.create(tin1, ndims, im_size, NULL);
                        
        // check input/out size and type
        if (nin1!=nin2 && nin2!=1)
            mexErrMsgTxt("Input arguments must have the same number of elements.");
        
        if (tin1!=tin2)
            mexErrMsgTxt("Input arguments must be of the same type.");
        
        int nout = gm->gputype.getNumel(OUT);
        
        if (tin1 !=gpuFLOAT)
            mexErrMsgTxt("Input must be GPUsingle");
                
        // I need the pointers to GPU memory
        CUdeviceptr d_IN1  = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(IN1));
        CUdeviceptr d_IN2 = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(IN2));
        CUdeviceptr d_OUT = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(OUT));
                
        hostdrv_pars_t gpuprhs[3];
        int gpunrhs = 3;
        gpuprhs[0] = hostdrv_pars(&d_IN1, sizeof(d_IN1), __alignof(d_IN1));
        gpuprhs[1] = hostdrv_pars(&d_IN2, sizeof(d_IN2), __alignof(d_IN2));
        gpuprhs[2] = hostdrv_pars(&d_OUT, sizeof(d_OUT), __alignof(d_OUT));
                
        int N = nin1;
        
//     mexPrintf("nin1: %d\n",N);
//     mexErrMsgTxt("stopping hre");
        
        if(nin2==1){
            drvfun = drvfunScalar;
        }else{
            drvfun = drvfunMatrices;
        }
                
        hostDriver(drvfun, N, gpunrhs, gpuprhs);
        
        // Finally make the output available to MATLAB
        plhs[0] = gm->gputype.createMxArray(OUT);
        
    }else{ // If we are maxing over dimensions then things are much more complicated.
//        mexPrintf("maxdim is: %d\n", maxdim);
        
//             mexPrintf("nin1: %d\n",nin1);

               // create GPUtype, with given dimensions (same size as input).
        int* out_size;
        out_size = (int*) mxCalloc(ndims, sizeof(int));
        out_size = (int*) gm->gputype.getSize(IN1);
        

        
        if (tin1 !=gpuFLOAT)
            mexErrMsgTxt("Input must be GPUsingle");
        
        // Variable for the new number of output dimensions.
        int newdims = 0;
        int* new_outsize; // The size to create teh new output matrix as.
        int* new_imsize; // The size to pass to the kernl (needs looping dimensions). 
        
        
        if(maxdim==1){
            drvfun = drvfunDim1;            
            newdims = 2;

            new_outsize = (int*) mxCalloc(newdims, sizeof(int));
            new_imsize = (int*) mxCalloc(newdims, sizeof(int));
                      
            //mexPrintf("out_size[0]: %d\n",out_size[0]);
            // Get the original image size we are reducing over.
            new_imsize[0] = out_size[0];

            
            // For reshaping later we also have to set out_size[0] = 1;
            out_size[0] = 1;
            
            // Resize the output for now to be 2D, preserving the first dimension.
            new_outsize[0] = out_size[0];
            new_outsize[1] = out_size[1];
            // Loop over the remaining dimensions to multiply their sizes.
            for(int i=2;i<ndims;i++){                
//                 mexPrintf("i: %d, out_size[i]: %i\n",i,out_size[i]);
                new_outsize[1] *= out_size[i];                
            }            
            // The sizes we pass to the kernel (need image size in dim 1).
            new_imsize[1] = new_outsize[1];
        }else if(maxdim==ndims){ // If doing over the last dimesnion.
            drvfun = drvfunDimLast;                                    
            newdims = 2;

            
//             mexPrintf("Pooling over last dimension:%d \n",maxdim);
            new_outsize = (int*) mxCalloc(newdims, sizeof(int));
            new_imsize = (int*) mxCalloc(newdims, sizeof(int));
                      
            //mexPrintf("out_size[0]: %d\n",out_size[0]);
            // Get the original image size we are reducing over.
            new_imsize[1] = out_size[maxdim-1];

            
            // For reshaping later we also have to set out_size[0] = 1;
            out_size[maxdim-1] = 1;
            
//                      mexPrintf("out_size to reshape to: %d x %d\n",out_size[0],out_size[1]);

            
            // Resize the output for now to be 2D, preserving the first dimension.
            new_outsize[0] = out_size[0];
            new_outsize[1] = out_size[maxdim-1];
            // Loop over the remaining dimensions to multiply their sizes.
            for(int i=1;i<ndims-1;i++){                
//                 mexPrintf("i: %d, out_size[i]: %i\n",i,out_size[i]);
                new_outsize[0] *= out_size[i];                
            }            
            // The sizes we pass to the kernel (need image size in dim 1).
            new_imsize[0] = new_outsize[0];        
            
            
//                         mexPrintf("out_size to reshape to: %d x %d\n", out_size[0], out_size[1]);
//             mexPrintf("mew_imsize to reshape to: %d x %d\n", new_imsize[0], new_imsize[1]);
//             mexPrintf("new_outsize to reshape to: %d x %d\n", new_outsize[0], new_outsize[1]);
            
        }else{ // Want to pool over an arbitrary middle dimension that is not the first or last dimension. 
            drvfun = drvfunDimMiddle;
            
            // Have first last and middle dimensions now.
            newdims = 3;

            new_outsize = (int*) mxCalloc(newdims, sizeof(int));
            new_imsize = (int*) mxCalloc(newdims, sizeof(int));
                      
            //mexPrintf("out_size[0]: %d\n",out_size[0]);
            // Get the original image size we are reducing over.
            new_imsize[1] = out_size[maxdim-1];

            
            // For reshaping later we also have to set out_size[0] = 1;
            out_size[maxdim-1] = 1;
            

            
            // Resize the output for now to be 2D, preserving the first dimension.
            new_outsize[0] = out_size[0];
            new_outsize[1] = out_size[maxdim-1]; // REduced dimension
            new_outsize[2] = out_size[maxdim];
            // Loop over the remaining dimensions to multiply their sizes.
            for(int i=1;i<maxdim-1;i++){                
                new_outsize[0] *= out_size[i];                
            }
            for(int i=maxdim+1;i<ndims;i++){
                new_outsize[2] *= out_size[i];                
            }
            
            // The sizes we pass to the kernel (need image size in dim 1).
            new_imsize[0] = new_outsize[0];  
            new_imsize[2] = new_outsize[2];
//             mexPrintf("out_size to reshape to: %d x %d x %d\n", out_size[0], out_size[1], out_size[2]);
//             mexPrintf("mew_imsize to reshape to: %d x %d x %d\n", new_imsize[0], new_imsize[1],new_imsize[2]);
//             mexPrintf("new_outsize to reshape to: %d x %d x %d\n", new_outsize[0], new_outsize[1], new_outsize[2]);

            
        }
        
        
        
        // Create the output matrices.
        GPUtype OUT = gm->gputype.create(tin1, newdims, new_outsize, NULL);
        gm->gputype.zeros(OUT);
        GPUtype IND = gm->gputype.create(tin1, newdims, new_outsize, NULL);
       // gm->gputype.zeros(IND);
        
        int nout = gm->gputype.getNumel(OUT);

        
//         mexPrintf("nout: %d,newdims %d, ndims %d\n",nout,newdims,ndims);
//          mexPrintf("new_imsize: %d x %d\n",new_imsize[0],new_imsize[1]);
//          mexPrintf("new_outsize: %d x %d\n",new_outsize[0],new_outsize[1]);
        
//         mexErrMsgTxt("");
        
                // I need the pointers to GPU memory
        CUdeviceptr d_IN1  = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(IN1));
        CUdeviceptr d_OUT = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(OUT));
        CUdeviceptr d_IND = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(IND));
       
        
        hostdrv_pars_t gpuprhs[3];
        int gpunrhs = 3;
        gpuprhs[0] = hostdrv_pars(&d_IN1, sizeof(d_IN1), __alignof(d_IN1));
        gpuprhs[1] = hostdrv_pars(&d_OUT, sizeof(d_OUT), __alignof(d_OUT));
        gpuprhs[2] = hostdrv_pars(&d_IND, sizeof(d_IND), __alignof(d_IND));
        
        int N = nout;
        
        
         hostDriver2(drvfun, N, gpunrhs, gpuprhs, (int)new_imsize[0], (int)new_imsize[1]);
        
//          mexPrintf("After hostDriver2");
         
         nout = gm->gputype.getNumel(OUT);
//          mexPrintf("nout: %d\n", nout);
         
         
         // Before we return the results, we have to setSize on them back to the original shape. 
         gm->gputype.setSize(OUT, ndims, out_size);
         gm->gputype.setSize(IND, ndims, out_size);
         
                         nout = gm->gputype.getNumel(OUT);
// mexPrintf("nout: %d\n",nout);
         
        // Finally make the output available to MATLAB
          plhs[0] = gm->gputype.createMxArray(OUT);
         if(nlhs>1)
             plhs[1] = gm->gputype.createMxArray(IND);
         
        
    }
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
}




//host driver
//void hostDriver(CUfunction drvfun, dim3 grid, dim3 threads, int shmem, int imgSizeX, int imgSizeY, int shmemX, int nrhs, hostdrv_pars_t *prhs) {
void hostDriver(CUfunction drvfun, int N, int nrhs, hostdrv_pars_t *prhs){
    //mexPrintf("threads.x: %d threads.y: %d threads.z %d\n",threads.x,threads.y,threads.z);
    
    
    unsigned int maxthreads = 65000;
    // Set threads per block here.
    unsigned int blocksdim1d = 256;
    dim3 threads(blocksdim1d, 1, 1);
    int nstreams = iDivUp(N, maxthreads*blocksdim1d);
    CUresult err = CUDA_SUCCESS;
    for (int str = 0; str < nstreams; str++) {
        int offset = str * maxthreads * blocksdim1d;
        int size = 0;
        if (str == (nstreams - 1))
            size = N - str * maxthreads * blocksdim1d;
        else
            size = maxthreads * blocksdim1d;
        
//         printf("size: %d, str: %d\n",size,str);
        
        
        int gridx = iDivUp(size, blocksdim1d); // number of x blocks
        
        // setup execution parameters
        
        if (CUDA_SUCCESS != (err = cuFuncSetBlockShape(drvfun, threads.x, threads.y, threads.y))) {
            mexErrMsgTxt("Error in cuFuncSetBlockShape");
        }
        
        if (CUDA_SUCCESS != cuFuncSetSharedSize(drvfun, 0)) {
            mexErrMsgTxt("Error in cuFuncSetSharedSize");
        }
        
        //mexPrintf("block shape ok\n");
        
        // add parameters
        int poffset = 0;
        
        // CUDA kernels interface
        // N: number of elements
        for (int p=0;p<nrhs;p++) {
            ALIGN_UP(poffset, prhs[p].align);
            if (CUDA_SUCCESS
                    != cuParamSetv(drvfun, poffset, prhs[p].par, prhs[p].psize)) {
                mexErrMsgTxt("Error in cuParamSetv");
            }
            poffset += prhs[p].psize;
        }
        
        ALIGN_UP(poffset, __alignof(size));
        if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, size)) {
            mexErrMsgTxt("Error in cuParamSeti");
        }
        poffset += sizeof(size);
        
        ALIGN_UP(poffset, __alignof(offset));
        if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, offset)) {
            mexErrMsgTxt("Error in cuParamSeti");
        }
        poffset += sizeof(offset);
        
//         ALIGN_UP(poffset, __alignof(beta));
//         if (CUDA_SUCCESS != cuParamSetv(drvfun, poffset, &beta, sizeof(beta))) {
//             mexErrMsgTxt("Error in cuParamSeti");
//         }
//         poffset += sizeof(beta);
        
//   if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, shmemX)) {
//     mexErrMsgTxt("Error in cuParamSeti");
//   }
//   poffset += sizeof(shmemX);
        
        if (CUDA_SUCCESS != cuParamSetSize(drvfun, poffset)) {
            mexErrMsgTxt("Error in cuParamSetSize");
        }
        
        err = cuLaunchGridAsync(drvfun, gridx, 1, 0);
        if (CUDA_SUCCESS != err) {
            mexErrMsgTxt("Error running kernel");
        }
        
    }
}






//host driver for Pooling over a dimension.
//void hostDriver(CUfunction drvfun, dim3 grid, dim3 threads, int shmem, int imgSizeX, int imgSizeY, int shmemX, int nrhs, hostdrv_pars_t *prhs) {
void hostDriver2(CUfunction drvfun, int N, int nrhs, hostdrv_pars_t *prhs, int imx, int imy){
    //mexPrintf("threads.x: %d threads.y: %d threads.z %d\n",threads.x,threads.y,threads.z);
    
    
    unsigned int maxthreads = 65000;
    // Set threads per block here.
    unsigned int blocksdim1d = 256;
    dim3 threads(blocksdim1d, 1, 1);
    int nstreams = iDivUp(N, maxthreads*blocksdim1d);
    CUresult err = CUDA_SUCCESS;
    for (int str = 0; str < nstreams; str++) {
        int offset = str * maxthreads * blocksdim1d;
        int size = 0;
        if (str == (nstreams - 1))
            size = N - str * maxthreads * blocksdim1d;
        else
            size = maxthreads * blocksdim1d;
        
//         printf("size: %d, str: %d\n",size,str);
        
        
        int gridx = iDivUp(size, blocksdim1d); // number of x blocks
        
        // setup execution parameters
        
        if (CUDA_SUCCESS != (err = cuFuncSetBlockShape(drvfun, threads.x, threads.y, threads.y))) {
            mexErrMsgTxt("Error in cuFuncSetBlockShape");
        }
        
        if (CUDA_SUCCESS != cuFuncSetSharedSize(drvfun, 0)) {
            mexErrMsgTxt("Error in cuFuncSetSharedSize");
        }
        
        //mexPrintf("block shape ok\n");
        
        // add parameters
        int poffset = 0;
        
        // CUDA kernels interface
        // N: number of elements
        for (int p=0;p<nrhs;p++) {
            ALIGN_UP(poffset, prhs[p].align);
            if (CUDA_SUCCESS
                    != cuParamSetv(drvfun, poffset, prhs[p].par, prhs[p].psize)) {
                mexErrMsgTxt("Error in cuParamSetv");
            }
            poffset += prhs[p].psize;
        }
        
        ALIGN_UP(poffset, __alignof(size));
        if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, size)) {
            mexErrMsgTxt("Error in cuParamSeti");
        }
        poffset += sizeof(size);
        
        ALIGN_UP(poffset, __alignof(offset));
        if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, offset)) {
            mexErrMsgTxt("Error in cuParamSeti");
        }
        poffset += sizeof(offset);
        
                ALIGN_UP(poffset, __alignof(imx));
        if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, imx)) {
            mexErrMsgTxt("Error in cuParamSeti");
        }
        poffset += sizeof(imx);
        
                ALIGN_UP(poffset, __alignof(imy));
        if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, imy)) {
            mexErrMsgTxt("Error in cuParamSeti");
        }
        poffset += sizeof(imy);
        
//                 ALIGN_UP(poffset, __alignof(imz));
//         if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, imz)) {
//             mexErrMsgTxt("Error in cuParamSeti");
//         }
//         poffset += sizeof(imz);
        
//         ALIGN_UP(poffset, __alignof(out_size));
//         if (CUDA_SUCCESS != cuParamSetv(drvfun, poffset, out_size, sizeof(int)*outdims)) {
//             mexErrMsgTxt("Error in cuParamSeti");
//         }
//         poffset += sizeof(out_size);
        
//   if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, shmemX)) {
//     mexErrMsgTxt("Error in cuParamSeti");
//   }
//   poffset += sizeof(shmemX);
        
        if (CUDA_SUCCESS != cuParamSetSize(drvfun, poffset)) {
            mexErrMsgTxt("Error in cuParamSetSize");
        }
        
        err = cuLaunchGridAsync(drvfun, gridx, 1, 0);
        if (CUDA_SUCCESS != err) {
            mexErrMsgTxt("Error running kernel");
        }
        
    }
}


















