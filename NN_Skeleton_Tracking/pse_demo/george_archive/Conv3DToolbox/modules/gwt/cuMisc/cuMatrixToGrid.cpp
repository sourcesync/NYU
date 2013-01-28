/* Code common to cuGridToMatrix and cuMatrixToGrid is in gridToMatrix.cpp */

#include "mex.h"
#include "gridToMatrix.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  _gtm(nlhs,plhs,nrhs,prhs,true);

}
