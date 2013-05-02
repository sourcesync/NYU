// blobslibtest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <stdlib.h>
#include "blobslib.h"

int _tmain(int argc, _TCHAR* argv[])
{
	blobparamstype parms;
	parms.rmin = RMIN;
	parms.rmax = RMAX;
	parms.gmin = GMIN;
	parms.gmax = GMAX;
	parms.bmin = BMIN;
	parms.bmax = BMAX;
	parms.gain = GAIN;
	parms.exposure = EXPOSURE;
	parms.erode = ERODE;
	parms.dilate = DILATE;
	parms.drawmode = LOGO;
	parms.flipmode = FLIP;
	parms.e_shape = MORPH_ESHAPE;
	parms.e_rows = MORPH_EROWS;
	parms.e_cols = MORPH_ECOLUMNS;
	parms.d_shape = MORPH_ESHAPE;
	parms.d_rows = MORPH_EROWS;
	parms.d_cols = MORPH_ECOLUMNS;
	parms.keyp = 0;

	if ( blobslib_init(0)<=0 )
	{
		printf("ERROR: Cannot init blobslib.\n");
		return 1;
	}

	int retv=0;
	blobtype *pblobs = (blobtype *)malloc( sizeof(blobtype)*MAX_BLOBS );
	int num_blobs;
	//while ( (retv = blobslib_do( pblobs, &num_blobs, 0 )) > 0 )
	while ( (retv = blobslib_doall( pblobs, &num_blobs, &parms )) > 0 )
	{
		printf("INFO: Blob count=%d\n", num_blobs);
		for ( int i=0; i< num_blobs; i++)
		{
			printf("INFO: Blob x=%f,y=%f,z=%f\n", pblobs[i].x, pblobs[i].y, pblobs[i].z);
		}
	}
		
	printf("INFO: Done.\n");
		
}

