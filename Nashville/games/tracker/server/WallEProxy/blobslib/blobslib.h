// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the BLOBSLIB_EXPORTS
// symbol defined on the command line. this symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// BLOBSLIB_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef BLOBSLIB_EXPORTS
#define BLOBSLIB_API __declspec(dllexport)
#else
#define BLOBSLIB_API __declspec(dllimport)
#endif

// This class is exported from the blobslib.dll
class BLOBSLIB_API Cblobslib {
public:
	Cblobslib(void);
	// TODO: add your methods here.
};

#define MAX_BLOBS 1000

//	Default parameter values.
#define RMIN 100
#define RMAX 256
#define GMIN 100
#define GMAX 256
#define BMIN 100
#define BMAX 256
#define GAIN 0
#define EXPOSURE 50
#define ERODE 0
#define DILATE 15
#define FLIP 0
#define MORPH_ESHAPE 0
#define MORPH_EROWS 3
#define MORPH_ECOLUMNS 3
#define MORPH_DSHAPE 0
#define MORPH_DROWS 3
#define MORPH_DCOLUMNS 3

//	Display modes...
#define RAW				0
#define THRESHOLD		1
#define SEGMENTATION	2
#define LABELS			3
#define TRACKS			4
#define BGMASK			5
#define XOR				6
#define NODISPLAY		100
#define LOGO			101

//	Strings...
#define DEFAULT_MASK_PATH		"c:\\Wally\\mask.jpg"

typedef struct
{
	float x;
	float y;
	float z;
} blobtype;

typedef struct
{
	int rmin;
	int rmax;
	int gmin;
	int gmax;
	int bmin;
	int bmax;
	int gain;
	int exposure;
	int erode;
	int dilate;
	int drawmode;
	int sig_setmask;
	int sig_resetmask;
	int flipmode;
	int e_shape;
	int e_rows;
	int e_cols;
	int d_shape;
	int d_rows;
	int d_cols;
	int sig_savemask;
	int sig_loadmask;
	char keyp;

} blobparamstype;

extern BLOBSLIB_API int nblobslib;

extern "C"
{
	BLOBSLIB_API int fnblobslib(void);
	BLOBSLIB_API int blobslib_init( int cam );
	BLOBSLIB_API int blobslib_do( blobtype *pblobs, int *numblobs, char keyp);
	BLOBSLIB_API int blobslib_doall( blobtype *pblobs, int *num_blobs, blobparamstype *params);
	BLOBSLIB_API int blobslib_free( blobtype **blobs );
	BLOBSLIB_API int blobslib_end(void);
}
