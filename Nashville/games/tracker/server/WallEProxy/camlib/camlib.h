// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the CAMLIB_EXPORTS
// symbol defined on the command line. this symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// CAMLIB_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef CAMLIB_EXPORTS
#define CAMLIB_API __declspec(dllexport)
#else
#define CAMLIB_API __declspec(dllimport)
#endif

// This class is exported from the camlib.dll
class CAMLIB_API Ccamlib {
public:
	Ccamlib(void);
	// TODO: add your methods here.
};

extern CAMLIB_API int ncamlib;

CAMLIB_API int fncamlib(void);
CAMLIB_API int camlib_init(int, int);
CAMLIB_API int camlib_frame(int, int);
