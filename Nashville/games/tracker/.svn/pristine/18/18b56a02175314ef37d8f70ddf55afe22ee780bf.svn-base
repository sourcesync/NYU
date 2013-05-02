// camlib.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "camlib.h"


#include "opencv/cv.h"
#include "opencv/highgui.h"

// This is an example of an exported variable
CAMLIB_API int ncamlib=0;

// This is an example of an exported function.
CAMLIB_API int fncamlib(void)
{
	return 42;
}

// This is the constructor of a class that has been exported.
// see camlib.h for the class definition
Ccamlib::Ccamlib()
{
	return;
}

#define MAX_CAPTURES 10

static CvCapture *capture[MAX_CAPTURES];
static CvSize imgSize[MAX_CAPTURES];
static int camid[MAX_CAPTURES];
static int num_captures = 0;

static int erode = 0;
static int dilate = 0;
static int c = 0;
static int t = 0;

CAMLIB_API int camlib_init(int cam, int draw)
{
	// Initiliaze capture and get first image...
	capture[num_captures] = cvCaptureFromCAM(cam);
	if (!capture) 
	{
		printf("ERROR: No capture.\n");
		return -1;
	}

  	IplImage *img = cvQueryFrame(capture[num_captures]);
	if ( !img )
	{
		printf("ERROR: No image.\n");
		return -1;
	}

  	imgSize[num_captures] = cvGetSize(img);
	camid[num_captures ] = cam;

	int id = num_captures;
	num_captures++;

	if (draw>0)
	{
		char winname[256];
		sprintf(winname,"cam%d",cam);
		cvNamedWindow(winname);
	}

	return id;
}



CAMLIB_API int camlib_frame(int id, int draw)
{
	IplImage *img = cvQueryFrame(capture[id]);
  	if (!img) 
	{
		printf("ERROR: Cannot get frame!");
		return -1;
	}
	
	// Create a gray scale with regions of interest...
	IplImage *gray = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
	cvCvtColor( img, gray, CV_BGR2GRAY );	

	// Threshold the grayscale to a binary...
	cvThreshold(gray, gray, 200, 255, CV_THRESH_BINARY);
	
	IplImage *contourImage = cvCloneImage(gray);

	IplImage *deImage = cvCloneImage( contourImage );
		
	if (erode>0)
	{
		cvErode(deImage,deImage,NULL,erode);
	}
	if (dilate>0)
	{
		cvDilate(deImage,deImage,NULL,dilate);
	}

	//	Show it...
	//cvShowImage( "image", img);
	
	if (draw)
	{
		char winname[256];
		sprintf(winname,"cam%d", camid[id]);
		cvShowImage( winname, deImage);
	}

	//	Release any temp resources...
	cvReleaseImage( &gray );
	cvReleaseImage( &contourImage );
	cvReleaseImage( &deImage );

    char k = cvWaitKey(10);
	if (k==27) 
	{
		return 0;
	}
	else if (k=='d') 
	{ 
		if ( dilate >0 ) dilate--;
		printf("dilate=%d\n", dilate);
	}
	else if (k=='D')  
	{ 
		dilate++;
		printf("dilate=%d\n", dilate);
	}
	else if (k=='e') 
	{ 
		if ( erode >0 ) erode--;
		printf("erode=%d\n", erode);
	}
	else if (k=='E')  
	{ 
		erode++;
		printf("erode=%d\n", erode);
	}
	else if (k=='c')
	{
		if (c==0) c=1;
		else c=0;
	}
	else if (k=='t')
	{
		t++;
	}

	return 1;
}


void cvShowManyImages(char* title, int nArgs, ...) {

    // img - Used for getting the arguments 
    IplImage *img;

    // DispImage - the image in which input images are to be copied
    IplImage *DispImage;

    int size;
    int i;
    int m, n;
    int x, y;

    // w - Maximum number of images in a row 
    // h - Maximum number of images in a column 
    int w, h;

    // scale - How much we have to resize the image
    float scale;
    int max;

    // If the number of arguments is lesser than 0 or greater than 12
    // return without displaying 
    if(nArgs <= 0) {
        printf("Number of arguments too small....\n");
        return;
    }
    else if(nArgs > 12) {
        printf("Number of arguments too large....\n");
        return;
    }
    // Determine the size of the image, 
    // and the number of rows/cols 
    // from number of arguments 
    else if (nArgs == 1) {
        w = h = 1;
        size = 300;
    }
    else if (nArgs == 2) {
        w = 2; h = 1;
        size = 300;
    }
    else if (nArgs == 3 || nArgs == 4) {
        w = 2; h = 2;
        size = 300;
    }
    else if (nArgs == 5 || nArgs == 6) {
        w = 3; h = 2;
        size = 200;
    }
    else if (nArgs == 7 || nArgs == 8) {
        w = 4; h = 2;
        size = 200;
    }
    else {
        w = 4; h = 3;
        size = 150;
    }

    // Create a new 3 channel image
    DispImage = cvCreateImage( cvSize(100 + size*w, 60 + size*h), 8, 3 );

    // Used to get the arguments passed
    va_list args;
    va_start(args, nArgs);

    // Loop for nArgs number of arguments
    for (i = 0, m = 20, n = 20; i < nArgs; i++, m += (20 + size)) {

        // Get the Pointer to the IplImage
        img = va_arg(args, IplImage*);

        // Check whether it is NULL or not
        // If it is NULL, release the image, and return
        if(img == 0) {
            printf("Invalid arguments");
            cvReleaseImage(&DispImage);
            return;
        }

        // Find the width and height of the image
        x = img->width;
        y = img->height;

        // Find whether height or width is greater in order to resize the image
        max = (x > y)? x: y;

        // Find the scaling factor to resize the image
        scale = (float) ( (float) max / size );

        // Used to Align the images
        if( i % w == 0 && m!= 20) {
            m = 20;
            n+= 20 + size;
        }

        // Set the image ROI to display the current image
        cvSetImageROI(DispImage, cvRect(m, n, (int)( x/scale ), (int)( y/scale )));

        // Resize the input image and copy the it to the Single Big Image
        cvResize(img, DispImage);

        // Reset the ROI in order to display the next image
        cvResetImageROI(DispImage);
    }

    // Create a new window, and show the Single Big Image
    cvNamedWindow( title, 1 );
    cvShowImage( title, DispImage);

    cvWaitKey();
    cvDestroyWindow(title);

    // End the number of arguments
    va_end(args);

    // Release the Image Memory
    cvReleaseImage(&DispImage);
}