// blobslib.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "blobslib.h"

// Cristóbal Carnero Liñán <grendel.ccl@gmail.com>

#include <iostream>
#include <sstream>
using namespace std;
//#include <iomanip>

#ifdef WIN32
#include <cv.h>
#include <highgui.h>
#include <cxcore.h>
#else
#include <opencv/cv.h>
#include <opencv/highgui.h>
#endif

#include "CLEyeMulticam.h"
#include <cvblob.h>
using namespace cvb;

// This is an example of an exported variable
BLOBSLIB_API int nblobslib=0;

// This is an example of an exported function.
BLOBSLIB_API int fnblobslib(void)
{
	return 42;
}

//	globals...
//static int drawmode = RAW;
//static int erode = ERODE;
//static int dilate = DILATE;
static int _lastgain = -1;
static int _lastexposure = -1;
//static int flipmode=FLIP;

static CvCapture *capture = NULL;
static CvFont font;
static IplConvKernel* emorphKernel = NULL;
static IplConvKernel* dmorphKernel = NULL;
static unsigned int frameNumber=0;
static IplImage *bg_mask = NULL;
static IplImage *inv_bg_mask = NULL;
static int process_events = 1;
static bool _running = false;
static HANDLE _hThread = NULL;
static GUID _cameraGUID;
static CLEyeCameraInstance _cam;
static CLEyeCameraColorMode _mode;
static CLEyeCameraResolution _resolution;
static float _fps;
static int w, h;
static IplImage *pCapImage;
static PBYTE pCapBuffer = NULL;
static IplImage *pr, *pg, *pb, *pa;
static int _last_emorph_rows;
static int _last_emorph_cols;
static int _last_emorph_shape=0;
static int _last_dmorph_rows;
static int _last_dmorph_cols;
static int _last_dmorph_shape=0;
static int _fullscreen = 0;
static IplImage *logonormal = NULL;
static IplImage *logohigh = NULL;

double GetRandomNormalized()
{
	return (double)(rand()-(RAND_MAX>>1))/(double)(RAND_MAX>>1);
}

//	init...
int blobslib_init(int cam)
{
	// CB CvTracks tracks;

	//	windowing...
	cvNamedWindow("gw_object_tracking", CV_WINDOW_AUTOSIZE);

	//	some preset images...
	logonormal = cvLoadImage( "C:\\c2c\\c2c_640x480.jpg");
	logohigh = cvLoadImage( "C:\\c2c\\c2c_1980x1200.jpg");

	if ( logohigh )
	{
		cvShowImage( "gw_object_tracking", logohigh );
	}

#ifdef OCVCAM
	//	input...
	capture = cvCaptureFromCAM(cam);
	if (!capture)
	{
		return -1;
	}

	cvGrabFrame(capture);
	IplImage *img = cvRetrieveFrame(capture);
	if (!img)
	{
		return -1;
	}
#else
	int i = cam;
	_cameraGUID = CLEyeGetCameraUUID(cam);
	printf("Camera %d GUID: [%08x-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x]\n", 
						i+1, _cameraGUID.Data1, _cameraGUID.Data2, _cameraGUID.Data3,
						_cameraGUID.Data4[0], _cameraGUID.Data4[1], _cameraGUID.Data4[2],
						_cameraGUID.Data4[3], _cameraGUID.Data4[4], _cameraGUID.Data4[5],
						_cameraGUID.Data4[6], _cameraGUID.Data4[7]);
	//_mode = rand()<(RAND_MAX>>1) ? CLEYE_COLOR_PROCESSED : CLEYE_MONO_PROCESSED; 
	_mode = CLEYE_COLOR_PROCESSED;

	//_resolution = rand()<(RAND_MAX>>1) ? CLEYE_VGA : CLEYE_QVGA;
	_resolution = CLEYE_VGA;

	_fps = 30;

	// Create camera instance
	_cam = CLEyeCreateCamera(_cameraGUID, _mode, _resolution, _fps);
	if(_cam == NULL)		return -1;

	// Get camera frame dimensions
	CLEyeCameraGetFrameDimensions(_cam, w, h);

	// Depending on color mode chosen, create the appropriate OpenCV image
	if(_mode == CLEYE_COLOR_PROCESSED || _mode == CLEYE_COLOR_RAW)
	{
		pCapImage = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, 4);
		pr = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, 1);
		pg = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, 1);
		pb = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, 1);
		pa = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, 1);
	}
	else
	{
		pCapImage = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, 1);
	}

	// Set some camera parameters
	CLEyeSetCameraParameter(_cam, CLEYE_GAIN, 0);
	CLEyeSetCameraParameter(_cam, CLEYE_EXPOSURE, 511);
 	CLEyeSetCameraParameter(_cam, CLEYE_ZOOM, 0 ); //(int)(GetRandomNormalized()*100.0));
 	CLEyeSetCameraParameter(_cam, CLEYE_ROTATION, 0); //(int)(GetRandomNormalized()*300.0));

	// Start capturing
	CLEyeCameraStart(_cam);
	cvGetImageRawData(pCapImage, &pCapBuffer);

#endif

	//	font...
	double hScale=1.0;
	double vScale=1.0;
	int    lineWidth=1;
	cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, hScale,vScale,0,lineWidth);

	//	morphological...
	emorphKernel = cvCreateStructuringElementEx( MORPH_EROWS, MORPH_ECOLUMNS, 1, 1, CV_SHAPE_ELLIPSE, NULL);
	_last_emorph_rows = MORPH_EROWS;
	_last_emorph_cols = MORPH_ECOLUMNS;
	_last_emorph_shape = 0;
	dmorphKernel = cvCreateStructuringElementEx( MORPH_DROWS, MORPH_DCOLUMNS, 1, 1, CV_SHAPE_ELLIPSE, NULL);
	_last_dmorph_rows = MORPH_DROWS;
	_last_dmorph_cols = MORPH_DCOLUMNS;
	_last_dmorph_shape = 0;
	//morphKernel = cvCreateStructuringElementEx(24, 24, 1, 1, CV_SHAPE_ELLIPSE, NULL);

	return 1;
}

int blobslib_do(blobtype *pblobs, int *num_blobs, char keyp)
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
	parms.drawmode = RAW;
	parms.flipmode = FLIP;
	parms.e_shape = MORPH_ESHAPE;
	parms.e_rows = MORPH_EROWS;
	parms.e_cols = MORPH_ECOLUMNS;
	parms.d_shape = MORPH_DSHAPE;
	parms.d_rows = MORPH_DROWS;
	parms.d_cols = MORPH_DCOLUMNS;
	parms.keyp = keyp;

	return blobslib_doall( pblobs, num_blobs, &parms );
}

int _show( IplImage *img, char *title, char *txt, int r, int g, int b)
{
	//IplImage *display = cvCloneImage(img);
	IplImage *display = NULL;
	if (_fullscreen)
	{
		display = cvCreateImage( cvSize(1280,1024), img->depth, img->nChannels );
	}
	else
	{
		display = cvCloneImage(img);
	}
	cvResize( img, display );
	if (txt)
	{
		cvPutText (display,txt,cvPoint(20,20), &font, cvScalar(r,g,b));
	}
	cvShowImage(title, display);
	cvReleaseImage( &display );
	return 1;
}

//	do...

int blobslib_doall( blobtype *pblobs, int *num_blobs, blobparamstype *parms )
{

	//while (cvGrabFrame(capture))
#ifdef OCVCAM
	if (!cvGrabFrame(capture) )
		return -1;
#endif
	{
#ifdef OCVCAM
		//	Get raw cam image...
		IplImage *img = cvRetrieveFrame(capture);
#else
		//	Show raw cam image...
		if ( parms->drawmode == LOGO )
		{
			if ( _fullscreen && logohigh )
				_show( logohigh, "gw_object_tracking", ".",0,0,0);
			else if ( !_fullscreen && logonormal )
				_show( logonormal, "gw_object_tracking", ".",0,0,0);
		}

		//	Set camera parameters first and be smart about it!!
		if ( _lastgain != parms->gain )
		{
			CLEyeSetCameraParameter(_cam, CLEYE_GAIN, parms->gain );
			_lastgain = parms->gain;
		}
		if ( _lastexposure != parms->exposure )
		{
			CLEyeSetCameraParameter(_cam, CLEYE_EXPOSURE, parms->exposure );
			_lastexposure = parms->exposure;
		}

		//	Get raw image...
		CLEyeCameraGetFrame(_cam, pCapBuffer);
		CvSize imgSize = cvGetSize( pCapImage );
		cvSplit( pCapImage, pb, pg, pr, NULL );
		IplImage *img  = cvCreateImage(imgSize, 8, 3);
		cvMerge( pb, pg, pr, NULL, img );
#endif

		//	Possibly flip raw image here...
		if ( parms->flipmode) cvFlip( img, NULL, 1);

		//	What is this ?
		IplImage *frame = cvCreateImage(imgSize, img->depth, img->nChannels);
		cvConvertScale(img, frame, 1, 0);
		
		//	Show raw cam image...
		if ( parms->drawmode == RAW )
		{
			_show( img, "gw_object_tracking", "raw", 255,255,0);
#if 0
			IplImage *display = cvCloneImage(img);
			cvPutText (display,"raw",cvPoint(20,20), &font, cvScalar(255,255,0));
			cvShowImage("gw_object_tracking", display);
			cvReleaseImage( &display );
#endif
		}
	
		// BG mask, first time init only...
		if ( ! bg_mask )
		{
			bg_mask = cvCreateImage(imgSize, 8, 1);
			cvZero( bg_mask );
			inv_bg_mask = cvCreateImage(imgSize, 8, 1);
			cvNot( bg_mask, inv_bg_mask );
		}

		// Show BG mask...
		if ( parms->drawmode==BGMASK )
		{
			_show( inv_bg_mask, "gw_object_tracking", "bg_mask", 0,0,0);
#if 0
			IplImage *display = cvCloneImage(inv_bg_mask);
			cvPutText (display,"bg mask",cvPoint(20,20), &font, cvScalar(0,0,0));
			cvShowImage("gw_object_tracking", display);
			cvReleaseImage( &display );
#endif
		}

		//	Figure out who does key processing...
		char k = 0;
		if ( (parms->keyp==0) || (parms->keyp=='0') ) // processed here...
		{
			k = cvWaitKey(1);
		}
		else  // caller sets these parameters
		{
			k = parms->keyp;
		}

		if ((k&0xff)==27)
		{
     		return 0;
		}
		else if ( k=='r' ) // show raw cam image...
		{
			parms->drawmode = RAW;
		}
		else if ( k=='h' ) // show thresholding...
		{
			parms->drawmode = THRESHOLD;
		}
		else if ( k=='s') // show segmentation...
		{
   			parms->drawmode = SEGMENTATION;
		}
		else if ( k=='l') // show lables...
		{
     		parms->drawmode = LABELS;
		}
		else if ( k=='t' ) // show tracks...
		{
			parms->drawmode = TRACKS;
		}
		else if ( k=='b' ) // show bg mask...
		{
			parms->drawmode = BGMASK;
		}
		else if ( k=='e' ) // decrease erosion...
		{
			if (parms->erode>0) parms->erode--;
			printf("INFO: erode=%d\n", parms->erode);
		}
		else if ( k=='E' ) // increase erosion...
		{
			parms->erode++;
			printf("INFO: erode=%d\n", parms->erode);
		}
		else if ( k=='d' ) // decrease dilation...
		{
			if (parms->dilate>0) parms->dilate--;
			printf("INFO: dilate=%d\n", parms->dilate);
		}
		else if ( k=='D' ) // increase dilation...
		{
			parms->dilate++;
			printf("INFO: dilate=%d\n", parms->dilate);
		}
		else if ( k=='p' ) // print info...
		{
			printf("INFO: erode=%d\n", parms->erode);
			printf("INFO: dilate=%d\n", parms->dilate);
			//cout << "yo" << "\n";
		}
		else if ( k=='w' )  // indicate desire to set bg mask here...
		{
			parms->sig_setmask = 1;
		}
		else if ( k=='z' ) // reset bg mask...
		{
			parms->sig_resetmask = 1;
		}
		else if ( k=='x' )
		{
			parms->flipmode = !parms->flipmode;	
		}
		else if ( k=='j' )
		{
			if ( parms->e_rows>2) parms->e_rows--;
		}
		else if ( k=='J' )
		{
			parms->e_rows++;
		}
		else if ( k=='k' )
		{
			if ( parms->e_cols>2) parms->e_cols--;
		}
		else if ( k=='K' )
		{
			parms->e_cols++;
		}
		else if ( k=='u' )
		{
			if ( parms->d_rows>2) parms->d_rows--;
		}
		else if ( k=='U' )
		{
			parms->d_rows++;
		}
		else if ( k=='i' )
		{
			if ( parms->d_cols>2) parms->d_cols--;
		}
		else if ( k=='I' )
		{
			parms->d_cols++;
		}
		else if ( k=='q' )
		{
			_fullscreen = !_fullscreen;
			parms->drawmode = LOGO;
		}
		else if ( k=='a' )
		{
			parms->drawmode = LOGO;
		}
		else if ( k=='g' )
		{
			parms->drawmode = NODISPLAY;
		}
		else if ( k=='7' )
		{
			parms->sig_savemask = 1;
		}
		else if ( k=='8' )
		{
			parms->sig_loadmask = 1;
		}

		//	Thresholding...
		IplImage *thresholded = cvCreateImage(imgSize, 8, 1);
		cvInRangeS( img, cvScalar( 
			parms->bmin, parms->gmin,parms->rmin,0), 
			cvScalar( parms->bmax,parms->gmax,parms->rmax), 
			thresholded );
		if ( parms->drawmode==THRESHOLD )
		{
			_show( thresholded, "gw_object_tracking", "threshold", 255, 255, 0 );
#if 0
			IplImage *display = cvCloneImage(thresholded);
			cvPutText (display,"threshold",cvPoint(20,20), &font, cvScalar(255,255,0));
    		cvShowImage("gw_object_tracking", display);
			cvReleaseImage( &display );
#endif
		}

		//	BG mask programmed here (note, it gets programmed after it gets calculated right above...
		if (parms->sig_setmask)
		{
			if (bg_mask) cvReleaseImage( &bg_mask);
			if (inv_bg_mask) cvReleaseImage( &inv_bg_mask);
			bg_mask = cvCloneImage( thresholded );
			inv_bg_mask = cvCloneImage( thresholded );
			cvNot( bg_mask, inv_bg_mask );
			parms->sig_setmask = 0; // reset signal!!
		}
		if (parms->sig_resetmask)
		{
			cvZero( bg_mask );
			cvNot( bg_mask, inv_bg_mask );
			parms->sig_resetmask = 0; // reset signal!!
		}

		//	load mask here...
		if (parms->sig_loadmask)
		{
			
#if 1
			IplImage *loaded_mask = cvLoadImage( DEFAULT_MASK_PATH, 0);
			if ( loaded_mask )
			{	
				if (bg_mask) cvReleaseImage( &bg_mask );
				if (inv_bg_mask) cvReleaseImage( &inv_bg_mask );
				bg_mask = loaded_mask;
				inv_bg_mask = cvCreateImage(imgSize, 8, 1);
				cvNot( bg_mask, inv_bg_mask );
			}
#endif
			parms->sig_loadmask = 0;
		}

		//	save mask here...
		if (parms->sig_savemask)
		{
			int p[3]; 
			p[0] = CV_IMWRITE_JPEG_QUALITY; 
			p[1] = 100; 
			p[2] = 0;
			cvSaveImage( DEFAULT_MASK_PATH, bg_mask, p); 
			parms->sig_savemask = 0;
		}
	
		//	Segmentation...
		IplImage *segmentated = cvCreateImage(imgSize, 8, 1);
		cvAnd( thresholded, inv_bg_mask, segmentated );
		//	Possibly deal with erode morph shape change...
		if ( (parms->e_shape != _last_emorph_shape) || 
			(parms->e_rows != _last_emorph_rows) ||
			(parms->e_cols != _last_emorph_cols) )
		{
			if ( emorphKernel!=NULL) cvReleaseStructuringElement( &emorphKernel );
			emorphKernel = cvCreateStructuringElementEx( 
				parms->e_rows, parms->e_cols, 
				1, 1, 
				CV_SHAPE_ELLIPSE, 
				NULL);
			_last_emorph_rows = parms->e_rows;
			_last_emorph_cols = parms->e_cols;
			_last_emorph_shape = parms->e_shape;
		}
		//	Possibly erode...
		if ( parms->erode>0)
		{
			cvErode(segmentated, segmentated, emorphKernel,parms->erode); 
		}
		//	Possibly deal with dilate morph shape change...
		if ( (parms->d_shape != _last_dmorph_shape) || 
			(parms->d_rows != _last_dmorph_rows) ||
			(parms->d_cols != _last_dmorph_cols) )
		{
			if ( dmorphKernel!=NULL) cvReleaseStructuringElement( &dmorphKernel );
			dmorphKernel = cvCreateStructuringElementEx( 
				parms->d_rows, parms->d_cols, 
				1, 1, 
				CV_SHAPE_ELLIPSE, 
				NULL);
			_last_dmorph_rows = parms->d_rows;
			_last_dmorph_cols = parms->d_cols;
			_last_dmorph_shape = parms->d_shape;
		}
		//	Possibly dilate...
		if ( parms->dilate > 0 )
		{
			cvDilate(segmentated, segmentated, dmorphKernel,parms->dilate); 
		}
		if ( parms->drawmode==SEGMENTATION )
		{
			_show(segmentated, "gw_object_tracking", "segmentated", 255,255,0);
#if 0
			IplImage *display = cvCloneImage(segmentated);
			cvPutText (display,"segmentated",cvPoint(20,20), &font, cvScalar(255,255,0));
    		cvShowImage("gw_object_tracking", display);
			cvReleaseImage( &display );
#endif
		}

		//	Labeling and blobbing...
		IplImage *labelImg = cvCreateImage(cvGetSize(frame), IPL_DEPTH_LABEL, 1);
		CvBlobs blobs;
		CvTracks tracks; //CB
		unsigned int result = cvLabel(segmentated, labelImg, blobs);  // secret sauce 3
		cvFilterByArea(blobs, 100, 1000000); //CB   secret sauce 4 (was 500 min)
		if ( parms->drawmode==LABELS)
		{
			IplImage *display = cvCreateImage( cvGetSize(frame), IPL_DEPTH_8U, 3);
			for (unsigned int y=0; y< (unsigned int)labelImg->height; y++)
			for (unsigned int x=0; x<(unsigned int)labelImg->width; x++)
			{
				int lbl = cvGetLabel(labelImg, x, y);
				unsigned char f = 255*lbl;
				cvSet2D( display, y, x, CV_RGB( f,f,f) );
			}

			_show( display, "gw_object_tracking", "labels", 255,255,0 );
#if 0
			cvPutText (display,"labels",cvPoint(20,20), &font, cvScalar(255,255,0));
			cvShowImage("gw_object_tracking", display);
			cvReleaseImage( &display );
#endif
			cvReleaseImage( &display );
		}

		//	Tracking...
		// CB  - GW put it back
		cvUpdateTracks(blobs, tracks, 200., 5);
		if ( parms->drawmode==TRACKS)
		{
			cvRenderTracks(tracks, frame, frame, CV_TRACK_RENDER_ID|CV_TRACK_RENDER_BOUNDING_BOX);
			_show( frame, "gw_object_tracking", "tracks", 255,255,0);
#if 0
			IplImage *display = cvCloneImage(frame);
			cvPutText (display,"tracks",cvPoint(20,20), &font, cvScalar(255,255,0));
			cvShowImage("gw_object_tracking", display);
			cvReleaseImage( &display );
#endif
		} 

		//	Possibly return blob info to caller...
		if (pblobs && num_blobs)
		{
			*num_blobs = 0;
			for (CvTracks::const_iterator it=tracks.begin(); it!=tracks.end(); ++it)
			{
				(*num_blobs)++;
			}

			if ( *num_blobs > 0 )
			{
				int b = 0;
				for (CvTracks::const_iterator it=tracks.begin(); it!=tracks.end(); ++it)
				{
					//cout << " - Centroid: (" << it->second->centroid.x << ", " << it->second->centroid.y << ")" << "\n";
					//printf("centroid %f %f\n", it->second->centroid.x, it->second->centroid.y );
					pblobs[ b ].x = (float)(it->second->centroid.x - imgSize.width/2.0f);
					pblobs[ b ].y = (float)it->second->area;
					pblobs[ b ].z = (float)(imgSize.height - it->second->centroid.y - imgSize.height/2.0f);  // make it so that z goes up...
					b++;
					if (b==MAX_BLOBS)
					{
						printf("WARNING: max blobs reached (%d)\n", MAX_BLOBS);
					}
				}
			}
		}

		//	Release resources...
		cvReleaseTracks(tracks);
		cvReleaseBlobs(blobs);
		cvReleaseImage(&labelImg);
		cvReleaseImage(&segmentated);
		cvReleaseImage(&thresholded);
		cvReleaseImage(&frame);
#ifdef OCVCAM
#else
		cvReleaseImage(&img);
#endif

		frameNumber++;

	}

	return 1;
}

int blobslib_free( blobtype **pblobs )
{
	if (pblobs)
	{
		blobtype *blobs = *pblobs;
		if (blobs)
		{
			free(blobs);
			return 1;
		}
	}
	return 0;
}

int cvblobs_end()
{
	cvReleaseStructuringElement(&emorphKernel);
	cvReleaseStructuringElement(&dmorphKernel);
	cvDestroyWindow("gw_object_tracking");

	return 0;
}
