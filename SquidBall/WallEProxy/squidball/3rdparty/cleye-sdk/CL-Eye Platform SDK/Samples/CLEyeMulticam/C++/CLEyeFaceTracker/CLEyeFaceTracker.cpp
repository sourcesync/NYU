//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// This file is part of CL-EyeMulticam SDK
//
// C++ CLEyeFaceTracker Sample Application
//
// For updates and file downloads go to: http://codelaboratories.com
//
// Copyright 2008-2010 (c) Code Laboratories, Inc. All rights reserved.
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "stdafx.h"

// Sample camera capture and processing class
class CLEyeCameraCapture
{
	CHAR _windowName[256];
	GUID _cameraGUID;
	CLEyeCameraInstance _cam;
	CLEyeCameraColorMode _mode;
	CLEyeCameraResolution _resolution;
	float _fps;
	HANDLE _hThread;
	bool _running;
public:
	CLEyeCameraCapture(LPSTR windowName, GUID cameraGUID, CLEyeCameraColorMode mode, CLEyeCameraResolution resolution, float fps) :
	_cameraGUID(cameraGUID), _cam(NULL), _mode(mode), _resolution(resolution), _fps(fps), _running(false)
	{
		strcpy(_windowName, windowName);
	}
	bool StartCapture()
	{
		_running = true;
		cvNamedWindow(_windowName, CV_WINDOW_AUTOSIZE);
		// Start CLEye image capture thread
		_hThread = CreateThread(NULL, 0, &CLEyeCameraCapture::CaptureThread, this, 0, 0);
		if(_hThread == NULL)
		{
			MessageBox(NULL,"Could not create capture thread","CLEyeMulticamTest", MB_ICONEXCLAMATION);
			return false;
		}
		return true;
	}
	void StopCapture()
	{
		if(!_running)	return;
		_running = false;
		WaitForSingleObject(_hThread, 1000);
		cvDestroyWindow(_windowName);
	}
	void IncrementCameraParameter(int param)
	{
		if(!_cam)	return;
		CLEyeSetCameraParameter(_cam, (CLEyeCameraParameter)param, CLEyeGetCameraParameter(_cam, (CLEyeCameraParameter)param)+10);
	}
	void DecrementCameraParameter(int param)
	{
		if(!_cam)	return;
		CLEyeSetCameraParameter(_cam, (CLEyeCameraParameter)param, CLEyeGetCameraParameter(_cam, (CLEyeCameraParameter)param)-10);
	}
	void Run()
	{
		int w, h;
		IplImage *pCapImage;
		PBYTE pCapBuffer = NULL;
		// Create camera instance
		_cam = CLEyeCreateCamera(_cameraGUID, _mode, _resolution, _fps);
		if(_cam == NULL)		return;
		// Get camera frame dimensions
		CLEyeCameraGetFrameDimensions(_cam, w, h);
		// Depending on color mode chosen, create the appropriate OpenCV image
		if(_mode == CLEYE_COLOR_PROCESSED || _mode == CLEYE_COLOR_RAW)
			pCapImage = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, 4);
		else
			pCapImage = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, 1);

		// Set some camera parameters
		CLEyeSetCameraParameter(_cam, CLEYE_GAIN, 10);
		CLEyeSetCameraParameter(_cam, CLEYE_EXPOSURE, 511);

		// Start capturing
		CLEyeCameraStart(_cam);

		CvMemStorage* storage = cvCreateMemStorage(0);
		// Get the current app path
		char strPathName[_MAX_PATH];
		GetModuleFileName(NULL, strPathName, _MAX_PATH);
		*(strrchr(strPathName, '\\') + 1) = '\0';
		// append the xml file name
		strcat(strPathName, "haarcascade_frontalface_default.xml");
		CvHaarClassifierCascade* cascade = cvLoadHaarClassifierCascade(strPathName, cvSize(24, 24));
		IplImage* image = cvCreateImage(cvSize(pCapImage->width, pCapImage->height), IPL_DEPTH_8U, 3);
		IplImage* temp = cvCreateImage(cvSize(pCapImage->width >> 1, pCapImage->height >> 1), IPL_DEPTH_8U, 3);
		// image capturing loop
		while(_running)
		{
			cvGetImageRawData(pCapImage, &pCapBuffer);
			CLEyeCameraGetFrame(_cam, pCapBuffer);

			cvConvertImage(pCapImage, image);

			cvPyrDown(image, temp, CV_GAUSSIAN_5x5);
			cvClearMemStorage(storage);

			if(cascade)
			{
				CvSeq* faces = cvHaarDetectObjects(temp, cascade, storage, 1.2, 2, CV_HAAR_DO_CANNY_PRUNING);
				for(int i = 0; i < faces->total; i++)
				{
					CvPoint pt1, pt2;
					CvRect* r = (CvRect*)cvGetSeqElem(faces, i);

					pt1.x = r->x * 2;
					pt2.x = (r->x + r->width) * 2;
					pt1.y = r->y * 2;
					pt2.y = (r->y + r->height) * 2;
					cvRectangle(image, pt1, pt2, CV_RGB(255, 0, 0), 3);
				}
			}
			cvShowImage(_windowName, image);
		}
		cvReleaseImage(&temp);
		cvReleaseImage(&image);

		// Stop camera capture
		CLEyeCameraStop(_cam);
		// Destroy camera object
		CLEyeDestroyCamera(_cam);
		// Destroy the allocated OpenCV image
		cvReleaseImage(&pCapImage);
		_cam = NULL;
	}
	static DWORD WINAPI CaptureThread(LPVOID instance)
	{
		// seed the rng with current tick count and thread id
		srand(GetTickCount() + GetCurrentThreadId());
		// forward thread to Capture function
		CLEyeCameraCapture *pThis = (CLEyeCameraCapture *)instance;
		pThis->Run();
		return 0;
	}
};

int _tmain(int argc, _TCHAR* argv[])
{
	CLEyeCameraCapture *cam = NULL;
	// Query for number of connected cameras
	int numCams = CLEyeGetCameraCount();
	if(numCams == 0)
	{
		printf("No PS3Eye cameras detected\n");
		return -1;
	}
	char windowName[64];
	// Query unique camera uuid
	GUID guid = CLEyeGetCameraUUID(0);
	printf("Camera GUID: [%08x-%04x-%04x-%02x%02x%02x%02x%02x%02x%02x%02x]\n", 
		guid.Data1, guid.Data2, guid.Data3,
		guid.Data4[0], guid.Data4[1], guid.Data4[2],
		guid.Data4[3], guid.Data4[4], guid.Data4[5],
		guid.Data4[6], guid.Data4[7]);
	sprintf(windowName, "Face Tracker Window");
	// Create camera capture object
	// Randomize resolution and color mode
	cam = new CLEyeCameraCapture(windowName, guid, CLEYE_COLOR_PROCESSED, CLEYE_VGA, 30);
	printf("Starting capture\n");
	cam->StartCapture();

	printf("Use the following keys to change camera parameters:\n"
		"\t'g' - select gain parameter\n"
		"\t'e' - select exposure parameter\n"
		"\t'z' - select zoom parameter\n"
		"\t'r' - select rotation parameter\n"
		"\t'+' - increment selected parameter\n"
		"\t'-' - decrement selected parameter\n");
	// The <ESC> key will exit the program
	CLEyeCameraCapture *pCam = NULL;
	int param = -1, key;
	while((key = cvWaitKey(0)) != 0x1b)
	{
		switch(key)
		{
		case 'g':	case 'G':	printf("Parameter Gain\n");		param = CLEYE_GAIN;		break;
		case 'e':	case 'E':	printf("Parameter Exposure\n");	param = CLEYE_EXPOSURE;	break;
		case 'z':	case 'Z':	printf("Parameter Zoom\n");		param = CLEYE_ZOOM;		break;
		case 'r':	case 'R':	printf("Parameter Rotation\n");	param = CLEYE_ROTATION;	break;
		case '+':	if(cam)		cam->IncrementCameraParameter(param);					break;
		case '-':	if(cam)		cam->DecrementCameraParameter(param);					break;
		}
	}
	cam->StopCapture();
	delete cam;
	return 0;
}

