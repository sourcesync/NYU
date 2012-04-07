//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// This file is part of CL-EyeMulticam SDK
//
// C++ CLEyeLatency Sample Application
//
// For updates and file downloads go to: http://codelaboratories.com
//
// Copyright 2008-2010 (c) Code Laboratories, Inc. All rights reserved.
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "stdafx.h"
#include <vector>
using namespace std;

PVOID latency = NULL;
#define OUTPUT_WINDOW	"Output Window"

// LED control class
class CLEyeLED
{
	bool _set, _reset;
	CLEyeCameraInstance _cam;
	HANDLE _hThread;
	bool _running;
public:
	CLEyeLED(CLEyeCameraInstance cam) : _cam(cam), _set(false), _reset(false), _running(false){}
	bool Start()
	{
		_running = true;
		// Start CLEye image capture thread
		_hThread = CreateThread(NULL, 0, &CLEyeLED::ThreadFunction, this, 0, 0);
		if(_hThread == NULL)
		{
			MessageBox(NULL,"Could not create capture thread","CLEyeLED", MB_ICONEXCLAMATION);
			return false;
		}
		CLEyeCameraLED(_cam, false);
		return true;
	}
	void Stop()
	{
		if(!_running)	return;
		_running = false;
		_set = true;
		_reset = true;
	}
	void Set()
	{
		_set = true;
	}
	void Reset()
	{
		_reset = true;
	}
	void Run()
	{
		while(_running)
		{
			while(!_set) Sleep(10);
			_set = false;
			if(!_running)	break;
			// image capturing loop
			int i = (rand() % 571) + 131;
			Sleep(i);
			CLEyeCameraLED(_cam, true);
			latency = ProfileMSStart();
			while(!_reset) Sleep(10);
			_reset = false;
			CLEyeCameraLED(_cam, false);
		};
	}
	static DWORD WINAPI ThreadFunction(LPVOID instance)
	{
		// seed the RNG with current tick count and thread id
		srand(GetTickCount() + GetCurrentThreadId());
		// forward thread to Capture function
		CLEyeLED *pThis = (CLEyeLED *)instance;
		pThis->Run();
		return 0;
	}
};

// Camera capture class
class CLEyeCapture
{
	GUID _cameraGUID;
	CLEyeCameraInstance _cam;
	CLEyeCameraColorMode _mode;
	CLEyeCameraResolution _resolution;
	int _fps;
	HANDLE _hThread;
	bool _running;
	double measuredCnt;
	bool _isColor;
public:
	double tMin, tMax, tAvg, curr;
public:
	CLEyeCapture(CLEyeCameraResolution resolution, CLEyeCameraColorMode mode, int fps) :
	  _resolution(resolution), 
		  _mode(mode), 
		  _fps(fps), 
		  _running(false)
	  {
		  _cameraGUID = CLEyeGetCameraUUID(0);
		  if(_mode == CLEYE_COLOR_PROCESSED || _mode == CLEYE_COLOR_RAW)
			  _isColor = true;
		  else
			  _isColor = false;
	  }
	  bool Start()
	  {
		  _running = true;
		  // Start CLEyeCapture image capture thread
		  _hThread = CreateThread(NULL, 0, &CLEyeCapture::CaptureThread, this, 0, 0);
		  if(_hThread == NULL)
		  {
			  MessageBox(NULL,"Could not create capture thread","CLEyeCapture", MB_ICONEXCLAMATION);
			  return false;
		  }
		  return true;
	  }
	  void Stop()
	  {
		  if(!_running)	return;
		  _running = false;
		  WaitForSingleObject(_hThread, 2000);
	  }
	  void Run()
	  {
		  int w, h;
		  IplImage *pCapImage;

		  // Create camera instances
		  _cam = CLEyeCreateCamera(_cameraGUID, _mode, _resolution, _fps);
		  if(_cam == NULL)	return;

		  CLEyeLED led(_cam);

		  // Get camera frame dimensions
		  CLEyeCameraGetFrameDimensions(_cam, w, h);
		  // Create the OpenCV images
		  pCapImage = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, _isColor ? 4 : 1);

		  // Set some camera parameters
		  CLEyeSetCameraParameter(_cam, CLEYE_GAIN, 0);
		  CLEyeSetCameraParameter(_cam, CLEYE_EXPOSURE, 511);

		  // Start capturing
		  CLEyeCameraStart(_cam);
		  Sleep(100);
		  led.Start();

		  tMin = 1000; 
		  tMax = 0;
		  tAvg = 0;
		  curr = 0;
		  measuredCnt = 0;
		  printf("\n");

		  // image capturing loop
		  while(_running)
		  {
			  PBYTE pCapBuffer;
			  // Capture camera images
			  cvGetImageRawData(pCapImage, &pCapBuffer);
			  CLEyeCameraGetFrame(_cam, pCapBuffer);
			  // find non-black frame
			  for(int i = 0; i < (w * h * (_isColor ? 4 : 1)); i++)
				  if(pCapBuffer[i] > 15)
				  {
					  if(latency != NULL)
					  {
							curr = ProfileMSEnd(latency);
							if(curr < tMin)	tMin = curr;
							if(curr > tMax)	tMax = curr;
							tAvg = ((tAvg * measuredCnt) + curr) / (measuredCnt+1);
							measuredCnt += 1;

							latency=NULL;
							led.Reset();
							led.Set();
					  }
					  else
					  {
						  led.Set();
					  }
					  break;
				  }
 			  cvShowImage(OUTPUT_WINDOW, pCapImage);
		  }
		  led.Stop();
		  Sleep(1000);
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
		  // seed the RNG with current tick count and thread id
		  srand(GetTickCount() + GetCurrentThreadId());
		  // forward thread to Capture function
		  CLEyeCapture *pThis = (CLEyeCapture *)instance;
		  pThis->Run();
		  return 0;
	  }
};

// list of formats to test
int formats[] = { CLEYE_MONO_PROCESSED, CLEYE_COLOR_PROCESSED, CLEYE_MONO_RAW, CLEYE_COLOR_RAW, CLEYE_BAYER_RAW };
char *formatName[] = { "CLEYE_MONO_PROCESSED", "CLEYE_COLOR_PROCESSED", "CLEYE_MONO_RAW", "CLEYE_COLOR_RAW", "CLEYE_BAYER_RAW" };
// list of QVGA frame rates to test
float ratesQvga[] = { 15, 20, 30, 40, 50, 60, 75, 90, 100, 120, 150, 187 };
// list of VGA frame rates to test
float ratesVga[] = { 15, 20, 30, 40, 50, 60, 75 };

int _tmain(int argc, _TCHAR* argv[])
{
	CLEyeCapture *cam = NULL;
	// Query for number of connected cameras
	int numCams = CLEyeGetCameraCount();
	if(numCams == 0)
	{
		printf("No PS3Eye cameras detected\n");
		return -1;
	}

	printf("CL-Eye Camera Latency Test\n");
	printf("Hit <ESC> to exit\n");
	cvNamedWindow(OUTPUT_WINDOW, CV_WINDOW_AUTOSIZE);

	// The <ESC> key will exit the program
	bool abort = false;

	// QVGA latency test
	for(int f = 0; f < sizeof(formats)/sizeof(int) && !abort; f++)
	{
		for(int r = 0; r < sizeof(ratesQvga)/sizeof(float) && !abort; r++)
		{
			printf("\n\nQVGA > %s > %g", formatName[f], ratesQvga[r]);
			latency = NULL;
			// Create camera capture object
			cam = new CLEyeCapture(CLEYE_QVGA, (CLEyeCameraColorMode)formats[f], ratesQvga[r]);
			// start capture
			cam->Start();
			// do 15 seconds measurement
			for(int i = 0; i < 30 && !abort; i++)
			{
				if(cvWaitKey(500) == 0x1b)	abort = true;
				printf("Latency Min: %gms  Max: %gms  Avg: %gms    \r", cam->tMin, cam->tMax, cam->tAvg);
			}
			cam->Stop();
			Sleep(1000);
			delete cam;
		}
	}

	// VGA latency test
	for(int f = 0; f < sizeof(formats)/sizeof(int) && !abort; f++)
	{
		for(int r = 0; r < sizeof(ratesVga)/sizeof(float) && !abort; r++)
		{
			printf("\n\nVGA > %s > %g", formatName[f], ratesVga[r]);
			latency = NULL;
			// Create camera capture object
			cam = new CLEyeCapture(CLEYE_VGA, (CLEyeCameraColorMode)formats[f], ratesVga[r]);
			// start capture
			cam->Start();
			// do 15 seconds measurement
			for(int i = 0; i < 30 && !abort; i++)
			{
				if(cvWaitKey(500) == 0x1b)	abort = true;
				printf("Latency Min: %gms  Max: %gms  Avg: %gms    \r", cam->tMin, cam->tMax, cam->tAvg);
			}
			cam->Stop();
			Sleep(1000);
			delete cam;
		}
	}
	cvDestroyWindow(OUTPUT_WINDOW);
	return 0;
}
