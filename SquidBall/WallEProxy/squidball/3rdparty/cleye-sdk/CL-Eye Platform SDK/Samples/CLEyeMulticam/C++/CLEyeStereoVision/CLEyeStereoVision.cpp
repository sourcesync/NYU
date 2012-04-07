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
#include <vector>
using namespace std;

// Sample camera capture and processing class
class CLEyeStereoCameraCapture
{
	CHAR _windowName[256];
	CHAR _depthWindowName[256];
	GUID _cameraGUID[2];
	CLEyeCameraInstance _cam[2];
	CLEyeCameraColorMode _mode;
	CLEyeCameraResolution _resolution;
	float _fps;
	HANDLE _hThread;
	bool _running;
public:
	CLEyeStereoCameraCapture(CLEyeCameraResolution resolution, float fps) :
	_mode(CLEYE_MONO_RAW), _resolution(resolution), _fps(fps), _running(false)
	{
		strcpy(_windowName, "Capture Window");
		strcpy(_depthWindowName, "Stereo Depth");
		for(int i = 0; i < 2; i++)
			_cameraGUID[i] = CLEyeGetCameraUUID(i);
	}
	bool StartCapture()
	{
		_running = true;
		cvNamedWindow(_windowName, CV_WINDOW_AUTOSIZE);
		cvNamedWindow(_depthWindowName, CV_WINDOW_AUTOSIZE);
		// Start CLEye image capture thread
		_hThread = CreateThread(NULL, 0, &CLEyeStereoCameraCapture::CaptureThread, this, 0, 0);
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
		cvDestroyWindow(_depthWindowName);
	}
	void IncrementCameraParameter(int param)
	{
		for(int i = 0; i < 2; i++)
		{
			if(!_cam[i])	continue;
			CLEyeSetCameraParameter(_cam[i], (CLEyeCameraParameter)param, CLEyeGetCameraParameter(_cam[i], (CLEyeCameraParameter)param)+10);
		}
	}
	void DecrementCameraParameter(int param)
	{
		for(int i = 0; i < 2; i++)
		{
			if(!_cam[i])	continue;
			CLEyeSetCameraParameter(_cam[i], (CLEyeCameraParameter)param, CLEyeGetCameraParameter(_cam[i], (CLEyeCameraParameter)param)-10);
		}
	}
	void Run()
	{
		int w, h;
		IplImage *pCapImage[2];
		IplImage *pDisplayImage;

		// Create camera instances
		for(int i = 0; i < 2; i++)
		{
			_cam[i] = CLEyeCreateCamera(_cameraGUID[i], _mode, _resolution, _fps);
			if(_cam[i] == NULL)	return;
			// Get camera frame dimensions
			CLEyeCameraGetFrameDimensions(_cam[i], w, h);
			// Create the OpenCV images
			pCapImage[i] = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, 1);

			// Set some camera parameters
			CLEyeSetCameraParameter(_cam[i], CLEYE_GAIN, 0);
			CLEyeSetCameraParameter(_cam[i], CLEYE_EXPOSURE, 511);

			// Start capturing
			CLEyeCameraStart(_cam[i]);
		}
		pDisplayImage = cvCreateImage(cvSize(w*2, h), IPL_DEPTH_8U, 1);

		// Get the current app path
		char strPathName[_MAX_PATH];
		GetModuleFileName(NULL, strPathName, _MAX_PATH);
		*(strrchr(strPathName, '\\') + 1) = '\0';

		Init(w, h);

		// image capturing loop
		while(_running)
		{
			PBYTE pCapBuffer;
			// Capture camera images
			for(int i = 0; i < 2; i++)
			{
				cvGetImageRawData(pCapImage[i], &pCapBuffer);
				CLEyeCameraGetFrame(_cam[i], pCapBuffer, (i==0)?2000:0);
			}

			if(calibrationStarted)
			{
				if(CalibrationAddSample(pCapImage[0], pCapImage[1]))
					printf("CalibrationAddSample succeeded\n");
				else
					printf("CalibrationAddSample failed\n");
			}
			else if(sampleCount && !calibrationDone)
				CalibrationEnd();
			
			if(calibrationDone)
			{
				if(StereoProcess(pCapImage[0], pCapImage[1]))
					cvShowImage(_depthWindowName, imageDepthNormalized);

				// Display stereo image
				for(int i = 0; i < 2; i++)
				{
					cvSetImageROI(pDisplayImage, cvRect(i*w, 0, w, h));
					cvCopy(imagesRectified[i], pDisplayImage);
				}
				cvResetImageROI(pDisplayImage);
			}
			else
			{
				// Display stereo image
				for(int i = 0; i < 2; i++)
				{
					cvSetImageROI(pDisplayImage, cvRect(i*w, 0, w, h));
					cvCopy(pCapImage[i], pDisplayImage);
				}
				cvResetImageROI(pDisplayImage);
			}
			cvShowImage(_windowName, pDisplayImage);
		}

		for(int i = 0; i < 2; i++)
		{
			// Stop camera capture
			CLEyeCameraStop(_cam[i]);
			// Destroy camera object
			CLEyeDestroyCamera(_cam[i]);
			// Destroy the allocated OpenCV image
			cvReleaseImage(&pCapImage[i]);
			_cam[i] = NULL;
		}
	}
	static DWORD WINAPI CaptureThread(LPVOID instance)
	{
		// seed the RNG with current tick count and thread id
		srand(GetTickCount() + GetCurrentThreadId());
		// forward thread to Capture function
		CLEyeStereoCameraCapture *pThis = (CLEyeStereoCameraCapture *)instance;
		pThis->Run();
		return 0;
	}

	int cornersX, cornersY, cornersN;
	int sampleCount;
	bool calibrationStarted;
	bool calibrationDone;

	CvSize imageSize;
	int imageWidth;
	int imageHeight;

	vector<CvPoint2D32f> ponintsTemp[2];
	vector<CvPoint3D32f> objectPoints;
	vector<CvPoint2D32f> points[2];
	vector<int> npoints;
	// matrices resulting from calibration (used for cvRemap to rectify images)
	CvMat *mx1,*my1,*mx2,*my2;

	CvMat* imagesRectified[2];
	CvMat  *imageDepth,*imageDepthNormalized;

	void Init(int imageWidth, int imageHeight)
	{
		imageSize = cvSize(imageWidth, imageHeight);
		mx1 = my1 = mx2 = my2 = 0;
		calibrationStarted = false;
		calibrationDone = false;
		imagesRectified[0] = imagesRectified[1] = imageDepth = imageDepthNormalized = 0;
		imageDepth = 0;
		sampleCount = 0;
	}

	void CalibrationStart(int cornerX, int cornerY)
	{
		cornersX = cornerX;
		cornersY = cornerY;
		cornersN = cornersX * cornersY;
		ponintsTemp[0].resize(cornersN);
		ponintsTemp[1].resize(cornersN);
		sampleCount = 0;
		calibrationStarted = true;
	}

	int CalibrationAddSample(IplImage* imageLeft,IplImage* imageRight)
	{
		if(!calibrationStarted) return false;

		IplImage* image[2] = {imageLeft, imageRight};

		int succeses = 0;
		for(int lr = 0; lr < 2;lr++)
		{
			int cornersDetected = 0;

			// find chessboard corners
			int result = cvFindChessboardCorners(image[lr], cvSize(cornersX, cornersY), &ponintsTemp[lr][0], &cornersDetected, 
									CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);

			if(result && cornersDetected == cornersN)
			{
				// Calibration will suffer without sub-pixel interpolation
				cvFindCornerSubPix(image[lr], &ponintsTemp[lr][0], cornersDetected, cvSize(11, 11), cvSize(-1,-1),
									cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 30, 0.01));
				succeses++;
			}
		}
		if(succeses == 2)
		{
			printf("CalibrationAddSample: Both samples found %d corners\n", cornersN);
			for(int lr = 0;lr < 2; lr++)
			{
				points[lr].resize((sampleCount+1)*cornersN);
				copy(ponintsTemp[lr].begin(), ponintsTemp[lr].end(),  points[lr].begin() + sampleCount*cornersN);
			}
			sampleCount++;
			return true;
		}
		else
			return false;
	}

	int CalibrationEnd()
	{
		printf("CalibrationEnd Started\n");
		calibrationStarted = false;

		double M1[3][3], M2[3][3], D1[5], D2[5];
		double R[3][3], T[3], E[3][3], F[3][3];
		CvMat _M1,_M2,_D1,_D2,_R,_T,_E,_F;

		_M1 = cvMat(3, 3, CV_64F, M1 );
		_M2 = cvMat(3, 3, CV_64F, M2 );
		_D1 = cvMat(1, 5, CV_64F, D1 );
		_D2 = cvMat(1, 5, CV_64F, D2 );
		_R = cvMat(3, 3, CV_64F, R );
		_T = cvMat(3, 1, CV_64F, T );
		_E = cvMat(3, 3, CV_64F, E );
		_F = cvMat(3, 3, CV_64F, F );

		objectPoints.resize(sampleCount*cornersN);

		for(int k=0;k<sampleCount;k++)
			for(int i = 0; i < cornersY; i++ )
				for(int j = 0; j < cornersX; j++ )
					objectPoints[k*cornersY*cornersX + i*cornersX + j] = cvPoint3D32f(i, j, 0);


		npoints.resize(sampleCount,cornersN);

		int N = sampleCount * cornersN;

		CvMat _objectPoints = cvMat(1, N, CV_32FC3, &objectPoints[0] );
		CvMat _imagePoints1 = cvMat(1, N, CV_32FC2, &points[0][0] );
		CvMat _imagePoints2 = cvMat(1, N, CV_32FC2, &points[1][0] );
		CvMat _npoints = cvMat(1, npoints.size(), CV_32S, &npoints[0] );
		cvSetIdentity(&_M1);
		cvSetIdentity(&_M2);
		cvZero(&_D1);
		cvZero(&_D2);

		cvStereoCalibrate( &_objectPoints, &_imagePoints1,
			&_imagePoints2, &_npoints,
			&_M1, &_D1, &_M2, &_D2,
			imageSize, &_R, &_T, &_E, &_F,
			cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5),
			CV_CALIB_FIX_ASPECT_RATIO + CV_CALIB_ZERO_TANGENT_DIST + CV_CALIB_SAME_FOCAL_LENGTH
			);

		cvUndistortPoints( &_imagePoints1, &_imagePoints1,&_M1, &_D1, 0, &_M1 );
		cvUndistortPoints( &_imagePoints2, &_imagePoints2,&_M2, &_D2, 0, &_M2 );

		double R1[3][3], R2[3][3];
		CvMat _R1 = cvMat(3, 3, CV_64F, R1);
		CvMat _R2 = cvMat(3, 3, CV_64F, R2);

		double H1[3][3], H2[3][3], iM[3][3];
		CvMat _H1 = cvMat(3, 3, CV_64F, H1);
		CvMat _H2 = cvMat(3, 3, CV_64F, H2);
		CvMat _iM = cvMat(3, 3, CV_64F, iM);

		cvStereoRectifyUncalibrated(&_imagePoints1,&_imagePoints2, &_F,imageSize,&_H1, &_H2, 3);
		cvInvert(&_M1, &_iM);
		cvMatMul(&_H1, &_M1, &_R1);
		cvMatMul(&_iM, &_R1, &_R1);
		cvInvert(&_M2, &_iM);
		cvMatMul(&_H2, &_M2, &_R2);
		cvMatMul(&_iM, &_R2, &_R2);

		cvReleaseMat(&mx1);
		cvReleaseMat(&my1);
		cvReleaseMat(&mx2);
		cvReleaseMat(&my2);
		mx1 = cvCreateMat( imageSize.height,imageSize.width, CV_32F );
		my1 = cvCreateMat( imageSize.height,imageSize.width, CV_32F );
		mx2 = cvCreateMat( imageSize.height,imageSize.width, CV_32F );
		my2 = cvCreateMat( imageSize.height,imageSize.width, CV_32F );

		cvInitUndistortRectifyMap(&_M1,&_D1,&_R1,&_M1,mx1,my1);
		cvInitUndistortRectifyMap(&_M2,&_D2,&_R2,&_M2,mx2,my2);

		calibrationDone = true;
		printf("CalibrationEnd Done\n");
		return true;
	}

	int StereoProcess(CvArr* imageSrcLeft, CvArr* imageSrcRight)
	{
		printf("StereoProcess Start\n");
		if(!calibrationDone) return false;

		if(!imagesRectified[0]) imagesRectified[0] = cvCreateMat( imageSize.height,imageSize.width, CV_8U );
		if(!imagesRectified[1]) imagesRectified[1] = cvCreateMat( imageSize.height,imageSize.width, CV_8U );
		if(!imageDepth) imageDepth = cvCreateMat( imageSize.height,imageSize.width, CV_16S );
		if(!imageDepthNormalized) imageDepthNormalized = cvCreateMat( imageSize.height,imageSize.width, CV_8U );

		// rectify images
		cvRemap(imageSrcLeft, imagesRectified[0] , mx1, my1 );
		cvRemap(imageSrcRight, imagesRectified[1] , mx2, my2 );

		CvStereoBMState *BMState = cvCreateStereoBMState();
		BMState->preFilterSize=41;
		BMState->preFilterCap=31;
		BMState->SADWindowSize=41;
		BMState->minDisparity=-64;
		BMState->numberOfDisparities=128;
		BMState->textureThreshold=10;
		BMState->uniquenessRatio=15;

		cvFindStereoCorrespondenceBM(imagesRectified[0], imagesRectified[1], imageDepth, BMState);
		cvNormalize(imageDepth, imageDepthNormalized, 0, 256, CV_MINMAX );

		cvReleaseStereoBMState(&BMState);
		printf("StereoProcess Done\n");
		return true;
	}

	void Calibration()
	{
		if(!calibrationStarted)	CalibrationStart(10, 7);
		else					calibrationStarted = false;
	}
};

int _tmain(int argc, _TCHAR* argv[])
{
	CLEyeStereoCameraCapture *cam = NULL;
	// Query for number of connected cameras
	int numCams = CLEyeGetCameraCount();
	if(numCams < 2)
	{
		printf("No PS3Eye cameras detected\n");
		return -1;
	}
	// Create camera capture object
	cam = new CLEyeStereoCameraCapture(CLEYE_QVGA, 30);
	printf("Starting capture\n");
	cam->StartCapture();

	printf("Use the following keys to change camera parameters:\n"
		"\t'g' - select gain parameter\n"
		"\t'e' - select exposure parameter\n"
		"\t'+' - increment selected parameter\n"
		"\t'-' - decrement selected parameter\n"
		"\t'c' - start/end chessboard camera calibration\n");
	// The <ESC> key will exit the program
	CLEyeStereoCameraCapture *pCam = NULL;
	int param = -1, key;
	while((key = cvWaitKey(0)) != 0x1b)
	{
		switch(key)
		{
			case 'g':	case 'G':	printf("Parameter Gain\n");		param = CLEYE_GAIN;		break;
			case 'e':	case 'E':	printf("Parameter Exposure\n");	param = CLEYE_EXPOSURE;	break;
			case '+':	if(cam)		cam->IncrementCameraParameter(param);					break;
			case '-':	if(cam)		cam->DecrementCameraParameter(param);					break;
			case 'c':	case 'C':	cam->Calibration();										break;
		}
	}
	cam->StopCapture();
	delete cam;
	return 0;
}

