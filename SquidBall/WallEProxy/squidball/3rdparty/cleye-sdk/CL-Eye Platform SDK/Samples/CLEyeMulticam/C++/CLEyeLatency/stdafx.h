// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once
#pragma warning(disable: 4996)

#include "targetver.h"

#include <stdio.h>
#include <conio.h>
#include <tchar.h>
#include <windows.h>

#include <cv.h>
#include <highgui.h>
#pragma comment(lib, "cv")
#pragma comment(lib, "cxcore")
#pragma comment(lib, "highgui")

#include "CLEyeMulticam.h"

static LARGE_INTEGER _frequencyPT;

static inline PVOID ProfileMSStart()
{
	PLARGE_INTEGER start = new LARGE_INTEGER;
	QueryPerformanceFrequency(&_frequencyPT);
	QueryPerformanceCounter(start);
	return (PVOID)start;
}

static inline double ProfileMSEnd(PVOID p)
{
	LARGE_INTEGER stop, diff;
	PLARGE_INTEGER start = (PLARGE_INTEGER)p;
	QueryPerformanceCounter(&stop);
	diff.QuadPart = stop.QuadPart - start->QuadPart;
	double timeMs = 1000.0 * ((double)diff.QuadPart / (double)_frequencyPT.QuadPart);
	delete start;
	return timeMs;
}
