//=============================================================================
// Copyright © 2011 Point Grey Research, Inc. All Rights Reserved.
//
// This software is the confidential and proprietary information of Point
// Grey Research, Inc. ("Confidential Information").  You shall not
// disclose such Confidential Information and shall use it only in
// accordance with the terms of the license agreement you entered into
// with PGR.
//
// PGR MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE
// SOFTWARE, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE, OR NON-INFRINGEMENT. PGR SHALL NOT BE LIABLE FOR ANY DAMAGES
// SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING
// THIS SOFTWARE OR ITS DERIVATIVES.
//=============================================================================

#include "stdafx.h"
#include "FlyCap2_MFC.h"
#include "VideoRecordingPage.h"

const char VideoFormatList[][MAX_PATH] = 
{
	"Uncompressed",
	"M-JPEG",
	"H.264"
};

const CString VideoRecordingPage::MJPEG_QUALITY_DEF = "75";
const CString VideoRecordingPage::H264_BITRATE_DEF = "1000000";
const CString VideoRecordingPage::VIDEO_FRAMERATE_DEF = "15.0";


// VideoRecordingPage dialog

IMPLEMENT_DYNAMIC(VideoRecordingPage, CDialog)

VideoRecordingPage::VideoRecordingPage(CWnd* pParent /*=NULL*/)
	: CDialog(VideoRecordingPage::IDD, pParent)
{

}

VideoRecordingPage::~VideoRecordingPage()
{
}

void VideoRecordingPage::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_COMBO_VIDEO_RECORD_TYPE, m_combo_VideoFormat);
	DDX_Control(pDX, IDC_GROUP_MJPEG, m_group_mjpegOptions);
	DDX_Control(pDX, IDC_STATIC_MJPEG_COMPRESSION, m_static_mjpegCompressionLevel);
	DDX_Control(pDX, IDC_EDIT_MJPEG_COMPRESSION, m_edit_mjpegCompressionLevel);
	DDX_Control(pDX, IDC_SPIN_MJPEG_COMPRESSION, m_spin_mjpegCompressionLevel);
	DDX_Control(pDX, IDC_GROUP_H264, m_group_h264Options);
	DDX_Control(pDX, IDC_STATIC_H264_BITRATE, m_static_h264Bitrate);
	DDX_Control(pDX, IDC_EDIT_H264_BITRATE, m_edit_h264Bitrate);
	DDX_Control(pDX, IDC_SPIN_H264_BITRATE, m_spin_h264Bitrate);
	DDX_Control(pDX, IDC_EDIT_FRAME_RATE, m_edit_frameRate);
	DDX_Control(pDX, IDC_BTN_SET_FRAME_RATE, m_btn_setFrameRate);
}

BEGIN_MESSAGE_MAP(VideoRecordingPage, CDialog)
	ON_CBN_SELCHANGE(IDC_COMBO_VIDEO_RECORD_TYPE, &VideoRecordingPage::OnCbnSelchangeComboVideoRecordType)
	ON_BN_CLICKED(IDC_BTN_SET_FRAME_RATE, &VideoRecordingPage::OnBnClickedBtnSetFrameRate)
END_MESSAGE_MAP()

// VideoRecordingPage message handlers

BOOL VideoRecordingPage::OnInitDialog()
{
	CDialog::OnInitDialog();

	for (int i = 0; i < NUM_VIDEO_FORMATS; i++)
	{
		m_combo_VideoFormat.AddString(VideoFormatList[i]);
	}
	m_combo_VideoFormat.SetCurSel(UNCOMPRESSED);
	OnCbnSelchangeComboVideoRecordType();

	m_edit_frameRate.SetWindowText("");

	// MJPEG Controls
	m_edit_mjpegCompressionLevel.SetWindowText(MJPEG_QUALITY_DEF);
	m_spin_mjpegCompressionLevel.SetRange(MJPEG_QUALITY_MIN, MJPEG_QUALITY_MAX);
	m_spin_mjpegCompressionLevel.SetBuddy(GetDlgItem(IDC_EDIT_MJPEG_COMPRESSION));

	// H.264 Controls
	m_edit_h264Bitrate.SetWindowText(H264_BITRATE_DEF);
	m_spin_h264Bitrate.SetRange32(H264_BITRATE_MIN, H264_BITRATE_MAX);
	m_spin_h264Bitrate.SetBuddy(GetDlgItem(IDC_EDIT_H264_BITRATE));

	// Frame rate control
	m_edit_frameRate.SetWindowText(VIDEO_FRAMERATE_DEF);

	return TRUE;
}

void VideoRecordingPage::GetSettings( VideoSettings* videoSettings )
{
	void* formatSettings = NULL;
	float framerate;
	
	GetFramerate(&framerate);

	switch(m_combo_VideoFormat.GetCurSel())
	{
	case UNCOMPRESSED:
		videoSettings->videoFormat = UNCOMPRESSED;
		videoSettings->aviOption.frameRate = framerate;
		break;
	case MJPEG:
		videoSettings->videoFormat = MJPEG;
		videoSettings->mjpgOption.frameRate = framerate;

		unsigned int quality;
		GetQuality(&quality);
		videoSettings->mjpgOption.quality = quality;		
		break;
	case H264:
		videoSettings->videoFormat = H264;
		videoSettings->h264Option.frameRate = framerate;

		unsigned int width, height, bitrate;
		GetBitrate(&bitrate);
		GetCurrentCameraResolution(&width, &height);

		videoSettings->h264Option.width = width;
		videoSettings->h264Option.height = height;
		videoSettings->h264Option.bitrate = bitrate;
		break;
	default:
		videoSettings->videoFormat = UNCOMPRESSED;
		videoSettings->aviOption.frameRate = framerate;
		break;
	}
}

BOOL VideoRecordingPage::ConvertToInt(CString* text, unsigned int* integer )
{
	errno = 0;
	*integer = _ttoi(text->GetBuffer());
	return ((errno == 0) || (*integer != 0));
}

BOOL VideoRecordingPage::GetQuality( unsigned int* quality )
{
	CString qualityTxt;
	m_edit_mjpegCompressionLevel.GetWindowText(qualityTxt);
	unsigned int qualityInt = atoi(qualityTxt);
	*quality = qualityInt;
	return (quality != 0);
}

BOOL VideoRecordingPage::GetBitrate(unsigned int* bitrate)
{
	CString bitrateTxt;
	m_edit_h264Bitrate.GetWindowText(bitrateTxt);
	bitrateTxt.Remove(',');
	unsigned int bitrateInt = atoi(bitrateTxt);
	*bitrate = bitrateInt;
	return (bitrate != 0);
}

BOOL VideoRecordingPage::GetFramerate( float* framerate )
{
	CString framerateTxt;
	m_edit_frameRate.GetWindowText(framerateTxt);

	errno = 0;
	*framerate = (float)atof(framerateTxt);
	return ((errno == 0) ||(*framerate != 0));
}

void VideoRecordingPage::DisplayMJPEGOptions(BOOL display)
{
	m_group_mjpegOptions.ShowWindow(display);
	m_static_mjpegCompressionLevel.ShowWindow(display);
	m_edit_mjpegCompressionLevel.ShowWindow(display);
	m_spin_mjpegCompressionLevel.ShowWindow(display);
}

void VideoRecordingPage::DisplayH264Options(BOOL display)
{
	m_group_h264Options.ShowWindow(display);
	m_static_h264Bitrate.ShowWindow(display);
	m_edit_h264Bitrate.ShowWindow(display);
	m_spin_h264Bitrate.ShowWindow(display);
}

void VideoRecordingPage::OnCbnSelchangeComboVideoRecordType()
{
	switch (m_combo_VideoFormat.GetCurSel())
	{
	case UNCOMPRESSED:
		DisplayMJPEGOptions(FALSE);
		DisplayH264Options(FALSE);
		break;
	case MJPEG:
		DisplayH264Options(FALSE);
		DisplayMJPEGOptions(TRUE);
		break;
	case H264:
		DisplayMJPEGOptions(FALSE);
		DisplayH264Options(TRUE);
		break;
	default:
		break;
	}
}

void VideoRecordingPage::OnBnClickedBtnSetFrameRate()
{
	FlyCapture2::Property prop;
	prop.type= FlyCapture2::FRAME_RATE;
	m_pCameraVid->GetProperty(&prop);
	float frameRateFlt = prop.absValue;

	char frameRateTxt[MAX_PATH];
	sprintf(frameRateTxt, "%4.3f", frameRateFlt);
	m_edit_frameRate.SetWindowText(frameRateTxt);
}

void VideoRecordingPage::StoreCameraPtr( FlyCapture2::CameraBase* m_pCamera )
{
	m_pCameraVid = m_pCamera;
}

BOOL VideoRecordingPage::GetCurrentCameraResolution( unsigned int* width, unsigned int* height )
{
	FlyCapture2::Error error;
	FlyCapture2::CameraInfo camInfo;


	error = ((FlyCapture2::Camera*)m_pCameraVid)->GetCameraInfo(&camInfo);

	if (camInfo.interfaceType == FlyCapture2::INTERFACE_GIGE)
	{
		FlyCapture2::GigEImageSettings gigeImageSettings;
		error = ((FlyCapture2::GigECamera*)m_pCameraVid)->GetGigEImageSettings(&gigeImageSettings);
		*width = gigeImageSettings.width;
		*height = gigeImageSettings.height;
	}
	else
	{
		FlyCapture2::VideoMode videoMode;
		FlyCapture2::FrameRate frameRate;
		FlyCapture2::Format7ImageSettings f7ImageSettings;

		error = ((FlyCapture2::Camera*)m_pCameraVid)->GetVideoModeAndFrameRate(&videoMode, &frameRate );

		switch(videoMode)
		{
		case FlyCapture2::VIDEOMODE_160x120YUV444:
			*width = 160;
			*height = 120;
			break;
		case FlyCapture2::VIDEOMODE_320x240YUV422:
			*width = 320;
			*height = 240;
			break;
		case FlyCapture2::VIDEOMODE_640x480YUV411:
		case FlyCapture2::VIDEOMODE_640x480YUV422:
		case FlyCapture2::VIDEOMODE_640x480RGB:
		case FlyCapture2::VIDEOMODE_640x480Y8:
		case FlyCapture2::VIDEOMODE_640x480Y16:
			*width = 640;
			*height = 480;
			break;
		case FlyCapture2::VIDEOMODE_800x600YUV422:
		case FlyCapture2::VIDEOMODE_800x600RGB:
		case FlyCapture2::VIDEOMODE_800x600Y8:
		case FlyCapture2::VIDEOMODE_800x600Y16:
			*width = 800;
			*height = 600;
			break;
		case FlyCapture2::VIDEOMODE_1024x768YUV422:
		case FlyCapture2::VIDEOMODE_1024x768RGB:
		case FlyCapture2::VIDEOMODE_1024x768Y8:
		case FlyCapture2::VIDEOMODE_1024x768Y16:
			*width = 1024;
			*height = 768;
			break;
		case FlyCapture2::VIDEOMODE_1280x960YUV422:
		case FlyCapture2::VIDEOMODE_1280x960RGB:
		case FlyCapture2::VIDEOMODE_1280x960Y8:
		case FlyCapture2::VIDEOMODE_1280x960Y16:
			*width = 1280;
			*height = 960;
			break;
		case FlyCapture2::VIDEOMODE_1600x1200YUV422:
		case FlyCapture2::VIDEOMODE_1600x1200RGB:
		case FlyCapture2::VIDEOMODE_1600x1200Y8:
		case FlyCapture2::VIDEOMODE_1600x1200Y16 :
			*width = 1600;
			*height = 1200;
			break;
		case FlyCapture2::VIDEOMODE_FORMAT7:
			unsigned int packetSize;
			float percentage;
			((FlyCapture2::Camera*)m_pCameraVid)->GetFormat7Configuration(&f7ImageSettings, &packetSize, &percentage);
			*width = f7ImageSettings.width;
			*height = f7ImageSettings.height;
			break;
		default:
			break;
		}
	}
	return TRUE;
}

void VideoRecordingPage::ValidateSettings( CString* errorList )
{
	switch(m_combo_VideoFormat.GetCurSel())
	{
	case MJPEG:
		unsigned int quality;
		if((!GetQuality(&quality)) || (quality < MJPEG_QUALITY_MIN) || (quality > MJPEG_QUALITY_MAX))
		{
			errorList->Append("Invalid JPEG Quality value specified.\n");
		}
		break;
	case H264:
		unsigned int bitrate;
		if ((!GetBitrate(&bitrate)) || (bitrate < H264_BITRATE_MIN) || (bitrate > H264_BITRATE_MAX))
		{
			errorList->Append("Invalid H.264 bitrate Quality value specified.\n");
		}
		break;
	default:
		break;
	}
}

void VideoRecordingPage::EnableControls(BOOL enable)
{
	m_combo_VideoFormat.EnableWindow(enable);
	m_edit_frameRate.EnableWindow(enable);
	m_btn_setFrameRate.EnableWindow(enable);

	m_edit_mjpegCompressionLevel.EnableWindow(enable);
	m_spin_mjpegCompressionLevel.EnableWindow(enable);
	m_edit_h264Bitrate.EnableWindow(enable);
}
