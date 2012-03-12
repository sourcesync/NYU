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
#include <afxsock.h>
#include <errno.h>
#include "FlyCap2_MFC.h"
#include "RecordingDialog.h"
#include "MySocket.h"

// RecordingDialog dialog
extern int socket_port;

const unsigned int ONE_MEG = 1024 * 1024;

const char RecordButtonStrings[][MAX_PATH] = 
{
	"Start Recording",
	"Stop Recording",
	"Abort Saving"
};

IMPLEMENT_DYNAMIC(RecordingDialog, CDialog)


RecordingDialog::RecordingDialog(CWnd* pParent /*=NULL*/)
: CDialog(RecordingDialog::IDD, pParent)
{
	m_intervalExpiredFlag = FALSE;
	m_durationExpiredFlag = FALSE;
	m_recorderTimerDuration = NULL;
	m_recorderTimerInterval = NULL;
	m_saveFrameLoopThread = NULL;
	m_currRecordingState = STOPPED;
}

RecordingDialog::~RecordingDialog()
{

}

void RecordingDialog::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_EDIT_NUM_FRAMES, m_edit_NumFrames);
	DDX_Control(pDX, IDC_EDIT_NTH_FRAMES, m_edit_NthFrame);
	DDX_Control(pDX, IDC_EDIT_NUM_SECONDS, m_edit_NumSeconds);
	DDX_Control(pDX, IDC_EDIT_NTH_SECONDS, m_edit_NthSecond);
	DDX_Control(pDX, IDC_STATIC_FRAME_COUNTER, m_static_FrameCounter);
	DDX_Control(pDX, IDC_TAB_OUTPUT_TYPE, m_tabCtrl_OutputType);

	DDX_Control(pDX, IDC_RADIO_NUM_FRAMES, m_radio_NumFrames);
	DDX_Control(pDX, IDC_RADIO_NTH_FRAME, m_radio_NthFrame);
	DDX_Control(pDX, IDC_RADIO_NUM_SECONDS, m_radio_NumSeconds);
	DDX_Control(pDX, IDC_RADIO_NTH_SECONDS, m_radio_NthSecond);
	DDX_Control(pDX, IDC_EDIT_NTH_TOTAL_FRAMES, m_edit_NthTotalFrames);
	DDX_Control(pDX, IDC_EDIT_NTH_TOTAL_SECONDS, m_edit_NthTotalSeconds);
	DDX_Control(pDX, IDC_BUTTON_START_STOP_VIDEO_RECORD, m_btn_StartStop);
	DDX_Control(pDX, IDC_EDIT_SAVE_FILE_PATH, m_edit_SaveFilePath);
	DDX_Control(pDX, IDC_STATIC_SAVE_COUNTER, m_static_savedImageCount);
	DDX_Control(pDX, IDCANCEL, m_btn_close);
	DDX_Control(pDX, IDC_CHK_CAPTURE_CORRUPT_FRAMES, m_chk_captureCorruptFrames);
	DDX_Control(pDX, IDC_STATIC_BUFFERED_COUNTER, m_static_bufferedCounter);
	DDX_Control(pDX, IDC_STATIC_AVAILABLE_MEMORY, m_static_availableMemory);
	DDX_Control(pDX, IDC_STATIC_TOTAL_MEMORY, m_static_totalMemory);
	DDX_Control(pDX, IDC_STATIC_MEMORY_LOAD, m_static_memoryLoad);
	DDX_Control(pDX, IDC_RADIO_BUFFERED_MODE, m_radio_bufferedMode);
	DDX_Control(pDX, IDC_RADIO_STREAMING_MODE, m_radio_streamingMode);
}


BEGIN_MESSAGE_MAP(RecordingDialog, CDialog)
	ON_BN_CLICKED(IDC_BUTTON_BROWSE, &RecordingDialog::OnBnClickedButtonBrowse)
	ON_NOTIFY(TCN_SELCHANGE, IDC_TAB_OUTPUT_TYPE, &RecordingDialog::OnTcnSelchangeTabOutputType)
	ON_BN_CLICKED(IDC_RADIO_NUM_FRAMES, &RecordingDialog::OnBnClickedRadioNumFrames)
	ON_BN_CLICKED(IDC_RADIO_NTH_FRAME, &RecordingDialog::OnBnClickedRadioNthFrame)
	ON_BN_CLICKED(IDC_RADIO_NUM_SECONDS, &RecordingDialog::OnBnClickedRadioNumSeconds)
	ON_BN_CLICKED(IDC_RADIO_NTH_SECONDS, &RecordingDialog::OnBnClickedRadioNthSeconds)
	ON_WM_SIZE()
	ON_BN_CLICKED(IDC_BUTTON_START_STOP_VIDEO_RECORD, &RecordingDialog::OnBnClickedButtonStartStopVideoRecord)
	ON_WM_TIMER()
	ON_WM_ERASEBKGND()
END_MESSAGE_MAP()

BOOL RecordingDialog::OnInitDialog()
{
	

	//AfxMessageBox("Failed to Initialize Sockets",MB_OK| MB_ICONSTOP);
	if(!AfxSocketInit())
	{
      AfxMessageBox("Failed to Initialize Sockets",MB_OK| MB_ICONSTOP);
	}
	m_socket = new MySocket(socket_port);
	m_socket->rc = this;


	if (!CDialog::OnInitDialog())
	{
		MessageBox("Failed to initialize Recording Dialog");
		return FALSE;
	}

	if (m_tabCtrl_OutputType.InsertItem(0, "Images") == -1)
	{
		MessageBox("Failed to insert Image Type tab page");
		return FALSE;
	}

	if (m_tabCtrl_OutputType.InsertItem(1, "Videos") == -1)
	{
		MessageBox("Failed to insert Video Type tab page");
		return FALSE;
	}

	if (m_imageRecordingPage.Create(IDD_TABPAGE_IMAGE_RECORD, &m_tabCtrl_OutputType) == FALSE)
	{
		MessageBox("Failed to create Image Type tab page");
		return FALSE;
	}

	if (m_videoRecordingPage.Create(IDD_TABPAGE_VIDEO_RECORD, &m_tabCtrl_OutputType) == FALSE)
	{
		MessageBox("Failed to create Video Type tab page");
		return FALSE;
	}

	CRect tabRect, itemRect;
	int nX, nY, nXc, nYc;
	m_tabCtrl_OutputType.GetClientRect(&tabRect);
	m_tabCtrl_OutputType.GetItemRect(0, &itemRect);
	nX  = itemRect.left;
	nY  = itemRect.bottom + 1;
	nXc = tabRect.right - itemRect.left - 2;
	nYc = tabRect.bottom - nY - 2;

	m_imageRecordingPage.SetWindowPos(&wndTop, nX, nY, nXc, nYc, SWP_SHOWWINDOW);
	m_videoRecordingPage.SetWindowPos(&wndTop, nX, nY, nXc, nYc, SWP_HIDEWINDOW);

	m_radio_NumFrames.SetCheck(BST_CHECKED);
	m_edit_NumFrames.SetWindowText("0");
	m_edit_SaveFilePath.SetWindowText("C:\\tmp\\fc2_save");

	m_radio_bufferedMode.SetCheck(BST_CHECKED);

	m_frameCounter = 0;
	m_streamingFrameCounter = 0;
	m_savedImageCounter = 0;

	UpdateSavingOptions();
	EnableControls();

	return TRUE;
}

// RecordingDialog message handlers

void RecordingDialog::OnBnClickedButtonBrowse()
{
	CFileDialog browseDialog(FALSE, NULL, "fc2_save", 0, NULL, 0);

	if(browseDialog.DoModal() == IDOK)
	{
		m_saveFilename = browseDialog.GetPathName();
		m_edit_SaveFilePath.SetWindowText(m_saveFilename);
	}
}

BOOL RecordingDialog::IsRecordingActive()
{
	return ((m_currRecordingState == STARTED) || (m_currRecordingState == STOPPING));
}

BOOL RecordingDialog::BufferFrame(FlyCapture2::Image* image)
{
	if (m_radio_bufferedMode.GetCheck())
	{
		// if memory usage above threshold, stop saving
		MEMORYSTATUSEX memStatus;
		memStatus.dwLength = sizeof(MEMORYSTATUSEX);

		GlobalMemoryStatusEx(&memStatus);
		if (memStatus.dwMemoryLoad >= 99)
		{
			ChangeState(SAVING);
			return FALSE;
		}
	}
	try
	{
		CSingleLock bufferLock(&m_recorderBuffer);
		bufferLock.Lock();
		if((m_grabMode == FlyCapture2::BUFFER_FRAMES) ||
			((m_grabMode == FlyCapture2::DROP_FRAMES) && (m_imageBuffer.empty())))
		{
			m_imageBuffer.push_back(*image);
			m_streamingFrameCounter++;

			if (m_imageBuffer.size() == 1)
			{
				SetEvent(m_recordingEvents[DATA_IN_BUFFER]);
			}
		}
	}
	catch (std::exception& e)
	{
		return FALSE;
	}
	return TRUE;
}

void RecordingDialog::OnTcnSelchangeTabOutputType(NMHDR *pNMHDR, LRESULT *pResult)
{
	switch (m_tabCtrl_OutputType.GetCurSel())
	{
	case OUTPUT_TYPE_IMAGE:
		m_videoRecordingPage.ShowWindow(SW_HIDE);
		m_imageRecordingPage.ShowWindow(SW_SHOW);
		break;

	case OUTPUT_TYPE_VIDEO:
		m_imageRecordingPage.ShowWindow(SW_HIDE);
		m_videoRecordingPage.ShowWindow(SW_SHOW);
		break;
	default:
		MessageBox("Unknown page index.");
		break;
	}
	*pResult = 0;
}

void RecordingDialog::OnBnClickedRadioNumFrames()
{
	UpdateSavingOptions();
}

void RecordingDialog::OnBnClickedRadioNthFrame()
{
	UpdateSavingOptions();
}

void RecordingDialog::OnBnClickedRadioNumSeconds()
{
	UpdateSavingOptions();
}

void RecordingDialog::OnBnClickedRadioNthSeconds()
{
	UpdateSavingOptions();
}

void RecordingDialog::UpdateSavingOptions()
{
	m_edit_NumFrames.EnableWindow(m_radio_NumFrames.GetCheck());
	m_edit_NthFrame.EnableWindow(m_radio_NthFrame.GetCheck());
	m_edit_NthTotalFrames.EnableWindow(m_radio_NthFrame.GetCheck());
	m_edit_NumSeconds.EnableWindow(m_radio_NumSeconds.GetCheck());
	m_edit_NthSecond.EnableWindow(m_radio_NthSecond.GetCheck());
	m_edit_NthTotalSeconds.EnableWindow(m_radio_NthSecond.GetCheck());
}

void RecordingDialog::OnBnClickedButtonStartStopVideoRecord()
{
	// Start/Stop/Abort button has been pressed...
	CSingleLock stateLock(&m_buttonState);

	if ( m_currRecordingState == STOPPED ||
		 m_currRecordingState == STOPPING )
	{
		StopRecording();
		DisableControls();
		CString errorList;
		if (ValidInput(&errorList))
		{
			time_t rawtime;
			struct tm * timeinfo;
			time( &rawtime );
			timeinfo = localtime( &rawtime );

			char timestamp[64];
			strftime( timestamp, 64, "%Y-%m-%d-%H%M%S", timeinfo );

			GetFilePath(&m_saveFilenameBase);
			m_saveFilenameBase.AppendFormat("_%s", timestamp);

			if(m_radio_bufferedMode.GetCheck())
			{
				m_grabMode = FlyCapture2::BUFFER_FRAMES;
			}
			else
			{
				m_grabMode = FlyCapture2::DROP_FRAMES;
			}
			
			if (CreateRecordingEvents())
			{
				StartRecording();
				m_saveFrameLoopThread = AfxBeginThread(ThreadSaveFrame, this);
			}
			else
			{
				AfxMessageBox("RecordingEvents creation failed... Aborting.", MB_OK);
				EnableControls();
			}
		}
		else
		{
			AfxMessageBox(errorList);
			EnableControls();
		}
	}
	else if (m_currRecordingState == STARTED)
	{
		ChangeState(SAVING);
	}
	else if (m_currRecordingState == SAVING)
	{
		ChangeState(STOPPED);
	}

	this->DisableControls();
	this->EnableControls();
	//this->Invalidate();
}

void RecordingDialog::OnTimer(UINT_PTR nIDEvent)
{
	switch (nIDEvent)
	{
	case TIMER_DURATION:
		// stop recording
		m_durationExpiredFlag = TRUE;
		break;
	case TIMER_INTERVAL:
		// signal interval
		m_intervalExpiredFlag = TRUE;
		break;
	default:
		break;
	}
	CDialog::OnTimer(nIDEvent);
}

RecordingDialog::SaveOptionType RecordingDialog::GetSaveType()
{
	if (m_radio_NumFrames.GetCheck())
		return NUMBER_OF_FRAMES;
	else if (m_radio_NthFrame.GetCheck())
		return EVERY_NTH_FRAME;
	else if (m_radio_NumSeconds.GetCheck())
		return NUMBER_OF_SECONDS;
	else if (m_radio_NthSecond.GetCheck())
		return EVERY_NTH_SECOND;

	return NUMBER_OF_FRAMES;
}

BOOL RecordingDialog::GetNumFrames(unsigned int* numFrames)
{
	CString numFramesTxt;
	m_edit_NumFrames.GetWindowText(numFramesTxt);

	return (!(numFramesTxt.IsEmpty()) && ConvertToInt(numFramesTxt, numFrames));
}

BOOL RecordingDialog::GetNthFrame(unsigned int* nthFrame)
{
	CString nthFramesTxt;
	m_edit_NthFrame.GetWindowText(nthFramesTxt);
	return (!(nthFramesTxt.IsEmpty()) && ConvertToInt(nthFramesTxt, nthFrame));
}

BOOL RecordingDialog::GetNthFrameTotal(unsigned int* nthFrameTotal)
{
	CString nthFramesTotalTxt;
	m_edit_NthTotalFrames.GetWindowText(nthFramesTotalTxt);
	return (!(nthFramesTotalTxt.IsEmpty()) && ConvertToInt(nthFramesTotalTxt, nthFrameTotal));
}

BOOL RecordingDialog::GetNumSeconds(unsigned int* numSeconds)
{
	CString numSecondsTxt;
	m_edit_NumSeconds.GetWindowText(numSecondsTxt);
	return (!(numSecondsTxt.IsEmpty()) && ConvertToInt(numSecondsTxt, numSeconds));
}

BOOL RecordingDialog::GetNthSecondsTotal(unsigned int* nthSecondsTotal)
{
	CString nthSecondsTotalTxt;
	m_edit_NthTotalSeconds.GetWindowText(nthSecondsTotalTxt);
	return (!(nthSecondsTotalTxt.IsEmpty()) && ConvertToInt(nthSecondsTotalTxt, nthSecondsTotal));
}

BOOL RecordingDialog::GetNthSecond(unsigned int* nthSecond)
{
	CString nthSecondTxt;
	m_edit_NthSecond.GetWindowText(nthSecondTxt);
	return (!(nthSecondTxt.IsEmpty()) && ConvertToInt(nthSecondTxt, nthSecond));
}

BOOL RecordingDialog::ConvertToInt(CString text, unsigned int* integer )
{
	errno = 0;
	*integer = _ttoi(text.GetBuffer());

	return ((errno == 0) || (*integer != 0));
}

void RecordingDialog::GetFilePath( CString* filename )
{
	m_edit_SaveFilePath.GetWindowText(*filename);
	filename->TrimLeft();
}

UINT RecordingDialog::ThreadSaveFrameHelper()
{
	CSingleLock bufferLock(&m_recorderBuffer);
	CSingleLock stateLock(&m_buttonState);

	FlyCapture2::Error error;
	FlyCapture2::Image image;

	if (m_tabCtrl_OutputType.GetCurSel() == OUTPUT_TYPE_IMAGE)
	{
		ImageRecordingPage::ImageSettings imageSettings;

		// get image saving settings
		GetImageSettings(&imageSettings);

		while(m_currRecordingState == STARTED)
		{
			DWORD waitResult = WaitForMultipleObjects(NUM_RECORDING_EVENTS, m_recordingEvents, FALSE, INFINITE);
			
			if ((waitResult - WAIT_OBJECT_0) == END_LIVE_RECORDING)
			{
				// stop button was hit
				break;
			}

			bufferLock.Lock();
			RetrieveNextImageFromBuffer(&image);

			if (m_imageBuffer.empty())
			{
				ResetEvent(m_recordingEvents[DATA_IN_BUFFER]);
			}
			bufferLock.Unlock();

			SaveImageToFile(&image, &imageSettings);
		}

		// save any remaining frames in buffer
		while(!(m_imageBuffer.empty()) && (m_currRecordingState == SAVING))
		{
			CSingleLock bufferLock(&m_recorderBuffer);

			bufferLock.Lock();
			RetrieveNextImageFromBuffer(&image);
			bufferLock.Unlock();

			SaveImageToFile(&image, &imageSettings);
		}
	}
	else if (m_tabCtrl_OutputType.GetCurSel() == OUTPUT_TYPE_VIDEO)
	{
		FlyCapture2::AVIRecorder aviRecorder;

		VideoRecordingPage::VideoSettings videoSettings;
		GetVideoSettings(&videoSettings);

		switch (videoSettings.videoFormat)
		{
		case VideoRecordingPage::UNCOMPRESSED:
			error = aviRecorder.AVIOpen(videoSettings.filename, &(videoSettings.aviOption));
			break;
		case VideoRecordingPage::MJPEG:
			error = aviRecorder.AVIOpen(videoSettings.filename, &(videoSettings.mjpgOption));
			break;
		case VideoRecordingPage::H264:
			error = aviRecorder.AVIOpen(videoSettings.filename, &(videoSettings.h264Option));
			break;
		default:
			throw RecordingException("Invalid Video Output Type Specified.");
			break;
		}

		if(error != FlyCapture2::PGRERROR_OK)
		{
			throw RecordingException(error.GetDescription());
		}

		while(m_currRecordingState == STARTED)
		{
			DWORD waitResult = WaitForMultipleObjects(NUM_RECORDING_EVENTS, m_recordingEvents, FALSE, INFINITE);

			if ((waitResult - WAIT_OBJECT_0) == END_LIVE_RECORDING)
			{
				// stop button was hit
				break;
			}

			bufferLock.Lock();
			RetrieveNextImageFromBuffer(&image);

			if (m_imageBuffer.empty())
			{
				ResetEvent(m_recordingEvents[DATA_IN_BUFFER]);
			}
			bufferLock.Unlock();

			SaveImageToVideo(&aviRecorder, &image);
		}

		// process any remaining frames in buffer
		while(!(m_imageBuffer.empty()) && (m_currRecordingState == SAVING))
		{
			bufferLock.Lock();
			RetrieveNextImageFromBuffer(&image);
			bufferLock.Unlock();

			SaveImageToVideo(&aviRecorder, &image);
		}
		// do cleanup
		aviRecorder.AVIClose();
		if(error != FlyCapture2::PGRERROR_OK)
		{
			// display error and quit
			char errMsg[256];
			sprintf(errMsg, "%s", error.GetDescription());
			AfxMessageBox(errMsg, MB_OK);
		}
	}

	// We cannot restart camera in this thread so we reset the dialog 
	// and put the recorder in a stopping state. 
	StoppingRecording();
	return 0;
}

UINT RecordingDialog::ThreadSaveFrame( void* pparam )
{
	if (pparam == NULL)
	{
		AfxEndThread(-1);
	}

	RecordingDialog* pDoc = (RecordingDialog*)pparam;

	UINT retVal = 0;

	try
	{
		retVal = pDoc->ThreadSaveFrameHelper();
	}
	catch (RecordingException& e)
	{
		pDoc->StoppingRecording();

		CString errMsg;
		errMsg.AppendFormat("%s\nStopping.", e.what());
		AfxMessageBox(errMsg, MB_OK);
		AfxEndThread(-1);
	}
	return retVal;
}

void RecordingDialog::DoRecording( FlyCapture2::Image* rawImage, BOOL isCorruptFrame )
{
	// if recording has started then buffer frame if it is wanted
	if (m_currRecordingState == STARTED)
	{
		if ( (!(m_chk_captureCorruptFrames.GetCheck()) && !isCorruptFrame) ||
			   (m_chk_captureCorruptFrames.GetCheck() && isCorruptFrame))
		{
			ProcessCurrentFrame(rawImage);
		}
	}
	
	if (m_currRecordingState == STOPPING)
	{
		StopRecording();
	}

	UpdateImageCounters();
}

void RecordingDialog::StopRecording()
{
	if (m_currRecordingState == STOPPED)
	{
		return;
	}

	FlyCapture2::Error error;
	FlyCapture2::FC2Config fc2Config;

	if (m_pCameraRec != NULL)
	{
		error = m_pCameraRec->GetConfiguration(&fc2Config);
	}

	if (fc2Config.grabMode != FlyCapture2::DROP_FRAMES)
	{
		// restart camera in drop frames mode
		CSingleLock controlLock(&m_startStopControl);
		controlLock.Lock();
		error = m_pCameraRec->StopCapture();
		fc2Config.grabMode = FlyCapture2::DROP_FRAMES;
		error = m_pCameraRec->SetConfiguration(&fc2Config);
		error = m_pCameraRec->StartCapture();
		controlLock.Unlock();
	}

	CSingleLock bufferLock(&m_recorderBuffer);
	bufferLock.Lock();
	m_imageBuffer.clear();
	bufferLock.Unlock();

	DeleteRecordingEvents();

	if (m_recorderTimerDuration != NULL)
		::KillTimer(m_hWnd, TIMER_DURATION);
	if (m_recorderTimerInterval != NULL)
		::KillTimer(m_hWnd, TIMER_INTERVAL);

	m_recorderTimerDuration = NULL;
	m_recorderTimerInterval = NULL;
	m_durationExpiredFlag = FALSE;
	m_intervalExpiredFlag = FALSE;

	ChangeState(STOPPED);
	EnableControls();
}

void RecordingDialog::StartRecording()
{
	SaveOptionType saveType = GetSaveType();

	if (saveType == NUMBER_OF_SECONDS)
	{
		unsigned int numSeconds;
		GetNumSeconds(&numSeconds);
		m_recorderTimerDuration = ::SetTimer(m_hWnd, TIMER_DURATION, numSeconds, (TIMERPROC)NULL);
	}
	else if (saveType == EVERY_NTH_SECOND)
	{
		unsigned int nthSecondsTotal, nthInterval;
		GetNthSecondsTotal(&nthSecondsTotal);
		GetNthSecond(&nthInterval);
		m_recorderTimerDuration = ::SetTimer(m_hWnd, TIMER_DURATION, nthSecondsTotal, (TIMERPROC)NULL);
		m_recorderTimerInterval = ::SetTimer(m_hWnd, TIMER_INTERVAL, nthInterval, (TIMERPROC)NULL);
	}

	// start the camera in buffered frame mode
	if(m_pCameraRec != NULL)
	{
		m_pCameraRec->StopCapture();
		FlyCapture2::FC2Config fc2Config;
		m_pCameraRec->GetConfiguration(&fc2Config);
		fc2Config.grabMode = m_grabMode;
		m_pCameraRec->SetConfiguration(&fc2Config);
		m_pCameraRec->StartCapture();
	}

	CSingleLock bufferLock(&m_recorderBuffer);
	bufferLock.Lock();
	m_imageBuffer.clear();
	bufferLock.Unlock();

	m_frameCounter = 0;
	m_streamingFrameCounter = 0;
	m_savedImageCounter = 0;

	UpdateImageCounters();

	ChangeState(STARTED);
}

void RecordingDialog::StoppingRecording()
{
	CSingleLock bufferLock(&m_recorderBuffer);
	bufferLock.Lock();
	m_imageBuffer.clear();
	bufferLock.Unlock();

	if (m_recorderTimerDuration != NULL)
		::KillTimer(m_hWnd, TIMER_DURATION);
	if (m_recorderTimerInterval != NULL)
		::KillTimer(m_hWnd, TIMER_INTERVAL);

	m_recorderTimerDuration = NULL;
	m_recorderTimerInterval = NULL;
	m_durationExpiredFlag = FALSE;
	m_intervalExpiredFlag = FALSE;

	ChangeState(STOPPED);
	EnableControls();
}

void RecordingDialog::ProcessCurrentFrame( FlyCapture2::Image* rawImage)
{
	FlyCapture2::Error error;
	SaveOptionType saveType = GetSaveType();
	CSingleLock stateLock(&m_buttonState);

	m_frameCounter++; 

	// if recMode == #ofFrames then 
	if (saveType == NUMBER_OF_FRAMES)
	{
		unsigned int numFrames;
		GetNumFrames(&numFrames);

		// push frame onto queue
		FlyCapture2::Image tmpImage;
		error = tmpImage.DeepCopy(rawImage);
		if(error == FlyCapture2::PGRERROR_OK)
		{
			BufferFrame(&tmpImage);

			if ((numFrames != 0) && 
			    (((m_grabMode == FlyCapture2::BUFFER_FRAMES) && (m_frameCounter >= numFrames)) ||
				((m_grabMode == FlyCapture2::DROP_FRAMES) && (m_streamingFrameCounter >= numFrames))))
			{
				ChangeState(SAVING);
			}
		}
	}
	// if recMode == every Nth Frame
	else if (saveType == EVERY_NTH_FRAME)
	{
		// if target not hit then:
		unsigned int nthFrame;
		GetNthFrame(&nthFrame);
		unsigned int nthFramesTotal;
		GetNthFrameTotal(&nthFramesTotal);

		// save last image of each interval
		if ((nthFrame == 1) || (m_frameCounter % nthFrame == 0))
		{
			// push frame onto queue
			FlyCapture2::Image tmpImage;
			error = tmpImage.DeepCopy(rawImage);
			if(error == FlyCapture2::PGRERROR_OK)
			{
				BufferFrame(&tmpImage);
			}
		}
		else
		{
			// do nothing, skip frame
		}

		if ((nthFramesTotal != 0) && (m_frameCounter >= nthFramesTotal))
		{
			ChangeState(SAVING);
		}
	}
	//    if recMode == # of Seconds
	else if (saveType == NUMBER_OF_SECONDS)
	{
		if (!m_durationExpiredFlag)
		{
			// push frame onto queue
			FlyCapture2::Image tmpImage;
			error = tmpImage.DeepCopy(rawImage);
			if (error == FlyCapture2::PGRERROR_OK)
			{
				BufferFrame(&tmpImage);
			}
		}
		else
		{
			// change state to saving
			ChangeState(SAVING);
		}
	}
	else if (saveType == EVERY_NTH_SECOND)
	{
		unsigned int nthTotalSeconds;
		GetNthSecondsTotal(&nthTotalSeconds);

		if ((nthTotalSeconds == 0) || (!m_durationExpiredFlag))
		{
			if(m_intervalExpiredFlag)
			{
				m_intervalExpiredFlag = FALSE;
				// push frame onto queue
				FlyCapture2::Image tmpImage;
				error = tmpImage.DeepCopy(rawImage);
				if (error == FlyCapture2::PGRERROR_OK)
				{
					BufferFrame(&tmpImage);
				}
			}
			else
			{
				// do nothing, skip frame
			}
		}
		else
		{
			// change state to saving
			ChangeState(SAVING);
		}
	}
}

void RecordingDialog::GetImageSettings( ImageRecordingPage::ImageSettings* imageSettings )
{
	strcpy(imageSettings->filename, m_saveFilenameBase);
	m_imageRecordingPage.GetSettings(imageSettings);
}

void RecordingDialog::GetVideoSettings( VideoRecordingPage::VideoSettings* videoSettings )
{
	strcpy(videoSettings->filename, m_saveFilenameBase);
	m_videoRecordingPage.GetSettings(videoSettings);
}

FlyCapture2::Error RecordingDialog::SaveImage( FlyCapture2::Image* tmp, ImageRecordingPage::ImageSettings* imageSettings, unsigned int count )
{
	char saveName[MAX_PATH];

	sprintf(saveName, "%s-%04d.%s", imageSettings->filename, count, imageSettings->fileExtension);
	switch (imageSettings->imageFormat)
	{

	case ImageRecordingPage::PGM:
		return tmp->Save(saveName, &(imageSettings->pgmOption));
		break;
	case ImageRecordingPage::PPM:
		return tmp->Save(saveName, &(imageSettings->ppmOption));
		break;
	case ImageRecordingPage::JPEG:
		return tmp->Save(saveName, &(imageSettings->jpgOption));
		break;
	case ImageRecordingPage::JPEG2000:
		return tmp->Save(saveName, &(imageSettings->jpg2Option));
		break;
	case ImageRecordingPage::TIFF:
		return tmp->Save(saveName, &(imageSettings->tiffOption));
		break;
	case ImageRecordingPage::PNG:
		return tmp->Save(saveName, &(imageSettings->pngOption));
		break;
	case ImageRecordingPage::BMP:
		return tmp->Save(saveName, FlyCapture2::BMP);
		break;
	case ImageRecordingPage::RAW:
		return tmp->Save(saveName, FlyCapture2::RAW);
		break;
	default:
		return tmp->Save(saveName, FlyCapture2::RAW);
		break;
	}
}

void RecordingDialog::StoreCamPtr( FlyCapture2::CameraBase* pCamera )
{
	m_pCameraRec = pCamera;
	m_videoRecordingPage.StoreCameraPtr(m_pCameraRec);
}

void RecordingDialog::UpdateImageCounters()
{
	CString counterString;

	counterString.Format("%d", m_frameCounter);
	m_static_FrameCounter.SetWindowText(counterString);

	counterString.Format("%d", m_imageBuffer.size());
	m_static_bufferedCounter.SetWindowText(counterString);

	counterString.Format("%d", m_savedImageCounter);
	m_static_savedImageCount.SetWindowText(counterString);

	MEMORYSTATUSEX memStatus;
	memStatus.dwLength = sizeof(MEMORYSTATUSEX);

	if(!GlobalMemoryStatusEx(&memStatus))
	{
		m_static_availableMemory.SetWindowText("N/A");
		m_static_totalMemory.SetWindowText("N/A");
		m_static_memoryLoad.SetWindowText("N/A");
	}
	else
	{
		counterString.Format("%5.2f MB", (float)(memStatus.ullAvailPhys / (float)ONE_MEG));
		m_static_availableMemory.SetWindowText(counterString);

		counterString.Format("%5.2f MB", (float)(memStatus.ullTotalPhys / (float)ONE_MEG));
		m_static_totalMemory.SetWindowText(counterString);

		counterString.Format("%d%%", memStatus.dwMemoryLoad);
		m_static_memoryLoad.SetWindowText(counterString);
	}
}

BOOL RecordingDialog::ValidInput(CString* errorList)
{
	ValidateFileName(errorList);
	ValidateSaveOptions(errorList);

	if (m_tabCtrl_OutputType.GetCurSel() == 0)
	{
		m_imageRecordingPage.ValidateSettings(errorList);
	}
	else
	{
		m_videoRecordingPage.ValidateSettings(errorList);
	}

	return errorList->IsEmpty();
}

void RecordingDialog::ValidateFileName( CString* errorList )
{
	CString filePath;
	CString dir;

	GetFilePath(&filePath);

	if (filePath.IsEmpty())
	{
		errorList->AppendFormat("Save file/path has not been specified\n");
		return;
	}
	

	unsigned int filePos = filePath.ReverseFind('\\');

	if (filePos != -1)
	{
		if (filePos > 2)
		{
			dir = filePath.Left(filePos+1);

			if (!CreateDirectory(dir, NULL))
			{
				DWORD lastError = GetLastError();

				if ( lastError == ERROR_ALREADY_EXISTS)
				{
					// dir exists
				}
				else
				{
					errorList->AppendFormat("Error creating save directory: 0x%08X\n", lastError);
				}
			}
		}
	}
}

void RecordingDialog::ValidateSaveOptions( CString* errorList )
{
	switch (GetSaveType())
	{
	case NUMBER_OF_FRAMES:

		unsigned int numFrames;
		if(!GetNumFrames(&numFrames))
		{
			errorList->Append("Invalid number of frames specified.\n");
		}
		break;
	case EVERY_NTH_FRAME:

		unsigned int nthFrame;
		unsigned int nthTotalFrames;

		if((!GetNthFrame(&nthFrame))                              || 
		   (!GetNthFrameTotal(&nthTotalFrames))                   || 
		   ((nthTotalFrames != 0) && (nthTotalFrames < nthFrame)) || 
		   (nthFrame == 0))
		{
			errorList->Append("Invalid frame interval/duration specified.\n");
		}
		break;
	case NUMBER_OF_SECONDS:
		unsigned int numSeconds;
		if(!GetNumSeconds(&numSeconds))
		{
			errorList->Append("Invalid number of ms specified.\n");
		}
		break;
	case EVERY_NTH_SECOND:
		unsigned int nthSecond;
		unsigned int nthTotalSeconds;

		if((!GetNthSecond(&nthSecond))                             || 
		   (!GetNthSecondsTotal(&nthTotalSeconds))                 || 
		   ((nthTotalSeconds!= 0) &&(nthTotalSeconds < nthSecond)) || 
		   (nthSecond == 0))
		{
			errorList->Append("Invalid ms interval/duration specified.\n");
		}
		break;
	default:
		errorList->Append("Unrecognized Save Type Specified.\n");
		break;
	}
}

void RecordingDialog::ValidateImageSettings( CString* errorList )
{
	ImageRecordingPage::ImageSettings imageSettings;

	m_imageRecordingPage.GetSettings(&imageSettings);
}

void RecordingDialog::EnableControls()
{
	m_edit_SaveFilePath.EnableWindow(TRUE);
	(GetDlgItem(IDC_BUTTON_BROWSE))->EnableWindow(TRUE);

	m_radio_NumFrames.EnableWindow(TRUE);
	m_radio_NthFrame.EnableWindow(TRUE);
	m_radio_NumSeconds.EnableWindow(TRUE);
	m_radio_NthSecond.EnableWindow(TRUE);

	m_chk_captureCorruptFrames.EnableWindow(TRUE);

	UpdateSavingOptions();

	m_radio_bufferedMode.EnableWindow(TRUE);
	m_radio_streamingMode.EnableWindow(TRUE);

	m_tabCtrl_OutputType.EnableWindow(TRUE);
	m_videoRecordingPage.EnableControls(TRUE);
	m_imageRecordingPage.EnableControls(TRUE);

	// counters are enabled opposite of other controls
	m_static_FrameCounter.EnableWindow(FALSE);
	m_static_bufferedCounter.EnableWindow(FALSE);
	m_static_savedImageCount.EnableWindow(FALSE);
	m_static_availableMemory.EnableWindow(FALSE);
	m_static_totalMemory.EnableWindow(FALSE);
	m_static_memoryLoad.EnableWindow(FALSE);
}

void RecordingDialog::DisableControls()
{
	m_edit_SaveFilePath.EnableWindow(FALSE);
	(GetDlgItem(IDC_BUTTON_BROWSE))->EnableWindow(FALSE);

	m_radio_NumFrames.EnableWindow(FALSE);
	m_edit_NumFrames.EnableWindow(FALSE);

	m_radio_NthFrame.EnableWindow(FALSE);
	m_edit_NthFrame.EnableWindow(FALSE);
	m_edit_NthTotalFrames.EnableWindow(FALSE);

	m_radio_NumSeconds.EnableWindow(FALSE);
	m_edit_NumSeconds.EnableWindow(FALSE);

	m_radio_NthSecond.EnableWindow(FALSE);
	m_edit_NthSecond.EnableWindow(FALSE);
	m_edit_NthTotalSeconds.EnableWindow(FALSE);

	m_radio_NthFrame.EnableWindow(FALSE);
	m_edit_NthFrame.EnableWindow(FALSE);

	m_radio_bufferedMode.EnableWindow(FALSE);
	m_radio_streamingMode.EnableWindow(FALSE);

	m_chk_captureCorruptFrames.EnableWindow(FALSE);

	m_tabCtrl_OutputType.EnableWindow(FALSE);
	m_videoRecordingPage.EnableControls(FALSE);
	m_imageRecordingPage.EnableControls(FALSE);

	// counters are enabled opposite of other controls
	m_static_FrameCounter.EnableWindow(TRUE);
	m_static_bufferedCounter.EnableWindow(TRUE);
	m_static_savedImageCount.EnableWindow(TRUE);
	m_static_availableMemory.EnableWindow(TRUE);
	m_static_totalMemory.EnableWindow(TRUE);
	m_static_memoryLoad.EnableWindow(TRUE);
}

BOOL RecordingDialog::IsCaptureCorrupt()
{
	return m_chk_captureCorruptFrames.GetCheck();
}

void RecordingDialog::RetrieveNextImageFromBuffer( FlyCapture2::Image* image )
{
	if(m_grabMode == FlyCapture2::BUFFER_FRAMES)
	{
		*image = m_imageBuffer.front();
		m_imageBuffer.erase(m_imageBuffer.begin());
	}
	else
	{
		*image = m_imageBuffer.back();
		m_imageBuffer.clear();
	}
}

void RecordingDialog::SaveImageToFile(FlyCapture2::Image* image, ImageRecordingPage::ImageSettings* imageSettings)
{
	FlyCapture2::PixelFormat pixelFormat = image->GetPixelFormat();
	
	BOOL isValidPGMPixelFormat =	pixelFormat == FlyCapture2::PIXEL_FORMAT_MONO8		|| 
									pixelFormat == FlyCapture2::PIXEL_FORMAT_RAW8		|| 	
									pixelFormat == FlyCapture2::PIXEL_FORMAT_MONO16		|| 	
									pixelFormat == FlyCapture2::PIXEL_FORMAT_RAW16		||
									pixelFormat == FlyCapture2::PIXEL_FORMAT_S_MONO16;

	FlyCapture2::Error error;

	if (imageSettings->imageFormat == ImageRecordingPage::RAW ||
		(imageSettings->imageFormat == ImageRecordingPage::PGM && isValidPGMPixelFormat))
	{
		error = SaveImage(image, imageSettings, m_savedImageCounter);
	}
	else
	{
		FlyCapture2::Image convertedImage;
		error = image->Convert(FlyCapture2::PIXEL_FORMAT_BGR, &convertedImage);
		if (error != FlyCapture2::PGRERROR_OK)
		{
			throw RecordingException(error.GetDescription());
		}
		error = SaveImage(&convertedImage, imageSettings, m_savedImageCounter);
	}
	
	if(error != FlyCapture2::PGRERROR_OK)
	{
		throw RecordingException(error.GetDescription());
	}

	m_savedImageCounter++;
	UpdateImageCounters();
}

void RecordingDialog::SaveImageToVideo(FlyCapture2::AVIRecorder* aviRecorder, FlyCapture2::Image* image )
{
	FlyCapture2::Error error = aviRecorder->AVIAppend(image);	
	if(error != FlyCapture2::PGRERROR_OK)
	{
		throw RecordingException(error.GetDescription());
	}

	m_savedImageCounter++;
	UpdateImageCounters();
}

BOOL RecordingDialog::CreateRecordingEvents()
{
	bool eventCreationSucceeded = TRUE;
	int i = 0;
	
	for (i = 0; i < NUM_RECORDING_EVENTS; i++)
	{
		m_recordingEvents[i] = CreateEvent(NULL, TRUE, FALSE, NULL);
		if (m_recordingEvents[i] == NULL)
		{
			eventCreationSucceeded = FALSE;
			break;
		}
	}

	if (!eventCreationSucceeded)
	{
		for ( ;i >= 0; i--)
		{
			CloseHandle(m_recordingEvents[i]);
		}
	}

	return eventCreationSucceeded;
}

void RecordingDialog::DeleteRecordingEvents()
{
	for (int i = 0; i < NUM_RECORDING_EVENTS; i++)
	{
		CloseHandle(m_recordingEvents[i]);
		m_recordingEvents[i] = NULL;
	}
}

void RecordingDialog::ChangeState(RecorderState state)
{
	CSingleLock stateLock(&m_buttonState);
	stateLock.Lock();
	m_currRecordingState = state;
	m_btn_StartStop.SetWindowText(RecordButtonStrings[state]);
	stateLock.Unlock();

	if (state == SAVING)
	{
		SetEvent(m_recordingEvents[END_LIVE_RECORDING]);
	}

	//gw
	this->Invalidate();
	//gw
}

void RecordingDialog::RC_Start(CString path)
{
	if ( m_currRecordingState == STOPPED )
	{
		if (path.GetLength()>0)
		{
			this->m_edit_SaveFilePath.SetWindowText( path );
		}
		this->OnBnClickedButtonStartStopVideoRecord();
	}
}

void RecordingDialog::RC_Stop()
{
	if (m_currRecordingState == STARTED)
	{
		this->OnBnClickedButtonStartStopVideoRecord();
	}
}

void RecordingDialog::RC_SetPath(CString path)
{
	this->m_edit_SaveFilePath.SetWindowText( path );
}

void RecordingDialog::RC_SetFormat(int fmt)
{
	this->m_imageRecordingPage.SetFormat(fmt);
}

#if 1

void RecordingDialog::OnNcPaint() 
{
		CDC* pDC = GetWindowDC( );
		
		//work out the coordinates of the window rectangle,
		CRect rect;
		GetWindowRect( &rect);
		rect.OffsetRect( -rect.left, -rect.top);
		
		//Draw a single line around the outside
		CBrush brush( RGB( 255, 0, 0));
		pDC->FrameRect( &rect, &brush);
		ReleaseDC( pDC);
}

BOOL RecordingDialog::OnEraseBkgnd(CDC* pDC)
{    
	CRect rect;    GetClientRect(&rect);  
	int r = 0;
	int g = 0;
	int b = 0;
	if (m_currRecordingState == STOPPED) // YELLOW
	{
		r = 255;
		g = 255;
	}
	else if (m_currRecordingState == STARTED)
	{
		g = 255;
	}
	CBrush myBrush(RGB(r, g, b));    // dialog background color   
	CBrush *pOld = pDC->SelectObject(&myBrush);    
	BOOL bRes  = pDC->PatBlt(0, 0, rect.Width(), rect.Height(), PATCOPY);    
	pDC->SelectObject(pOld);    // restore old brush    
	return bRes;                       // CDialog::OnEraseBkgnd(pDC);}
}

#endif