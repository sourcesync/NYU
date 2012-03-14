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
#include "FlyCap2_MFCDoc.h"
#include "FlyCap2_MFCView.h"
using namespace FlyCapture2;
#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CFlyCap2_MFCDoc

IMPLEMENT_DYNCREATE(CFlyCap2_MFCDoc, CDocument)

BEGIN_MESSAGE_MAP(CFlyCap2_MFCDoc, CDocument)
	ON_COMMAND(ID_CAMERACONTROL_TOGGLECAMERACONTROL, &CFlyCap2_MFCDoc::OnToggleCameraControl)
    ON_COMMAND(ID_FILE_SAVE_AS, &CFlyCap2_MFCDoc::OnFileSaveAs)
	ON_COMMAND(ID_FILE_STARTIMAGETRANSFER, &CFlyCap2_MFCDoc::OnStartImageTransfer)
	ON_COMMAND(ID_FILE_STOPIMAGETRANSFER, &CFlyCap2_MFCDoc::OnStopImageTransfer)
	ON_COMMAND(ID_COLORPROCESSINGALGORITHM_NONE, &CFlyCap2_MFCDoc::OnColorAlgorithmNone)
	ON_COMMAND(ID_COLORPROCESSINGALGORITHM_NEARESTNEIGHBOR, &CFlyCap2_MFCDoc::OnColorAlgorithmNearestNeighbor)
	ON_COMMAND(ID_COLORPROCESSINGALGORITHM_EDGESENSING, &CFlyCap2_MFCDoc::OnColorAlgorithmEdgeSensing)
	ON_COMMAND(ID_COLORPROCESSINGALGORITHM_HQLINEAR, &CFlyCap2_MFCDoc::OnColorAlgorithmHQLinear)
	ON_COMMAND(ID_COLORPROCESSINGALGORITHM_DIRECTIONALFILTER, &CFlyCap2_MFCDoc::OnColorAlgorithmDirectionalFilter)
	ON_COMMAND(ID_COLORPROCESSINGALGORITHM_RIGOROUS, &CFlyCap2_MFCDoc::OnColorAlgorithmRigorous)
	ON_COMMAND(ID_COLORPROCESSINGALGORITHM_IPP, &CFlyCap2_MFCDoc::OnColorAlgorithmIPP)

	ON_UPDATE_COMMAND_UI(ID_FILE_STARTIMAGETRANSFER, &CFlyCap2_MFCDoc::OnUpdateStartImageTransferBtn)
	ON_UPDATE_COMMAND_UI(ID_FILE_STOPIMAGETRANSFER, &CFlyCap2_MFCDoc::OnUpdateFileStopImageTransferBtn)
	ON_UPDATE_COMMAND_UI(ID_VIEW_ENABLEOPENGL, &CFlyCap2_MFCDoc::OnUpdateViewEnableOpenGL)
	ON_COMMAND(ID_HISTGRAM_BTN, &CFlyCap2_MFCDoc::OnToggleHistgram)
	ON_COMMAND(ID_FILE_GRAB_AVI, &CFlyCap2_MFCDoc::OnToggleRecorder)
	ON_COMMAND(ID_VIEW_EVENTSTAT, &CFlyCap2_MFCDoc::OnViewEventStat)
	ON_UPDATE_COMMAND_UI(ID_CAMERACONTROL_TOGGLECAMERACONTROL, &CFlyCap2_MFCDoc::OnUpdateCameraControlToggleButton)
	ON_UPDATE_COMMAND_UI(ID_HISTGRAM_BTN, &CFlyCap2_MFCDoc::OnUpdateHistgramBtn)
	ON_UPDATE_COMMAND_UI(ID_FILE_GRAB_AVI, &CFlyCap2_MFCDoc::OnUpdateRecordingBtn)
	ON_UPDATE_COMMAND_UI(ID_VIEW_EVENTSTAT, &CFlyCap2_MFCDoc::OnUpdateEventStatsBtn)
END_MESSAGE_MAP()


// CFlyCap2_MFCDoc construction/destruction

CFlyCap2_MFCDoc::CFlyCap2_MFCDoc()
{
	InitBitmapStruct( _DEFAULT_WINDOW_X, _DEFAULT_WINDOW_Y );

	m_continueGrabThread = false;
	m_threadDoneEvent = CreateEvent( NULL, FALSE, FALSE, NULL );
	m_uiFilterIndex = 0;

	EnableOpenGL(false);
	m_componentsInitialized = false;
	m_isSelectingNewCamera = false;
	m_grabLoopThread = NULL;
	m_pCamera = NULL;
	RegisterCallbacks();
}

CFlyCap2_MFCDoc::~CFlyCap2_MFCDoc()
{
	CloseHandle( m_threadDoneEvent );
	m_eventStatDlg.DestroyWindow();
	m_histogramDlg.DestroyWindow();
	m_recordingDlg.DestroyWindow();

    if (m_pCamera)
    {
        delete m_pCamera;
        m_pCamera = NULL;
    }  
	UnregisterCallbacks();
}
void CFlyCap2_MFCDoc::InitializeComponents()
{
	if (m_componentsInitialized == true)
	{
		m_eventStatDlg.ShowWindow(SW_HIDE);
		m_histogramDlg.ShowWindow(SW_HIDE);
		m_eventStatDlg.CleanUpCounters();
		m_histogramDlg.StopUpdate();
		m_recordingDlg.ShowWindow(SW_HIDE);
		return;
	}
    POSITION pos = GetFirstViewPosition();
    CView* pView = GetNextView(pos); //Get current view
    if (m_eventStatDlg.Create(EventStatDialog::IDD, pView) == FALSE)
	{
		TRACE0("Failed to create event statistics dialog box\n");
	    CString csMessage;
        csMessage.Format(
            "Failed to create event statistics dialog box.");
        AfxMessageBox( csMessage, MB_ICONSTOP );
		m_componentsInitialized = false;
		return;
	}
	if (m_histogramDlg.Create(HistogramDialog::IDD, pView) == FALSE)
	{
		TRACE0("Failed to create histogram dialog box\n");
		CString csMessage;
        csMessage.Format(
            "Failed to create histogram dialog box.");
        AfxMessageBox( csMessage, MB_ICONSTOP );
		m_componentsInitialized = false;
		return;
	}
	if (m_recordingDlg.Create(RecordingDialog::IDD, pView) == FALSE)
	{
		TRACE0("Failed to create recording dialog box\n");
		CString csMessage;
        csMessage.Format(
            "Failed to create recording dialog box.");
        AfxMessageBox( csMessage, MB_ICONSTOP );
		m_componentsInitialized = false;
		return;
	}
	m_componentsInitialized = true;
}


void CFlyCap2_MFCDoc::InitBitmapStruct( int cols, int rows )
{
   BITMAPINFOHEADER* pheader = &m_bitmapInfo.bmiHeader;
   
   // Initialize permanent data in the bitmapinfo header.
   pheader->biSize          = sizeof( BITMAPINFOHEADER );
   pheader->biPlanes        = 1;
   pheader->biCompression   = BI_RGB;
   pheader->biXPelsPerMeter = 100;
   pheader->biYPelsPerMeter = 100;
   pheader->biClrUsed       = 0;
   pheader->biClrImportant  = 0;
   
   // Set a default window size.
   pheader->biWidth    = cols;
   pheader->biHeight   = -rows;
   pheader->biBitCount = 32;
   
   m_bitmapInfo.bmiHeader.biSizeImage = 0;
}

BOOL CFlyCap2_MFCDoc::OnNewDocument()
{
	m_isSelectingNewCamera = true;
	Error error;
	if (!CDocument::OnNewDocument())
		return FALSE;
	// (SDI documents will reuse this document)

	// Set the default image processing parameters
    Image::SetDefaultColorProcessing( NEAREST_NEIGHBOR );
    Image::SetDefaultOutputFormat( PIXEL_FORMAT_BGRU );

	//reset previous camera event counters
	m_previousTransmitFailures = 0;
	m_previousRecoveryCount = 0;
	m_previousSkippedImages = 0;

    // If entering this function from File->New Camera, stop the grab thread
    // first before doing anything else
    if ( m_continueGrabThread == true )
    {
        m_continueGrabThread = false;      

		if (m_pCamera->IsConnected())
		{
			m_pCamera->StopCapture();
		}

        DWORD dwRet = WaitForSingleObject( m_threadDoneEvent, 5000 );
        if ( dwRet == WAIT_TIMEOUT )
        {
            // Timed out while waiting for thread to exit
			//m_grabLoopThread->PostThreadMessageA( WM_QUIT,0,0);// Force thread exit
			TerminateThread(m_grabLoopThread->m_hThread,0);// Force thread exit
			TRACE("Grab thread - force exit\n");
            delete m_grabLoopThread;
            m_grabLoopThread = NULL;
        }
		
        m_pCamera->Disconnect();
    }
    
	m_camCtlDlg.Hide();
	m_camCtlDlg.Disconnect();

	//initialize components
	InitializeComponents();

    // Let the user select a camera
    bool okSelected;
    PGRGuid arGuid[64];
    unsigned int size = 64;
    CameraSelectionDlg camSlnDlg;
    camSlnDlg.ShowModal( &okSelected, arGuid, &size );
    if ( okSelected != true )
    {
        return FALSE;
    }
	if (Start(arGuid[0]) == false)
	{
		return FALSE;
	}
	//m_histogramDlg.StartUpdate();

	m_isSelectingNewCamera = false;
    return TRUE;
}

CString CFlyCap2_MFCDoc::GetTitleString()
{
	CString title;
	if (m_pCamera == NULL)
	{
		title.Format("FlyCap2");
	}
    else
    {
        title.Format(
            "FlyCap2 %s - %s %s (%u)",
            GetVersionString(),
            m_cameraInfo.vendorName,
            m_cameraInfo.modelName,
            m_cameraInfo.serialNumber );
    }

	return title;
}

CString CFlyCap2_MFCDoc::GetVersionString()
{
	FC2Version version;
	const Error errorVer = Utilities::GetLibraryVersion(&version);
	if (errorVer != PGRERROR_OK)
	{
		return "0.0.0.0";
	}

	CString verStr;
	verStr.Format(
		"%u.%u.%u.%u",
		version.major,
		version.minor,
		version.type,
		version.build);
	return verStr;
}

void CFlyCap2_MFCDoc::OnCloseDocument(void)
{
	m_histogramDlg.StopUpdate();
	if(m_recordingDlg.IsRecordingActive())
	{
		m_recordingDlg.StopRecording();
	}
	
    m_continueGrabThread = false; 
	if (m_grabLoopThread != NULL)
	{
		DWORD dwRet = WaitForSingleObject( m_threadDoneEvent, 5000 );
		if ( dwRet == WAIT_TIMEOUT )
		{
			// Timed out while waiting for thread to exit
			//m_grabLoopThread->PostThreadMessageA( WM_QUIT,0,0);// Force thread exit
			TerminateThread(m_grabLoopThread->m_hThread,0);// Force thread exit
			m_pCamera->StopCapture();//camera might not stop due to force exit thread
			delete m_grabLoopThread;
            m_grabLoopThread = NULL;
		}
	}	
    
    m_camCtlDlg.Hide();
    m_camCtlDlg.Disconnect();

    if (m_pCamera != NULL)
	{
		m_pCamera->Disconnect();
        delete m_pCamera;
        m_pCamera = NULL;
	}

    CDocument::OnCloseDocument(); 
}

void CFlyCap2_MFCDoc::UpdateHistogramWindow()
{
	if ( m_histogramDlg.IsWindowVisible() == TRUE )
    {
		CSingleLock dataLock(&m_csRawImageData);
		if (dataLock.IsLocked())
		{
			return;
		}
		if ( dataLock.Lock() ==TRUE)
		{
			m_histogramDlg.SetImageForStatistics(m_rawImage);
			dataLock.Unlock();			
		}
		
    }
}

UINT CFlyCap2_MFCDoc::ThreadGrabImage( void* pparam )
{
    TRACE("Grab thread - start\n");

    CFlyCap2_MFCDoc* pDoc = ((CFlyCap2_MFCDoc*)pparam);
    const UINT uiRetval = pDoc->DoGrabLoop();   
    if( uiRetval != 0 )
    {
        CString csMessage;
        csMessage.Format(
            "The grab thread has encountered a problem and had to terminate." );
        AfxMessageBox( csMessage, MB_ICONSTOP );

        // Signal that the thread has died.
        SetEvent( pDoc->m_threadDoneEvent );      
    }

    TRACE("Grab thread - exit\n");

    return uiRetval;
}



UINT 
CFlyCap2_MFCDoc::DoGrabLoop()
{
	Error error;
	CString csMessage;
	BOOL isCorruptFrame = FALSE;

    // Start of main grab loop
    while( m_continueGrabThread )
    {
		Image buffImage;
		error = m_pCamera->RetrieveBuffer( &buffImage );
        if (error != PGRERROR_OK)
        {
			if (error == PGRERROR_IMAGE_CONSISTENCY_ERROR)
			{
				AddEvent(IMAGE_CONSISTENCY_ERRORS);
				if(m_recordingDlg.IsRecordingActive())
				{
					m_recordingDlg.DoRecording(&buffImage, true);
				}
			}
			else
			{
				time_t rawtime;
				struct tm * timeinfo;
				time( &rawtime );
				timeinfo = localtime( &rawtime );

				char currTimeStr[128];
				sprintf(currTimeStr, "%s", asctime(timeinfo));
				currTimeStr[strlen(currTimeStr) - 1] = '\0';

				char errorMsg[1024];
				sprintf( 
					errorMsg, 
					"%s: Grab loop had an error: %s\n",
					currTimeStr,
					error.GetDescription() );
				TRACE(errorMsg);
			}
            continue;
        }
		
		if(m_recordingDlg.IsRecordingActive())
		{
			m_recordingDlg.DoRecording(&buffImage, false);
		}

		CSingleLock dataLock(&m_csRawImageData);
		dataLock.Lock();
        m_rawImage = buffImage;
		dataLock.Unlock();
		AddEvent(TOTAL_NUMBER_OF_FRAMES);

        // Check to see if the thread should die.
        if (!m_continueGrabThread)
        {
            break;
        }

        // Update current framerate.
        m_processedFrameRate.NewFrame();

        // We try to detect whether the view is getting behind on servicing
        // the invalidate requests we send to it.  If there is still an 
        // invalid area, don't bother color processing this frame.
        bool skipProcessing = false;
        POSITION pos = GetFirstViewPosition();
        while (pos != NULL)
        {
            if (GetUpdateRect(GetNextView(pos)->GetSafeHwnd(), NULL, FALSE) != 0)
            {
                skipProcessing = true;
            }
        }

        // Check to see if the thread should die.
        if( !m_continueGrabThread )
        {
            break;
        }

        if (!skipProcessing)
        {
            // Do post processing on the image.
            unsigned int rows,cols,stride;
            PixelFormat format;
            m_rawImage.GetDimensions(&rows, &cols, &stride, &format);    

            CSingleLock dataLock(&m_csData);
            dataLock.Lock();

			if (m_enableOpenGL)
			{
				error = m_rawImage.Convert(PIXEL_FORMAT_BGR, &m_processedImage); 
			}
			else
			{
				error = m_rawImage.Convert(PIXEL_FORMAT_BGRU, &m_processedImage); 
			}

            if (error != PGRERROR_OK)
            {
				AddEvent(IMAGE_CONVERSION_ERRORS);
                csMessage.Format(
                    "Convert Failure: %s", error.GetDescription());
                continue;
            }
            dataLock.Unlock();        
            InitBitmapStruct(cols, rows); 
			RedrawAllViews();             
        }		
    }
	
    // End of main grab loop
    SetEvent(m_threadDoneEvent);

    return 0;
}


// CFlyCap2_MFCDoc diagnostics

#ifdef _DEBUG
void CFlyCap2_MFCDoc::AssertValid() const
{
	CDocument::AssertValid();
}

void CFlyCap2_MFCDoc::Dump(CDumpContext& dc) const
{
	CDocument::Dump(dc);
}
#endif //_DEBUG


// CFlyCap2_MFCDoc commands

void CFlyCap2_MFCDoc::RedrawAllViews()
{
    POSITION pos = GetFirstViewPosition();
    while ( pos != NULL )
    {
		InvalidateRect( GetNextView( pos )->GetSafeHwnd(), NULL, TRUE );
    }    
	if (m_histogramDlg.IsWindowVisible() == TRUE)
	{
		m_histogramDlg.InvalidateRect(NULL,TRUE);
	}
}

double CFlyCap2_MFCDoc::GetProcessedFrameRate()
{
    return m_processedFrameRate.GetFrameRate();
}

double CFlyCap2_MFCDoc::GetRequestedFrameRate()
{
	// Set up the frame rate data
    Property prop;
    prop.type = FRAME_RATE;
    if ( m_pCamera == NULL)
    {
        return 0.0;
    }
    else
    {
	    Error error = m_pCamera->GetProperty( &prop );
        return (error == PGRERROR_OK) ? prop.absValue : 0.0;
    }
}

unsigned char* CFlyCap2_MFCDoc::GetProcessedPixels()
{
    return m_processedImage.GetData();
}

void CFlyCap2_MFCDoc::GetImageSize( unsigned int* pWidth, unsigned int* pHeight )
{
	/*
	//this sometimes return a expired information, so get rid of this
	*pWidth = abs(m_bitmapInfo.bmiHeader.biWidth);
    *pHeight = abs(m_bitmapInfo.bmiHeader.biHeight);
	*/

	//This will be called in update status bar function (in MainFrm.cpp), 
	//so it must be locked before use.
	CSingleLock dataLock(&m_csRawImageData);
	dataLock.Lock();
	*pWidth = m_rawImage.GetCols();
	*pHeight = m_rawImage.GetRows();
	dataLock.Unlock();
}


Image CFlyCap2_MFCDoc::GetConvertedImage()
{
    return m_processedImage;
}

bool CFlyCap2_MFCDoc::IsGrabThreadRunning()
{
	return m_continueGrabThread;
}

void CFlyCap2_MFCDoc::OnToggleCameraControl()
{
    if ( m_camCtlDlg.IsVisible() == true )
    {
        m_camCtlDlg.Hide();
    }
    else
    {
        m_camCtlDlg.Show();
    }
}

void CFlyCap2_MFCDoc::OnFileSaveAs()
{
    Error   error;
    CString csMessage;
    JPEGOption JPEG_Save_Option;
    PNGOption  PNG_Save_Option;
    Image tempImage;

    CSingleLock dataLock(&m_csRawImageData);
	dataLock.Lock();
    tempImage.DeepCopy(&m_rawImage);
    dataLock.Unlock();

    // Define the list of filters to include in the SaveAs dialog.
    const unsigned int uiNumFilters = 8;
    const CString arcsFilter[uiNumFilters] = {
        "Windows Bitmap (*.bmp)|*.bmp" , 
        "Portable Pixelmap (*.ppm)|*.ppm" , 
        "Portable Greymap (raw image) (*.pgm)|*.pgm" , 
        "Independent JPEG Group (*.jpg, *.jpeg)|*.jpg; *.jpeg" , 
        "Tagged Image File Format (*.tiff)|*.tiff" , 
        "Portable Network Graphics (*.png)|*.png" , 
        "Raw data (*.raw)|*.raw" , 
        "All Files (*.*)|*.*" };

    CString csFilters;

    // Keep track of which filter should be selected as default.
    // m_uiFilterIndex is set to what was previously used (0 if this is first time).
    for ( int i = 0; i < (uiNumFilters - 1); i++ )
    {
        csFilters += arcsFilter[(m_uiFilterIndex + i) % (uiNumFilters - 1)];
        csFilters += "|";      
    }
    // Always finish with All Files and a ||.
    csFilters += arcsFilter[uiNumFilters - 1];
    csFilters += "||"; 

    time_t rawtime;
    struct tm * timeinfo;
    time( &rawtime );
    timeinfo = localtime( &rawtime );

    char timestamp[64];
    strftime( timestamp, 64, "%Y-%m-%d-%H%M%S", timeinfo );

    char tempFilename[128];
    sprintf( tempFilename, "%u-%s", m_cameraInfo.serialNumber, timestamp );

    CFileDialog fileDialog( 
        FALSE, 
        "bmp", 
        tempFilename, 
        OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT, 
        csFilters,
        AfxGetMainWnd() );

    if( fileDialog.DoModal() == IDOK )
    {
        PixelFormat tempPixelFormat = tempImage.GetPixelFormat();
        Error rawError;
        ImageFileFormat saveImageFormat;
        CString csExt = fileDialog.GetFileExt();

        // Check file extension
        if( csExt.CompareNoCase("bmp") == 0 )
        {
            saveImageFormat = FlyCapture2::BMP;
        }
        else if( csExt.CompareNoCase("ppm") == 0 )
        {
            saveImageFormat = FlyCapture2::PPM;
        }
        else if( csExt.CompareNoCase("pgm") == 0 )
        {
            saveImageFormat = FlyCapture2::PGM;
        }
        else if( csExt.CompareNoCase("jpeg") == 0 || csExt.CompareNoCase("jpg") == 0 )
        {
            saveImageFormat = FlyCapture2::JPEG; 
            JPEG_Save_Option.progressive = false;
            JPEG_Save_Option.quality = 100; //Superb quality.
        }
        else if( csExt.CompareNoCase("tiff") == 0 )
        {
            saveImageFormat = FlyCapture2::TIFF;
        }
        else if( csExt.CompareNoCase("png") == 0 )
        {
            saveImageFormat = FlyCapture2::PNG;
            PNG_Save_Option.interlaced = false;
            PNG_Save_Option.compressionLevel = 9; //Best compression
        }
        else if( csExt.CompareNoCase("raw") == 0 )
        {
            saveImageFormat = FlyCapture2::RAW;
        }
        else
        {
            AfxMessageBox( "Invalid file type" );
        }

        if ( saveImageFormat == FlyCapture2::RAW )
        {
            rawError = tempImage.Save( fileDialog.GetPathName(), FlyCapture2::RAW );
            if ( rawError != PGRERROR_OK )
            {
                ShowErrorMessageDialog( "Failed to save image.", rawError);   
            }
        }
        else if ( saveImageFormat == FlyCapture2::PGM )
        {
            PixelFormat tempPixelFormat = tempImage.GetPixelFormat();
            if (tempPixelFormat == PIXEL_FORMAT_MONO8 ||
                tempPixelFormat == PIXEL_FORMAT_MONO12 ||
                tempPixelFormat == PIXEL_FORMAT_MONO16 ||
                tempPixelFormat == PIXEL_FORMAT_RAW8 ||
                tempPixelFormat == PIXEL_FORMAT_RAW12 ||
                tempPixelFormat == PIXEL_FORMAT_RAW16)
            {
                Error error = tempImage.Save( fileDialog.GetPathName(), saveImageFormat );
                if ( error != PGRERROR_OK )
                {
                    ShowErrorMessageDialog( "Failed to convert image", error );           
                }                
            }
            else
            {
                AfxMessageBox( "Invalid file format.\r\nNon mono / raw images cannot be saved as PGM.", MB_ICONSTOP );        
            }
        }
        else
        {
            Error conversionError;
            Image convertedImage;
            conversionError = tempImage.Convert( &convertedImage );
            if ( conversionError != PGRERROR_OK )
            {
                ShowErrorMessageDialog( "Failed to convert image", conversionError );   
                return;
            }

            Error convertedError;
            convertedError = convertedImage.Save( fileDialog.GetPathName(), saveImageFormat );
            if ( convertedError != PGRERROR_OK )
            {         
                ShowErrorMessageDialog( "Failed to save image", convertedError );           
            }
        }  
    }
}

void CFlyCap2_MFCDoc::OnStartImageTransfer()
{
	if ( m_pCamera->IsConnected() != true )
    {
        OnNewDocument();
        return;
    }
	Error error = m_pCamera->StartCapture();
    if( error != PGRERROR_OK )
    {
		CString csMessage;
        csMessage.Format(
            "StartCapture Failure: %s", error.GetDescription() );
        AfxMessageBox( csMessage, MB_ICONSTOP );
        return;
    }
	// Start the grab thread
    m_continueGrabThread = true;   
    m_grabLoopThread = AfxBeginThread( ThreadGrabImage, this );

	
}

void CFlyCap2_MFCDoc::OnStopImageTransfer()
{
	Stop();
	RedrawAllViews();//refresh screen and show PRG logo
}

bool CFlyCap2_MFCDoc::Start( PGRGuid guid )
{
	
    InterfaceType ifType;
    Error error = m_busMgr.GetInterfaceTypeFromGuid( &guid, &ifType );
    if ( error != PGRERROR_OK )
    {   
        return false;
    }    
	if (m_pCamera !=NULL)
	{
		delete m_pCamera;
        m_pCamera = NULL;
	}

    if ( ifType == INTERFACE_GIGE )
    {
        m_pCamera = new GigECamera;
    }
    else
    {
        m_pCamera = new Camera;
    }

    // Connect to selected camera
    error = m_pCamera->Connect( &guid );
    if( error != PGRERROR_OK )
    {
        CString csMessage;
        csMessage.Format(
            "Connect Failure: %s", error.GetDescription() );
        AfxMessageBox( csMessage, MB_ICONSTOP );
        return false;
    }

    error = m_pCamera->GetCameraInfo( &m_cameraInfo );
    if( error != PGRERROR_OK )
    {
        CString csMessage;
        csMessage.Format(
            "CameraInfo Failure: %s", error.GetDescription() );
        AfxMessageBox( csMessage, MB_ICONSTOP );

        return false;
    }

    // Connect the camera control dialog to the camera object
    m_camCtlDlg.Connect( m_pCamera );
	//RegisterCallbacks();
	ForcePGRY16Mode();

    OnStartImageTransfer();	
	return true;
}

bool CFlyCap2_MFCDoc::Stop()
{
	if (m_continueGrabThread == false)
	{
		return false;
	}

	 // Stop the image capture
	m_continueGrabThread=false;
	DWORD dw = WaitForSingleObject(m_grabLoopThread->m_hThread,1000);
	if (dw != WAIT_OBJECT_0)
	{
		if (dw == WAIT_TIMEOUT)
		{
			TerminateThread(m_grabLoopThread->m_hThread,0);// Force thread exit
			TRACE("Grab thread - force exit\n");
		}
		else
		{
			TRACE("Grab thread - has an error\n");
		}

        delete m_grabLoopThread;
        m_grabLoopThread = NULL;
	}

    ASSERT(m_pCamera != NULL);
	if (m_pCamera->IsConnected())
	{
		Error error = m_pCamera->StopCapture();
		if( error != PGRERROR_OK)
		{
			// This may fail when the camera was removed, so don't show 
			// an error message

			/*csMessage.Format(
				"Stop Failure: %s", error.GetDescription() );
			AfxMessageBox( csMessage, MB_ICONSTOP );*/
		}
	}

	RedrawAllViews();// Refresh drawing area to show the PGR logo

	// Hide the camera control dialog
	m_camCtlDlg.Hide();

	if (m_histogramDlg.IsWindowVisible() == TRUE)
	{
		m_histogramDlg.ShowWindow(SW_HIDE);
	}

	if (m_eventStatDlg.IsWindowVisible() == TRUE)
	{
		m_eventStatDlg.ShowWindow(SW_HIDE);
	}

	if (m_recordingDlg.IsWindowVisible())
	{
		m_recordingDlg.ShowWindow(SW_HIDE);
	}

	return true;
}
void CFlyCap2_MFCDoc::RegisterCallbacks()
{
	Error error;

    // Register arrival callbacks
    error = m_busMgr.RegisterCallback( &CFlyCap2_MFCDoc::OnBusArrival, ARRIVAL, this, &m_cbArrivalHandle );    
    if ( error != PGRERROR_OK )
    {
        ShowErrorMessageDialog( "Failed to register callback", error );     
    } 

    // Register removal callbacks
    error = m_busMgr.RegisterCallback( &CFlyCap2_MFCDoc::OnBusRemoval, REMOVAL, this, &m_cbRemovalHandle );    
    if ( error != PGRERROR_OK )
    {
        ShowErrorMessageDialog( "Failed to register callback", error );     
    } 

	// Register reset callbacks
    error = m_busMgr.RegisterCallback( &CFlyCap2_MFCDoc::OnBusReset, BUS_RESET, this, &m_cbResetHandle );    
    if ( error != PGRERROR_OK )
    {
        ShowErrorMessageDialog( "Failed to register callback", error );     
    } 
}
void CFlyCap2_MFCDoc::UnregisterCallbacks()
{
	Error error;

    // Unregister arrival callback
    error = m_busMgr.UnregisterCallback( m_cbArrivalHandle );
    if ( error != PGRERROR_OK )
    {
        ShowErrorMessageDialog( "Failed to unregister callback", error );     
    }   

    // Unregister removal callback
    error = m_busMgr.UnregisterCallback( m_cbRemovalHandle );
    if ( error != PGRERROR_OK )
    {
        ShowErrorMessageDialog( "Failed to unregister callback", error );     
    }   

	// Unregister reset callback
    error = m_busMgr.UnregisterCallback( m_cbResetHandle );
    if ( error != PGRERROR_OK )
    {
        ShowErrorMessageDialog( "Failed to unregister callback", error );     
    }
}


void CFlyCap2_MFCDoc::OnBusReset( void* pParam, unsigned int serialNumber )
{
	CFlyCap2_MFCDoc* pDoc =  static_cast<CFlyCap2_MFCDoc*>(pParam);
	pDoc->AddEvent(NUMBER_OF_BUS_RESETS);
}

void CFlyCap2_MFCDoc::OnBusArrival( void* pParam, unsigned int serialNumber )
{
    CFlyCap2_MFCDoc* pDoc =  static_cast<CFlyCap2_MFCDoc*>(pParam);
    pDoc->m_arrQueue.push(serialNumber);
    pDoc->OnBusArrivalEvent();
}


void CFlyCap2_MFCDoc::OnBusRemoval( void* pParam , unsigned int serialNumber)
{
    CFlyCap2_MFCDoc* pDoc =  static_cast<CFlyCap2_MFCDoc*>(pParam);
    pDoc->m_remQueue.push(serialNumber);
    pDoc->OnBusRemovalEvent();
}

void CFlyCap2_MFCDoc::OnBusRemovalEvent()
{
    unsigned int serialNumber;
    serialNumber = m_remQueue.front();
    m_remQueue.pop();
    if( m_cameraInfo.serialNumber == serialNumber )
    {
		Stop();
		m_pCamera->Disconnect();
    }
	AddEvent(NUMBER_OF_BUS_REMOVALS);
}

void CFlyCap2_MFCDoc::OnBusArrivalEvent()
{
    unsigned int serialNumber;
    serialNumber = m_arrQueue.front();
    m_arrQueue.pop();
	AddEvent(NUMBER_OF_BUS_ARRIVALS);
    // Currently on Arrival all camera handles get updated.
    // We need to stop the stream just like in the OnBusRemoval.
    // TODO: once proper bus reset handlign is implemented we can 
    // remove this stop here.
}

void CFlyCap2_MFCDoc::ForcePGRY16Mode()
{
    Error error;
    const unsigned int k_imageDataFmtReg = 0x1048;
    unsigned int value = 0;
    error = m_pCamera->ReadRegister( k_imageDataFmtReg, &value );
    if ( error != PGRERROR_OK )
    {
        // Error
    }

    value &= ~(0x1 << 0);

    error = m_pCamera->WriteRegister( k_imageDataFmtReg, value );
    if ( error != PGRERROR_OK )
    {
        // Error
    }
}

bool CFlyCap2_MFCDoc::IsOpenGLEnabled()
{
	return m_enableOpenGL;
}

void CFlyCap2_MFCDoc::ShowErrorMessageDialog(char* mainTxt, Error error, bool detailed)
{
	char tempStr[1024];
	if ( detailed == true )
    { 
        sprintf(
            tempStr,
            "Source: %s(%u) Built: %s - %s\n",
            error.GetFilename(),
            error.GetLine(),
            error.GetBuildDate(),
            error.GetDescription() );

        Error cause = error.GetCause();
        while( cause.GetType() != PGRERROR_UNDEFINED )
        {
            sprintf(
                tempStr,
                "+-> From: %s(%d) Built: %s - %s\n",
                cause.GetFilename(),
                cause.GetLine(),
                cause.GetBuildDate(),
                cause.GetDescription() );
            cause = cause.GetCause();
		}
	}
	else
	{
		 sprintf(tempStr,error.GetDescription());
	}
	CString csMessage;
    csMessage.Format(
    "%s: %s", mainTxt,tempStr  );
    AfxMessageBox( csMessage, MB_ICONSTOP );
}

void CFlyCap2_MFCDoc::UncheckAllColorProcessingAlgorithm()
{
	CMenu *pMenu = AfxGetMainWnd()->GetMenu();
	//Go to color processing algorithm menu
	pMenu = pMenu->GetSubMenu(2); //go to setting menu
#if (_DEBUG)
	//validate position
	if (pMenu == NULL)
	{
		//Bug: Menu item not found
		//It means the menu item has been changed by others
		//this will cause some problems of updating menu item
		//to resolve this you need you open resource view of IDR_MAINFRAME
		//put the "Color processing algorithm" position to the first in setting menu
		//and put the "Setting" position to the 3rd in main menu
		DebugBreak();
	}
#endif
	pMenu = pMenu->GetSubMenu(0); //go to color processing algorithm menu
#if (_DEBUG)
	//validate menu item is right or not
	if (pMenu == NULL || pMenu->GetMenuItemID(0) != ID_COLORPROCESSINGALGORITHM_NONE)
	{
		//Bug: this item is not color processing algorithm
		//It means the menu item has been changed by others
		//this will cause some problems of updating menu item
		//to resolve this you need you open resource view of IDR_MAINFRAME
		//put the "Color processing algorithm" position to the first in setting menu
		//and put the "Setting" position to the 3rd in main menu
		DebugBreak();
	}
#endif
	for (int i = 0; i < 7;i++)
	{
		pMenu->CheckMenuItem(i,MF_UNCHECKED | MF_BYPOSITION);
	}
}


void CFlyCap2_MFCDoc::OnColorAlgorithmNone()
{
	UncheckAllColorProcessingAlgorithm();
	CMenu *pMenu = AfxGetMainWnd()->GetMenu();
	pMenu->CheckMenuItem(ID_COLORPROCESSINGALGORITHM_NONE,MF_CHECKED | MF_BYCOMMAND);
	Image::SetDefaultColorProcessing(NO_COLOR_PROCESSING);
}

void CFlyCap2_MFCDoc::OnColorAlgorithmNearestNeighbor()
{
	UncheckAllColorProcessingAlgorithm();
	CMenu *pMenu = AfxGetMainWnd()->GetMenu();
	pMenu->CheckMenuItem(ID_COLORPROCESSINGALGORITHM_NEARESTNEIGHBOR,MF_CHECKED | MF_BYCOMMAND);
	Image::SetDefaultColorProcessing(NEAREST_NEIGHBOR);
}

void CFlyCap2_MFCDoc::OnColorAlgorithmEdgeSensing()
{
	UncheckAllColorProcessingAlgorithm();
	CMenu *pMenu = AfxGetMainWnd()->GetMenu();
	pMenu->CheckMenuItem(ID_COLORPROCESSINGALGORITHM_EDGESENSING,MF_CHECKED | MF_BYCOMMAND);
	Image::SetDefaultColorProcessing(EDGE_SENSING);
}

void CFlyCap2_MFCDoc::OnColorAlgorithmHQLinear()
{
	UncheckAllColorProcessingAlgorithm();
	CMenu *pMenu = AfxGetMainWnd()->GetMenu();
	pMenu->CheckMenuItem(ID_COLORPROCESSINGALGORITHM_HQLINEAR,MF_CHECKED | MF_BYCOMMAND);
	Image::SetDefaultColorProcessing(HQ_LINEAR);
}

void CFlyCap2_MFCDoc::OnColorAlgorithmDirectionalFilter()
{
	UncheckAllColorProcessingAlgorithm();
	CMenu *pMenu = AfxGetMainWnd()->GetMenu();
	pMenu->CheckMenuItem(ID_COLORPROCESSINGALGORITHM_DIRECTIONALFILTER,MF_CHECKED | MF_BYCOMMAND);
	Image::SetDefaultColorProcessing(DIRECTIONAL_FILTER);
}

void CFlyCap2_MFCDoc::OnColorAlgorithmRigorous()
{
	UncheckAllColorProcessingAlgorithm();
	CMenu *pMenu = AfxGetMainWnd()->GetMenu();
	pMenu->CheckMenuItem(ID_COLORPROCESSINGALGORITHM_RIGOROUS,MF_CHECKED | MF_BYCOMMAND);
	Image::SetDefaultColorProcessing(RIGOROUS);
}

void CFlyCap2_MFCDoc::OnColorAlgorithmIPP()
{
	UncheckAllColorProcessingAlgorithm();
	CMenu *pMenu = AfxGetMainWnd()->GetMenu();
	pMenu->CheckMenuItem(ID_COLORPROCESSINGALGORITHM_IPP,MF_CHECKED | MF_BYCOMMAND);
	Image::SetDefaultColorProcessing(IPP);
}
void CFlyCap2_MFCDoc::OnUpdateStartImageTransferBtn(CCmdUI *pCmdUI)
{
	if (m_isSelectingNewCamera == false)
	{
		pCmdUI->Enable(m_continueGrabThread ? FALSE : TRUE );
	}
	else
	{
		pCmdUI->Enable(FALSE);
	}
}

void CFlyCap2_MFCDoc::OnUpdateFileStopImageTransferBtn(CCmdUI *pCmdUI)
{
	if (m_isSelectingNewCamera == false)
	{
		pCmdUI->Enable(m_continueGrabThread ? TRUE : FALSE);
	}
	else
	{
		pCmdUI->Enable(FALSE);
	}
}

void CFlyCap2_MFCDoc::EnableOpenGL(bool bEnable)
{
	m_enableOpenGL = bEnable;
}

void CFlyCap2_MFCDoc::OnUpdateViewEnableOpenGL(CCmdUI *pCmdUI)
{
	pCmdUI->SetCheck(m_enableOpenGL ? TRUE:FALSE);
}

InformationPane::InformationPaneStruct CFlyCap2_MFCDoc::GetRawImageInformation()
{
	Error error;
	InformationPane::InformationPaneStruct infoStruct;
	infoStruct.fps.requestedFrameRate = GetRequestedFrameRate();
	infoStruct.fps.processedFrameRate = GetProcessedFrameRate();
    if (m_pCamera == NULL)
    {
        return infoStruct;
    }

	CSingleLock dataLock(&m_csRawImageData);
	dataLock.Lock();

	// Set up the timestamp data
	infoStruct.timestamp = m_rawImage.GetTimeStamp();

	// Set up the image info data
	m_rawImage.GetDimensions( 
				&infoStruct.imageInfo.height, 
				&infoStruct.imageInfo.width, 
				&infoStruct.imageInfo.stride, 
				&infoStruct.imageInfo.pixFmt );
	
	// Set up the embedded image info data
	const unsigned int k_frameInfoReg = 0x12F8;
	unsigned int frameInfoRegVal = 0;
	error = m_pCamera->ReadRegister( k_frameInfoReg, &frameInfoRegVal );
	if (error == PGRERROR_OK &&
        (frameInfoRegVal >> 31) != 0)
	{
		ImageMetadata metadata = m_rawImage.GetMetadata();
        dataLock.Unlock();

        const int k_numEmbeddedInfo = 10;  
		unsigned int* pEmbeddedInfo = infoStruct.embeddedInfo.arEmbeddedInfo;

		for (int i=0; i < k_numEmbeddedInfo; i++)
		{
			switch (i)
			{
			case 0: pEmbeddedInfo[i] = metadata.embeddedTimeStamp; break;
			case 1: pEmbeddedInfo[i] = metadata.embeddedGain; break;
			case 2: pEmbeddedInfo[i] = metadata.embeddedShutter; break;
			case 3: pEmbeddedInfo[i] = metadata.embeddedBrightness; break;
			case 4: pEmbeddedInfo[i] = metadata.embeddedExposure; break;
			case 5: pEmbeddedInfo[i] = metadata.embeddedWhiteBalance; break;
			case 6: pEmbeddedInfo[i] = metadata.embeddedFrameCounter; break;
			case 7: pEmbeddedInfo[i] = metadata.embeddedStrobePattern; break;
			case 8: pEmbeddedInfo[i] = metadata.embeddedGPIOPinState; break;
			case 9: pEmbeddedInfo[i] = metadata.embeddedROIPosition; break;
			}
		}
	} 
    else
    {
	    dataLock.Unlock();
    }

	// Set up the diagnostics info
	const unsigned int k_frameSkippedReg = 0x12C0;
	unsigned int frameSkippedRegVal = 0;
	error = m_pCamera->ReadRegister( k_frameSkippedReg, &frameSkippedRegVal );
	if (error != PGRERROR_OK  || 
		m_cameraInfo.interfaceType != INTERFACE_USB3 || 
        m_cameraInfo.iidcVer < 132 ||
		(m_cameraInfo.iidcVer >= 132 && (frameSkippedRegVal & 0x80000000) == 0))
	{
		infoStruct.diagnostics.skippedFrames = -1;
	}
	else
	{
		int skippedImages = frameSkippedRegVal & 0x7FFFFFFF;
		infoStruct.diagnostics.skippedFrames = skippedImages;
		int newEvents = skippedImages - m_previousSkippedImages;
		if (newEvents > 0)
		{
			for (int i = 0; i < newEvents; i++)
			{
				AddEvent(SKIPPED_IMAGES);
			}
		}
		m_previousSkippedImages = skippedImages;
	}

	const unsigned int k_linkRecoveryCountReg = 0x12C4;
	unsigned int linkRecoveryCountRegVal = 0;
	error = m_pCamera->ReadRegister( k_linkRecoveryCountReg, &linkRecoveryCountRegVal );
	if (error != PGRERROR_OK  ||
		m_cameraInfo.interfaceType != INTERFACE_USB3 ||
        m_cameraInfo.iidcVer < 132 ||
		(m_cameraInfo.iidcVer >= 132 && (linkRecoveryCountRegVal & 0x80000000) == 0))
	{
		infoStruct.diagnostics.linkRecoveryCount = -1;
	}
	else
	{
		int recoveryCount = linkRecoveryCountRegVal & 0x7FFFFFFF;
		infoStruct.diagnostics.linkRecoveryCount = recoveryCount;
		int newEvents = recoveryCount - m_previousRecoveryCount;
		if (newEvents > 0)
		{
			for (int i = 0; i < newEvents; i++)
			{
				AddEvent(RECOVERY_COUNT);
			}
		}
		m_previousRecoveryCount = recoveryCount;
	}

	const unsigned int k_transmitFailureReg = 0x12FC;
	unsigned int transmitFailureRegVal = 0;
	error = m_pCamera->ReadRegister( k_transmitFailureReg, &transmitFailureRegVal );
	if ( error != PGRERROR_OK  || 
        (m_cameraInfo.iidcVer >= 132 && (transmitFailureRegVal & 0x80000000) == 0 ))
	{
		infoStruct.diagnostics.transmitFailures = -1;
	}
	else
	{
		int transmitFailuresValue = transmitFailureRegVal & 0x7FFFFFFF;
		infoStruct.diagnostics.transmitFailures = transmitFailuresValue;
		int newFailures = transmitFailuresValue - m_previousTransmitFailures;
		if (newFailures > 0)
		{
			for (int i = 0; i < newFailures; i++)
			{
				AddEvent(TRANSMIT_FAILURES);
			}
		}
		m_previousTransmitFailures = transmitFailuresValue;
	}

	const unsigned int k_initializeTimeReg = 0x12E0;   
	unsigned int initializeTimeRegVal = 0;
	error = m_pCamera->ReadRegister( k_initializeTimeReg, &initializeTimeRegVal );
	if ( error != PGRERROR_OK )
	{
		infoStruct.diagnostics.timeSinceInitialization = "";
	}
	else
	{
		unsigned int numHours = 0;
		unsigned int numMins = 0;
		unsigned int numSeconds = 0;

		ParseTimeRegister( initializeTimeRegVal, numHours, numMins, numSeconds );

		char timeStr[512];
		sprintf( 
			timeStr,
			"%uh %um %us",
			numHours,
			numMins,
			numSeconds );

		infoStruct.diagnostics.timeSinceInitialization = timeStr;
	}

	const unsigned int k_busResetTimeReg = 0x12E4;  
	unsigned int busResetTimeRegVal = 0;
	error = m_pCamera->ReadRegister( k_busResetTimeReg, &busResetTimeRegVal );
	if ( error != PGRERROR_OK )
	{
		infoStruct.diagnostics.timeSinceLastBusReset = "";
	}
	else
	{
		unsigned int numHours = 0;
		unsigned int numMins = 0;
		unsigned int numSeconds = 0;

		ParseTimeRegister( busResetTimeRegVal, numHours, numMins, numSeconds );

		char timeStr[512];
		sprintf( 
			timeStr,
			"%uh %um %us",
			numHours,
			numMins,
			numSeconds );

		infoStruct.diagnostics.timeSinceLastBusReset = timeStr;
	}
	return infoStruct;
}

void CFlyCap2_MFCDoc::ParseTimeRegister( 
    unsigned int timeRegVal, 
    unsigned int& hours, 
    unsigned int& mins, 
    unsigned int& seconds )
{
    hours = timeRegVal / (60 * 60);
    mins = (timeRegVal - (hours * 60 * 60)) / 60;
    seconds = timeRegVal - (hours * 60 * 60) - (mins * 60);
}

void CFlyCap2_MFCDoc::OnViewEventStat()
{
	ToggleEventStatDialog();
}
void CFlyCap2_MFCDoc::ToggleEventStatDialog()
{
	if (m_eventStatDlg.IsWindowVisible() == TRUE)
	{
		m_eventStatDlg.ShowWindow(SW_HIDE);
	}
	else
	{
		m_eventStatDlg.ShowWindow(SW_SHOW);
	}
}

void CFlyCap2_MFCDoc::AddEvent(EventType eventType)
{
	m_eventStatDlg.AddEvent(eventType);
}

void CFlyCap2_MFCDoc::OnToggleHistgram()
{
	if (m_histogramDlg.IsWindowVisible() == TRUE)
	{
		m_histogramDlg.ShowWindow(SW_HIDE);
		m_histogramDlg.StopUpdate();
	}
	else
	{
		m_histogramDlg.StartUpdate();
		m_histogramDlg.ShowWindow(SW_SHOW);
	}
}

void CFlyCap2_MFCDoc::OnToggleRecorder()
{
	if (m_recordingDlg.IsWindowVisible())
	{
		m_recordingDlg.ShowWindow(SW_HIDE);
	}
	else
	{
		m_recordingDlg.ShowWindow(SW_SHOW);
		m_recordingDlg.StoreCamPtr(m_pCamera);
	}
}

bool CFlyCap2_MFCDoc::HasBadEventRecently()
{
	if (m_eventStatDlg.IsWindowVisible() == FALSE)
	{
		// if the event data window is visible
		// the update function will be automatically 
		// called in EventStatDialog::OnTimer();
		// but if it is not visible, we need to 
		// update event data manually in order to 
		// check recent bad event
		m_eventStatDlg.UpdateEventsData();
	}
	return m_eventStatDlg.HasBadEventRecently();
}
void CFlyCap2_MFCDoc::OnUpdateCameraControlToggleButton(CCmdUI *pCmdUI)
{
    if ( m_pCamera->IsConnected() == true )
    {
        pCmdUI->Enable(TRUE);
	    pCmdUI->SetCheck(m_camCtlDlg.IsVisible()?TRUE:FALSE);
    }
    else
    {
        pCmdUI->Enable(FALSE);
        pCmdUI->SetCheck(FALSE);
    }
}

void CFlyCap2_MFCDoc::OnUpdateHistgramBtn(CCmdUI *pCmdUI)
{
    if ( m_pCamera->IsConnected() == true )
    {
        pCmdUI->Enable(TRUE);
	    pCmdUI->SetCheck(m_histogramDlg.IsWindowVisible());
    }
    else
    {
        pCmdUI->Enable(FALSE);
        pCmdUI->SetCheck(FALSE);
    }
}

void CFlyCap2_MFCDoc::OnUpdateRecordingBtn(CCmdUI *pCmdUI)
{
    if ( m_pCamera->IsConnected() == true )
    {
        pCmdUI->Enable(TRUE);
	    pCmdUI->SetCheck(m_recordingDlg.IsWindowVisible());
    }
    else
    {
        pCmdUI->Enable(FALSE);
        pCmdUI->SetCheck(FALSE);
    }
}

void CFlyCap2_MFCDoc::OnUpdateEventStatsBtn(CCmdUI *pCmdUI)
{
	pCmdUI->SetCheck(m_eventStatDlg.IsWindowVisible());
}
