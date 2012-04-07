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
#include "MainFrm.h"
#include <vector>
#include <io.h>
#include <GL/gl.h>
#include <GL/glu.h>
using namespace FlyCapture2;

// in order to get function prototypes from glext.h, define GL_GLEXT_PROTOTYPES before including glext.h
#define GL_GLEXT_PROTOTYPES
#include "glext.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// function pointers for PBO Extension
// Windows needs to get function pointers from ICD OpenGL drivers,
// because opengl32.dll does not support extensions higher than v1.1.
PFNGLGENBUFFERSARBPROC pglGenBuffersARB = 0;                     // VBO Name Generation Procedure
PFNGLBINDBUFFERARBPROC pglBindBufferARB = 0;                     // VBO Bind Procedure
PFNGLBUFFERDATAARBPROC pglBufferDataARB = 0;                     // VBO Data Loading Procedure
PFNGLDELETEBUFFERSARBPROC pglDeleteBuffersARB = 0;               // VBO Deletion Procedure
#define glGenBuffersARB           pglGenBuffersARB
#define glBindBufferARB           pglBindBufferARB
#define glBufferDataARB           pglBufferDataARB
#define glDeleteBuffersARB        pglDeleteBuffersARB

int GetMinimumPowerOfTwo(int in)
{
    int i = 1;
    while ( i < in)
    {
        i *= 2;
    }

    return i;
}

void OutputGLError( char* pszLabel )
{
    GLenum errorno = glGetError();

    if ( errorno != GL_NO_ERROR )
    {
        char msg[256];
        sprintf( msg,
            "%s had error: #(%d) %s\r\n", 
            pszLabel, 
            errorno, 
            gluErrorString( errorno ) );
        AfxMessageBox( msg, MB_OK);
    }
}

// CFlyCap2_MFCView

IMPLEMENT_DYNCREATE(CFlyCap2_MFCView, CFormView)

BEGIN_MESSAGE_MAP(CFlyCap2_MFCView, CFormView)
	ON_WM_CREATE()
    ON_WM_DESTROY()
    ON_WM_SIZE()
	ON_WM_ERASEBKGND()
	ON_COMMAND(ID_VIEW_SHOWINFORMATIONPANEL, &CFlyCap2_MFCView::OnViewShowInformationPanel)
	ON_UPDATE_COMMAND_UI(ID_VIEW_DRAWIMAGE, &CFlyCap2_MFCView::OnUpdateViewDrawImage)
	ON_COMMAND(ID_VIEW_DRAWIMAGE, &CFlyCap2_MFCView::OnViewDrawImage)
	ON_COMMAND(ID_VIEW_CHANGECROSSHAIRCOLOR, &CFlyCap2_MFCView::OnViewChangeCrosshairColor)
	ON_COMMAND(ID_VIEW_DRAWCROSSHAIR, &CFlyCap2_MFCView::OnDrawCrosshairClick)
	ON_UPDATE_COMMAND_UI(ID_VIEW_DRAWCROSSHAIR, &CFlyCap2_MFCView::OnUpdateViewDrawCrosshair)
	ON_UPDATE_COMMAND_UI(ID_VIEW_FULLSCREEN, &CFlyCap2_MFCView::OnUpdateViewFullscreen)
	ON_WM_GETMINMAXINFO()
	ON_COMMAND(ID_VIEW_ENABLEOPENGL, &CFlyCap2_MFCView::OnViewEnableOpenGL)
	ON_COMMAND(ID_ZOOM_IN, &CFlyCap2_MFCView::OnZoomIn)
	ON_COMMAND(ID_ZOOM_OUT, &CFlyCap2_MFCView::OnZoomOut)
	ON_COMMAND(ID_BTN_GETORGSIZE, &CFlyCap2_MFCView::OnBtnOriginalImageSize)
	ON_UPDATE_COMMAND_UI(ID_ZOOM_IN, &CFlyCap2_MFCView::OnUpdateZoomIn)
	ON_UPDATE_COMMAND_UI(ID_ZOOM_OUT, &CFlyCap2_MFCView::OnUpdateZoomOut)
	ON_UPDATE_COMMAND_UI(ID_BTN_GETORGSIZE, &CFlyCap2_MFCView::OnUpdateOriginalZoom)
	ON_WM_LBUTTONDOWN()
	ON_WM_LBUTTONUP()
	ON_WM_HSCROLL()
	ON_WM_VSCROLL()
	ON_WM_MOUSEMOVE()
	ON_UPDATE_COMMAND_UI(ID_VIEW_STRETCHTOFIT, &CFlyCap2_MFCView::OnUpdateViewStretchToFit)
	ON_COMMAND(ID_VIEW_STRETCHTOFIT, &CFlyCap2_MFCView::OnViewStretchToFit)
	ON_WM_MOUSEWHEEL()
	ON_COMMAND(ID_FILE_NEWCAMERA, &CFlyCap2_MFCView::OnFileNewCamera)
	ON_UPDATE_COMMAND_UI(ID_VIEW_SHOWINFORMATIONPANEL, &CFlyCap2_MFCView::OnUpdateViewShowInformationPanel)
END_MESSAGE_MAP()


// CFlyCap2_MFCView construction/destruction

CFlyCap2_MFCView::CFlyCap2_MFCView()
	: CFormView(CFlyCap2_MFCView::IDD)
{
	m_isStreaming = false;
	m_viewInitialized=false;
	m_isFullScreen = false;
	m_openGLInitialized = false; 
	m_camInfoDataInitialized = false;
	m_enableCameraInformationPanel = true;
	m_isDrawingImage = true;
	m_showCrosshair = false;
	m_isStretchToFit = false;
	m_currentSizeX = 0;
	m_currentSizeY = 0;

    m_hHand = (HCURSOR)LoadImage( 
        GetModuleHandle( NULL ), 
        MAKEINTRESOURCE( IDC_CURSOR_GRAB ), 
        IMAGE_CURSOR, 
        0, 
        0, 
        LR_MONOCHROME );

    m_hArrow = (HCURSOR)LoadImage( 
        NULL, 
        MAKEINTRESOURCE( IDC_ARROW ),
        IMAGE_CURSOR, 
        0, 
        0,
        LR_MONOCHROME );

    m_zoomLevelTable.push_back(12.5);
    m_zoomLevelTable.push_back(16.7);
    m_zoomLevelTable.push_back(25);
    m_zoomLevelTable.push_back(33);
    m_zoomLevelTable.push_back(50);
    m_zoomLevelTable.push_back(75);
    m_zoomLevelTable.push_back(100);
    m_zoomLevelTable.push_back(125);
    m_zoomLevelTable.push_back(150);
    m_zoomLevelTable.push_back(200);
    m_zoomLevelTable.push_back(300);
    m_zoomLevelTable.push_back(400);
    m_zoomLevelTable.push_back(600);
    m_zoomLevelTable.push_back(800);
    m_zoomLevelTable.push_back(1200);
    m_zoomLevelTable.push_back(1600);
    m_zoomLevelTable.push_back(2400);
    m_zoomLevelTable.push_back(3200);    

    for (unsigned int i=0; i < m_zoomLevelTable.size(); i++)
    {
        if (m_zoomLevelTable[i] == 100.0f)
        {
            m_zoomLevelIndex = i;
        }
    }
}

void CFlyCap2_MFCView::ResetViewConfig()
{
	if (m_isFullScreen == true)
	{
		//restore window size
		((CMainFrame*)GetParentFrame())->ToggleView();
	}

	m_isDrawingImage = true;
	OnInitialUpdate();
}

CFlyCap2_MFCView::~CFlyCap2_MFCView()
{
	glDeleteTextures( sk_maxNumTextures, m_imageTextures );
}

void CFlyCap2_MFCView::DoDataExchange(CDataExchange* pDX)
{
	CFormView::DoDataExchange(pDX);
}

BOOL CFlyCap2_MFCView::PreCreateWindow(CREATESTRUCT& cs)
{
	//  the CREATESTRUCT cs
	return CFormView::PreCreateWindow(cs);
}

void CFlyCap2_MFCView::OnInitialUpdate()
{
	CFormView::OnInitialUpdate();
	CFlyCap2_MFCDoc* pDoc = GetDocument();
    ASSERT_VALID(pDoc);

	CTreeCtrl* pCamInfoTreeView  = (CTreeCtrl *)GetDlgItem(IDC_INFOTREE); 
    m_openGLInitialized = InitializeOpenGL();
	m_camInfoDataInitialized = m_infoPane.Initialize(pCamInfoTreeView);
	m_colorCrosshair = 0x000000FF;//set initial crosshair color to red
	//set application to a new title
	AfxGetMainWnd()->SetWindowText(pDoc->GetTitleString());
	if (m_viewInitialized == false)
	{
		m_zoomLevel = 1.0;
		m_prevZoomLevel = 1.0;
		m_logo.LoadBitmapA(IDB_BITMAP_LOGO);//Load PGR Logo Bitmap
		BITMAP bmp;
        const int bitmapRetVal = m_logo.GetBitmap(&bmp);
		if (bitmapRetVal == 0)
        {
            MessageBox("Failed to load Pt Grey logo.\n");
			m_viewInitialized = false;
            return;
        }
		m_logo.SetBitmapDimension(bmp.bmWidth,bmp.bmHeight);
		// Resize the window to properly display the image
		unsigned int width,height;
		pDoc->GetImageSize(&width, &height);
        UpdateCameraInfoPanel();

        //Adjust initial frame size
        static const int sk_additionalHeight = 150; //including: toolbar height, caption height, status bar height
        int preferredWidth = width + sk_camInfoWidth+sk_scrollbarThickness+(GetSystemMetrics(SM_CXFRAME) * 2);
        if (preferredWidth > 1280)//make sure initial width is not too large
        {
            preferredWidth = 1280;
        }
        int preferredHeight = height+sk_scrollbarThickness+sk_additionalHeight+ (GetSystemMetrics(SM_CYFRAME) * 2);
        if (preferredHeight > 960)//make sure initial height is not too large
        {
            preferredHeight = 960;
        }
        GetParentFrame()->SetWindowPos(NULL,0,0,preferredWidth,preferredHeight,SWP_NOMOVE);
		m_viewInitialized = true;
        AdjustDrawingArea();
	}
}

// CFlyCap2_MFCView diagnostics

#ifdef _DEBUG
void CFlyCap2_MFCView::AssertValid() const
{
	CFormView::AssertValid();
}

void CFlyCap2_MFCView::Dump(CDumpContext& dc) const
{
	CFormView::Dump(dc);
}

CFlyCap2_MFCDoc* CFlyCap2_MFCView::GetDocument() const // non-debug version is inline
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CFlyCap2_MFCDoc)));
	return (CFlyCap2_MFCDoc*)m_pDocument;
}
#endif //_DEBUG

void CFlyCap2_MFCView::OnDraw(CDC* pDC)
{
	CFormView::OnDraw(pDC);
	CStatic* pDrawingArea = (CStatic*)GetDlgItem(IDC_IMAGEAREA);
	CFlyCap2_MFCDoc* pDoc = GetDocument();
    ASSERT_VALID(pDoc);

    if (!pDoc || pDrawingArea == NULL || m_viewInitialized == false)
	{
        return;
	}

	// Update current streaming status
	if (pDoc->IsGrabThreadRunning() != m_isStreaming)
	{
		 m_isStreaming = pDoc->IsGrabThreadRunning();
		 AdjustDrawingArea();
	}

	CDC* pImageDC = pDrawingArea->GetDC();//Device context for drawing
	if (pImageDC == NULL )
	{
        TRACE("Unable to get DC for drawing area\n");
		return;
	}

	unsigned char* pImagePixels = NULL;

	RECT imageDrawingAreaRect;
	pDrawingArea->GetWindowRect(&imageDrawingAreaRect);

	const int imageDrawingAreaWidth = imageDrawingAreaRect.right - imageDrawingAreaRect.left;
	const int imageDrawingAreaHeight = imageDrawingAreaRect.bottom - imageDrawingAreaRect.top;
	
	if (m_isStreaming)
	{
		if (m_isDrawingImage == false)
		{
			m_displayedFrameRate.SetFrameRate(0);
			pImagePixels = m_holdImage.GetData();
		}
		else
		{
			pImagePixels = pDoc->GetProcessedPixels();
			m_displayedFrameRate.NewFrame();
		}
	}
	else
	{
        // Draw the Pt Grey logo
		CDC dc;
		dc.CreateCompatibleDC(pImageDC);
		CBitmap* pOldBitmap = dc.SelectObject(&m_logo);
		CSize bmpSize = m_logo.GetBitmapDimension();
		const BOOL retVal = pImageDC->StretchBlt(
			0, 
			0, 
			imageDrawingAreaWidth,
			imageDrawingAreaHeight,
			&dc, 
			0, 
			0,
			bmpSize.cx, 
			bmpSize.cy,
			SRCCOPY);
        if (!retVal)
        {
            TRACE("Failed to draw Pt Grey logo\n");
        }
		dc.SelectObject(pOldBitmap);
		dc.DeleteDC();
		return;		
	}

    // Transfer the RGB buffer to graphics card.
    const int width = pDoc->m_bitmapInfo.bmiHeader.biWidth;
    const int height = ::abs( pDoc->m_bitmapInfo.bmiHeader.biHeight );

    CSingleLock dataLock( &pDoc->m_csData );
    dataLock.Lock();	
	
	int posX, posXMax, posXMin;
	int posY, posYMax, posYMin;

	CScrollBar* hScrollbar = (CScrollBar*)GetDlgItem(IDC_HSCROLLBAR);		
	CScrollBar* vScrollbar = (CScrollBar*)GetDlgItem(IDC_VSCROLLBAR);

    ASSERT(hScrollbar != NULL);
    ASSERT(vScrollbar != NULL);

	if (hScrollbar != NULL &&
		vScrollbar != NULL &&
		m_isStretchToFit == false &&
		m_isFullScreen == false	)
	{
		posX= hScrollbar->GetScrollPos();
		posY= vScrollbar->GetScrollPos();
		hScrollbar->GetScrollRange(&posXMin, &posXMax);
		vScrollbar->GetScrollRange(&posYMin, &posYMax);
	}
	else
	{
		posX = 0;
		posXMax = 0;
		posXMin = 0;
	    posY = 0;
		posYMax = 0;
		posYMin = 0;
	}

	const int imageWidth = pDoc->m_bitmapInfo.bmiHeader.biWidth - posXMax;
	const int imageHeight = abs(pDoc->m_bitmapInfo.bmiHeader.biHeight ) - posYMax;

    if( pImagePixels == NULL )
    {
        return;
    }
    
	if ( pDoc->IsOpenGLEnabled())
    {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        BindGL( );
        glEnable( GL_TEXTURE_2D );
        glMatrixMode( GL_MODELVIEW );
        glLoadIdentity();
		GLdouble scaleSizeX = (double)pDoc->m_bitmapInfo.bmiHeader.biWidth / (double)imageWidth;
		GLdouble scaleSizeY = (double)abs(pDoc->m_bitmapInfo.bmiHeader.biHeight) / (double)imageHeight;
		GLdouble translateX = (double)(0 - posX) / pDoc->m_bitmapInfo.bmiHeader.biWidth;
		GLdouble translateY = (double)(posYMax - posY) /pDoc->m_bitmapInfo.bmiHeader.biHeight;
		glScaled(scaleSizeX,scaleSizeY,0.0);
		glTranslated(translateX,translateY,0.0);
		DrawOGLImage(width, height, pImagePixels);
        SwapBuffers( pImageDC->m_hDC );
        UnbindGL( );
    }
    else
    {			
		// Non-OpenGL drawing
		// Draw a double buffered image to screen
		CDC dcMem; //memory buffer
		CBitmap bufferBitmap; //buffer image
		dcMem.CreateCompatibleDC(pImageDC);
		if (bufferBitmap.CreateCompatibleBitmap(
			pImageDC, 
			imageDrawingAreaWidth, 
			imageDrawingAreaHeight) == FALSE)
		{
			//create object fail
			return;
		}
		CBitmap* pOldBitmap = dcMem.SelectObject(&bufferBitmap);
		SetStretchBltMode(dcMem.GetSafeHdc(),COLORONCOLOR);

		// resize image size and draw image to window
		if( StretchDIBits(
			dcMem.GetSafeHdc(),
			0,
			0,
			imageDrawingAreaWidth,
			imageDrawingAreaHeight,
			posX,
			(posYMax - posY),
			imageWidth,
			imageHeight,
			pImagePixels, 
			&pDoc->m_bitmapInfo,
            DIB_RGB_COLORS,
			SRCCOPY	) == 0)
		{
			m_isDrawingImage = false;
		}

		if ( m_showCrosshair )
		{
			// Set drawing points
			CPoint crosshairCenter;
			crosshairCenter.SetPoint((imageDrawingAreaWidth)/2, (imageDrawingAreaHeight)/2);
			crosshairCenter.x += static_cast<LONG>((double)(-posX + posXMax/2)*m_zoomLevel);
			crosshairCenter.y += static_cast<LONG>((double)(-posY + posYMax/2)*m_zoomLevel);
			int crosshairLength =static_cast<int>(m_zoomLevel * ((double)width) * CROSSHAIR_LENGTH);
			
			// Set drawing pen
			CPen pen(PS_SOLID, sk_crosshairThickness , m_colorCrosshair);
			CPen* pOldPen = dcMem.SelectObject( &pen );

			// Draw the crosshair
			dcMem.MoveTo(crosshairCenter);
			dcMem.LineTo(crosshairCenter.x,crosshairCenter.y - crosshairLength);
			dcMem.MoveTo(crosshairCenter);
			dcMem.LineTo(crosshairCenter.x,crosshairCenter.y + crosshairLength);
			dcMem.MoveTo(crosshairCenter);
			dcMem.LineTo(crosshairCenter.x + crosshairLength,crosshairCenter.y);
			dcMem.MoveTo(crosshairCenter);
			dcMem.LineTo(crosshairCenter.x - crosshairLength,crosshairCenter.y );

			//restore pen configuration
			dcMem.SelectObject( pOldPen );
		}
	
		//paint buffer to image DC
		pImageDC->BitBlt(0,0,imageDrawingAreaWidth,imageDrawingAreaHeight,&dcMem,0,0,SRCCOPY);
		
		// Clean up memory
		dcMem.SelectObject(pOldBitmap);
		bufferBitmap.DeleteObject();
		ReleaseDC(&dcMem);
    }		

    dataLock.Unlock();	
	if (pDrawingArea->ReleaseDC(pImageDC) == 0)
    {
        TRACE("Unable to release image drawing area\n");
    }
}

void CFlyCap2_MFCView::DrawOGLImage(int width, int height, const unsigned char* pImagePixels)
{
    double validTextureWidth = 1.0;
    double validTextureHeight = 1.0;
    bool useTiledTextures = false;
    Image covertedImage = GetDocument()->GetConvertedImage();
    const float bytesPerPixel = covertedImage.GetBitsPerPixel() / 8.0f;
    const PixelFormat imagePixelFormat = covertedImage.GetPixelFormat();
    GLenum errorno;
    glBindTexture( GL_TEXTURE_2D, m_imageTextures[0] );

    if (m_PBOSupported)
    {
        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, m_PBO);
        errorno = glGetError();

        if (errorno != GL_NO_ERROR)
        {
            m_PBOSupported = false;
        }
        else
        {
            glBufferDataARB( 
                GL_PIXEL_UNPACK_BUFFER_ARB, 
                (unsigned int)(width * height * bytesPerPixel), 
                pImagePixels,
                GL_STREAM_DRAW_ARB);
            errorno = glGetError();

            if (errorno != GL_NO_ERROR)
            {
                m_PBOSupported = false;
            }
        }
    }

    switch (imagePixelFormat)
    {
    case PIXEL_FORMAT_MONO8: 
        glTexImage2D(
            GL_TEXTURE_2D, 
            0, 
            GL_LUMINANCE, 
            width, 
            height, 
            0, 
            GL_LUMINANCE, 
            GL_UNSIGNED_BYTE, 
            m_PBOSupported ? NULL : pImagePixels ); 
        break;    
    case PIXEL_FORMAT_MONO16: 
        glTexImage2D(
            GL_TEXTURE_2D, 
            0, 
            GL_LUMINANCE16, 
            width, 
            height, 
            0, 
            GL_LUMINANCE, 
            GL_UNSIGNED_SHORT, 
            m_PBOSupported ? NULL : pImagePixels ); 
        break;

    default: 
        glTexImage2D(
            GL_TEXTURE_2D, 
            0, 
            GL_RGB, 
            width, 
            height, 
            0, 
            GL_BGR_EXT, 
            GL_UNSIGNED_BYTE,
            m_PBOSupported ? NULL : pImagePixels );
        break;
    }
    errorno = glGetError();
    if ( errorno != GL_NO_ERROR)
    {
        // Attempt to fall back and use a power-of-two sized texture.
        // This is for older cards that don't support more arbitrary
        // texture sizes.

        const int textureWidth = GetMinimumPowerOfTwo(width);
        const int textureHeight = GetMinimumPowerOfTwo(height);
        validTextureWidth = (double)width / textureWidth;
        validTextureHeight = (double)height / textureHeight;

        glTexImage2D(
            GL_TEXTURE_2D, 
            0, 
            GL_RGB, 
            textureWidth, 
            textureHeight, 
            0, 
            GL_BGR_EXT, 
            GL_UNSIGNED_BYTE,
            NULL );
        errorno = glGetError();
        if ( errorno != GL_NO_ERROR)
        {
            // The graphics doesn't seem to support this texture size.
            // Images must be split and then tiled.
            useTiledTextures = true;
        }
        else
        {
            glTexSubImage2D( 
                GL_TEXTURE_2D, 
                0, 
                0, 
                0,
                width, 
                height,
                GL_BGR_EXT, 
                GL_UNSIGNED_BYTE, 
                pImagePixels );
            errorno = glGetError();
            if ( errorno != GL_NO_ERROR)
            {
                // Error
            }
        }
    }

    if ( useTiledTextures)
    {
        //
        // The image is split into multiple textures.
        //
        int bytesPerPixel = 3;
        int tileSize = 1024;
        int horizResidual = width % tileSize;
        int vertResidual = height % tileSize;
        int numHorizTextures = width / tileSize + ( horizResidual > 0);
        int numVertTextures = height / tileSize + ( vertResidual > 0);

        unsigned char *tileBuffer = new unsigned char [ tileSize * tileSize * bytesPerPixel];
        for ( int tileY = 0; tileY < numVertTextures ; tileY++)
        {
            for ( int tileX = 0; tileX < numHorizTextures; tileX++)
            {
                int subTexHeight = tileSize;
                if (  tileY == numVertTextures - 1 && vertResidual > 0)
                    subTexHeight = vertResidual;

                int subTexWidth = tileSize;
                if ( tileX == numHorizTextures - 1 && horizResidual > 0)
                    subTexWidth = horizResidual;

                // copy image buffer to the tile
                for ( int line = 0; line < subTexHeight; line++)
                {
                    memcpy( tileBuffer + line * tileSize * bytesPerPixel, 
                        pImagePixels + ( ( line + tileSize * tileY) * width + tileSize * tileX) * bytesPerPixel, 
                        subTexWidth * bytesPerPixel);
                }

                int texId = tileY * numHorizTextures + tileX;
                if ( texId >= sk_maxNumTextures)
                    continue;

                glBindTexture( GL_TEXTURE_2D, m_imageTextures[ texId] );
                glTexImage2D(
                    GL_TEXTURE_2D, 
                    0, 
                    GL_RGB, 
                    tileSize, 
                    tileSize, 
                    0, 
                    GL_BGR_EXT, 
                    GL_UNSIGNED_BYTE,
                    tileBuffer );

                double x_begin = (double)tileSize / width * tileX;
                double x_end = (double)tileSize / width * ( tileX + 1);
                double y_begin = 1.0 - (double)tileSize / height * ( tileY + 1);
                double y_end = 1.0 - (double)tileSize / height * tileY;

                glBegin( GL_QUADS );

                glTexCoord2d( 0.0, 1.0 );
                glVertex2d( x_begin, y_begin );

                glTexCoord2d( 1.0, 1.0 );
                glVertex2d( x_end, y_begin );

                glTexCoord2d( 1.0, 0.0 );
                glVertex2d( x_end, y_end );

                glTexCoord2d( 0.0, 0.0 );
                glVertex2d( x_begin, y_end );

                glEnd();
            }
        }

        delete [] tileBuffer;
    }
    else
    {
        // Just one texture
        glBegin( GL_QUADS );

        glTexCoord2d( 0.0, validTextureHeight );
        glVertex2d( 0.0, 0.0 );

        glTexCoord2d( validTextureWidth, validTextureHeight );
        glVertex2d( 1.0, 0.0 );

        glTexCoord2d( validTextureWidth, 0.0 );
        glVertex2d( 1.0, 1.0 );

        glTexCoord2d( 0.0, 0.0 );
        glVertex2d( 0.0, 1.0 );

        glEnd();
    }

    if ( m_showCrosshair )
    {

        const double aspectRatio = ((double)width/(double)height);
        glTranslated(0.5, 0.5, 0.0);
        glScaled(1.0, aspectRatio, 0.0);       
        glDisable(GL_TEXTURE_2D);

        glColor3d(
            (float)GetRValue(m_colorCrosshair)/256.0f,
            (float)GetGValue(m_colorCrosshair)/256.0f,
            (float)GetBValue(m_colorCrosshair)/256.0f);
        const GLfloat length = static_cast<GLfloat>(CROSSHAIR_LENGTH);
        const float center = 0.0f;

        glLineWidth((GLfloat)sk_crosshairThickness );
        glBegin( GL_LINES );

        glVertex2f( center - length, center );
        glVertex2f( center + length, center );

        glVertex2f( center, center - length );
        glVertex2f( center, center + length );

        glEnd();
    }
}

double CFlyCap2_MFCView::GetDisplayedFrameRate()
{
    return m_displayedFrameRate.GetFrameRate();
}

RECT CFlyCap2_MFCView::GetDisplaySize()
{
	RECT rect;
	GetWindowRect(&rect);
	if (m_enableCameraInformationPanel == true)
	{
		rect.left += sk_camInfoWidth;
	}
	return rect;
}

int CFlyCap2_MFCView::GetXOffset()
{
	if (m_enableCameraInformationPanel == true && 
        m_isFullScreen == false)
	{
        return sk_camInfoWidth;
	}
	else
	{
		return 0;
	}
}

double CFlyCap2_MFCView::GetZoomLevel()
{
	return m_zoomLevel;
}

void CFlyCap2_MFCView::GetPixelPositionFromImage(int* pX, int* pY)
{
	CStatic* pDrawingArea = (CStatic*)GetDlgItem(IDC_IMAGEAREA);
	CScrollBar* hScrollbar = (CScrollBar*) GetDlgItem(IDC_HSCROLLBAR);
	CScrollBar* vScrollbar = (CScrollBar*) GetDlgItem(IDC_VSCROLLBAR);	

    ASSERT(pDrawingArea != NULL);
    ASSERT(hScrollbar != NULL);
    ASSERT(vScrollbar != NULL);

	if (pDrawingArea == NULL || hScrollbar == NULL || vScrollbar == NULL)
	{
		//the view has not been initialized yet...
		*pX = -1;
		*pY = -1;
		return;
	}

	CPoint result;
	if (GetCursorPos(&result) == FALSE)
    {
        TRACE("GetCursorPos() failed\n");
		*pX = -1;
		*pY = -1;
		return;
    }

	CRect imageDrawingAreaRect;
	pDrawingArea->GetClientRect(&imageDrawingAreaRect);
	pDrawingArea->ScreenToClient(&result);

	if (result.x >= imageDrawingAreaRect.left && 
		result.x < imageDrawingAreaRect.right &&
		result.y >= imageDrawingAreaRect.top && 
		result.y < imageDrawingAreaRect.bottom )
	{	
		const int xOffset = hScrollbar->GetScrollPos();
		const int yOffset = vScrollbar->GetScrollPos();

		int xMax,xMin,yMax,yMin;
		hScrollbar->GetScrollRange(&xMin,&xMax);
		vScrollbar->GetScrollRange(&yMin,&yMax);

		unsigned int imageWidth, imageHeight;
		GetDocument()->GetImageSize(&imageWidth,&imageHeight);  //get the image size

		const double actualWidth = static_cast<double>(imageWidth) - (xMax - xMin); // the actual image width which shown on the screen
		const double actualHeight = static_cast<double>(imageHeight) - (yMax - yMin); // the actual image width which shown on the screen
		const double xScaleRatio = actualWidth /(double)imageDrawingAreaRect.Width();
		const double yScaleRatio = actualHeight/(double)imageDrawingAreaRect.Height();
		*pX =(int)(((double)result.x * xScaleRatio ) )+ xOffset;
		*pY =(int)(((double)result.y * yScaleRatio) )+ yOffset;
	}
	else
	{
		*pX = -1;
		*pY = -1;
	}
}

int CFlyCap2_MFCView::OnCreate( LPCREATESTRUCT lpCreateStruct )
{
    if (CFormView::OnCreate(lpCreateStruct) == -1)
    {
        return -1;
    }

    return 0;
}

void CFlyCap2_MFCView::OnDestroy()
{
    CFormView::OnDestroy();

    if (m_openGLInitialized)
    {
        // Make the RC non-current
        UnbindGL( );

        // Delete the rendering context
        if ( ::wglDeleteContext( m_hRC ) == FALSE )
        {
            MessageBox("Could not Make RC non-Current.");
        }

        // Delete DC
        if ( m_pDC )
        {
            delete m_pDC;
            m_pDC = NULL;
        }        
    }

	DestroyCursor( m_hHand );
	DestroyCursor( m_hArrow );
	m_logo.DeleteObject();
	m_logo.Detach();
}

void CFlyCap2_MFCView::GetMinSize(unsigned int* width, unsigned int* height)
{
	GetDocument()->GetImageSize(width, height);	
	*width = *width + GetXOffset();
}

void CFlyCap2_MFCView::OnSize( UINT nType, int cx, int cy )
{
    CFormView::OnSize(nType, cx, cy);
	AdjustViewSize(cx,cy);
}

void CFlyCap2_MFCView::AdjustViewSize(int cx,int cy)
{
	//adjust height of camera information panel
	CTreeCtrl* pCamInfoTreeView  = (CTreeCtrl *)GetDlgItem(IDC_INFOTREE); 
	if (pCamInfoTreeView != NULL)
	{
		pCamInfoTreeView->MoveWindow(0,0,GetXOffset(),cy,FALSE);
		pCamInfoTreeView->InvalidateRect(NULL,FALSE);
		pCamInfoTreeView->GetUpdateRect(NULL);
	}

	AdjustDrawingArea();

	m_currentSizeX = cx;
	m_currentSizeY = cy;
}

void CFlyCap2_MFCView::AdjustDrawingArea()
{
	CStatic* pDrawingArea = (CStatic*)GetDlgItem(IDC_IMAGEAREA);
	CFlyCap2_MFCDoc* pDoc = GetDocument();	
	if (pDrawingArea == NULL || pDoc == NULL || m_viewInitialized == false)
	{
		return;
	}

	const int xOffset = GetXOffset();
    unsigned int width = 0;
    unsigned int height = 0;

	//the picture size of logo and grabbing image are different, 
	//we need to figure out which size we are using
	if (m_isStreaming == true)
	{
		pDoc->GetImageSize(&width, &height);
	}
	else
	{
		CSize logoSize = m_logo.GetBitmapDimension();
		width = logoSize.cx;
		height = logoSize.cy;
	}

    CRect viewRect;
	GetWindowRect(&viewRect);//get view rectangle (including camera information panel)

    int displayWidth = 0;
    int displayHeight = 0;
    int x = 0;
    int y = 0;

	if (m_isStretchToFit == false && m_isFullScreen == false && m_isStreaming)
	{
		//Turn off resize to fit mode(size of drawing view is depends on image size)
		displayWidth = static_cast<int>(width * m_zoomLevel);
		displayHeight = static_cast<int>(height* m_zoomLevel);
		x = ((abs(viewRect.left - viewRect.right) - displayWidth - xOffset - sk_scrollbarThickness)/2 ) + xOffset ;
		y = ((abs(viewRect.bottom - viewRect.top) - displayHeight - sk_scrollbarThickness)/2 );
		
		//check to see if x and y is out of boundary
        if (x < xOffset)
		{
			x = xOffset;
		}

		if (y < 0)
		{
			y = 0;
		}
		
		const int maxWidth = abs(viewRect.Width()) - xOffset - sk_scrollbarThickness;
		const int maxHeight = abs(viewRect.Height()) - sk_scrollbarThickness;
		CScrollBar* hScrollbar = (CScrollBar*)GetDlgItem(IDC_HSCROLLBAR);
		if (hScrollbar !=NULL)
		{
			//Adjust horizontal scroll bar
			if (displayWidth > maxWidth)
			{					
				const int hScrollbarMax = static_cast<int>((double)(displayWidth - maxWidth ) / m_zoomLevel);
				const int hScrollbarMin = 0;
				int oldScrollbarMax = 0;
				int oldScrollbarMin = 0;
				hScrollbar->GetScrollRange(&oldScrollbarMin, &oldScrollbarMax);
				hScrollbar->SetScrollRange(hScrollbarMin, hScrollbarMax, FALSE);
				int scrollbarAdjustment = ((hScrollbarMax - hScrollbarMin) - (oldScrollbarMax - oldScrollbarMin)) / 2;
				hScrollbar->SetScrollPos(hScrollbar->GetScrollPos() + scrollbarAdjustment,TRUE);
				displayWidth = maxWidth;

				hScrollbar->SetWindowPos(
					NULL,
					xOffset,
					maxHeight, //keep it in the bottom of view
					displayWidth,
					sk_scrollbarThickness - 3,//3 pixel is for 3D board frame style
					SWP_SHOWWINDOW);
			}
			else
			{
				//There is enough space to display, so hide this scroll bar
				hScrollbar->SetWindowPos(NULL,0,0,0,0,SWP_HIDEWINDOW);
				hScrollbar->SetScrollPos(0,FALSE);
				hScrollbar->SetScrollRange(0,0,FALSE);

			}
		}
		
		CScrollBar* vScrollbar = (CScrollBar*)GetDlgItem(IDC_VSCROLLBAR);
		if (vScrollbar !=NULL)
		{
			//Adjust vertical scroll bar
			if (displayHeight > maxHeight)
			{					
				const int vScrollbarMax = static_cast<int>((double)(displayHeight - maxHeight)/ m_zoomLevel);
				const int vScrollbarMin = 0;
				int oldScrollbarMax = 0;
				int oldScrollbarMin = 0;
				vScrollbar->GetScrollRange(&oldScrollbarMin, &oldScrollbarMax);
				vScrollbar->SetScrollRange(vScrollbarMin, vScrollbarMax, FALSE);
				int scrollbarAdjustment = ((vScrollbarMax - vScrollbarMin) - (oldScrollbarMax - oldScrollbarMin)) / 2;
				vScrollbar->SetScrollPos(vScrollbar->GetScrollPos() + scrollbarAdjustment, TRUE);
				displayHeight = maxHeight;

				vScrollbar->SetWindowPos(
					NULL,
					xOffset + maxWidth,  //keep it in the right side of view
					0, 
					sk_scrollbarThickness - 3,//3 pixel is for 3D board frame style
					displayHeight,
					SWP_SHOWWINDOW);
			}
			else
			{
				//There is enough space to display, so hide this scroll bar
				vScrollbar->SetWindowPos(NULL,0,0,0,0,SWP_HIDEWINDOW);
				vScrollbar->SetScrollPos(0,FALSE);
				vScrollbar->SetScrollRange(0,0,FALSE);
			}
		}
	}
	else
	{
		//Turn on resize to fit mode(size of drawing view is depends on window size)
		const int maxWidth = abs(viewRect.Width()) - xOffset;
		const int maxHeight = abs(viewRect.Height());

		const double scaleX = (double)maxWidth / (double)width; //get max horizontal scale
		const double scaleY = (double)maxHeight / (double)height; //get max vertical scale

        if (!m_isStretchToFit)
        {
            // Figure out what zoom level to use
            const float testLevel = (scaleX < scaleY ? scaleX : scaleY) * 100.0f;

            // Round down to the nearest value in the zoom level vector
            for (unsigned int i=0; i < m_zoomLevelTable.size() - 1; i++)
            {
                const float currVal = m_zoomLevelTable[i];
                const float nextVal = m_zoomLevelTable[i+1];
                if (currVal <= testLevel && 
                    testLevel < nextVal )
                {
                    // Use this value as the zoom level - effectively rounding down
                    if(pDoc->IsGrabThreadRunning())
					{
						m_zoomLevel = 100 / 100.0f; //Fix for Bug 16625
					}
					else
					{
						m_zoomLevel = m_zoomLevelTable[i] / 100.0f;
					}
                }
            }
        }
        else
        {
            m_zoomLevel = scaleX < scaleY ? scaleX : scaleY;
        }        

        displayWidth = static_cast<int>(width * m_zoomLevel);
        displayHeight = static_cast<int>(height* m_zoomLevel);

		if (scaleX < scaleY)
		{		
			y = ((abs(viewRect.Height()) - displayHeight)/2 );
			x = xOffset;
		}
		else
		{
			x = ((abs(viewRect.Width()) - displayWidth - xOffset)/2 ) + xOffset;
			y = 0;
		}


		if (m_viewInitialized == true)
		{
			CScrollBar* vScrollbar = (CScrollBar*)GetDlgItem(IDC_VSCROLLBAR);
			CScrollBar* hScrollbar = (CScrollBar*)GetDlgItem(IDC_HSCROLLBAR);
			vScrollbar->SetWindowPos(NULL, 0, 0, 0, 0, SWP_HIDEWINDOW);
			hScrollbar->SetWindowPos(NULL, 0, 0, 0, 0, SWP_HIDEWINDOW);
		}
	}

	//adjust size of image area
	pDrawingArea->MoveWindow(x, y, displayWidth, displayHeight, TRUE);
	if (pDoc->IsOpenGLEnabled())
	{
		BindGL();
		::glViewport(0, 0, displayWidth, displayHeight);
		UnbindGL();
	}
}


BOOL CFlyCap2_MFCView::OnEraseBkgnd( CDC* pDC )
{
	CFormView::OnEraseBkgnd(pDC);

    return TRUE;
}

bool CFlyCap2_MFCView::InitializeOpenGL()
{
    if ( m_openGLInitialized )
    {
        // Nothing to be done here
        return true;
    }

	CStatic* pDrawingArea = (CStatic*)GetDlgItem(IDC_IMAGEAREA);
	if (pDrawingArea == NULL)
	{
		MessageBox("Error Obtaining Image drawing area control.");
        return false;
	}
	//
    // Get a DC for the Client Area
    m_pDC = new CClientDC( pDrawingArea);
    if ( m_pDC == NULL )
    {
        MessageBox("Error Obtaining DC");
        return false;
    }
	

    // Set Pixel Format
    if ( !SetupPixelFormat() )
    {
        return false;
    }

    // Create Rendering Context
    m_hRC = ::wglCreateContext( m_pDC->GetSafeHdc() );
    if( m_hRC == NULL )
    {
        MessageBox("Error Creating RC.");;
        return false;
    }
	
	glClearColor(0.0f, 0.0f, 0.0f, 0.5f);
	glClearDepth(1.0f);							// Depth Buffer Setup
	glEnable(GL_DEPTH_TEST);						// Enables Depth Testing
	glDepthFunc(GL_LEQUAL);							// The Type Of Depth Testing To Do
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    // initialize OGL texture
    BindGL();

    // check PBO is supported
    const char *extensions = (const char*)glGetString(GL_EXTENSIONS);
    if ( strstr( extensions, "GL_ARB_pixel_buffer_object") != NULL)
    {
        // get pointers to GL functions
        glGenBuffersARB = (PFNGLGENBUFFERSARBPROC)wglGetProcAddress("glGenBuffersARB");
        glBindBufferARB = (PFNGLBINDBUFFERARBPROC)wglGetProcAddress("glBindBufferARB");
        glBufferDataARB = (PFNGLBUFFERDATAARBPROC)wglGetProcAddress("glBufferDataARB");
        glDeleteBuffersARB = (PFNGLDELETEBUFFERSARBPROC)wglGetProcAddress("glDeleteBuffersARB");

        if ( glGenBuffersARB == 0 || glBindBufferARB == 0 || glBufferDataARB == 0 || glDeleteBuffersARB == 0 )
        {
            // failed to get function pointer
        }
        else
        {
            m_PBOSupported = true;
            glGenBuffersARB( 1, &m_PBO);
        }
    }

    glEnable( GL_BLEND );
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );

	InitializeImageTexture();

    glShadeModel( GL_FLAT );

    // initialize matrices
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    gluOrtho2D( 0, 1, 0, 1 );

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

    UnbindGL();

    return true; 
}

void CFlyCap2_MFCView::InitializeImageTexture()
{
    glGenTextures( sk_maxNumTextures, m_imageTextures );

    bool useClampToEdge = true;
    if ( atof((const char*)glGetString(GL_VERSION)) < 1.15)
    {
        useClampToEdge = false;
    }

    for ( unsigned int i = 0; i < sk_maxNumTextures; i++)
    {
        glBindTexture( GL_TEXTURE_2D, m_imageTextures[ i] );

        if ( useClampToEdge)
        {
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
        }
        else
        {
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );
        }
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );

        glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
    }
}

bool CFlyCap2_MFCView::SetupPixelFormat()
{
    // Fill in the Pixel Format Descriptor
    PIXELFORMATDESCRIPTOR pfd;
    memset( &pfd, 0x0, sizeof( PIXELFORMATDESCRIPTOR ) );

    pfd.nSize = sizeof( PIXELFORMATDESCRIPTOR );
    pfd.nVersion = 1;
    pfd.dwFlags =	
        PFD_DOUBLEBUFFER |
        PFD_SUPPORT_OPENGL |
        PFD_DRAW_TO_WINDOW;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 24;
    pfd.cAlphaBits = 0;
    pfd.cDepthBits = 0;

    int nPixelFormat = ::ChoosePixelFormat( m_pDC->m_hDC, &pfd );
    if( nPixelFormat == 0 )
    {
        ASSERT( FALSE );
        return false;
    }

    if( !::SetPixelFormat( m_pDC->m_hDC, nPixelFormat, &pfd ) )
    {
        ASSERT( FALSE );
        return false;
    }

    return true;
}

void CFlyCap2_MFCView::BindGL()
{
    if( !::wglMakeCurrent( m_pDC->m_hDC, m_hRC ) )
    {
        MessageBox("Error binding OpenGL.");
    }
}

void CFlyCap2_MFCView::UnbindGL()
{
    if( !::wglMakeCurrent( m_pDC->m_hDC, NULL ) )
    {
        MessageBox("Error unbinding OpenGL.");
    }
}
void CFlyCap2_MFCView::OnViewShowInformationPanel()
{
	m_enableCameraInformationPanel = (!m_enableCameraInformationPanel);
	UpdateCameraInfoPanel();
}

void CFlyCap2_MFCView::OnUpdateViewShowInformationPanel(CCmdUI *pCmdUI)
{
	if (m_isFullScreen == true)
	{
		pCmdUI->SetCheck(FALSE);
		pCmdUI->Enable(FALSE);
	}
	else
	{
		pCmdUI->Enable(TRUE);
		pCmdUI->SetCheck(m_enableCameraInformationPanel ? TRUE:FALSE); 
	}
}


void CFlyCap2_MFCView::UpdateCameraInfoPanel()
{
    CTreeCtrl* pCamInfoTreeView  = (CTreeCtrl *)GetDlgItem(IDC_INFOTREE); 
    if (pCamInfoTreeView != NULL)
    {
        if (m_enableCameraInformationPanel == true && m_isFullScreen == false)
        {
            pCamInfoTreeView->MoveWindow(0,0,sk_camInfoWidth,m_currentSizeY,TRUE);
        }
        else
        {
            pCamInfoTreeView->MoveWindow(0,0,0,0,TRUE);
        }

        AdjustDrawingArea();
    }
}

void CFlyCap2_MFCView::SetToFullScreenMode()
{
	m_isFullScreen = true;
	CTreeCtrl* pCamInfoTreeView  = (CTreeCtrl *)GetDlgItem(IDC_INFOTREE); 
	CMenu *pMenu = AfxGetMainWnd()->GetMenu();
	RECT clientRect;
    GetClientRect( &clientRect );
	if (pCamInfoTreeView != NULL)
	{
		pMenu->CheckMenuItem(ID_VIEW_SHOWINFORMATIONPANEL,MF_UNCHECKED| MF_BYCOMMAND);
		if (GetDocument()->IsOpenGLEnabled())
		{
			BindGL( );
			::glViewport( 0, 0, clientRect.right, clientRect.bottom );
			UnbindGL( );
		}

		pCamInfoTreeView->MoveWindow(0,0,0,0,TRUE);
	}

	AdjustDrawingArea();
}

void CFlyCap2_MFCView::SetToWindowedMode()
{
	m_isFullScreen = false;
	UpdateCameraInfoPanel();
	CMenu *pMenu = AfxGetMainWnd()->GetMenu();
	if (m_enableCameraInformationPanel == true)
	{
		pMenu->CheckMenuItem(ID_VIEW_SHOWINFORMATIONPANEL,MF_CHECKED| MF_BYCOMMAND);
		CTreeCtrl* pCamInfoTreeView  = (CTreeCtrl *)GetDlgItem(IDC_INFOTREE); 
        pCamInfoTreeView->InvalidateRect(NULL,TRUE);
        pCamInfoTreeView->GetUpdateRect(NULL,FALSE);
    }	
}

void CFlyCap2_MFCView::UpdateCameraInfoData()
{
    if (m_camInfoDataInitialized == false)
    {
        //Bad!
        return;
    }

    CFlyCap2_MFCDoc* pDoc = GetDocument();
    CTreeCtrl* pCamInfoTreeView  = (CTreeCtrl *)GetDlgItem(IDC_INFOTREE); 
    if (m_isStreaming == true)
    {
        // Update image information
        static unsigned int s_prevWidth, s_prevHeight = 0;
        InformationPane::InformationPaneStruct infoStruct = pDoc->GetRawImageInformation();
        infoStruct.fps.displayedFrameRate = GetDisplayedFrameRate();
		//pCamInfoTreeView->SetRedraw(FALSE);
        m_infoPane.UpdateInformationPane(pCamInfoTreeView, infoStruct );
		//pCamInfoTreeView->SetRedraw(TRUE);
        // Check to see if the image size has changed
        if (infoStruct.imageInfo.height!= s_prevHeight || 
            infoStruct.imageInfo.width != s_prevWidth)
        {
            AdjustDrawingArea();
            s_prevWidth = infoStruct.imageInfo.width;
            s_prevHeight = infoStruct.imageInfo.height;
        }
    }	
}


void CFlyCap2_MFCView::OnUpdateViewDrawImage(CCmdUI *pCmdUI)
{
	pCmdUI->SetCheck(m_isDrawingImage ? TRUE:FALSE); 
}

void CFlyCap2_MFCView::OnViewDrawImage()
{
	m_isDrawingImage = !m_isDrawingImage;
	if (m_isDrawingImage == false)
	{
		//hold the last image
		CFlyCap2_MFCDoc* pDoc = GetDocument();
		m_holdImage = pDoc->GetConvertedImage();
	}
}

void CFlyCap2_MFCView::OnViewChangeCrosshairColor()
{
	CColorDialog dlg;
	if (dlg.DoModal() == IDOK) 
	{
		m_colorCrosshair = dlg.GetColor();
	}	
}

void CFlyCap2_MFCView::OnDrawCrosshairClick()
{
	m_showCrosshair = (!m_showCrosshair);
}

void CFlyCap2_MFCView::OnUpdateViewDrawCrosshair(CCmdUI *pCmdUI)
{
	pCmdUI->SetCheck( m_showCrosshair ? TRUE:FALSE); 
}

void CFlyCap2_MFCView::OnUpdateViewFullscreen(CCmdUI *pCmdUI)
{
	pCmdUI->SetCheck( m_isFullScreen ? TRUE:FALSE); 
}

void CFlyCap2_MFCView::OnUpdateViewStretchToFit(CCmdUI *pCmdUI)
{
	if (m_isFullScreen == false)
	{
		pCmdUI->SetCheck( m_isStretchToFit ? TRUE:FALSE); 
	}
	else
	{
		pCmdUI->SetCheck(FALSE);
		pCmdUI->Enable(FALSE);
	}
}

void CFlyCap2_MFCView::OnViewStretchToFit()
{
	if (m_isStretchToFit == true)
	{
		m_zoomLevel = m_prevZoomLevel;
		m_isStretchToFit = false;
	}
	else
	{
		m_prevZoomLevel = m_zoomLevel;
		m_isStretchToFit = true;
	}

	AdjustDrawingArea();
}

void CFlyCap2_MFCView::OnViewEnableOpenGL()
{
	if (m_isDrawingImage == false)
	{
		MessageBox("Please enable \"View -> Draw Image\" before selecting this option.", "Unable to change the drawing method", MB_OK);
		return;
	}

	CFlyCap2_MFCDoc* pDoc = GetDocument();
	pDoc->EnableOpenGL(!pDoc->IsOpenGLEnabled());
	AdjustDrawingArea();
}

void CFlyCap2_MFCView::OnZoomIn()
{
	if (m_isFullScreen || m_isStretchToFit)
	{
		return;
	}
    
    if ((m_zoomLevelIndex + 1) == m_zoomLevelTable.size())
    {
        return;
    }

    m_zoomLevelIndex++;
    m_zoomLevel = m_zoomLevelTable[m_zoomLevelIndex] / 100.0f;

	AdjustDrawingArea();
}

void CFlyCap2_MFCView::OnZoomOut()
{
	if (m_isFullScreen || m_isStretchToFit)
	{
		return;
	}

    if (m_zoomLevelIndex == 0)
    {
        return;
    }

    m_zoomLevelIndex--;
    m_zoomLevel = m_zoomLevelTable[m_zoomLevelIndex] / 100.0f;

	AdjustDrawingArea();
}

void CFlyCap2_MFCView::OnUpdateZoomIn(CCmdUI *pCmdUI)
{
	if (m_viewInitialized == false ||
		m_zoomLevel >= MAX_ZOOM_LEVEL || 
		m_isFullScreen || 
        m_isStretchToFit)
	{
		pCmdUI->Enable(FALSE);
	}
	else
	{
		pCmdUI->Enable(TRUE);
	}
}

void CFlyCap2_MFCView::OnUpdateZoomOut(CCmdUI *pCmdUI)
{

	if (m_viewInitialized == false ||
		m_zoomLevel <= MIN_ZOOM_LEVEL || 
		m_isFullScreen || 
        m_isStretchToFit)
	{
		pCmdUI->Enable(FALSE);
	}
	else
	{
		pCmdUI->Enable(TRUE);
	}
}


void CFlyCap2_MFCView::OnUpdateOriginalZoom(CCmdUI *pCmdUI)
{

	if (m_viewInitialized == false ||
		m_isFullScreen || 
        m_isStretchToFit)
	{
		pCmdUI->Enable(FALSE);
	}
	else
	{
		pCmdUI->Enable(TRUE);
	}
}


void CFlyCap2_MFCView::OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar)
{
	//ChangeScrollbarPositionValue(nSBCode, nPos, pScrollBar);
	//CFormView::OnHScroll(nSBCode, nPos, pScrollBar);
	ChangeScrollbarPositionValue(nSBCode, nPos, (CScrollBar*)GetDlgItem(IDC_HSCROLLBAR));
	CFormView::OnHScroll(nSBCode, nPos, (CScrollBar*)GetDlgItem(IDC_HSCROLLBAR));
}

void CFlyCap2_MFCView::OnVScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar)
{
	//ChangeScrollbarPositionValue(nSBCode, nPos, pScrollBar);
	//CFormView::OnVScroll(nSBCode, nPos, pScrollBar);
	ChangeScrollbarPositionValue(nSBCode, nPos, (CScrollBar*)GetDlgItem(IDC_VSCROLLBAR));
	CFormView::OnVScroll(nSBCode, nPos, (CScrollBar*)GetDlgItem(IDC_VSCROLLBAR));
}

void CFlyCap2_MFCView::ChangeScrollbarPositionValue(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar)
{
	int currentPosition = pScrollBar->GetScrollPos();
	int maxVal, minVal;
	pScrollBar->GetScrollRange(&minVal,&maxVal);

	// Determine the new position of scroll box.
	switch (nSBCode)
	{
	case SB_LEFT:      // Scroll to far left.
		currentPosition = minVal;
		break;

	case SB_RIGHT:      // Scroll to far right.
		currentPosition = maxVal;
		break;

	case SB_ENDSCROLL:   // End scroll.
		break;

	case SB_LINELEFT:      // Scroll left.
		if (currentPosition > minVal)
			currentPosition--;
		break;

	case SB_LINERIGHT:   // Scroll right.
		if (currentPosition < maxVal)
			currentPosition++;
		break;

	case SB_PAGELEFT:    // Scroll one page left.
		{
			// Get the page size. 
			if (currentPosition > minVal)
			{
				currentPosition = max(minVal, currentPosition - sk_scrollbarPageChangeValue);
			}
		}
		break;

	case SB_PAGERIGHT:      // Scroll one page right
		{
			// Get the page size. 
			/*SCROLLINFO   info;
			pScrollBar->GetScrollInfo(&info, SIF_ALL);
			*/
			if (currentPosition < maxVal)
			{
				currentPosition = min(maxVal, currentPosition + sk_scrollbarPageChangeValue);
			}
		}
		break;

	case SB_THUMBPOSITION: // Scroll to absolute position. nPos is the position
		currentPosition = nPos;      // of the scroll box at the end of the drag operation.
		break;

	case SB_THUMBTRACK:   // Drag scroll box to specified position. nPos is the
		currentPosition = nPos;     // position that the scroll box has been dragged to.
		break;
	}
	pScrollBar->SetScrollPos(currentPosition);
}

//Mouse activities
void CFlyCap2_MFCView::OnLButtonDown(UINT nFlags, CPoint point)
{
	SetCapture();
	if (GetDocument()->IsGrabThreadRunning() == TRUE)
	{
		SetCursor( m_hHand );
	}
	CFormView::OnLButtonDown(nFlags, point);
}

void CFlyCap2_MFCView::OnLButtonUp(UINT nFlags, CPoint point)
{
	ReleaseCapture();
	SetCursor( m_hArrow );
	CFormView::OnLButtonUp(nFlags, point);
}

void CFlyCap2_MFCView::OnMouseMove(UINT nFlags, CPoint point)
{
	static const double sk_imageMoveSpeed = 0.5;
	static CPoint prevClick = point;

	if( (MK_LBUTTON & nFlags)	&& 
		m_isStretchToFit == false &&
		m_isFullScreen == false	&&
		GetDocument()->IsGrabThreadRunning() == TRUE)
	{
		CScrollBar* hScrollbar = (CScrollBar*) GetDlgItem(IDC_HSCROLLBAR);	
		if (hScrollbar != NULL)
		{
			int posX = hScrollbar->GetScrollPos();

			int posXMax, posXMin;
			hScrollbar->GetScrollRange(&posXMin, &posXMax);

			const int addX = (int) ((double)(prevClick.x - point.x) / (m_zoomLevel * sk_imageMoveSpeed));
			if (addX != 0)
			{
				posX +=addX;
			}
			else
			{
				posX += (int) (((prevClick.x - point.x)/2) * sk_imageMoveSpeed);
			}
			if (posX > posXMax)
			{
				posX = posXMax;
			}
			else if (posX < posXMin)
			{
				posX = posXMin;
			}

			hScrollbar->SetScrollPos(posX);
		}
		CScrollBar* vScrollbar = (CScrollBar*) GetDlgItem(IDC_VSCROLLBAR);
		if (hScrollbar != NULL)
		{
			int posY = vScrollbar->GetScrollPos();
			int posYMax, posYMin;
			vScrollbar->GetScrollRange(&posYMin, &posYMax);
			int addY = (int) ((double)(prevClick.y - point.y) / (m_zoomLevel* sk_imageMoveSpeed));
			if (addY != 0)
			{
				posY += addY;
			}
			else
			{
				posY += (int)(((prevClick.y - point.y) / 2) * sk_imageMoveSpeed);
			}
			if (posY > posYMax)
			{
				posY = posYMax;
			}
			else if (posY < posYMin)
			{
				posY = posYMin;
			}
			vScrollbar->SetScrollPos(posY);
		}
	}
	prevClick = point;
}

BOOL CFlyCap2_MFCView::OnMouseWheel(UINT nFlags, short zDelta, CPoint pt)
{
	if (m_isStreaming == false)
	{
		return TRUE;
	}
	if (zDelta >= WHEEL_DELTA)
	{
		OnZoomIn();
	}
	else
	{
        OnZoomOut();		
	}

	return CFormView::OnMouseWheel(nFlags, zDelta, pt);
}

void CFlyCap2_MFCView::OnFileNewCamera()
{
	GetParentFrame()->ShowWindow(SW_HIDE);
	if (GetDocument()->OnNewDocument() == FALSE)
	{
		//if user click cancel or fail to create document, then exit application
		AfxGetMainWnd()->PostMessage(WM_CLOSE);
	}
	else
	{
		ResetViewConfig();
		GetParentFrame()->ShowWindow(SW_SHOW);
	}
}

void CFlyCap2_MFCView::OnBtnOriginalImageSize()
{
	m_zoomLevel = 1.00;
	AdjustDrawingArea();
}
