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
#ifndef PGR_FC2_INFORMATIONPANE_H
#define PGR_FC2_INFORMATIONPANE_H

#include <string>

class InformationPane
{
public:
    struct FPSStruct
    {
        double processedFrameRate;
        double displayedFrameRate;
        double requestedFrameRate;

        FPSStruct()
        {
            processedFrameRate = 0.0;
            displayedFrameRate = 0.0;
            requestedFrameRate = 0.0;
        }
    };

    struct ImageInfoStruct
    {
        unsigned int width;
        unsigned int height;
        unsigned int stride;
		FlyCapture2::PixelFormat pixFmt;

        ImageInfoStruct()
        {
            width = 0;
            height = 0;
            stride = 0;
			pixFmt = FlyCapture2::NUM_PIXEL_FORMATS;
        }        
    };

    struct EmbeddedImageInfoStruct
    {
        union
        {            
            unsigned int arEmbeddedInfo[10];

            struct
            {
                unsigned int timestamp;
                unsigned int gain;
                unsigned int shutter;
                unsigned int brightness;
                unsigned int exposure;
                unsigned int whiteBalance;
                unsigned int frameCounter;
                unsigned int strobePattern;
                unsigned int GPIOPinState;
                unsigned int ROIPosition;
            } Individual;
        };

        EmbeddedImageInfoStruct()
        {
            memset(arEmbeddedInfo, 0x0, 10);
        }
    };

    struct DiagnosticsStruct
    {
        int skippedFrames;
        int linkRecoveryCount;
        int transmitFailures;
        std::string timeSinceInitialization;
        std::string timeSinceLastBusReset;

        DiagnosticsStruct()
        {
            skippedFrames = -1;
            linkRecoveryCount = -1;
            transmitFailures = -1;
            timeSinceInitialization = "";
            timeSinceLastBusReset = "";
        }
    };

    struct InformationPaneStruct
    {
        FPSStruct fps; 
		FlyCapture2::TimeStamp timestamp;
        ImageInfoStruct imageInfo;
        EmbeddedImageInfoStruct embeddedInfo;
        DiagnosticsStruct diagnostics;
    };

    InformationPane();
    virtual ~InformationPane(void);
	bool Initialize(CTreeCtrl* pCamInfoTreeView);
    void UpdateInformationPane(CTreeCtrl* pCamInfoTreeView, InformationPaneStruct infoStruct );    

protected:

private:   
    static std::string GetPixelFormatStr( FlyCapture2::PixelFormat pixFmt); 
    CFont m_monospaceFont;

    // FPS    
    HTREEITEM m_pLblDisplayedFPS;
    HTREEITEM m_pLblProcessedFPS;
    HTREEITEM m_pLblRequestedFPS;

    // Timestamp   
    HTREEITEM m_pLblTimestampSeconds;
    HTREEITEM m_pLblTimestampMicroseconds;
    HTREEITEM m_pLbl1394CycleTimeSeconds;
    HTREEITEM m_pLbl1394CycleTimeCount;
    HTREEITEM m_pLbl1394CycleTimeOffset;

    // Image info
    HTREEITEM m_pLblImageWidth;
    HTREEITEM m_pLblImageHeight;
    HTREEITEM m_pLblImagePixFmt;
    HTREEITEM m_pLblImageBitsPerPixel;

    // Embedded image info
    HTREEITEM m_pLblEmbeddedGain;
    HTREEITEM m_pLblEmbeddedShutter;
    HTREEITEM m_pLblEmbeddedBrightness;
    HTREEITEM m_pLblEmbeddedExposure;
    HTREEITEM m_pLblEmbeddedWhiteBalance;
    HTREEITEM m_pLblEmbeddedFrameCounter;
    HTREEITEM m_pLblEmbeddedStrobePattern;
    HTREEITEM m_pLblEmbeddedGPIOPinState;
    HTREEITEM m_pLblEmbeddedROIPosition;

    // Diagnostics
    HTREEITEM m_pLblSkippedFrames;
    HTREEITEM m_pLblLinkRecoveryCount;
    HTREEITEM m_pLblTransmitFailures;
    HTREEITEM m_pLblTimeSinceInitialization;
    HTREEITEM m_pLblTimeSinceLastBusReset;

	
    void UpdateFrameRateInfo(CTreeCtrl* pCamInfoTreeView, FPSStruct fps );
	void UpdateTimestampInfo(CTreeCtrl* pCamInfoTreeView, FlyCapture2::TimeStamp timestamp );
	void UpdateImageInfo(CTreeCtrl* pCamInfoTreeView, ImageInfoStruct imageInfo );
    void UpdateEmbeddedInfo(CTreeCtrl* pCamInfoTreeView, EmbeddedImageInfoStruct embeddedInfo );
    void UpdateDiagnostics(CTreeCtrl* pCamInfoTreeView, DiagnosticsStruct diagnostics );

};

#endif // PGR_FC2_INFORMATIONPANE_H
