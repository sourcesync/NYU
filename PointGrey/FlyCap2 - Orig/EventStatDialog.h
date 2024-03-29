//=============================================================================
// Copyright � 2011 Point Grey Research, Inc. All Rights Reserved.
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

#include "Resource.h"
#include <time.h>
#pragma once
#include "afxwin.h"
#include "afxcmn.h"

// EventStatDialog dialog
using namespace std;

enum EventType
{
		TOTAL_NUMBER_OF_FRAMES = 0,
		IMAGE_CONSISTENCY_ERRORS,
		IMAGE_CONVERSION_ERRORS,
		TRANSMIT_FAILURES,
		RECOVERY_COUNT,
		SKIPPED_IMAGES,
		NUMBER_OF_BUS_RESETS,
		NUMBER_OF_BUS_ARRIVALS,
		NUMBER_OF_BUS_REMOVALS,
		NUMBER_OF_EVENT_TYPES
};

class EventStatDialog : public CDialog
{
	DECLARE_DYNAMIC(EventStatDialog)
private:
	static const int sk_numOfColumns = 4;//number of data columns (exclude event type name column)
	static const UINT sk_dataRefreshDelay = 100;
public:
	// Dialog Data
	enum { IDD = IDD_DIALOG_EVENT };
	void UpdateEventsData();
	bool HasBadEventRecently();

protected:
	enum TimeSliceType
	{
		//the type must be sorted
		LAST_10S = 0,
		LAST_30S,
		LAST_1MIN,
		LAST_5MINS,
		LAST_10MINS,
		LAST_15MINS,
		NUMBER_OF_TIME_SLICE_TYPES
	};
    bool m_turnOnEventCollection;
	bool m_hasBadEventRecently;
	bool IsBadEvent(EventType eventType);
	int GetTime(TimeSliceType timeSliceType);
	void GetTimeString(TimeSliceType timeSliceType, char* pResultString);
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
	void UpdateColumnHeader();
	DECLARE_MESSAGE_MAP()
	CListCtrl m_eventTable;
	deque<time_t> m_data[NUMBER_OF_EVENT_TYPES];
	TimeSliceType m_columnsData[sk_numOfColumns];
	time_t m_currentTime;
	UINT_PTR m_timer; 
	
	// Critical section to protect access to the event statistics data
    CCriticalSection m_csEventData;
	void InitializeColumns();
	CComboBox m_timeSliceComboBox;
    CButton m_eventCollectionCheckBox;
	static void GetEventTypeString(EventType eventType, char* pResultString);
	
public:
	EventStatDialog(CWnd* pParent = NULL);   // standard constructor
	virtual ~EventStatDialog();
	virtual BOOL OnInitDialog();
	void AddEvent(EventType eventType);
	void CleanUpCounters();
	afx_msg void OnTimer(UINT_PTR nIDEvent);
	virtual BOOL DestroyWindow();
	afx_msg void OnCbnSelchangeComboTimeslice();
    afx_msg void OnBnClickedTurnOn();    
};
