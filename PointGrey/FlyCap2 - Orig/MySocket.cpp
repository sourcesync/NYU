// Receiving Socket Class implementation
// UdpReceiveSocket.cpp : implementation file
//
#include "stdafx.h"

#include <afxsock.h>

//#include "UdpTestApp.h"
#include "MySocket.h"

class RecordingDialog;

// UdpReceiveSocket
MySocket::MySocket(int port)
{
	  rc = NULL;

      // Just specify input PORT#, local machine is assumed
	  if (port<=0) port = 9122;

#if 0
	  char msg[100];
	  sprintf(msg,"PORT=%d\n", port);
	  AfxMessageBox(msg,MB_OK| MB_ICONSTOP);
#endif

      BOOL bRet = Create(port,SOCK_DGRAM,FD_READ);
      if (bRet != TRUE)
      {
             UINT uErr = GetLastError();
             TCHAR szError[256];
             wsprintf(szError, "Server Receive Socket Create() failed: %d", uErr);
             AfxMessageBox(szError);
      }
}

MySocket::~MySocket()
{
}

// UdpReceiveSocket member functions
void MySocket::OnReceive(int nErrorCode)   
{
  static int i=0;

  //AfxMessageBox("received!");

  i++;

  TCHAR buff[4096];
  int nRead;

  CString strSendersIp;

  UINT uSendersPort;

  // Could use Receive here if you don't need the senders address & port
  //nRead = ReceiveFromEx(buff, 4096, strSendersIp, uSendersPort); 

  nRead = Receive(buff, 4096, 0 );

  switch (nRead)
  {
  case 0:       // Connection was closed.
     Close();      
     break;
  case SOCKET_ERROR:
     if (GetLastError() != WSAEWOULDBLOCK) 
     {
        AfxMessageBox ("Error occurred");
        Close();
     }
     break;
  default: // Normal case: Receive() returned the # of bytes received.
     buff[nRead] = 0; //terminate the string (assuming a string for this example)
     CString strReceivedData(buff);       // This is the input data   

	 char stuff[100];
	 sprintf(stuff,"NREAD %d\n",nRead);
	 //AfxMessageBox(stuff);
	 //AfxMessageBox(strReceivedData);

		//	Get the command code...
		int colon = strReceivedData.Find(":");
		if (colon>0)
		{
			//	Get the command...
			CString _cmd = (LPCTSTR)strReceivedData.Mid(0,colon);
			CT2CA pszCharacterString (_cmd);
			
			//AfxMessageBox (pszCharacterString);
			
			int cmd = atoi( pszCharacterString );
			int len = strReceivedData.GetLength();

			if (cmd==0)
			{
				rc->RC_Start( strReceivedData.Mid(colon+1,len ));
			}
			else if (cmd==1)
			{
				rc->RC_Stop();
			}
			else if (cmd==2)
			{
				rc->RC_SetPath( strReceivedData.Mid(colon+1,len) );
			}
			else if (cmd==3)
			{
				CString _arg = (LPCTSTR)strReceivedData.Mid(colon+1,len);
				CT2CA astr (_arg);
				int arg = atoi( astr );

				rc->RC_SetFormat(arg);
			}
		}

  }
  CAsyncSocket::OnReceive(nErrorCode);

  //AfxMessageBox("YO!",MB_OK| MB_ICONSTOP);

 
}