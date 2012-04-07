#pragma once

// Receiving Socket Class definition
#pragma once
#include <afxtempl.h>
#include <afxsock.h>

#include "RemoteControl.h"

// UdpReceiveSocket command target
class MySocket : public CAsyncSocket
{
      void OnReceive(int nErrorCode);
public:
      MySocket(int port);
      virtual ~MySocket();

	  RemoteControl *rc;
};