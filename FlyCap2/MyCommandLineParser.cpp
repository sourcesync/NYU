#include "StdAfx.h"
#include "MyCommandLineParser.h"

MyCommandLineParser::MyCommandLineParser(void)
{
	m_pp = FALSE;
	m_port = 0;
}

MyCommandLineParser::~MyCommandLineParser(void)
{
}

void MyCommandLineParser::ParseParam(const char* pszParam, BOOL bFlag, BOOL bLast)
  {
#if 0
    if(0 == strcmp(pszParam, "z"))
    {
		m_pp = TRUE;
	}
#endif

	if(bFlag) 
	{
		CString sParam(pszParam);

		if (sParam.Left(2) == "p:") 
		{
			CString sTemp;
			sTemp = sParam.Right(sParam.GetLength() - 2);
			m_port = atoi(sTemp);
			m_pp = TRUE;
			return;
		}
	}

  }
