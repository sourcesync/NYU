#pragma once

class MyCommandLineParser : public CCommandLineInfo
{

public:
	BOOL m_pp;
	int m_port;

public:
	MyCommandLineParser(void);

	virtual void ParseParam(const char* pszParam, BOOL bFlag, BOOL bLast);

public:
	~MyCommandLineParser(void);
};
