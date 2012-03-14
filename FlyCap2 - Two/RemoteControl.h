#pragma once

class RemoteControl
{
public:
	virtual void RC_Start( CString path ) = 0;
	virtual void RC_Stop() = 0;
	virtual void RC_SetPath( CString path ) = 0;
	virtual void RC_SetFormat( int fmt ) = 0;
};