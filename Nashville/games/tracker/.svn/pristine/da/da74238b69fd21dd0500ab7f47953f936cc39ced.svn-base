// camlibtest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "camlib.h"

int main(int argc, char* argv)
{
	//	test
	//int a = fncamlib(0);
	//printf("out=%d\n", a);
	
	/*
	int id0 = camlib_init(0,0);
	if (id0<0) return 1;
	*/

	int id1 = camlib_init(1,1);
	if (id1<0) return 1;
	int id2 = camlib_init(3,1);
	if (id2<0) return 1;

	while (1)
	{
		/*
		int retv = camlib_frame(id0,0);
		*/

		int ret0 = camlib_frame(id1,1);
		int ret1 = camlib_frame(id2,1);
	}

	printf("done\n");
}

