#pragma once
#ifndef _DEM_CUBE_WALL_IO_H_
#define _DEM_CUBE_WALL_IO_H_

#include <stdio.h>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <assert.h>
#include<cuda_runtime.h>

#include "dem_parameters.h"

using namespace std;

class DEMWallCubeIo
{
private:
	Cube* wall_host;

	double3* Pos;
	double3* Vel;
	double3* AngSpd;
	double4* Quat;

public:
	DEMWallCubeIo();
	~DEMWallCubeIo();

	Cube* GetWallParameters() { return wall_host; }

	double3* GetPos() { return Pos; }
	double3* GetVel() { return Vel; }
	double3* GetAngSpd() { return AngSpd; }
	double4* GetQuat() { return Quat; }

	void LoadInfo();
};



#endif


