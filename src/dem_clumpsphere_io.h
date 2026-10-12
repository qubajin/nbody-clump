#pragma once
#ifndef _DEM_CLUMPSPHERE_IO_H_
#define _DEM_CLUMPSPHERE_IO_H_

#include"dem_parameters.h"
#include"dem_constant.h"

#include<cuda_runtime.h>

#include<string>
#include <sstream>
#include <fstream>
#include <iostream>
#include<iomanip>
#include <algorithm>
#include<assert.h>
#include<cmath>

using namespace std;

class  DEMClumpSphereIo
{
private:
	double3* Pos;
	double3* Vel;
	double3* AngSpd;
	double4* Quat;

	double4* Property; 

	string iniconfile;

public:

	DEMClumpSphereIo();
	~DEMClumpSphereIo();

	double3* GetPos() { return Pos; }
	double3* GetVel() { return Vel; }
	double3* GetAngSpd() { return AngSpd; }
	double4* GetQuat() { return Quat; }
	double4* GetProperty() { return Property; }

	void LoadPhase(Parameters* Par);

    // Binary functions use the completed step number without adding one.
    void SaveClumps_Bin(int ii, Parameters* Par);
    void SaveClumps_Bin(int ii, Parameters* Par, double3* IndPos, double3* IndVel, double4* IndQuat, double3* IndAngSpd);

	void SaveClumps(int ii, Parameters* Par);

	void SaveClumps(int ii, Parameters* Par, double3* IndPos, double3* IndVel, double4* IndQuat, double3* IndAngSpd);
};



#endif
