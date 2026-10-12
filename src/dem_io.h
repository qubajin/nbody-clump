#pragma once

#ifndef _DEM_IO_H_
#define _DEM_IO_H_

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

class  DEMSphereIo
{
private:

	Parameters* Par;
	GlobalFlg* Glbflg;

	int* ParticleID;
	int* sortedParticleID;

	int* Type;

	double3* Pos;
	double3* Vel;
	double3* AngSpd;
	double4* Quat;

	double* Radius;
	double4* Property; // Inertia, Mass

	double* cellSize;
	int3* cellNum;
	int* cellOffset;   //hgrid

	double3* CacheFi;
	double3* CacheTi;

	string bodyfilefolder;
	string iniconfile;

	//
	int* Anei;
	double3* ChainFcn;
	double3* ChainCtp;

public:
	DEMSphereIo();
	~DEMSphereIo();

	Parameters* GetParameters() { return Par; }
	GlobalFlg* GetGlobalFlag() { return Glbflg; }
	double3* GetPos() { return Pos; }
	double3* GetVel() { return Vel; }
	double3* GetAngSpd() { return AngSpd; }
	double4* GetQuat() { return Quat; }

	double3* GetCacheFi() { return CacheFi; }
	double3* GetCacheTi() { return CacheTi; }

	int* GetParticleID() { return ParticleID; }
	int* GetSortedParticleID() { return sortedParticleID; }
	int* GetType() { return Type; }

	double* GetRadius() { return Radius; }
	double4* GetProperty() { return Property; }

	int* GetAnei() { return Anei; }
	double3* GetChainFcn() { return  ChainFcn; }
	double3* GetChainCtp() { return ChainCtp; }

	double* GetCellSize() { return cellSize; }
	int3* GetCellNum() { return cellNum; }
	int* GetCellOffset() { return cellOffset; }

	void LoadParameters();

	void LoadGridLevels();

	void LoadPhase();

	void SetMeshSize();

	void AllocateCpuMemory();

    // Binary functions use the completed step number without adding one.
    void SaveParticles_Bin(int ii);
    void SaveParticles_Bin(int ii, double3* IndPos, double3* IndVel, double4* IndQuat, double3* IndAngSpd);
    void SaveContInfo_Bin(int ii);

	void SaveParticles(int ii);

	void SaveParticles(int ii, double3* IndPos, double3* IndVel, double4* IndQuat, double3* IndAngSpd);

	void SaveContInfo(int ii);
};




#endif

