#pragma once
#ifndef _DEM_SUBSPHERE_H_
#define _DEM_SUBSPHERE_H_

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

class  DEMSubSphereIo
{
	Parameters* Par;
	GlobalFlg* Glbflg;

	int* ParticleID;
	int* sortedParticleID;

	int* ClumpId;
	double3* Pos;
	double3* LocalPos;
	double* Radius;
	double* Mass;

	double3* CacheFi;     
	double3* CacheTi;

	string bodyfilefolder;
	string iniconfile;
	
	int* Anei;          
	double3* ChainFcn;
	double3* ChainCtp;

	double3* Fin;      
	double3* Fit;
	double3* Tin;
	double3* Tit;

public:
	DEMSubSphereIo();
	~DEMSubSphereIo();

	int* GetParticleID() { return ParticleID; }
	int* GetSortedParticleID() { return sortedParticleID; }

	Parameters* GetParameters() { return Par; }
	GlobalFlg* GetGlobalFlag() { return Glbflg; }

	int* GetClumpId() { return ClumpId; }
	double3* GetPos() { return Pos; }
	double3* GetLocalPos() { return LocalPos; }
	double* GetRadius() { return Radius; }
	double* GetMass() { return Mass; }

	int* GetAnei() { return Anei; }
	double3* GetChainFcn() { return  ChainFcn; }
	double3* GetChainCtp() { return ChainCtp; }

	double3* GetCacheFi() { return  CacheFi; }
	double3* GetCacheTi() { return  CacheTi; }

	double3* GetFin() { return Fin; }
	double3* GetFit() { return Fit; }
	double3* GetTin() { return Tin; }
	double3* GetTit() { return Tit; }

	void LoadParameters();

	void LoadPhase();

	void SetMeshSize();

	void AllocateCpuMemory();

    // Binary functions take the completed step number.
    void SaveParticles_Bin(int ii, GlobalFlg* Glb);
    void SaveContInfo_Bin(int ii);

	void SaveParticles(int ii, GlobalFlg* Glb);

	void SaveContInfo(int ii);
};





#endif
