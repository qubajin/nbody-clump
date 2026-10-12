#pragma once
#ifndef _DEM_CLUMP_H_
#define _DEM_CLUMP_H_

#include<iostream>
#include<sstream>
#include<fstream>
#include<iomanip>
#include<cmath>
#include<ctime>
#include<vector>
#include<string>
#include<cuda_runtime.h>
#include<boost/variant.hpp>
#include<boost/algorithm/string.hpp>
#include<boost/filesystem.hpp>
#include<helper_cuda.h>
#include<thrust/extrema.h>
#include<chrono>

#include"dem_parameters.h"
#include"dem_grid.h"
#include"dem_force.h"
#include"dem_cylind_wall.h"
#include"dem_cylind_wall_io.h"
#include"dem_penetrator.h"
#include"dem_penetrator_io.h"
#include"dem_cube_wall_io.h"
#include"dem_cube_wall.h"

#include"dem_clumpsphere_io.h"
#include"dem_clumpsphere.h"
#include"dem_subsphere_io.h"
#include"type.h"

class DEMClumpMain
{
private:
	Parameters* par_host;
	Parameters* par_device;

	GlobalFlg* Glb;
	
	DEMSubSphereIo* io;

	DEMClumpSphereIo* clump_io;
	DEMClumpSphere* clump;

	DEMGrid* grid;
	DEMForce* force;

	DEMWallCylinderIo* wall_io;
	DEMWallCylinder* wall;

	DEMWallCubeIo* cube_io;
	DEMWallCube* cube;

	DEMPeneCylinderIo* pene_io;
	DEMPeneCylinder* pene;

	int* ClumpId;

	double3* Pos;
	double3* LocalPos;
	double3* CumPos;

	double* Radius;    
	double* Mass;

	double3* totF;
	double3* totT;

	//
	void AllocateGpuMemory();

	void CopyDataFromHostToDevice();

	void UpdatePos(int* ClumpId, double3* ClumpPos, double4* ClumpQuat);

	void ForceZero();

	void SumAllForces(double3* Fc, double3* Tc, double3* Fw, double3* Tw, double3* Fi, double3* Ti);

	void CopyDataToIo(GlobalFlg* Glb, double3* CacheFc, double3* CacheTc);

	void CopyInfoToIo(size_t pitch_int, size_t pitch_double3, int* Anei, double3* ChainFcn, double3* ChainCtp);

	void pointerExch(double3** pt1, double3** pt2);

public:
	DEMClumpMain();

	~DEMClumpMain();

	void RunSimulationClump();
};




#endif
