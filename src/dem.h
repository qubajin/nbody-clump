#pragma once
#ifndef _DEM_H_
#define _DEM_H_

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
#include"dem_hgrid.h"
#include"dem_io.h"
#include"dem_force.h"
#include"dem_cylind_wall.h"
#include"dem_cylind_wall_io.h"
#include"dem_penetrator.h"
#include"dem_penetrator_io.h"
#include"dem_cube_wall_io.h"
#include"dem_cube_wall.h"

class DEMmain
{
private:
	Parameters* par_host;
	Parameters* par_device;

	GlobalFlg* Glb;

	DEMSphereIo* io;
	DEMGrid* grid;
	DEMForce* force;

	DEMHGrid* hgrid;

	DEMWallCylinderIo* wall_io;
	DEMWallCylinder* wall;

	DEMWallCubeIo* cube_io;
	DEMWallCube* cube;

	DEMPeneCylinderIo* pene_io;
	DEMPeneCylinder* pene;

	int* Type;

	double3* Pos;
	double3* Vel;
	double3* AngSpd;
	double4* Quat;

	double3* CumPos;

	double* Radius;     // Equivalent radius
	double4* Property;  // Inertia, Mass

	//
	void AllocateGpuMemory();

	void CopyDataFromHostToDevice();

	void CopyDataToIo();

	void CopyDataToIo(double3* CntFc, double3* CntTc);

	void CopyDataToIo_HGrid(double3* CntFc, double3* CntTc);

	void CopyInfoToIo(size_t pitch_int, size_t pitch_double3, int* Anei, double3* ChainFcn, double3* ChainCtp);

	void pointerExch(double3** pt1, double3** pt2);

public:
	DEMmain();

	~DEMmain();

	void RunSimulation();

	void RunSimulation2();

	void RunSimulation_Split();

	void RunSimulation_HGrid();


};




#endif
