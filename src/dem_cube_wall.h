#pragma once
#ifndef _DEM_CUBE_WALL_H_
#define _DEM_CUBE_WALL_H_

#include<cuda_runtime.h>
#include<helper_cuda.h>

#include"dem_parameters.h"
#include"dem_cube_wall_io.h"
#include"type.h"

class DEMWallCube
{
private:
	Cube* wall_device;

	double3* Pos;
	double3* Vel;
	double3* AngSpd;
	double4* Quat;

	double3* Fc;
	double3* Tc;

	double3* Dv_UpX;
	double3* Dt_UpX;
	double3* Dr_UpX;
	double3* Dv_pre_UpX;
	double3* Dt_pre_UpX;
	double3* Dr_pre_UpX;

	double3* Dv_UpY;
	double3* Dt_UpY;
	double3* Dr_UpY;
	double3* Dv_pre_UpY;
	double3* Dt_pre_UpY;
	double3* Dr_pre_UpY;

	double3* Dv_UpZ;
	double3* Dt_UpZ;
	double3* Dr_UpZ;
	double3* Dv_pre_UpZ;
	double3* Dt_pre_UpZ;
	double3* Dr_pre_UpZ;

	double3* Dv_LowX;
	double3* Dt_LowX;
	double3* Dr_LowX;
	double3* Dv_pre_LowX;
	double3* Dt_pre_LowX;
	double3* Dr_pre_LowX;

	double3* Dv_LowY;
	double3* Dt_LowY;
	double3* Dr_LowY;
	double3* Dv_pre_LowY;
	double3* Dt_pre_LowY;
	double3* Dr_pre_LowY;

	double3* Dv_LowZ;
	double3* Dt_LowZ;
	double3* Dr_LowZ;
	double3* Dv_pre_LowZ;
	double3* Dt_pre_LowZ;
	double3* Dr_pre_LowZ;

	void pointerExch(double3** pt1, double3** pt2);

public:

	DEMWallCube();
	~DEMWallCube();

	Cube* GetWallParameters() { return wall_device; }

	double3* GetPos() { return Pos; }
	double3* GetVel() { return Vel; }
	double3* GetAngSpd() { return AngSpd; }
	double4* GetQuat() { return Quat; }

	double3* GetFc() { return Fc; }
	double3* GetTc() { return Tc; }

	double3* GetDvUpX() { return Dv_UpX; }
	double3* GetDtUpX() { return Dt_UpX; }
	double3* GetDrUpX() { return Dr_UpX; }
	double3* GetDvPreUpX() { return Dv_pre_UpX; }
	double3* GetDtPreUpX() { return Dt_pre_UpX; }
	double3* GetDrPreUpX() { return Dr_pre_UpX; }

	double3* GetDvUpY() { return Dv_UpY; }
	double3* GetDtUpY() { return Dt_UpY; }
	double3* GetDrUpY() { return Dr_UpY; }
	double3* GetDvPreUpY() { return Dv_pre_UpY; }
	double3* GetDtPreUpY() { return Dt_pre_UpY; }
	double3* GetDrPreUpY() { return Dr_pre_UpY; }

	double3* GetDvUpZ() { return Dv_UpZ; }
	double3* GetDtUpZ() { return Dt_UpZ; }
	double3* GetDrUpZ() { return Dr_UpZ; }
	double3* GetDvPreUpZ() { return Dv_pre_UpZ; }
	double3* GetDtPreUpZ() { return Dt_pre_UpZ; }
	double3* GetDrPreUpZ() { return Dr_pre_UpZ; }

	double3* GetDvLowX() { return Dv_LowX; }
	double3* GetDtLowX() { return Dt_LowX; }
	double3* GetDrLowX() { return Dr_LowX; }
	double3* GetDvPreLowX() { return Dv_pre_LowX; }
	double3* GetDtPreLowX() { return Dt_pre_LowX; }
	double3* GetDrPreLowX() { return Dr_pre_LowX; }

	double3* GetDvLowY() { return Dv_LowY; }
	double3* GetDtLowY() { return Dt_LowY; }
	double3* GetDrLowY() { return Dr_LowY; }
	double3* GetDvPreLowY() { return Dv_pre_LowY; }
	double3* GetDtPreLowY() { return Dt_pre_LowY; }
	double3* GetDrPreLowY() { return Dr_pre_LowY; }

	double3* GetDvLowZ() { return Dv_LowZ; }
	double3* GetDtLowZ() { return Dt_LowZ; }
	double3* GetDrLowZ() { return Dr_LowZ; }
	double3* GetDvPreLowZ() { return Dv_pre_LowZ; }
	double3* GetDtPreLowZ() { return Dt_pre_LowZ; }
	double3* GetDrPreLowZ() { return Dr_pre_LowZ; }

	void ForceZero(Parameters* par_host);

	void ForceExchange(Parameters* par_host);

	void AllocateMemory(Parameters* par_host, DEMWallCubeIo* wall_io);

	void CalcContactForce(Parameters* par_host, Parameters* par_device, double3* PartPos, double3* PartVel,
						  double3* PartAngSpd, double* PartRadius, double4* PartProperty);

	void CalcContactForce_Clump(Parameters* par_host, Parameters* par_device, int* ClumpId, double3* PartPos, double3* PartLocalPos, double* Radius, double* Mass,
								double3* ClumpVel, double3* ClumpAngSpd, double4* ClumpQuat);

};

#endif
