#pragma once
#ifndef _DEM_CYLIND_WALL_H_
#define _DEM_CYLIND_WALL_H_

#include<cuda_runtime.h>
#include<helper_cuda.h>

#include"dem_parameters.h"
#include"dem_cylind_wall_io.h"
#include"type.h"


class  DEMWallCylinder
{
private:

	Cylinder* wall_device;

	double3* Pos;
	double3* Vel;
	double3* AngSpd;
	double4* Quat;

	double3* Fc;
	double3* Tc;

	double3* Dv_cap;
	double3* Dv_pre_cap;

	double3* Dt_cap;
	double3* Dt_pre_cap;

	double3* Dr_cap;
	double3* Dr_pre_cap;

	double3* Dv_up;
	double3* Dv_pre_up;

	double3* Dt_up;
	double3* Dt_pre_up;

	double3* Dr_up;
	double3* Dr_pre_up;

	double3* Dv_side;
	double3* Dv_pre_side;

	double3* Dt_side;
	double3* Dt_pre_side;

	double3* Dr_side;
	double3* Dr_pre_side;

	void pointerExch(double3** pt1, double3** pt2);

public:
	DEMWallCylinder();
	~DEMWallCylinder();

	Cylinder* GetCylinderParameters() { return wall_device; }

	double3* GetPos() { return Pos; }
	double3* GetVel() { return Vel; }
	double3* GetAngSpd() { return AngSpd; }
	double4* GetQuat() { return Quat; }

	double3* GetFc() { return Fc; }
	double3* GetTc() { return Tc; }

	double3* GetDvCap() { return Dv_cap; }
	double3* GetDvPreCap() { return Dv_pre_cap; }

	double3* GetDtCap() { return Dt_cap; }
	double3* GetDtPreCap() { return Dt_pre_cap; }

	double3* GetDrCap() { return Dr_cap; }
	double3* GetDrPreCap() { return Dr_pre_cap; }

	double3* GetDvUp() { return Dv_up; }
	double3* GetDvPreUp() { return Dv_pre_up; }

	double3* GetDtUp() { return Dt_up; }
	double3* GetDtPreUp() { return Dt_pre_up; }

	double3* GetDrUp() { return Dr_up; }
	double3* GetDrPreUp() { return Dr_pre_up; }

	double3* GetDvSide() { return Dv_side; }
	double3* GetDvPreSide() { return Dv_pre_side; }

	double3* GetDtSide() { return Dt_side; }
	double3* GetDtPreSide() { return Dt_pre_side; }

	double3* GetDrSide() { return Dr_side; }
	double3* GetDrPreSide() { return Dr_pre_side; }

	void ForceZero(Parameters* par_host);

	void ForceExchange(Parameters* par_host);

	void AllocateMemory(Parameters* par_host, DEMWallCylinderIo* cyd_io);

	//void CreateContList(Parameters* par_host, Parameters* par_device, double3* PartPos, double* PartRaduis, int* part_fix);

	void CalcContactForce(Parameters* par_host, Parameters* par_device, double3* PartPos, double3* PartVel, double3* PartAngSpd, double* PartRadius, double4* PartProperty);

};


#endif
