#pragma once
#ifndef _DEM_PENETRATOR_H_
#define _DEM_PENETRATOR_H_

#include<cuda_runtime.h>
#include<helper_cuda.h>

#include"dem_parameters.h"
#include"dem_subsphere_io.h"
#include"dem_penetrator_io.h"
#include"type.h"

#include "thrust/sort.h"
#include "thrust/device_ptr.h"
#include "thrust/scan.h"


class  DEMPeneCylinder
{
private:

	Cylinder* pene_device;

	double3* Pos;
	double3* Vel;
	double3* AngSpd;
	double4* Quat;

	double3* AngSpdB;
	double3* AngSpdDotB;

	double3* Fc;
	double3* Tc;
	double3* peneTc;

	double3* Fcn;
	double3* Fct;
	double3* Tcn;
	double3* Tct;

	double3* Cache;

	double3* totFc;
	double3* totTc;

	double3* Dv;
	double3* Dv_pre;

	double3* Dt;
	double3* Dt_pre;

	double3* Dr;
	double3* Dr_pre;

	double3* RVo;
	double3* RVn;

	double3* CacheFc;
	double3* CacheTc;

	int* LcFlg;
	int* Lc;
	int* Nsum;

	void pointerExch(double3** pt1, double3** pt2);

public:
	 DEMPeneCylinder();
	~ DEMPeneCylinder();

	Cylinder* GetPeneParameters() { return pene_device; }

	double3* GetPos() { return Pos; }
	double3* GetVel() { return Vel; }
	double3* GetAngSpd() { return AngSpd; }
	double4* GetQuat() { return Quat; }

	double3* GetAngSpdB() { return AngSpdB; }
	double3* GetAngSpdDotB() { return AngSpdDotB; }

	double3* GetFc() { return Fc; }
	double3* GetTc() { return Tc; }
	double3* GetPeneTc() { return peneTc; }

	double3* GetFcn() { return Fcn; }
	double3* GetFct() { return Fct; }
	double3* GetTcn() { return Tcn; }
	double3* GetTct() { return Tct; }

	double3* GetTotFc() { return totFc; }
	double3* GetTotTc() { return totTc; }

	double3* GetCache() { return Cache; }

	double3* GetCacheFc() { return CacheFc; }
	double3* GetCacheTc() { return CacheTc; }

	double3* GetDv() { return Dv; }
	double3* GetDvPre() { return Dv_pre; }

	double3* GetDt() { return Dt; }
	double3* GetDtPre() { return Dt_pre; }

	double3* GetDr() { return Dr; }
	double3* GetDrPre() { return Dr_pre; }

	double3* GetRVn() { return RVn; }
	double3* GetRVo() { return RVo; }

	int* GetLcFlg() { return LcFlg; }
	int* GetLc() { return Lc; }
	int* GetNsum() { return Nsum; }

	void CopyDataToIo(DEMPeneCylinderIo* cyd_io);

	void ForceZero(Parameters* par_host);

	void ForceExchange(Parameters* par_host);

	void AllocateMemory(Parameters* par_host, DEMPeneCylinderIo* pene_io);

	void CopyDataToIo(Parameters* par_host, DEMPeneCylinderIo* cyd_io, DEMSubSphereIo* io);

	void CreateContList(Parameters* par_host, Parameters* par_device, double3* PartPos, double* PartRaduis, int* part_fix);

	void CalcContactForce(Parameters* par_host, Parameters* par_device, double3* PartPos, double3* PartVel, double3* PartAngSpd, double* PartRadius, double4* PartProperty);

	void SumForce(Parameters* par_host, double3* F, double3* finalF);

	void Integrate_Before(Parameters* par_device);

	void Integrate_After(int count, Parameters* par_device);

	void CalcContactForce_Clump(Parameters* par_host, Parameters* par_device, int* ClumpId, double3* PartPos, double3* PartLocalPos, double* PartRadius, double* PartMass,
								double3* ClumpVel, double3* ClumpAngSpd, double4* ClumpQuat);

};


#endif