#pragma once
#ifndef _DEM_FORCE_H_
#define _DEM_FORCE_H_

#include<cuda_runtime.h>

#include"dem_parameters.h"

#include"type.h"

class DEMForce
{
private:

	size_t pitch;

	double3* Fc;
	double3* Tc;

	double3* Dv;
	double3* Dv_pre;

	double3* Dt;
	double3* Dt_pre;

	double3* Dr;
	double3* Dr_pre;

	double3* CacheFc;
	double3* CacheTc;
	double3* CacheTc2;

	//
	double3* CacheFcn;
	double3* CacheCtp;

	double3* ChainFcn;
	double3* ChainCtp;

	void pointerExch(double3** pt1, double3** pt2);

public:
	DEMForce();
	~DEMForce();

	size_t GetPitch() { return pitch; }

	double3* GetFc() { return Fc; }
	double3* GetTc() { return Tc; }

	double3* GetCacheFc() { return CacheFc; }
	double3* GetCacheTc() { return CacheTc; }
	double3* GetCacheTc2() { return CacheTc2; }

	//
	double3* GetDv() { return Dv; }
	double3* GetDvPre() { return Dv_pre; }

	double3* GetDt() { return Dt; }
	double3* GetDtPre() { return Dt_pre; }

	double3* GetDr() { return Dr; }
	double3* GetDrPre() { return Dr_pre; }

	//
	double3* GetCahceFcn() { return CacheFcn; }
	double3* GetChainFcn() { return ChainFcn; }
	double3* GetChainCtp() { return ChainCtp; }

	void ForceConfig(Parameters* par_host, GlobalFlg* Glb);

	void ForceZero(Parameters* par_host, GlobalFlg* Glb);

	void ForceExchange(Parameters* par_host);

	void CalcContactForce(Parameters* par_host, Parameters* par_device, int* Lpa, int* Lpb, double3* Pos, double3* Vel, double3* AngSpd, double4* Quat, double4* Property, double* Radius,
						  int* Aref, int* Ng, int* Ni, size_t pitch, int* part_fix);

	void CalcFcnChain(Parameters* par_host, Parameters* par_device, int* Aref, int* Ng, int* Ni, size_t pitch_int);

	void Integrate_Before(Parameters* par_host, Parameters* par_device, double3* Pos, double3* Vel, double3* AngSpd, double3* CumPos, double* Radius, double4* Property,
				          double3* Fw, double3* Tw, double3* Fi, double3* Ti, int* part_fix);

	void Attitude(Parameters* par_host, Parameters* par_device, double3* AngSpd, double4* Quat, double4* Property, double3* Tw, double3* Ti);

	void Integrate_After(Parameters* par_host, Parameters* par_device, double3* Vel, double3* AngSpd, double4* Property,
						 double3* Fw, double3* Tw, double3* Fi, double3* Ti);

	//
	void Integrate_Before(Parameters* par_host, Parameters* par_device, double3* Pos, double3* Vel, double3* AngSpd, double3* CumPos, double* Radius, double4* Property,
					      double3* Fw, double3* Tw, double3* Fi, double3* Ti, int*Type, int* part_fix);

	void Attitude(Parameters* par_host, Parameters* par_device, double3* AngSpd, double4* Quat, double4* Property, double3* Tw, double3* Ti, int* Type);

	void Integrate_After(Parameters* par_host, Parameters* par_device, double3* Vel, double3* AngSpd, double4* Property,
						 double3* Fw, double3* Tw, double3* Fi, double3* Ti, int* Type);

	//
	void CalcContactForce_Clump(Parameters* par_host, Parameters* par_device, int* Lpa, int* Lpb, int* ClumpId, double3* Pos, double3* LocalPos, double* Mass, double* Radius,
								double3* ClumpVel, double3* ClumpAngSpd, double4* ClumpQuat, int* Aref, int* Ng, int* Ni, size_t pitch, int* part_fix);

};




#endif // !_DEM_FORCE_H_
