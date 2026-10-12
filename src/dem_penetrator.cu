#include"dem_penetrator.h"

DEMPeneCylinder::DEMPeneCylinder()
{
	pene_device = NULL;
	Pos = NULL;
	Vel = NULL;
	AngSpd = NULL;
	Quat = NULL;

	AngSpdB = NULL;
	AngSpdDotB = NULL;

	Fc = NULL;
	Tc = NULL;
	peneTc = NULL;
	Fcn = NULL;
	Fct = NULL;
	Tcn = NULL;
	Tct = NULL;
	Cache = NULL;
	totFc = NULL;
	totTc = NULL;
	CacheFc = NULL;
	CacheTc = NULL;

	RVn = NULL;
	RVo = NULL;

	Dv = NULL;
	Dt = NULL;
	Dr = NULL;

	Dt_pre = NULL;
	Dv_pre = NULL;
	Dr_pre = NULL;

	LcFlg = NULL;
	Lc = NULL;
	Nsum = NULL;
}

DEMPeneCylinder::~DEMPeneCylinder()
{
	if (pene_device != NULL) cudaFree(pene_device);
	if (Pos != NULL) cudaFree(Pos);
	if (Vel != NULL) cudaFree(Vel);
	if (AngSpd != NULL) cudaFree(AngSpd);
	if (Quat != NULL) cudaFree(Quat);
	if (AngSpdB != NULL) cudaFree(AngSpdB);
	if (AngSpdDotB != NULL) cudaFree(AngSpdDotB);

	if (Fc != NULL) cudaFree(Fc);
	if (Tc != NULL) cudaFree(Tc);
	if (peneTc != NULL) cudaFree(peneTc);
	if (Fcn != NULL) cudaFree(Fcn);
	if (Fct != NULL) cudaFree(Fct);
	if (Tcn != NULL) cudaFree(Tcn);
	if (Tct != NULL) cudaFree(Tct);
	if (totFc != NULL) cudaFree(totFc);
	if (totTc != NULL) cudaFree(totTc);
	if (Cache != NULL) cudaFree(Cache);
	if (CacheFc != NULL) cudaFree(CacheFc);
	if (CacheTc != NULL) cudaFree(CacheTc);

	if (RVn != NULL) cudaFree(RVn);
	if (RVo != NULL) cudaFree(RVo);

	if (Dv != NULL) cudaFree(Dv);
	if (Dv_pre != NULL) cudaFree(Dv_pre);
	if (Dt != NULL) cudaFree(Dt);
	if (Dt_pre != NULL) cudaFree(Dt_pre);
	if (Dr != NULL) cudaFree(Dr);
	if (Dr_pre != NULL) cudaFree(Dr_pre);

	if (LcFlg != NULL) cudaFree(LcFlg);
	if (Lc != NULL) cudaFree(Lc);
	if (Nsum != NULL) cudaFree(Nsum);
}

__global__ void DEMPeneCylinder_cuk(double3* Val)
{
	for (int i = 0; i < 1; i++)
	{
		printf("PrintDEMWallCylinder: %f %f %f\n", Val[i].x, Val[i].y, Val[i].z);
	}
}

void DEMPeneCylinder::AllocateMemory(Parameters* par_host, DEMPeneCylinderIo* pene_io)
{
	checkCudaErrors(cudaMalloc((void**)&pene_device, sizeof(Cylinder)));

	checkCudaErrors(cudaMalloc((void**)&Pos, sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&Vel, sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&AngSpd, sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&Quat, sizeof(double4)));

	checkCudaErrors(cudaMalloc((void**)&AngSpdB, sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&AngSpdDotB, sizeof(double3)));

	checkCudaErrors(cudaMalloc((void**)&Fc, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&Tc, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&peneTc, par_host->ParticleNum * sizeof(double3)));

	checkCudaErrors(cudaMalloc((void**)&Fcn, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&Fct, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&Tcn, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&Tct, par_host->ParticleNum * sizeof(double3)));

	checkCudaErrors(cudaMemset(Fc, 0.0, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMemset(Tc, 0.0, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMemset(peneTc, 0.0, par_host->ParticleNum * sizeof(double3)));

	checkCudaErrors(cudaMalloc((void**)&totFc, sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&totTc, sizeof(double3)));

	checkCudaErrors(cudaMemset(totFc, 0.0, sizeof(double3)));
	checkCudaErrors(cudaMemset(totTc, 0.0, sizeof(double3)));
	checkCudaErrors(cudaMemset(AngSpdB, 0.0, sizeof(double3)));
	checkCudaErrors(cudaMemset(AngSpdDotB, 0.0, sizeof(double3)));

	checkCudaErrors(cudaMalloc((void**)&Cache, 128 * sizeof(double3)));

	checkCudaErrors(cudaMalloc((void**)&CacheFc, par_host->OutputInterval * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&CacheTc, par_host->OutputInterval * sizeof(double3)));

	checkCudaErrors(cudaMalloc((void**)&RVn, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&RVo, par_host->ParticleNum * sizeof(double3)));

	checkCudaErrors(cudaMalloc((void**)&Dv, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&Dv_pre, par_host->ParticleNum * sizeof(double3)));

	checkCudaErrors(cudaMalloc((void**)&Dt, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&Dt_pre, par_host->ParticleNum * sizeof(double3)));

	checkCudaErrors(cudaMalloc((void**)&Dr, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&Dr_pre, par_host->ParticleNum * sizeof(double3)));

	checkCudaErrors(cudaMemset(Dv, 0.0, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMemset(Dv_pre, 0.0, par_host->ParticleNum * sizeof(double3)));

	checkCudaErrors(cudaMemset(Dt, 0.0, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMemset(Dt_pre, 0.0, par_host->ParticleNum * sizeof(double3)));

	checkCudaErrors(cudaMemset(Dr, 0.0, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMemset(Dr_pre, 0.0, par_host->ParticleNum * sizeof(double3)));


	checkCudaErrors(cudaMalloc((void**)&LcFlg, par_host->ParticleNum * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&Lc, par_host->ParticleNum * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&Nsum, par_host->ParticleNum * sizeof(int)));

	//
	checkCudaErrors(cudaMemcpy(pene_device, pene_io->GetPeneParameters(), sizeof(Cylinder), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(Pos, pene_io->GetPos(), sizeof(double3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(Vel, pene_io->GetVel(), sizeof(double3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(Quat, pene_io->GetQuat(), sizeof(double4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(AngSpd, pene_io->GetAngSpd(), sizeof(double3), cudaMemcpyHostToDevice));

	/*DEMPeneCylinder_cuk << <1, 1 >> > (Tc);
	cudaDeviceSynchronize();*/
}

void DEMPeneCylinder::pointerExch(double3** pt1, double3** pt2)
{
	double3* tmp;
	tmp = *pt1;
	*pt1 = *pt2;
	*pt2 = tmp;
}


void DEMPeneCylinder::ForceZero(Parameters* par_host)
{
	checkCudaErrors(cudaMemset(Fc, 0.0, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMemset(Tc, 0.0, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMemset(peneTc, 0.0, par_host->ParticleNum * sizeof(double3)));

	checkCudaErrors(cudaMemset(Fcn, 0.0, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMemset(Fct, 0.0, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMemset(Tcn, 0.0, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMemset(Tct, 0.0, par_host->ParticleNum * sizeof(double3)));
}

void DEMPeneCylinder::ForceExchange(Parameters* par_host)
{
	pointerExch(&Dv, &Dv_pre);
	pointerExch(&Dt, &Dt_pre);
	pointerExch(&Dr, &Dr_pre);

	checkCudaErrors(cudaMemset(Dv, 0.0, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMemset(Dt, 0.0, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMemset(Dr, 0.0, par_host->ParticleNum * sizeof(double3)));

}

void DEMPeneCylinder::CopyDataToIo(DEMPeneCylinderIo* cyd_io)
{
	checkCudaErrors(cudaMemcpy(cyd_io->GetPos(), Pos, sizeof(double3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(cyd_io->GetVel(), Vel, sizeof(double3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(cyd_io->GetQuat(), Quat, sizeof(double4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(cyd_io->GetAngSpd(), AngSpd, sizeof(double3), cudaMemcpyDeviceToHost));

}

void DEMPeneCylinder::CopyDataToIo(Parameters* par_host, DEMPeneCylinderIo* cyd_io, DEMSubSphereIo *io)
{
	checkCudaErrors(cudaMemcpy(cyd_io->GetPos(), Pos, sizeof(double3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(cyd_io->GetVel(), Vel, sizeof(double3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(cyd_io->GetQuat(), Quat, sizeof(double4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(cyd_io->GetAngSpd(), AngSpd, sizeof(double3), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(io->GetFin(), Fcn, par_host->ParticleNum * sizeof(double3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(io->GetFit(), Fct, par_host->ParticleNum * sizeof(double3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(io->GetTin(), Tcn, par_host->ParticleNum * sizeof(double3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(io->GetTit(), Tct, par_host->ParticleNum * sizeof(double3), cudaMemcpyDeviceToHost));

}
