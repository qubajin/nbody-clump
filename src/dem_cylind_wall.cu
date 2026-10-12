#include"dem_cylind_wall.h"

DEMWallCylinder::DEMWallCylinder()
{
	wall_device = NULL;
	Pos = NULL;
	Vel = NULL;
	AngSpd = NULL;
	Quat = NULL;

	Fc = NULL;
	Tc = NULL;

	Dv_cap = NULL;
	Dv_pre_cap = NULL;

	Dt_cap = NULL;
	Dt_pre_cap = NULL;

	Dr_cap = NULL;
	Dr_pre_cap = NULL;

	Dv_side = NULL;
	Dv_pre_side = NULL;

	Dt_side = NULL;
	Dt_pre_side = NULL;

	Dr_side = NULL;
	Dr_pre_side = NULL;
}

DEMWallCylinder::~DEMWallCylinder()
{
	if (Pos != NULL) cudaFree(Pos);
	if (Vel != NULL) cudaFree(Vel);
	if (AngSpd != NULL) cudaFree(AngSpd);
	if (Quat != NULL) cudaFree(Quat);

	if (Fc != NULL) cudaFree(Fc);
	if (Tc != NULL) cudaFree(Tc);

	if (Dv_cap != NULL) cudaFree(Dv_cap);
	if (Dv_pre_cap != NULL) cudaFree(Dv_pre_cap);

	if (Dt_cap != NULL) cudaFree(Dt_cap);
	if (Dt_pre_cap != NULL) cudaFree(Dt_pre_cap);

	if (Dr_cap != NULL) cudaFree(Dr_cap);
	if (Dr_pre_cap != NULL) cudaFree(Dr_pre_cap);

	if (Dv_side != NULL) cudaFree(Dv_side);
	if (Dv_pre_side != NULL) cudaFree(Dv_pre_side);

	if (Dt_side != NULL) cudaFree(Dt_side);
	if (Dt_pre_side != NULL) cudaFree(Dt_pre_side);

	if (Dr_side != NULL) cudaFree(Dr_side);
	if (Dr_pre_side != NULL) cudaFree(Dr_pre_side);

}

void DEMWallCylinder::pointerExch(double3** pt1, double3** pt2)
{
	double3* tmp;
	tmp = *pt1;
	*pt1 = *pt2;
	*pt2 = tmp;
}

void DEMWallCylinder::ForceExchange(Parameters* par_host)
{
	//printf("Zero!\n");

	pointerExch(&Dv_cap, &Dv_pre_cap);
	pointerExch(&Dt_cap, &Dt_pre_cap);
	pointerExch(&Dr_cap, &Dr_pre_cap);

	pointerExch(&Dv_up, &Dv_pre_up);
	pointerExch(&Dt_up, &Dt_pre_up);
	pointerExch(&Dr_up, &Dr_pre_up);

	pointerExch(&Dv_side, &Dv_pre_side);
	pointerExch(&Dt_side, &Dt_pre_side);
	pointerExch(&Dr_side, &Dr_pre_side);

	checkCudaErrors(cudaMemset(Dv_cap, 0.0, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMemset(Dt_cap, 0.0, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMemset(Dr_cap, 0.0, par_host->ParticleNum * sizeof(double3)));

	checkCudaErrors(cudaMemset(Dv_up, 0.0, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMemset(Dt_up, 0.0, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMemset(Dr_up, 0.0, par_host->ParticleNum * sizeof(double3)));

	checkCudaErrors(cudaMemset(Dv_side, 0.0, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMemset(Dt_side, 0.0, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMemset(Dr_side, 0.0, par_host->ParticleNum * sizeof(double3)));
}

void DEMWallCylinder::ForceZero(Parameters* par_host)
{
	checkCudaErrors(cudaMemset(Fc, 0.0, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMemset(Tc, 0.0, par_host->ParticleNum * sizeof(double3)));
}

__global__ void PrintDEMWallCylinder_cuk( double3* Val)
{
	for (int i = 0; i < 1; i++)
	{
		printf("PrintDEMWallCylinder: %f %f %f\n", Val[i].x, Val[i].y, Val[i].z);
	}
}


void DEMWallCylinder::AllocateMemory(Parameters* par_host, DEMWallCylinderIo* wall_io)
{
	checkCudaErrors(cudaMalloc((void**)&wall_device, sizeof(Cylinder)));

	checkCudaErrors(cudaMalloc((void**)&Pos, sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&Vel, sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&AngSpd, sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&Quat, sizeof(double4)));

	checkCudaErrors(cudaMalloc((void**)&Fc, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&Tc, par_host->ParticleNum * sizeof(double3)));

	checkCudaErrors(cudaMalloc((void**)&Dv_cap, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&Dv_pre_cap, par_host->ParticleNum * sizeof(double3)));

	checkCudaErrors(cudaMalloc((void**)&Dt_cap, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&Dt_pre_cap, par_host->ParticleNum * sizeof(double3)));

	checkCudaErrors(cudaMalloc((void**)&Dr_cap, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&Dr_pre_cap, par_host->ParticleNum * sizeof(double3)));

	checkCudaErrors(cudaMalloc((void**)&Dv_up, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&Dv_pre_up, par_host->ParticleNum * sizeof(double3)));

	checkCudaErrors(cudaMalloc((void**)&Dt_up, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&Dt_pre_up, par_host->ParticleNum * sizeof(double3)));

	checkCudaErrors(cudaMalloc((void**)&Dr_up, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&Dr_pre_up, par_host->ParticleNum * sizeof(double3)));

	checkCudaErrors(cudaMalloc((void**)&Dv_side, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&Dv_pre_side, par_host->ParticleNum * sizeof(double3)));

	checkCudaErrors(cudaMalloc((void**)&Dt_side, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&Dt_pre_side, par_host->ParticleNum * sizeof(double3)));

	checkCudaErrors(cudaMalloc((void**)&Dr_side, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&Dr_pre_side, par_host->ParticleNum * sizeof(double3)));

	checkCudaErrors(cudaMemset(Fc, 0.0, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMemset(Tc, 0.0, par_host->ParticleNum * sizeof(double3)));

	checkCudaErrors(cudaMemset(Dv_cap, 0.0, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMemset(Dv_pre_cap, 0.0, par_host->ParticleNum * sizeof(double3)));

	checkCudaErrors(cudaMemset(Dt_cap, 0.0, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMemset(Dt_pre_cap, 0.0, par_host->ParticleNum * sizeof(double3)));

	checkCudaErrors(cudaMemset(Dr_cap, 0.0, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMemset(Dr_pre_cap, 0.0, par_host->ParticleNum * sizeof(double3)));

	checkCudaErrors(cudaMemset(Dv_up, 0.0, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMemset(Dv_pre_up, 0.0, par_host->ParticleNum * sizeof(double3)));

	checkCudaErrors(cudaMemset(Dt_up, 0.0, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMemset(Dt_pre_up, 0.0, par_host->ParticleNum * sizeof(double3)));

	checkCudaErrors(cudaMemset(Dr_up, 0.0, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMemset(Dr_pre_up, 0.0, par_host->ParticleNum * sizeof(double3)));

	checkCudaErrors(cudaMemset(Dv_side, 0.0, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMemset(Dv_pre_side, 0.0, par_host->ParticleNum * sizeof(double3)));

	checkCudaErrors(cudaMemset(Dt_side, 0.0, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMemset(Dt_pre_side, 0.0, par_host->ParticleNum * sizeof(double3)));

	checkCudaErrors(cudaMemset(Dr_side, 0.0, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMemset(Dr_pre_side, 0.0, par_host->ParticleNum * sizeof(double3)));

	//
	checkCudaErrors(cudaMemcpy(wall_device, wall_io->GetWallParameters(), sizeof(Cylinder), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(Pos, wall_io->GetPos(), sizeof(double3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(Vel, wall_io->GetVel(), sizeof(double3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(Quat, wall_io->GetQuat(), sizeof(double4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(AngSpd, wall_io->GetAngSpd(), sizeof(double3), cudaMemcpyHostToDevice));

	/*PrintDEMWallCylinder_cuk << <1, 1 >> > (Vel);
	cudaDeviceSynchronize();*/

}