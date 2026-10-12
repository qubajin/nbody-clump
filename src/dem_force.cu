#include"dem_force.h"

#include <helper_cuda.h>
#include <cub/cub.cuh>

DEMForce::DEMForce()
{
	pitch = 0;

	Fc = NULL;
	Tc = NULL;

	CacheFc = NULL;
	CacheTc = NULL;
	CacheTc2 = NULL;



}

DEMForce::~DEMForce()
{
	if (Fc != NULL) cudaFree(Fc);
	if (Tc != NULL) cudaFree(Tc);

	if (CacheFc != NULL) cudaFree(CacheFc);
	if (CacheTc != NULL) cudaFree(CacheTc);
	if (CacheTc2 != NULL) cudaFree(CacheTc2);

}

void DEMForce::ForceConfig(Parameters* par_host, GlobalFlg* Glb)
{
	checkCudaErrors(cudaMalloc((void**)&Fc, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&Tc, par_host->ParticleNum * sizeof(double3)));

	checkCudaErrors(cudaMalloc((void**)&Dv, par_host->ParticleNum * par_host->BranchLmtLen / 2 * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&Dv_pre, par_host->ParticleNum * par_host->BranchLmtLen / 2 * sizeof(double3)));

	checkCudaErrors(cudaMalloc((void**)&Dt, par_host->ParticleNum * par_host->BranchLmtLen / 2 * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&Dt_pre, par_host->ParticleNum * par_host->BranchLmtLen / 2 * sizeof(double3)));

	checkCudaErrors(cudaMalloc((void**)&Dr, par_host->ParticleNum * par_host->BranchLmtLen / 2 * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&Dr_pre, par_host->ParticleNum * par_host->BranchLmtLen / 2 * sizeof(double3)));

	checkCudaErrors(cudaMemset(Dv, 0.0, par_host->ParticleNum * par_host->BranchLmtLen / 2 * sizeof(double3)));
	checkCudaErrors(cudaMemset(Dv_pre, 0.0, par_host->ParticleNum * par_host->BranchLmtLen / 2 * sizeof(double3)));

	checkCudaErrors(cudaMemset(Dt, 0.0, par_host->ParticleNum * par_host->BranchLmtLen / 2 * sizeof(double3)));
	checkCudaErrors(cudaMemset(Dt_pre, 0.0, par_host->ParticleNum * par_host->BranchLmtLen / 2 * sizeof(double3)));

	checkCudaErrors(cudaMemset(Dr, 0.0, par_host->ParticleNum * par_host->BranchLmtLen / 2 * sizeof(double3)));
	checkCudaErrors(cudaMemset(Dr_pre, 0.0, par_host->ParticleNum * par_host->BranchLmtLen / 2 * sizeof(double3)));

	checkCudaErrors(cudaMalloc((void**)&CacheFc, par_host->ParticleNum * (par_host->BranchLmtLen / 2) * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&CacheTc, par_host->ParticleNum * (par_host->BranchLmtLen / 2) * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&CacheTc2, par_host->ParticleNum * (par_host->BranchLmtLen / 2) * sizeof(double3)));

	//
	checkCudaErrors(cudaMalloc((void**)&CacheFcn, par_host->ParticleNum * (par_host->BranchLmtLen / 2) * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&CacheCtp, par_host->ParticleNum * (par_host->BranchLmtLen / 2) * sizeof(double3)));

	checkCudaErrors(cudaMallocPitch(&ChainFcn, &pitch, par_host->BranchLmtLen * sizeof(double3), par_host->ParticleNum));
	checkCudaErrors(cudaMallocPitch(&ChainCtp, &pitch, par_host->BranchLmtLen * sizeof(double3), par_host->ParticleNum));

}

__global__ void kernel_Dt(double3* Dt, double3* Dt_pre)
{
	int id = 521;
	printf("%f %f %f %f %f %f\n", Dt_pre[id].x, Dt_pre[id].y, Dt_pre[id].z, Dt[id].x, Dt[id].y, Dt[id].z);
}


void DEMForce::ForceZero(Parameters* par_host, GlobalFlg* Glb)
{
	checkCudaErrors(cudaMemset(Fc, 0.0, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMemset(Tc, 0.0, par_host->ParticleNum * sizeof(double3)));

	checkCudaErrors(cudaMemset(CacheFc, 0.0, par_host->ParticleNum * (par_host->BranchLmtLen / 2) * sizeof(double3)));
	checkCudaErrors(cudaMemset(CacheTc, 0.0, par_host->ParticleNum * (par_host->BranchLmtLen / 2) * sizeof(double3)));
	checkCudaErrors(cudaMemset(CacheTc2, 0.0, par_host->ParticleNum * (par_host->BranchLmtLen / 2) * sizeof(double3)));

	checkCudaErrors(cudaMemset(CacheFcn, 0.0, par_host->ParticleNum * (par_host->BranchLmtLen / 2) * sizeof(double3)));
	checkCudaErrors(cudaMemset(CacheCtp, 0.0, par_host->ParticleNum * (par_host->BranchLmtLen / 2) * sizeof(double3)));
}

void DEMForce::ForceExchange(Parameters* par_host)
{
	pointerExch(&Dv, &Dv_pre);
	pointerExch(&Dt, &Dt_pre);
	pointerExch(&Dr, &Dr_pre);

	checkCudaErrors(cudaMemset(Dv, 0.0, par_host->ParticleNum * par_host->BranchLmtLen / 2 * sizeof(double3)));
	checkCudaErrors(cudaMemset(Dt, 0.0, par_host->ParticleNum * par_host->BranchLmtLen / 2 * sizeof(double3)));
	checkCudaErrors(cudaMemset(Dr, 0.0, par_host->ParticleNum * par_host->BranchLmtLen / 2 * sizeof(double3)));

}


void DEMForce::pointerExch(double3** pt1, double3** pt2)
{
	double3* tmp;
	tmp = *pt1;
	*pt1 = *pt2;
	*pt2 = tmp;
}
