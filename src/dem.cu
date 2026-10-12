#include "dem.h"

DEMmain::DEMmain()
{
	par_host = NULL;
	par_device = NULL;

	wall_io = NULL;
	wall = NULL;

	pene_io = NULL;
	pene = NULL;

	io = NULL;
	grid = NULL;
	force = NULL;

	Pos = NULL;
	Vel = NULL;
	AngSpd = NULL;
	Quat = NULL;

	CumPos = NULL;

	Radius = NULL;
	Property = NULL;

}

DEMmain::~DEMmain()
{
	if (Pos != NULL) cudaFree(Pos);
	if (Vel != NULL) cudaFree(Vel);
	if (AngSpd != NULL) cudaFree(AngSpd);
	if (Quat != NULL) cudaFree(Quat);

	if (CumPos != NULL)cudaFree(CumPos);

	if (Radius != NULL) cudaFree(Radius);
	if (Property != NULL)cudaFree(Property);

	if (Radius != NULL) cudaFree(Radius);
	if (Property != NULL) cudaFree(Property);

}


void DEMmain::AllocateGpuMemory()
{
	checkCudaErrors(cudaMalloc((void**)&par_device, sizeof(Parameters)));

	checkCudaErrors(cudaMalloc((void**)&Pos, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&Vel, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&AngSpd, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&Quat, par_host->ParticleNum * sizeof(double4)));

	checkCudaErrors(cudaMalloc((void**)&CumPos, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMemset(CumPos, 0, par_host->ParticleNum * sizeof(double3)));

	checkCudaErrors(cudaMalloc((void**)&Radius, par_host->ParticleNum * sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&Property, par_host->ParticleNum * sizeof(double4)));


	checkCudaErrors(cudaMalloc((void**)&Type, par_host->ParticleNum * sizeof(int)));

}

void DEMmain::CopyDataFromHostToDevice()
{
	checkCudaErrors(cudaMemcpy(par_device, par_host, sizeof(Parameters), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(Pos, io->GetPos(), par_host->ParticleNum * sizeof(double3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(Vel, io->GetVel(), par_host->ParticleNum * sizeof(double3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(AngSpd, io->GetAngSpd(), par_host->ParticleNum * sizeof(double3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(Quat, io->GetQuat(), par_host->ParticleNum * sizeof(double4), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(Radius, io->GetRadius(), par_host->ParticleNum * sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(Property, io->GetProperty(), par_host->ParticleNum * sizeof(double4), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(Type, io->GetType(), par_host->ParticleNum * sizeof(int), cudaMemcpyHostToDevice));
}

void DEMmain::CopyInfoToIo(size_t pitch_int, size_t pitch_double3, int* Anei, double3* ChainFcn, double3* ChainCtp)
{
	checkCudaErrors(cudaMemcpy2D(io->GetAnei(), sizeof(int) * par_host->BranchLmtLen, Anei, pitch_int, sizeof(int) * par_host->BranchLmtLen, par_host->ParticleNum, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy2D(io->GetChainFcn(), sizeof(double3) * par_host->BranchLmtLen, ChainFcn, pitch_double3, sizeof(double3) * par_host->BranchLmtLen, par_host->ParticleNum, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy2D(io->GetChainCtp(), sizeof(double3) * par_host->BranchLmtLen, ChainCtp, pitch_double3, sizeof(double3) * par_host->BranchLmtLen, par_host->ParticleNum, cudaMemcpyDeviceToHost));
}


void DEMmain::pointerExch(double3** pt1, double3** pt2)
{
	double3* tmp;
	tmp = *pt1;
	*pt1 = *pt2;
	*pt2 = tmp;
}

void DEMmain::CopyDataToIo()
{
	checkCudaErrors(cudaMemcpy(io->GetPos(), Pos, par_host->ParticleNum * sizeof(double3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(io->GetVel(), Vel, par_host->ParticleNum * sizeof(double3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(io->GetAngSpd(), AngSpd, par_host->ParticleNum * sizeof(double3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(io->GetQuat(), Quat, par_host->ParticleNum * sizeof(double4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(io->GetProperty(), Property, par_host->ParticleNum * sizeof(double4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(io->GetRadius(), Radius, par_host->ParticleNum * sizeof(double), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(io->GetParticleID(), grid->GetPartFix(), par_host->ParticleNum * sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(io->GetSortedParticleID(), grid->GetPartIdx(), par_host->ParticleNum * sizeof(int), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(io->GetType(), Type, par_host->ParticleNum * sizeof(int), cudaMemcpyDeviceToHost));

}

__global__ void PrintFc(double3* CntFc)
{
	for (int ii = 0; ii < 5000; ii++)
	{
		if (Length(CntFc[ii]) != 0)
			printf("ii: %d %f %f %f\n", ii, CntFc[ii].x, CntFc[ii].y, CntFc[ii].z);
	}
}


void DEMmain::CopyDataToIo(double3* CntFc, double3* CntTc)
{
	checkCudaErrors(cudaMemcpy(io->GetPos(), Pos, par_host->ParticleNum * sizeof(double3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(io->GetVel(), Vel, par_host->ParticleNum * sizeof(double3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(io->GetAngSpd(), AngSpd, par_host->ParticleNum * sizeof(double3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(io->GetQuat(), Quat, par_host->ParticleNum * sizeof(double4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(io->GetProperty(), Property, par_host->ParticleNum * sizeof(double4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(io->GetRadius(), Radius, par_host->ParticleNum * sizeof(double), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(io->GetParticleID(), grid->GetPartFix(), par_host->ParticleNum * sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(io->GetSortedParticleID(), grid->GetPartIdx(), par_host->ParticleNum * sizeof(int), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(io->GetType(), Type, par_host->ParticleNum * sizeof(int), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(io->GetCacheFi(), CntFc, par_host->OutputInterval * sizeof(double3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(io->GetCacheTi(), CntTc, par_host->OutputInterval * sizeof(double3), cudaMemcpyDeviceToHost));


	/*PrintFc << <1, 1 >> > (CntFc);
	cudaDeviceSynchronize();*/
}


void DEMmain::CopyDataToIo_HGrid(double3* CntFc, double3* CntTc)
{
	checkCudaErrors(cudaMemcpy(io->GetPos(), Pos, par_host->ParticleNum * sizeof(double3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(io->GetVel(), Vel, par_host->ParticleNum * sizeof(double3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(io->GetAngSpd(), AngSpd, par_host->ParticleNum * sizeof(double3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(io->GetQuat(), Quat, par_host->ParticleNum * sizeof(double4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(io->GetProperty(), Property, par_host->ParticleNum * sizeof(double4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(io->GetRadius(), Radius, par_host->ParticleNum * sizeof(double), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(io->GetParticleID(), hgrid->GetPartFix(), par_host->ParticleNum * sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(io->GetSortedParticleID(), hgrid->GetPartIdx(), par_host->ParticleNum * sizeof(int), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(io->GetType(), Type, par_host->ParticleNum * sizeof(int), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(io->GetCacheFi(), CntFc, par_host->OutputInterval * sizeof(double3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(io->GetCacheTi(), CntTc, par_host->OutputInterval * sizeof(double3), cudaMemcpyDeviceToHost));


	/*PrintFc << <1, 1 >> > (CntFc);
	cudaDeviceSynchronize();*/
}
