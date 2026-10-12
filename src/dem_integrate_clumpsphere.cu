#include"dem_clumpsphere.h"

__global__ void Integrate_Before_clump_cuk(Parameters* par, double3* Pos, double3* Vel, double3* AngSpd, double4* Property,
									       double3* totF, double3* totT)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= par->ClumpNum) return;

	double Dt;
	double3 dist, F, T;
	double3 pos, vel, angspd, angspd_be;

	pos = Pos[i];
	vel = Vel[i];
	angspd = AngSpd[i];

	angspd_be = angspd;

	Dt = par->StepSize;
	F = totF[i];
	T = totT[i];

	F = (1.0 / Property[i].w) * F;
	F.z = F.z + par->Gravity;

	T.x = (1.0 / Property[i].x) * T.x;
	T.y = (1.0 / Property[i].y) * T.y;
	T.z = (1.0 / Property[i].z) * T.z;


	dist = Dt * vel + 0.5 * Dt * Dt * F;
	pos = pos + dist;
	vel = vel + 0.5 * Dt * F;
	angspd = angspd + 0.5 * Dt * T;


	Pos[i] = pos;
	Vel[i] = vel;
	AngSpd[i] = angspd;

	//if (Length(T) != 0) printf("T1: %f %f %f %f %f %f\n", angspd_be.x, angspd_be.y, angspd_be.z, angspd.x, angspd.y, angspd.z);

}

__global__ void Attitude_clump_cuk(Parameters* par, double3* AngSpd, double4* Quat, double4* Property, double3* totT)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= par->ClumpNum) return;

	double3 W, Wb, Wdotb, T;
	double4 quat, quatDot, quatDotDot;
	double DCM[3][3];
	double Dt, norm;

	Dt = par->StepSize;

	W = AngSpd[i];
	T = totT[i];

	T.x = (1.0 / Property[i].x) * T.x;
	T.y = (1.0 / Property[i].y) * T.y;
	T.z = (1.0 / Property[i].z) * T.z;

	quat = Quat[i];

	//printf("att1: %f %f %f %f %f %f %f\n", quat.x, quat.y, quat.z, quat.w, T.x, T.y, T.z);

	Quat2DCM(quat, DCM);
	Map(DCM, W, Wb);
	Map(DCM, T, Wdotb);

	quatDot.x = 0.50 * (Wb.x * quat.w - Wb.y * quat.z + Wb.z * quat.y);
	quatDot.y = 0.50 * (Wb.x * quat.z + Wb.y * quat.w - Wb.z * quat.x);
	quatDot.z = 0.50 * (-Wb.x * quat.y + Wb.y * quat.x + Wb.z * quat.w);
	quatDot.w = 0.50 * (-Wb.x * quat.x - Wb.y * quat.y - Wb.z * quat.z);

	quatDotDot.x = (Wdotb.x * quat.w - Wdotb.y * quat.z + Wdotb.z * quat.y) * 0.50 + (Wb.x * quatDot.w - Wb.y * quatDot.z + Wb.z * quatDot.y) * 0.50;
	quatDotDot.y = (Wdotb.x * quat.z + Wdotb.y * quat.w - Wdotb.z * quat.x) * 0.50 + (Wb.x * quatDot.z + Wb.y * quatDot.w - Wb.z * quatDot.x) * 0.50;
	quatDotDot.z = (-Wdotb.x * quat.y + Wdotb.y * quat.x + Wdotb.z * quat.w) * 0.50 + (-Wb.x * quatDot.y + Wb.y * quatDot.x + Wb.z * quatDot.w) * 0.50;
	quatDotDot.w = (-Wdotb.x * quat.x - Wdotb.y * quat.y - Wdotb.z * quat.z) * 0.50 + (-Wb.x * quatDot.x - Wb.y * quatDot.y - Wb.z * quatDot.z) * 0.50;

	quat = quat + Dt * quatDot + 0.5 * Dt * Dt * quatDotDot;
	norm = sqrt(quat.x * quat.x + quat.y * quat.y + quat.z * quat.z + quat.w * quat.w);

	quat = (1.0 / norm) * quat;

	Quat[i] = quat;

	//printf("att2: %f %f %f %f %f %f %f\n", quat.x, quat.y, quat.z, quat.w, T.x, T.y, T.z);
}



void DEMClumpSphere::Integrate_Before(Parameters* par_host, Parameters* par_device, double3* totF, double3* totT)
{
	dim3 block(128, 1);
	dim3 cuda_grid_size((par_host->ClumpNum + block.x - 1) / block.x, 1);

	Integrate_Before_clump_cuk << < cuda_grid_size, block >> > (par_device, Pos, Vel, AngSpd, Property,
																totF, totT);
	getLastCudaError("Integrate_Before_clump_cuk failed...     \n");
	cudaDeviceSynchronize();


	Attitude_clump_cuk << < cuda_grid_size, block >> > (par_device, AngSpd, Quat, Property, totT);
	getLastCudaError("Attitude_clump_cuk failed...     \n");
	cudaDeviceSynchronize();
}

__global__ void Integrate_After_clump_cuk(Parameters* par, double3* Vel, double3* AngSpd, double4* Property, double3* totF, double3* totT)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= par->ClumpNum) return;

	double Dt;
	double3 F, T, angspd, angspd_be;

	angspd_be = AngSpd[i];


	Dt = par->StepSize;
	F = totF[i];
	T = totT[i];

	F = (1.0 / Property[i].w) * F;
	F.z = F.z + par->Gravity;

	T.x = (1.0 / Property[i].x) * T.x;
	T.y = (1.0 / Property[i].y) * T.y;
	T.z = (1.0 / Property[i].z) * T.z;

	Vel[i] = Vel[i] + 0.5 * Dt * F;
	AngSpd[i] = AngSpd[i] + 0.5 * Dt * T;

	angspd = AngSpd[i];

	//if(Length(T)!=0) printf("T2: %f %f %f %f %f %f\n", angspd_be.x, angspd_be.y, angspd_be.z, angspd.x, angspd.y, angspd.z);

	//printf("afterT: %f %f %f %f %f %f\n", sortedAngSpd[i].x, sortedAngSpd[i].y, sortedAngSpd[i].z, AngSpd[i].x, AngSpd[i].y, AngSpd[i].z);

}


void DEMClumpSphere::Integrate_After(Parameters* par_host, Parameters* par_device, double3* totF, double3* totT)
{
	dim3 block(128, 1);
	dim3 cuda_grid_size((par_host->ParticleNum + block.x - 1) / block.x, 1);

	Integrate_After_clump_cuk << <cuda_grid_size, block >> > (par_device, Vel, AngSpd, Property, totF, totT);
	cudaDeviceSynchronize();
}
