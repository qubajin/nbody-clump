#include"dem.h"

__global__ void Integrate_Before_cuk(Parameters* par, double3* Pos, double3* Vel, double3* AngSpd, double3* CumPos, double* Radius, double4* Property,
								     double3* Fc, double3* Tc, double3* Fw, double3* Tw, double3* Fi, double3* Ti, int* part_fix)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= par->ParticleNum) return;

	double Dt;
	double3 dist, F, T;
	double3 pos, vel, angspd, cumpos, angspd_be;

	pos = Pos[i];
	vel = Vel[i];
	angspd = AngSpd[i];
	cumpos = CumPos[i];

	angspd_be = angspd;
	
	Dt = par->StepSize;
	F = Fc[i] + Fw[i];
	T = Tc[i] + Tw[i];

	F = (1.0 / Property[i].w) * F;
	F.z = F.z + par->Gravity;

	T.x = (1.0 / Property[i].x) * T.x;
	T.y = (1.0 / Property[i].y) * T.y;
	T.z = (1.0 / Property[i].z) * T.z;


	dist = Dt * vel + 0.5 * Dt * Dt * F;
	cumpos = cumpos + dist;
	pos = pos + dist;
	vel = vel + 0.5 * Dt * F;
	angspd = angspd + 0.5 * Dt * T;

	if (Length(cumpos) > par->Thresh && par->Update == 0)
	{
		atomicAdd(&par->Update, 1);
	}

	Pos[i] = pos;
	Vel[i] = vel;
	AngSpd[i] = angspd;
	CumPos[i] = cumpos;

	if(Length(T)!=0) printf("T1: %f %f %f %f %f %f\n", angspd_be.x, angspd_be.y, angspd_be.z, angspd.x, angspd.y, angspd.z);

	//printf("pos: %f %f %f %f %f %f\n", pos.x, pos.y, pos.z, angspd.x, angspd.y, angspd.z);

	/*if (Length(Fi[i]) > 0.0)
		printf("F: %d %f %f %f %f %f %f %f %f %f\n", part_fix[i], Fc[i].x, Fc[i].y, Fc[i].z, Fi[i].x, Fi[i].y, Fi[i].z, vel.x, vel.y, vel.z);*/

}

__global__ void Attitude_cuk(Parameters* par, double3* AngSpd, double4* Quat, double4*  Property, double3* Tc, double3* Tw, double3* Ti)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= par->ParticleNum) return;

	double3 W, Wb, Wdotb, T;
	double4 quat, quatDot, quatDotDot;
	double DCM[3][3];
	double Dt, norm;

	Dt = par->StepSize;

	W = AngSpd[i];
	T = Tc[i] + Tw[i];

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

	quatDotDot.x = ( Wdotb.x * quat.w - Wdotb.y * quat.z + Wdotb.z * quat.y) * 0.50 + ( Wb.x * quatDot.w - Wb.y * quatDot.z + Wb.z * quatDot.y) * 0.50;
	quatDotDot.y = ( Wdotb.x * quat.z + Wdotb.y * quat.w - Wdotb.z * quat.x) * 0.50 + ( Wb.x * quatDot.z + Wb.y * quatDot.w - Wb.z * quatDot.x) * 0.50;
	quatDotDot.z = (-Wdotb.x * quat.y + Wdotb.y * quat.x + Wdotb.z * quat.w) * 0.50 + (-Wb.x * quatDot.y + Wb.y * quatDot.x + Wb.z * quatDot.w) * 0.50;
	quatDotDot.w = (-Wdotb.x * quat.x - Wdotb.y * quat.y - Wdotb.z * quat.z) * 0.50 + (-Wb.x * quatDot.x - Wb.y * quatDot.y - Wb.z * quatDot.z) * 0.50;

	quat = quat + Dt * quatDot + 0.5 * Dt * Dt * quatDotDot;
	norm = sqrt(quat.x * quat.x + quat.y * quat.y + quat.z * quat.z + quat.w * quat.w);

	quat = (1.0 / norm) * quat;

	Quat[i] = quat;

	//printf("att2: %f %f %f %f %f %f %f\n", quat.x, quat.y, quat.z, quat.w, T.x, T.y, T.z);
}


void DEMForce::Integrate_Before(Parameters* par_host, Parameters* par_device, double3* Pos, double3* Vel, double3* AngSpd, double3* CumPos, double* Radius, double4* Property,
							    double3* Fw, double3* Tw, double3* Fi, double3* Ti, int* part_fix)
{

	dim3 block(128, 1);
	dim3 cuda_grid_size((par_host->ParticleNum + block.x - 1) / block.x, 1);

	Integrate_Before_cuk << < cuda_grid_size, block >> > (par_device, Pos, Vel, AngSpd, CumPos, Radius, Property,
														  Fc, Tc, Fw, Tw, Fi, Ti, part_fix);
	getLastCudaError("Integrate_Before_cuk failed...     \n");
	cudaDeviceSynchronize();
}

void DEMForce::Attitude(Parameters* par_host, Parameters* par_device, double3* AngSpd, double4* Quat, double4* Property, double3* Tw, double3* Ti)
{
	dim3 block(128, 1);
	dim3 cuda_grid_size((par_host->ParticleNum + block.x - 1) / block.x, 1);

	Attitude_cuk << < cuda_grid_size, block >> > (par_device, AngSpd, Quat, Property, Tc, Tw, Ti);
	cudaDeviceSynchronize();
}

__global__ void Integrate_After_cuk(Parameters* par, double3* Vel, double3* AngSpd, double4* Property,
									double3* Fc, double3* Tc, double3* Fw, double3* Tw, double3* Fi, double3* Ti)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= par->ParticleNum) return;

	double Dt;
	double3 F, T, angspd, angspd_be;

	angspd_be = AngSpd[i];

	Dt = par->StepSize;
	F = Fc[i] + Fw[i] + Fi[i];
	T = Tc[i] + Tw[i] + Ti[i];

	F = (1.0 / Property[i].w) * F;
	F.z = F.z + par->Gravity;

	T.x = (1.0 / Property[i].x) * T.x;
	T.y = (1.0 / Property[i].y) * T.y;
	T.z = (1.0 / Property[i].z) * T.z;

	Vel[i] = Vel[i] + 0.5 * Dt * F;
	AngSpd[i] = AngSpd[i] + 0.5 * Dt * T;

	angspd = AngSpd[i];


	if (Length(T) != 0) printf("T2: %f %f %f %f %f %f\n", angspd_be.x, angspd_be.y, angspd_be.z, angspd.x, angspd.y, angspd.z);


	//printf("afterT: %f %f %f %f %f %f\n", sortedAngSpd[i].x, sortedAngSpd[i].y, sortedAngSpd[i].z, AngSpd[i].x, AngSpd[i].y, AngSpd[i].z);

}


void DEMForce::Integrate_After(Parameters* par_host, Parameters* par_device, double3* Vel, double3* AngSpd, double4* Property,
							   double3* Fw, double3* Tw, double3* Fi, double3* Ti)
{
	dim3 block(128, 1);
	dim3 cuda_grid_size((par_host->ParticleNum + block.x - 1) / block.x, 1);

	Integrate_After_cuk << <cuda_grid_size, block >> > (par_device, Vel, AngSpd, Property,
														Fc, Tc, Fw, Tw, Fi, Ti);
	cudaDeviceSynchronize();
}

