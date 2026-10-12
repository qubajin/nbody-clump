#include"dem.h"

__global__ void Integrate_Before_WithGlue_cuk(Parameters* par, double3* Pos, double3* Vel, double3* AngSpd, double3* CumPos, double* Radius, double4* Property,
											  double3* Fc, double3* Tc, double3* Fw, double3* Tw, double3* Fi, double3* Ti, int* Type, int* part_fix)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= par->ParticleNum || Type[i] == 2) return;

	double Dt;
	double3 dist, F, T;
	double3 pos, vel, angspd, cumpos;

	pos = Pos[i];
	vel = Vel[i];
	angspd = AngSpd[i];
	cumpos = CumPos[i];

	Dt = par->StepSize;

	if (Type[i] == 0)
	{
		
		F = Fc[i] + Fw[i];
		T = Tc[i] + Tw[i];

		//printf("beforeT: %f %f %f\n", Tw[i].x, Tw[i].y, Tw[i].z);

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

		if (Length(cumpos) > 0.10 * Radius[i] && par->Update == 0)
		{
			atomicAdd(&par->Update, 1);
		}	
	}

	if(Type[i] == 1)
	{
		double Omega = 0.2;

		vel.x = -Omega * pos.y;
		vel.y = -Omega * pos.x;
		vel.z = 0.0;

		dist = Dt * vel;
		cumpos = cumpos + dist;
		pos = pos + dist;
	}

	Pos[i] = pos;
	Vel[i] = vel;
	AngSpd[i] = angspd;
	CumPos[i] = cumpos;

}

__global__ void Attitude_WithGlue_cuk(Parameters* par, double3* AngSpd, double4* Quat, double4* Property, double3* Tc, double3* Tw, double3* Ti, int* Type)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= par->ParticleNum || Type[i] != 0) return;

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


void DEMForce::Integrate_Before(Parameters* par_host, Parameters* par_device, double3* Pos, double3* Vel, double3* AngSpd, double3* CumPos, double* Radius, double4* Property,
								double3* Fw, double3* Tw, double3* Fi, double3* Ti, int* Type, int* part_fix)
{

	dim3 block(128, 1);
	dim3 cuda_grid_size((par_host->ParticleNum + block.x - 1) / block.x, 1);

	Integrate_Before_WithGlue_cuk << < cuda_grid_size, block >> > (par_device, Pos, Vel, AngSpd, CumPos, Radius, Property,
																   Fc, Tc, Fw, Tw, Fi, Ti, Type, part_fix);
	cudaDeviceSynchronize();
}

void DEMForce::Attitude(Parameters* par_host, Parameters* par_device, double3* AngSpd, double4* Quat, double4* Property, double3* Tw, double3* Ti, int* Type)
{
	dim3 block(128, 1);
	dim3 cuda_grid_size((par_host->ParticleNum + block.x - 1) / block.x, 1);

	Attitude_WithGlue_cuk << < cuda_grid_size, block >> > (par_device, AngSpd, Quat, Property, Tc, Tw, Ti, Type);
	cudaDeviceSynchronize();
}

__global__ void Integrate_After_WithGlue_cuk(Parameters* par, double3* Vel, double3* AngSpd, double4* Property,
											 double3* Fc, double3* Tc, double3* Fw, double3* Tw, double3* Fi, double3* Ti, int* Type)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= par->ParticleNum || Type[i] != 0) return;

	double Dt;
	double3 F, T;

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

}


void DEMForce::Integrate_After(Parameters* par_host, Parameters* par_device, double3* Vel, double3* AngSpd, double4* Property,
							   double3* Fw, double3* Tw, double3* Fi, double3* Ti, int* Type)
{
	dim3 block(128, 1);
	dim3 cuda_grid_size((par_host->ParticleNum + block.x - 1) / block.x, 1);

	Integrate_After_WithGlue_cuk << <cuda_grid_size, block >> > (par_device, Vel, AngSpd, Property,
																 Fc, Tc, Fw, Tw, Fi, Ti, Type);
	cudaDeviceSynchronize();
}


