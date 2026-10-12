#include"dem_penetrator.h"
#include"dem_sphere_intruder.cuh"
#include <cooperative_groups.h>

using namespace cooperative_groups;


__global__ void ForcePene_cuk(Parameters* par, double3* Pos, double3* Vel, double3* AngSpd, double* Radius, double4* Property,
                              Cylinder* cyd, double3* PenePos, double3* PeneVel, double3* PeneAngSpd,
                              double3* Dv_pre, double3* Dv, double3* Dt_pre, double3* Dt, double3* Dr_pre, double3* Dr,
                              int* LcFlg, double3* RVo, double3* RVn, double3* Fc, double3* Tc, double3* peneTc)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int cp = par->ParticleNum;

    if (i >= cp) return;

    if (LcFlg[i] == 0) return;

	
    double3 Dist, DistU, Vrel, Vrot, Vtot, Vnor, Vtan;
    double3 normal_force, tangential_force, tangential_history;
    double3 rolling_moment, rolling_history;
    double3 twisting_moment, twisting_history;
    double3 cohesive_force;
    double3 Ds, Dtheta, DthetaR, DthetaT, H, Mr, Mt;
    double3 Vwall;

    double DistS, DistL, DistR, normal_forceL, tangential_forceL, tangential_historyL;
    double rolling_momentL, rolling_historyL, twisting_momentL, twisting_historyL;
    double Ap, Rij, Mij, Kn, Cn, Ks, Cs, Kr, Cr, Kt, Ct;
    double Dn, DsL, DthetaL, DthetaRL, DthetaTL, ERR;

    double Ymod = par->Ymod;
	SphereIntruderContact sphereContact;

 
    Dist = RVn[i];
    H = Dv_pre[i];
    Mr = Dr_pre[i];
    Mt = Dt_pre[i];

    DistS = Dist.x * Dist.x + Dist.y * Dist.y + Dist.z * Dist.z;
    DistL = sqrt(DistS);
    Dn = Radius[i] - DistL;
	if (cyd->Shape == PENETRATOR_SPHERE)
	{
		sphereContact = GetSphereIntruderContact(
			Pos[i], Radius[i], PenePos[0], cyd->Radius, cyd->Dirct);
		Dist = sphereContact.normalGapVector;
		DistL = sphereContact.particleArm;
		Dn = sphereContact.overlap;
	}

    if (Dn > 0.0)
    {
		if (cyd->Shape == PENETRATOR_SPHERE)
		{
			DistU = sphereContact.normal;
		}
		else
		{
			DistR = 1.0 / DistL;
			DistU = DistR * Dist;
		}
        Ap = DistL;
        Rij = Radius[i];
		if (cyd->Shape == PENETRATOR_SPHERE)
		{
			Rij = Radius[i] * cyd->Radius / (Radius[i] + cyd->Radius);
		}
        Mij = Property[i].w;
        if (cyd->MotionMode == PENETRATOR_GRAVITY)
        {
            Mij = Property[i].w * cyd->Property.w /
                  (Property[i].w + cyd->Property.w);
        }
        Kn = 2.0 * Ymod * sqrt(Rij * Dn) / (3.0 * (1.0 - par->Prat * par->Prat));
        Cn = par->Damping * sqrt(Kn * Mij);

        Ks = 2.0 * Ymod / (1.0 + par->Prat) / (2.0 - par->Prat) * sqrt(Rij * Dn);
        Cs = par->Damping * sqrt(Ks * Mij);

        Kr = 0.25 * Kn * (par->Beta * Rij) * (par->Beta * Rij);
        Cr = 0.25 * Cn * (par->Beta * Rij) * (par->Beta * Rij);
        Kt = 0.50 * Ks * (par->Beta * Rij) * (par->Beta * Rij);
        Ct = 0.50 * Cs * (par->Beta * Rij) * (par->Beta * Rij);

        Vwall = Cross(PeneAngSpd[0], RVo[i]);
        Vrel = Vel[i] - PeneVel[0];
        Vrot = Ap * Cross(DistU, AngSpd[i]) - Vwall;
        Vtot = Vrel + Vrot;

        //printf("Vwall: %f %f %f %f %f %f %f %f %f\n", Vwall.x, Vwall.y, Vwall.z, Vrot.x, Vrot.y, Vrot.z, Vtot.x, Vtot.y, Vtot.z);

        ERR = Dot(DistU, Vtot);
        Vnor = ERR * DistU;
        Vtan = Vtot - Vnor;

        normal_force = Kn * Dn * DistU + Cn * Vnor;
        normal_forceL = Length(normal_force);

        Ds = par->StepSize * Vtan;
        DsL = Length(Ds);

        tangential_force = -Ks * Ds + Cs * Vtan + H;
        tangential_forceL = Length(tangential_force);

        if (tangential_forceL > normal_forceL * par->Frcoe)
        {
            tangential_force = par->Frcoe * normal_forceL * (1.0 / tangential_forceL) * tangential_force;
            tangential_history = tangential_force;
        }
        else
        {
            tangential_history = tangential_force - Cs * Vtan;
        }

        cohesive_force = - par->Cohesion * (par->Beta * Rij) * (par->Beta * Rij) * DistU;

        Fc[i] = normal_force + tangential_force + cohesive_force;
        Tc[i] = -Ap * Cross(DistU, tangential_force);
        peneTc[i] = -1.0 * Cross(RVo[i], (normal_force + tangential_force + cohesive_force));

        //printf("Fc: %d %f %f %f\n", i, Fc[i].x, Fc[i].y, Fc[i].z);
       
        Dtheta = par->StepSize * (AngSpd[i] - PeneAngSpd[0]);
        DthetaL = Dot(Dtheta, DistU);

        DthetaT = DthetaL * DistU;
        DthetaTL = Length(DthetaT);

        DthetaR = Dtheta - DthetaT;
        DthetaRL = Length(DthetaR);

        rolling_moment = Kr * DthetaR + Mr;
        rolling_momentL = Length(rolling_moment);

        if (rolling_momentL > 0.525 * par->Beta * Rij * normal_forceL)
        {
            rolling_moment = 0.525 * par->Beta * Rij * normal_forceL * (1.0 / rolling_momentL) * rolling_moment;
            rolling_history = rolling_moment;
        }
        else
        {
            rolling_history = rolling_moment;
            rolling_moment = rolling_moment - Cr * (1.0 / par->StepSize) * DthetaR;
        }

        twisting_moment = Kt * DthetaT + Mt;
        twisting_momentL = Length(twisting_moment);

        if (twisting_momentL > 0.65 * par->Frcoe * par->Beta * Rij * normal_forceL)
        {
            twisting_moment = 0.65 * par->Frcoe * par->Beta * Rij * normal_forceL * (1.0 / twisting_momentL) * twisting_moment;
            twisting_history = twisting_moment;
        }
        else
        {
            twisting_history = twisting_moment;
            twisting_moment = twisting_moment - Ct * (1.0 / par->StepSize) * DthetaT;
        }

        Tc[i] = Tc[i] - rolling_moment - twisting_moment;
        peneTc[i] = peneTc[i] + rolling_moment + twisting_moment;

        Dv[i] = tangential_history;
        Dt[i] = twisting_history;
        Dr[i] = rolling_history;

    }
}

void DEMPeneCylinder::CalcContactForce(Parameters* par_host, Parameters* par_device, double3* PartPos, double3* PartVel, double3* PartAngSpd, double* PartRadius, double4* PartProperty)
{
    int cp = par_host->ParticleNum;

    dim3 block(128, 1);
    dim3 cuda_grid_size((cp + block.x - 1) / block.x, 1);

    ForcePene_cuk << <cuda_grid_size, block >> > (par_device, PartPos, PartVel, PartAngSpd, PartRadius, PartProperty,
                                                  pene_device, Pos, Vel, AngSpd,
                                                  Dv_pre, Dv, Dt_pre, Dt, Dr_pre, Dr,
                                                  LcFlg, RVo, RVn, Fc, Tc, peneTc);
    cudaDeviceSynchronize();

}

__global__ void SumForce_cuk(const double3* F, double3* Cache, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    extern __shared__ double3 s_f[];

    double3 f = make_double3(0.0, 0.0, 0.0);
    const int stride = blockDim.x * gridDim.x;

    for (int n = bid * blockDim.x + tid; n < N; n += stride)
    {
        f = f + F[n];
        //printf("stride: %d %d %d %f %f %f\n", bid, stride, n, y.x, y.y, y.z);
    }
    s_f[tid] = f;
    __syncthreads();

    //printf("f: %d %d %f %f %f\n", bid, tid, s_f[tid].x, s_f[tid].y, s_f[tid].z);

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
    {
        if (tid < offset)
        {
            s_f[tid] = s_f[tid] + s_f[tid + offset];
        }
        __syncthreads();
    }

    f = s_f[tid];

    //if (Length(f)!=0) printf("f: %d %d %f %f %f\n", bid, tid, f.x, f.y, f.z);

    thread_block_tile<32> g = tiled_partition<32>(this_thread_block());
    for (int i = g.size() >> 1; i > 0; i >>= 1)
    {
        f.x += g.shfl_down(f.x, i);
        f.y += g.shfl_down(f.y, i);
        f.z += g.shfl_down(f.z, i);
    }

    if (tid == 0)
    {
        Cache[bid] = f;
    }
}



void DEMPeneCylinder::SumForce(Parameters* par_host, double3* F, double3* finalF)
{
    int block = 128;
    int cuda_grid_size = 128;
    int nb = par_host->ParticleNum;

    SumForce_cuk << <cuda_grid_size, block, sizeof(double3)* block >> > (F, Cache, nb);
    SumForce_cuk << <1, 1024, sizeof(double3) * 1024 >> > (Cache, Cache, cuda_grid_size);

    cudaMemcpy(finalF, Cache, sizeof(double3), cudaMemcpyDeviceToDevice);

}
