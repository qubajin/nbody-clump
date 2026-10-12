#include "dem_force.h"
#include <helper_cuda.h>

__global__ void ContactForce_cuk(Parameters* par, int* Lpa, int* Lpb, double3* Pos, double3* Vel,
                                 double3* AngSpd, double4* Property, double* Radius,
                                 double3* Dv_pre, double3* Dv, double3* Dt_pre, double3* Dt,
                                 double3* Dr_pre, double3* Dr, double3* CacheFc, double3* CacheTc,
                                 double3* CacheTc2, double3* CacheFcn, double3* CacheCpt,
                                 int* part_fix) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int cp = par->ParticleNum * (par->BranchLmtLen / 2);

    // Check the index before accessing contact arrays.
    if (i >= cp)
        return;
    if (Lpa[i] < 0 || Lpb[i] < 0)
        return;

    double3 Dist, DistU, Vrel, Vrot, Vtot, Vnor, Vtan;
    double3 normal_force, tangential_force, tangential_history;
    double3 rolling_moment, rolling_history;
    double3 twisting_moment, twisting_history;
    double3 cohesive_force, gravity_force;
    double3 Ds, Dtheta, DthetaR, DthetaT, H, Mr, Mt, RV;

    double DistS, DistL, DistR, normal_forceL, tangential_forceL, tangential_historyL;
    double rolling_momentL, rolling_historyL, twisting_momentL, twisting_historyL;
    double Ap, An, Rij, Mij, Iij, Kn, Cn, Ks, Cs, Kr, Cr, Kt, Ct, lnCOR;
    double Dn, DsL, DthetaL, DthetaRL, DthetaTL, ERR;

    bool slipping, rolling, twisting, touching;
    int particleI, particleJ;

    particleI = Lpa[i];
    particleJ = Lpb[i];

    Dist = Pos[particleJ] - Pos[particleI];
    H = Dv_pre[i];
    Mr = Dr_pre[i];
    Mt = Dt_pre[i];

    touching = false;
    slipping = false;
    rolling = false;
    twisting = false;

    DistS = Dist.x * Dist.x + Dist.y * Dist.y + Dist.z * Dist.z;
    DistL = sqrt(DistS);
    Dn = Radius[particleI] + Radius[particleJ] - DistL;

    if (Dn > 0.0) {
        /*if (Radius[particleI] != Radius[particleJ])
        {
            printf("id: %d %d %d %d\n", particleI, particleJ, part_fix[particleI],
        part_fix[particleJ]);
        }*/

        // if (Dn > 0.02 * Radius[particleI] || Dn > 0.02 * Radius[particleJ])
        //{
        //     printf("id: %d %d %d %d %f %f %f\n", particleI, particleJ, part_fix[particleI],
        //     part_fix[particleJ],
        //                                                      Dn, Radius[particleI],
        //                                                      Radius[particleJ]);
        //     /*printf("id: %d %d %d %d %f %f %f %f %f %f %f\n", particleI, particleJ,
        //     part_fix[particleI], part_fix[particleJ],
        //                                                      Dn, Pos[particleI].x,
        //                                                      Pos[particleI].y, Pos[particleI].z,
        //                                                          Pos[particleJ].x,
        //                                                          Pos[particleJ].y,
        //                                                          Pos[particleJ].z);*/
        // }

        DistR = 1.0 / DistL;
        DistU = DistR * Dist;
        Ap = 0.50 *
             (Radius[particleI] * Radius[particleI] - Radius[particleJ] * Radius[particleJ] +
              DistS) *
             DistR;
        An = DistL - Ap;
        Rij = Radius[particleI] * Radius[particleJ] / (Radius[particleI] + Radius[particleJ]);
        Mij = Property[particleI].w * Property[particleJ].w /
              (Property[particleI].w + Property[particleJ].w);
        Kn = 2.0 * par->Ymod * sqrt(Rij * Dn) / (3.0 * (1.0 - par->Prat * par->Prat));
        Cn = par->Damping * sqrt(Kn * Mij);

        Ks = 2.0 * par->Ymod / (1.0 + par->Prat) / (2.0 - par->Prat) * sqrt(Rij * Dn);
        Cs = par->Damping * sqrt(Ks * Mij);

        Kr = 0.25 * Kn * (par->Beta * Rij) * (par->Beta * Rij);
        Cr = 0.25 * Cn * (par->Beta * Rij) * (par->Beta * Rij);
        Kt = 0.50 * Ks * (par->Beta * Rij) * (par->Beta * Rij);
        Ct = 0.50 * Cs * (par->Beta * Rij) * (par->Beta * Rij);

        Vrel = Vel[particleJ] - Vel[particleI];
        Vrot = An * Cross(DistU, AngSpd[particleJ]) + Ap * Cross(DistU, AngSpd[particleI]);
        Vtot = Vrel + Vrot;

        ERR = Dot(DistU, Vtot);
        Vnor = ERR * DistU;
        Vtan = Vtot - Vnor;

        normal_force = Kn * Dn * DistU + Cn * Vnor;
        normal_forceL = Length(normal_force);

        double3 tmp1 = Kn * Dn * DistU;
        double3 tmp2 = Cn * Vnor;

        // printf("%f %f %f %f %f\n", tmp1.x, tmp2.x, Vnor.x, Cn, par->Damping);

        // printf("Fc: %f %f %f %f %f %f %f\n", -tmp1.x, -tmp1.y, -tmp1.z, -tmp2.x, -tmp2.y,
        // -tmp2.z, Dn);

        // if (Length(AngSpd[particleJ]) > 50.0 || Length(AngSpd[particleI]) > 50.0)
        //{
        //     printf("Fc: %f %f %f %f %f %f %f %f %f %f\n", normal_forceL, tmp1.x, tmp1.y, tmp1.z,
        //     tmp2.x, tmp2.y, tmp2.z, Vtot.x, Vtot.y, Vtot.z);
        //     //printf("Fc: %f %f %f %f %f %f %f\n", normal_forceL, Vtan.x, Vtan.y, Vtan.z, Vnor.x,
        //     Vnor.y, Vnor.z);
        // }

        Ds = par->StepSize * Vtan;
        DsL = Length(Ds);

        tangential_force = -Ks * Ds + Cs * Vtan + H;
        tangential_forceL = Length(tangential_force);

        if (tangential_forceL > normal_forceL * par->Frcoe) {
            tangential_force =
                par->Frcoe * normal_forceL * (1.0 / tangential_forceL) * tangential_force;
            tangential_history = tangential_force;
        } else {
            tangential_history = tangential_force - Cs * Vtan;
        }

        cohesive_force = par->Cohesion * (par->Beta * Rij) * (par->Beta * Rij) * DistU;

        CacheFc[i] = -1.0 * normal_force - tangential_force + cohesive_force;
        CacheTc[i] = -Ap * Cross(DistU, tangential_force);
        CacheTc2[i] = -An * Cross(DistU, tangential_force);

        Dtheta = par->StepSize * (AngSpd[particleI] - AngSpd[particleJ]);
        DthetaL = Dot(Dtheta, DistU);

        DthetaT = DthetaL * DistU;
        DthetaTL = Length(DthetaT);

        DthetaR = Dtheta - DthetaT;
        DthetaRL = Length(DthetaR);

        rolling_moment = Kr * DthetaR + Mr;
        rolling_momentL = Length(rolling_moment);

        if (rolling_momentL > 0.525 * par->Beta * Rij * normal_forceL) {
            rolling_moment =
                0.525 * par->Beta * Rij * normal_forceL * (1.0 / rolling_momentL) * rolling_moment;
            rolling_history = rolling_moment;
        } else {
            rolling_history = rolling_moment;
            rolling_moment = rolling_moment - Cr * (1.0 / par->StepSize) * DthetaR;
        }

        twisting_moment = Kt * DthetaT + Mt;
        twisting_momentL = Length(twisting_moment);

        if (twisting_momentL > 0.65 * par->Frcoe * par->Beta * Rij * normal_forceL) {
            twisting_moment = 0.65 * par->Frcoe * par->Beta * Rij * normal_forceL *
                              (1.0 / twisting_momentL) * twisting_moment;
            twisting_history = twisting_moment;
        } else {
            twisting_history = twisting_moment;
            twisting_moment = twisting_moment - Ct * (1.0 / par->StepSize) * DthetaT;
        }

        CacheTc[i] = CacheTc[i] - rolling_moment - twisting_moment;
        CacheTc2[i] = CacheTc2[i] + rolling_moment + twisting_moment;

        Dv[i] = tangential_history;
        Dt[i] = twisting_history;
        Dr[i] = rolling_history;

        CacheFcn[i] = -1.0 * normal_force;
        CacheCpt[i] = Pos[particleI] + Ap * DistU;

        /*if (Radius[particleI] != Radius[particleJ])
            printf("CacheFc[i]: %d %d %f %f %f\n", particleI, particleJ, CacheFc[i].x, CacheFc[i].y,
           CacheFc[i].z);*/
    }
}

__global__ void SumContForce(Parameters* par, double3* CacheFc, double3* CacheTc, double3* CacheTc2,
                             double3* Fc, double3* Tc, int* Aref, int* Ng, int* Ni, size_t pitch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= par->ParticleNum)
        return;

    int Cmax = par->BranchLmtLen;
    int* row_Aref = (int*)((char*)Aref + i * pitch);

    for (int Igrid = 0; Igrid < Ng[i]; Igrid++) {
        Fc[i] = Fc[i] + CacheFc[row_Aref[Igrid]];
        Tc[i] = Tc[i] + CacheTc[row_Aref[Igrid]];
        // if (Length(Fc[i]) != 0) printf("i: %d %d Fc: %f %f %f\n", i, Igrid, Fc[i].x, Fc[i].y,
        // Fc[i].z);
    }

    for (int Igrid = Cmax - Ni[i]; Igrid < Cmax; Igrid++) {
        Fc[i] = Fc[i] - CacheFc[row_Aref[Igrid]];
        Tc[i] = Tc[i] + CacheTc2[row_Aref[Igrid]]; //!!!!!!!!
        // if (Length(Fc[i]) != 0) printf("i: %d %d Fc: %f %f %f\n", i, Igrid, Fc[i].x, Fc[i].y,
        // Fc[i].z);
    }

    // if(Length(Fc[i])!=0)
}

void DEMForce::CalcContactForce(Parameters* par_host, Parameters* par_device, int* Lpa, int* Lpb,
                                double3* Pos, double3* Vel, double3* AngSpd, double4* Quat,
                                double4* Property, double* Radius, int* Aref, int* Ng, int* Ni,
                                size_t pitch, int* part_fix) {
    int cp = par_host->ParticleNum * (par_host->BranchLmtLen / 2);

    dim3 block(128, 1);
    dim3 cuda_grid_size = dim3((cp + block.x - 1) / block.x, 1);

    ContactForce_cuk<<<cuda_grid_size, block, 0>>>(
        par_device, Lpa, Lpb, Pos, Vel, AngSpd, Property, Radius, Dv_pre, Dv, Dt_pre, Dt, Dr_pre,
        Dr, CacheFc, CacheTc, CacheTc2, CacheFcn, CacheCtp, part_fix);
    cudaDeviceSynchronize();

    cuda_grid_size.x = (par_host->ParticleNum + block.x - 1) / block.x;

    SumContForce<<<cuda_grid_size, block, 0>>>(par_device, CacheFc, CacheTc, CacheTc2, Fc, Tc, Aref,
                                               Ng, Ni, pitch);
    cudaDeviceSynchronize();
}

__global__ void CalFcnChain_cuk(Parameters* par, int* Aref, int* Ng, int* Ni, size_t pitch_int,
                                size_t pitch_double3, double3* CacheFcn, double3* CacheCtp,
                                double3* ChainFcn, double3* ChainCtp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= par->ParticleNum)
        return;

    int Cmax = par->BranchLmtLen;

    int* row_Aref = (int*)((char*)Aref + i * pitch_int);

    double3* row_ChainFcn = (double3*)((char*)ChainFcn + i * pitch_double3);
    double3* row_ChainCtp = (double3*)((char*)ChainCtp + i * pitch_double3);

    for (int Igrid = 0; Igrid < Ng[i]; Igrid++) {
        row_ChainFcn[Igrid] = CacheFcn[row_Aref[Igrid]];
        row_ChainCtp[Igrid] = CacheCtp[row_Aref[Igrid]];
        // if (i == 3128) printf("i: %d %d %f %f %f %f %f %f\n", i, row_Aref[Igrid],
        // row_ChainFcn[Igrid].x, row_ChainFcn[Igrid].y, row_ChainFcn[Igrid].z,
        // row_ChainCtp[Igrid].x, row_ChainCtp[Igrid].y, row_ChainCtp[Igrid].z);
    }

    for (int Igrid = Cmax - Ni[i]; Igrid < Cmax; Igrid++) {
        row_ChainFcn[Igrid] = -1.0 * CacheFcn[row_Aref[Igrid]];
        row_ChainCtp[Igrid] = CacheCtp[row_Aref[Igrid]];
        // if (i == 3128) printf("i: %d %d %f %f %f %f %f %f\n", i, row_Aref[Igrid],
        // row_ChainFcn[Igrid].x, row_ChainFcn[Igrid].y, row_ChainFcn[Igrid].z,
        // row_ChainCtp[Igrid].x, row_ChainCtp[Igrid].y, row_ChainCtp[Igrid].z);
    }
}

void DEMForce::CalcFcnChain(Parameters* par_host, Parameters* par_device, int* Aref, int* Ng,
                            int* Ni, size_t pitch_int) {
    dim3 block(128, 1);
    dim3 cuda_grid_size((par_host->ParticleNum + block.x - 1) / block.x, 1);

    checkCudaErrors(cudaMemset2D(ChainFcn, pitch, 0.0, par_host->BranchLmtLen * sizeof(double3),
                                 par_host->ParticleNum));
    checkCudaErrors(cudaMemset2D(ChainCtp, pitch, 0.0, par_host->BranchLmtLen * sizeof(double3),
                                 par_host->ParticleNum));

    CalFcnChain_cuk<<<cuda_grid_size, block, 0>>>(par_device, Aref, Ng, Ni, pitch_int, pitch,
                                                  CacheFcn, CacheCtp, ChainFcn, ChainCtp);
    getLastCudaError("CalFcnChain_cuk failed...\n");
    checkCudaErrors(cudaDeviceSynchronize());
}
