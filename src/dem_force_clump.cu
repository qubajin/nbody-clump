#include "dem_force.h"

__global__ void ContactForce_clump_cuk(Parameters* par, int* Lpa, int* Lpb, int* ClumpId,
                                       double3* Pos, double3* LocalPos, double* Mass,
                                       double* Radius, double3* Dv_pre, double3* Dv,
                                       double3* Dt_pre, double3* Dt, double3* Dr_pre, double3* Dr,
                                       double3* ClumpVel, double3* ClumpAngSpd, double4* ClumpQuat,
                                       double3* CacheFc, double3* CacheTc, double3* CacheTc2,
                                       double3* CacheFcn, double3* CacheCpt, int* part_fix) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int cp = par->ParticleNum * (par->BranchLmtLen / 2);

    // Check the index before accessing contact arrays.
    if (i >= cp)
        return;
    if (Lpa[i] < 0 || Lpb[i] < 0)
        return;
    if (ClumpId[Lpa[i]] == ClumpId[Lpb[i]])
        return;

    double3 ARMi, ARMj, Dist, DistU, Vrel, Vrot, Vtot, Vnor, Vtan;
    double3 normal_force, tangential_force, tangential_history;
    double3 rolling_moment, rolling_history;
    double3 twisting_moment, twisting_history;
    double3 cohesive_force, gravity_force;
    double3 Ds, Dtheta, DthetaR, DthetaT, H, Mr, Mt, RV;

    double DistS, DistL, DistR, normal_forceL, tangential_forceL, tangential_historyL;
    double rolling_momentL, rolling_historyL, twisting_momentL, twisting_historyL;
    double Ap, An, Rij, Mij, Iij, Kn, Cn, Ks, Cs, Kr, Cr, Kt, Ct, lnCOR;
    double Dn, DsL, DthetaL, DthetaRL, DthetaTL, ERR;

    double DCMi[3][3], DCMj[3][3];

    bool slipping, rolling, twisting, touching;
    int particleI, particleJ, clumpI, clumpJ;

    particleI = Lpa[i];
    particleJ = Lpb[i];

    clumpI = ClumpId[Lpa[i]];
    clumpJ = ClumpId[Lpb[i]];

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
        DistR = 1.0 / DistL;
        DistU = DistR * Dist;
        Ap = 0.50 *
             (Radius[particleI] * Radius[particleI] - Radius[particleJ] * Radius[particleJ] +
              DistS) *
             DistR;
        An = DistL - Ap;
        Rij = Radius[particleI] * Radius[particleJ] / (Radius[particleI] + Radius[particleJ]);
        Mij = Mass[particleI] * Mass[particleJ] / (Mass[particleI] + Mass[particleJ]);
        Kn = 2.0 * par->Ymod * sqrt(Rij * Dn) / (3.0 * (1.0 - par->Prat * par->Prat));
        Cn = par->Damping * sqrt(Kn * Mij);

        Ks = 2.0 * par->Ymod / (1.0 + par->Prat) / (2.0 - par->Prat) * sqrt(Rij * Dn);
        Cs = par->Damping * sqrt(Ks * Mij);

        Kr = 0.25 * Kn * (par->Beta * Rij) * (par->Beta * Rij);
        Cr = 0.25 * Cn * (par->Beta * Rij) * (par->Beta * Rij);
        Kt = 0.50 * Ks * (par->Beta * Rij) * (par->Beta * Rij);
        Ct = 0.50 * Cs * (par->Beta * Rij) * (par->Beta * Rij);

        Quat2IvDCM(ClumpQuat[clumpI], DCMi);
        Quat2IvDCM(ClumpQuat[clumpJ], DCMj);
        Map(DCMi, LocalPos[particleI], ARMi);
        Map(DCMj, LocalPos[particleJ], ARMj);

        ARMi = ARMi + Ap * DistU;
        ARMj = ARMj - An * DistU;

        Vrel = ClumpVel[clumpJ] - ClumpVel[clumpI];
        Vrot = Cross(ClumpAngSpd[clumpJ], ARMj) - Cross(ClumpAngSpd[clumpI], ARMi);
        Vtot = Vrel + Vrot;

        ERR = Dot(DistU, Vtot);
        Vnor = ERR * DistU;
        Vtan = Vtot - Vnor;

        normal_force = Kn * Dn * DistU + Cn * Vnor;
        normal_forceL = Length(normal_force);

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

        // printf("CacheFc: %f %f %f %f \n", Dn, normal_force.x, normal_force.y, normal_force.z);

        Dtheta = par->StepSize * (ClumpAngSpd[clumpI] - ClumpAngSpd[clumpJ]);
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
    }
}

__global__ void SumContForce_clump(Parameters* par, double3* CacheFc, double3* CacheTc,
                                   double3* CacheTc2, double3* Fc, double3* Tc, int* Aref, int* Ng,
                                   int* Ni, size_t pitch) {
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

void DEMForce::CalcContactForce_Clump(Parameters* par_host, Parameters* par_device, int* Lpa,
                                      int* Lpb, int* ClumpId, double3* Pos, double3* LocalPos,
                                      double* Mass, double* Radius, double3* ClumpVel,
                                      double3* ClumpAngSpd, double4* ClumpQuat, int* Aref, int* Ng,
                                      int* Ni, size_t pitch, int* part_fix) {
    int cp = par_host->ParticleNum * (par_host->BranchLmtLen / 2);

    dim3 block(128, 1);
    dim3 cuda_grid_size = dim3((cp + block.x - 1) / block.x, 1);

    ContactForce_clump_cuk<<<cuda_grid_size, block, 0>>>(
        par_device, Lpa, Lpb, ClumpId, Pos, LocalPos, Mass, Radius, Dv_pre, Dv, Dt_pre, Dt, Dr_pre,
        Dr, ClumpVel, ClumpAngSpd, ClumpQuat, CacheFc, CacheTc, CacheTc2, CacheFcn, CacheCtp,
        part_fix);
    cudaDeviceSynchronize();

    cuda_grid_size.x = (par_host->ParticleNum + block.x - 1) / block.x;

    SumContForce_clump<<<cuda_grid_size, block, 0>>>(par_device, CacheFc, CacheTc, CacheTc2, Fc, Tc,
                                                     Aref, Ng, Ni, pitch);
    cudaDeviceSynchronize();
}