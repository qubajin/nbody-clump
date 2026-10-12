#include"dem_cylind_wall.h"

__device__ void ForceWall_cuk(Parameters* par, double3* Pos, double3* Vel, double3* AngSpd, double* Radius, double4* Property,
                              Cylinder* cyd, double3* WallPos, double3* WallVel, double3* WallAngSpd,
                              double3* Dv_pre,double3* Dt_pre,double3* Dr_pre,
                              double3* Dv, double3* Dt, double3* Dr,
                              double3 RVo, double3 RVn, double Dn, double3* Fc, double3* Tc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= par->ParticleNum) return;

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
    double DsL, DthetaL, DthetaRL, DthetaTL, ERR;

    //printf("id: %d %d\n", i, id);

    Dist = RVn;

    DistS = Dist.x * Dist.x + Dist.y * Dist.y + Dist.z * Dist.z;
    DistL = sqrt(DistS);

    H = Dv_pre[i];
    Mr = Dr_pre[i];
    Mt = Dt_pre[i];

    DistR = 1.0 / DistL;
    DistU = DistR * Dist;
    Ap = DistL;
    Rij = Radius[i];
    Mij = Property[i].w;
    Kn = 2.0 * par->Ymod * sqrt(Rij * Dn) / (3.0 * (1.0 - par->Prat * par->Prat));
    Cn = par->Damping * sqrt(Kn * Mij);

    Ks = 2.0 * par->Ymod / (1.0 + par->Prat) / (2.0 - par->Prat) * sqrt(Rij * Dn);
    Cs = 2.0 * par->Damping * sqrt(Ks * Mij);

    Kr = 0.25 * Kn * (par->Beta * Rij) * (par->Beta * Rij);
    Cr = 0.25 * Cn * (par->Beta * Rij) * (par->Beta * Rij);
    Kt = 0.50 * Ks * (par->Beta * Rij) * (par->Beta * Rij);
    Ct = 0.50 * Cs * (par->Beta * Rij) * (par->Beta * Rij);

    Vwall = Cross(WallAngSpd[0], RVo);
    Vrel = Vel[i] - WallVel[0];
    Vrot = Ap * Cross(DistU, AngSpd[i]) - Vwall;
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

    int tanFlg = 0;

    if (tangential_forceL > normal_forceL * par->Frcoe)
    {
        tangential_force = par->Frcoe * normal_forceL * (1.0 / tangential_forceL) * tangential_force;
        tangential_history = tangential_force;
        tanFlg = 1;
    }
    else
    {
        tangential_history = tangential_force - Cs * Vtan;
    }

    Fc[i] = Fc[i] + normal_force + tangential_force;
    Tc[i] = Tc[i] -Ap * Cross(DistU, tangential_force);

    //printf("Dt:%d %f %f %f %f %f %f\n", tanFlg, Dv_pre[i].x, Dv_pre[i].y, Dv_pre[i].z, tangential_history.x, tangential_history.y, tangential_history.z);

    //printf("Wall: %f %f %f %f %f %f %f %f %f\n", Fc[i].x, Fc[i].y, Fc[i].z, Tc[i].x, Tc[i].y, Tc[i].z, Vel[i].x, Vel[i].y, Vel[i].z);
    //printf("Wall: %d %f %f %f %f %f %f\n", tanFlg, tangential_force.x, tangential_force.y, tangential_force.z, Vel[i].x, Vel[i].y, Vel[i].z);
    //printf("Wall: %d %f %f %f %f %f %f %f %f %f %f %f\n", tanFlg, Ks, Cs, Ds.x, Ds.y, Ds.z, Vtan.x, Vtan.y, Vtan.z, H.x, H.y, H.z);

    Dtheta = par->StepSize * (AngSpd[i] - WallAngSpd[0]);
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

    Dv[i] = tangential_history;
    Dt[i] = twisting_history;
    Dr[i] = rolling_history;

    //printf("Dt:%d %f %f %f %f %f %f\n", tanFlg, Dv_pre[i].x, Dv_pre[i].y, Dv_pre[i].z, tangential_history.x, tangential_history.y, tangential_history.z);


}

__global__ void PrintVal2(Parameters* par, double3* Tmp)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= par->ParticleNum) return;

    for (i = 0; i < par->ParticleNum; i++)
    {
        printf("Val: %d %f %f %f\n", i, Tmp[i].x, Tmp[i].y, Tmp[i].z);
    }
}

__global__ void CylindWallInteraction_cuk(Parameters* par, double3* Pos, double3* Vel, double3* AngSpd, double* Radius, double4* Property,
                                          Cylinder* cyd, double3* WallPos, double3* WallVel, double3* WallAngSpd,
                                          double3* Dv_pre_cap, double3* Dt_pre_cap, double3* Dr_pre_cap,
                                          double3* Dv_cap, double3* Dt_cap, double3* Dr_cap, 
                                          double3* Dv_pre_up, double3* Dt_pre_up, double3* Dr_pre_up,
                                          double3* Dv_up, double3* Dt_up, double3* Dr_up,
                                          double3* Dv_pre_side, double3* Dt_pre_side, double3* Dr_pre_side,
                                          double3* Dv_side, double3* Dt_side, double3* Dr_side,
                                          double3* Fc, double3* Tc)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= par->ParticleNum) return;

    double3 RV, RVb, RVt, RVn, RVo;
    double3 WallVector;
    double ERR, norm, DistL, Dn, DistS;
    double d_cap, d_side;

    WallVector = cyd->Dirct;
    RV = Pos[i] - WallPos[0];
    ERR = Dot(RV, WallVector);
    RVn = make_double3(0.0, 0.0, 0.0);
    RVo = make_double3(0.0, 0.0, 0.0);

    RVb = RV - ERR * WallVector;
    RVt = ERR * WallVector;
    norm = Length(RVb);

    d_cap = abs(ERR) - cyd->Height + Radius[i];
    d_side = norm - cyd->Radius + Radius[i];

    if (d_cap >= 1e-14 && ERR < 0)
    {
        RVo = (cyd->Height / abs(ERR)) * RVt + RVb;
        RVn = RV - RVo;
        DistL = Length(RVn);
        Dn = Radius[i] - DistL;
        if (Dn > 0.0)
        {
            ForceWall_cuk(par, Pos, Vel, AngSpd, Radius, Property,
                          cyd, WallPos, WallVel, WallAngSpd,
                          Dv_pre_cap, Dt_pre_cap, Dr_pre_cap,                         
                          Dv_cap, Dt_cap, Dr_cap,
                          RVo, RVn, Dn, Fc, Tc);
            //printf("con1: %f %f %f\n", d_cap, d_side, ERR);
        }
        
    }

    if (d_cap >= 1e-14 && ERR > 0)
    {
        RVo = (cyd->Height / abs(ERR)) * RVt + RVb;
        RVn = RV - RVo;
        DistL = Length(RVn);
        Dn = Radius[i] - DistL;
        if (Dn > 0.0)
        {
            ForceWall_cuk(par, Pos, Vel, AngSpd, Radius, Property,
                         cyd, WallPos, WallVel, WallAngSpd,
                         Dv_pre_up, Dt_pre_up, Dr_pre_up,
                         Dv_up, Dt_up, Dr_up,
                         RVo, RVn, Dn, Fc, Tc);
            //printf("con2: %f %f %f\n", d_cap, d_side, ERR);
        }
        
    }

    if (d_side >= 1e-14)
    {
        RVo = (cyd->Radius / norm) * RVb + RVt;
        RVn = RV - RVo;
        DistL = Length(RVn);
        Dn = Radius[i] - DistL;
        if (Dn > 0.0)
        {
            ForceWall_cuk(par, Pos, Vel, AngSpd, Radius, Property,
                          cyd, WallPos, WallVel, WallAngSpd,
                          Dv_pre_side, Dt_pre_side, Dr_pre_side,
                          Dv_side, Dt_side, Dr_side,
                          RVo, RVn, Dn, Fc, Tc);
            //printf("con3: %f %f %f\n", d_cap, d_side, ERR);
        }
        
    }

}

__global__ void PrintHist(double3* Dv_pre, double3* Dv, double3* sortedDv)
{
    int i = 0;
    if (Length(Dv_pre[i]) != 0)
        printf("%f %f %f %f %f %f %f %f %f\n", Dv_pre[i].x, Dv_pre[i].y, Dv_pre[i].z, Dv[i].x, Dv[i].y, Dv[i].z, sortedDv[i].x, sortedDv[i].y, sortedDv[i].z);
}



void DEMWallCylinder::CalcContactForce(Parameters* par_host, Parameters* par_device, double3* PartPos, double3* PartVel, double3* PartAngSpd, double* PartRadius, double4* PartProperty)
{
    int cp = par_host->ParticleNum;

    dim3 block(128, 1);
    dim3 cuda_grid_size((cp + block.x - 1) / block.x, 1);

   /* PrintHist << <1, 1 >> > (Dv_pre_cap, Dv_cap, sortedDv_cap);
    cudaDeviceSynchronize();*/

    CylindWallInteraction_cuk << <cuda_grid_size, block >> > (par_device, PartPos, PartVel, PartAngSpd, PartRadius, PartProperty,
                                                              wall_device, Pos, Vel, AngSpd,
                                                              Dv_pre_cap, Dt_pre_cap, Dr_pre_cap,                                                         
                                                              Dv_cap, Dt_cap, Dr_cap,
                                                              Dv_pre_up, Dt_pre_up, Dr_pre_up,                                                           
                                                              Dv_up, Dt_up, Dr_up,
                                                              Dv_pre_side, Dt_pre_side, Dr_pre_side,                                                             
                                                              Dv_side, Dt_side, Dr_side,
                                                              Fc, Tc);
    cudaDeviceSynchronize();

    //PrintHist << <1, 1 >> > (Dv_pre_cap, Dv_cap, sortedDv_cap);
    //cudaDeviceSynchronize();
   
}