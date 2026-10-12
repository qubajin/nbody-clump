#include"dem_cube_wall.h"

__device__ void ForceWall_Cube_cuk(Parameters* par, double3* Pos, double3* Vel, double3* AngSpd, double* Radius, double4* Property,
                                   double3* WallPos, double3* WallVel, double3* WallAngSpd,
                                   double3* Dv_pre, double3* Dt_pre, double3* Dr_pre,
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

    //printf("Vnor: %f %f %f %f %f %f %f\n", Ap, DistU.x, DistU.y, DistU.z, AngSpd[i].x, AngSpd[i].y, AngSpd[i].z);


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
    Tc[i] = Tc[i] - Ap * Cross(DistU, tangential_force);

    //printf("pos: %f %f %f %f %f %f %f\n", Pos[i].x, Pos[i].y, Pos[i].z, Dn, Vnor.x, Vnor.y, Vnor.z);

    //printf("wall: %f %f %f %f %f %f\n", normal_force.x, normal_force.y, normal_force.z, tangential_force.x, tangential_force.y, tangential_force.z);

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

   /* Tc[i] = Tc[i] - rolling_moment - twisting_moment;

    Dv[i] = tangential_history;
    Dt[i] = twisting_history;
    Dr[i] = rolling_history;*/

    //printf("Force: %d %f %f %f %f %f %f %f %f %f %f %f\n", tanFlg, -Ks * Ds.x, -Ks * Ds.y, -Ks * Ds.z, Cs * Vtan.x, Cs * Vtan.y, Cs * Vtan.z, H.x, H.y, H.z, tangential_forceL, normal_forceL * par->Frcoe);
    //printf("force: %d %f %f %f %f %f %f %f\n", i, Dn, DistU.x, DistU.y, DistU.z, RVo.x, RVo.y, RVo.z);
}



__global__ void CubeWallInteraction_cuk(Parameters* par, double3* Pos, double3* Vel, double3* AngSpd, double* Radius, double4* Property,
                                        Cube* cub, double3* WallPos, double3* WallVel, double3* WallAngSpd,
                                        double3* Dv_UpX, double3* Dt_UpX, double3* Dr_UpX,
                                        double3* Dv_UpY, double3* Dt_UpY, double3* Dr_UpY,
                                        double3* Dv_UpZ, double3* Dt_UpZ, double3* Dr_UpZ,
                                        double3* Dv_LowX, double3* Dt_LowX, double3* Dr_LowX,
                                        double3* Dv_LowY, double3* Dt_LowY, double3* Dr_LowY,
                                        double3* Dv_LowZ, double3* Dt_LowZ, double3* Dr_LowZ,
                                        double3* Dv_pre_UpX, double3* Dt_pre_UpX, double3* Dr_pre_UpX,
                                        double3* Dv_pre_UpY, double3* Dt_pre_UpY, double3* Dr_pre_UpY,
                                        double3* Dv_pre_UpZ, double3* Dt_pre_UpZ, double3* Dr_pre_UpZ,
                                        double3* Dv_pre_LowX, double3* Dt_pre_LowX, double3* Dr_pre_LowX,
                                        double3* Dv_pre_LowY, double3* Dt_pre_LowY, double3* Dr_pre_LowY,
                                        double3* Dv_pre_LowZ, double3* Dt_pre_LowZ, double3* Dr_pre_LowZ,
                                        double3* Fc, double3* Tc)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= par->ParticleNum) return;

    double ERRx, ERRy, ERRz, DistL, Dn;
    double3 RV, RVtx, RVty, RVtz, RVbx, RVby, RVbz;
    double3 RVo, RVn;
    double3 Vectorx, Vectory, Vectorz;

    Vectorx = make_double3(1.0, 0.0, 0.0);
    Vectory = make_double3(0.0, 1.0, 0.0);
    Vectorz = make_double3(0.0, 0.0, 1.0);

    RV = Pos[i] - WallPos[0];
    ERRx = RV.x;
    ERRy = RV.y;
    ERRz = RV.z;

    RVbx = RV - ERRx * Vectorx;
    RVtx = ERRx * Vectorx;

    RVby = RV - ERRy * Vectory;
    RVty = ERRy * Vectory;

    RVbz = RV - ERRz * Vectorz;
    RVtz = ERRz * Vectorz;
  
    if (ERRx > 0 && ERRx + Radius[i] - cub->SideLength.x > 0)
    {
        //UpX
        RVo = (cub->SideLength.x / abs(ERRx)) * RVtx + RVbx;
        RVn = RV - RVo;
        DistL = Length(RVn);
        Dn = Radius[i] - DistL;

        ForceWall_Cube_cuk(par, Pos, Vel, AngSpd, Radius, Property,
                           WallPos, WallVel, WallAngSpd,
                           Dv_pre_UpX, Dt_pre_UpX, Dr_pre_UpX,
                           Dv_UpX, Dt_UpX, Dr_UpX,
                           RVo, RVn, Dn, Fc, Tc);
    }

    if (ERRx < 0 && ERRx - Radius[i] + cub->SideLength.x < 0)
    {
        //DownX
        RVo = (cub->SideLength.x / abs(ERRx)) * RVtx + RVbx;
        RVn = RV - RVo;
        DistL = Length(RVn);
        Dn = Radius[i] - DistL;

        ForceWall_Cube_cuk(par, Pos, Vel, AngSpd, Radius, Property,
                           WallPos, WallVel, WallAngSpd,
                           Dv_pre_LowX, Dt_pre_LowX, Dr_pre_LowX,
                           Dv_LowX, Dt_LowX, Dr_LowX,
                           RVo, RVn, Dn, Fc, Tc);
    }

    if (ERRy > 0 && ERRy + Radius[i] - cub->SideLength.y > 0)
    {
        //UpY
        RVo = (cub->SideLength.y / abs(ERRy)) * RVty + RVby;
        RVn = RV - RVo;
        DistL = Length(RVn);
        Dn = Radius[i] - DistL;

        ForceWall_Cube_cuk(par, Pos, Vel, AngSpd, Radius, Property,
                           WallPos, WallVel, WallAngSpd,
                           Dv_pre_UpY, Dt_pre_UpY, Dr_pre_UpY,
                           Dv_UpY, Dt_UpY, Dr_UpY,
                           RVo, RVn, Dn, Fc, Tc);
    }

    if (ERRy < 0 && ERRy - Radius[i] + cub->SideLength.y < 0)
    {
        //DownY
        RVo = (cub->SideLength.y / abs(ERRy)) * RVty + RVby;
        RVn = RV - RVo;
        DistL = Length(RVn);
        Dn = Radius[i] - DistL;

        ForceWall_Cube_cuk(par, Pos, Vel, AngSpd, Radius, Property,
                           WallPos, WallVel, WallAngSpd,
                           Dv_pre_LowY, Dt_pre_LowY, Dr_pre_LowY,
                           Dv_LowY, Dt_LowY, Dr_LowY,
                           RVo, RVn, Dn, Fc, Tc);
    }

    if (ERRz > 0 && ERRz + Radius[i] - cub->SideLength.z > 0)
    {
        //UpZ
        RVo = (cub->SideLength.z / abs(ERRz)) * RVtz + RVbz;
        RVn = RV - RVo;
        DistL = Length(RVn);
        Dn = Radius[i] - DistL;

        ForceWall_Cube_cuk(par, Pos, Vel, AngSpd, Radius, Property,
                           WallPos, WallVel, WallAngSpd,
                           Dv_pre_UpZ, Dt_pre_UpZ, Dr_pre_UpZ,
                           Dv_UpZ, Dt_UpZ, Dr_UpZ,
                           RVo, RVn, Dn, Fc, Tc);
    }

    if (ERRz < 0 && ERRz - Radius[i] + cub->SideLength.z < 0)
    {
        //DownZ
        RVo = (cub->SideLength.z / abs(ERRz)) * RVtz + RVbz;
        RVn = RV - RVo;
        DistL = Length(RVn);
        Dn = Radius[i] - DistL;

        ForceWall_Cube_cuk(par, Pos, Vel, AngSpd, Radius, Property,
                           WallPos, WallVel, WallAngSpd,
                           Dv_pre_LowZ, Dt_pre_LowZ, Dr_pre_LowZ,
                           Dv_LowZ, Dt_LowZ, Dr_LowZ,
                           RVo, RVn, Dn, Fc, Tc);
    }

}

void DEMWallCube::CalcContactForce(Parameters* par_host, Parameters* par_device, double3* PartPos, double3* PartVel, double3* PartAngSpd, double* PartRadius, double4* PartProperty)
{
    int cp = par_host->ParticleNum;

    dim3 block(128, 1);
    dim3 cuda_grid_size((cp + block.x - 1) / block.x, 1);

    CubeWallInteraction_cuk << <cuda_grid_size, block >> > (par_device, PartPos, PartVel, PartAngSpd, PartRadius, PartProperty,
                                                            wall_device, Pos, Vel, AngSpd,
                                                            Dv_UpX, Dt_UpX, Dr_UpX,
                                                            Dv_UpY, Dt_UpY, Dr_UpY,
                                                            Dv_UpZ, Dt_UpZ, Dr_UpZ,
                                                            Dv_LowX, Dt_LowX, Dr_LowX,
                                                            Dv_LowY, Dt_LowY, Dr_LowY,
                                                            Dv_LowZ, Dt_LowZ, Dr_LowZ,
                                                            Dv_pre_UpX, Dt_pre_UpX, Dr_pre_UpX,
                                                            Dv_pre_UpY, Dt_pre_UpY, Dr_pre_UpY,
                                                            Dv_pre_UpZ, Dt_pre_UpZ, Dr_pre_UpZ,
                                                            Dv_pre_LowX, Dt_pre_LowX, Dr_pre_LowX,
                                                            Dv_pre_LowY, Dt_pre_LowY, Dr_pre_LowY,
                                                            Dv_pre_LowZ, Dt_pre_LowZ, Dr_pre_LowZ,
                                                            Fc, Tc);
    cudaDeviceSynchronize();
}