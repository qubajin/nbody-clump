#include "dem_cube_wall.h"

DEMWallCube::DEMWallCube() {
    wall_device = NULL;
    Pos = NULL;
    Vel = NULL;
    AngSpd = NULL;
    Quat = NULL;

    Fc = NULL;
    Tc = NULL;

    Dv_UpX = NULL;
    Dt_UpX = NULL;
    Dr_UpX = NULL;
    Dv_pre_UpX = NULL;
    Dt_pre_UpX = NULL;
    Dr_pre_UpX = NULL;

    Dv_UpY = NULL;
    Dt_UpY = NULL;
    Dr_UpY = NULL;
    Dv_pre_UpY = NULL;
    Dt_pre_UpY = NULL;
    Dr_pre_UpY = NULL;

    Dv_UpZ = NULL;
    Dt_UpZ = NULL;
    Dr_UpZ = NULL;
    Dv_pre_UpZ = NULL;
    Dt_pre_UpZ = NULL;
    Dr_pre_UpZ = NULL;

    Dv_LowX = NULL;
    Dt_LowX = NULL;
    Dr_LowX = NULL;
    Dv_pre_LowX = NULL;
    Dt_pre_LowX = NULL;
    Dr_pre_LowX = NULL;

    Dv_LowY = NULL;
    Dt_LowY = NULL;
    Dr_LowY = NULL;
    Dv_pre_LowY = NULL;
    Dt_pre_LowY = NULL;
    Dr_pre_LowY = NULL;

    Dv_LowZ = NULL;
    Dt_LowZ = NULL;
    Dr_LowZ = NULL;
    Dv_pre_LowZ = NULL;
    Dt_pre_LowZ = NULL;
    Dr_pre_LowZ = NULL;
}

DEMWallCube::~DEMWallCube() {
    if (Pos != NULL)
        cudaFree(Pos);
    if (Vel != NULL)
        cudaFree(Vel);
    if (AngSpd != NULL)
        cudaFree(AngSpd);
    if (Quat != NULL)
        cudaFree(Quat);

    if (Fc != NULL)
        cudaFree(Fc);
    if (Tc != NULL)
        cudaFree(Tc);

    if (Dv_UpX != NULL)
        cudaFree(Dv_UpX);
    if (Dt_UpX != NULL)
        cudaFree(Dt_UpX);
    if (Dr_UpX != NULL)
        cudaFree(Dr_UpX);
    if (Dv_pre_UpX != NULL)
        cudaFree(Dv_pre_UpX);
    if (Dt_pre_UpX != NULL)
        cudaFree(Dt_pre_UpX);
    if (Dr_pre_UpX != NULL)
        cudaFree(Dr_pre_UpX);

    if (Dv_UpY != NULL)
        cudaFree(Dv_UpY);
    if (Dt_UpY != NULL)
        cudaFree(Dt_UpY);
    if (Dr_UpY != NULL)
        cudaFree(Dr_UpY);
    if (Dv_pre_UpY != NULL)
        cudaFree(Dv_pre_UpY);
    if (Dt_pre_UpY != NULL)
        cudaFree(Dt_pre_UpY);
    if (Dr_pre_UpY != NULL)
        cudaFree(Dr_pre_UpY);

    if (Dv_UpZ != NULL)
        cudaFree(Dv_UpZ);
    if (Dt_UpZ != NULL)
        cudaFree(Dt_UpZ);
    if (Dr_UpZ != NULL)
        cudaFree(Dr_UpZ);
    if (Dv_pre_UpZ != NULL)
        cudaFree(Dv_pre_UpZ);
    if (Dt_pre_UpZ != NULL)
        cudaFree(Dt_pre_UpZ);
    if (Dr_pre_UpZ != NULL)
        cudaFree(Dr_pre_UpZ);

    if (Dv_LowX != NULL)
        cudaFree(Dv_LowX);
    if (Dt_LowX != NULL)
        cudaFree(Dt_LowX);
    if (Dr_LowX != NULL)
        cudaFree(Dr_LowX);
    if (Dv_pre_LowX != NULL)
        cudaFree(Dv_pre_LowX);
    if (Dt_pre_LowX != NULL)
        cudaFree(Dt_pre_LowX);
    if (Dr_pre_LowX != NULL)
        cudaFree(Dr_pre_LowX);

    if (Dv_LowY != NULL)
        cudaFree(Dv_LowY);
    if (Dt_LowY != NULL)
        cudaFree(Dt_LowY);
    if (Dr_LowY != NULL)
        cudaFree(Dr_LowY);
    if (Dv_pre_LowY != NULL)
        cudaFree(Dv_pre_LowY);
    if (Dt_pre_LowY != NULL)
        cudaFree(Dt_pre_LowY);
    if (Dr_pre_LowY != NULL)
        cudaFree(Dr_pre_LowY);

    if (Dv_LowZ != NULL)
        cudaFree(Dv_LowZ);
    if (Dt_LowZ != NULL)
        cudaFree(Dt_LowZ);
    if (Dr_LowZ != NULL)
        cudaFree(Dr_LowZ);
    if (Dv_pre_LowZ != NULL)
        cudaFree(Dv_pre_LowZ);
    if (Dt_pre_LowZ != NULL)
        cudaFree(Dt_pre_LowZ);
    if (Dr_pre_LowZ != NULL)
        cudaFree(Dr_pre_LowZ);
}

void DEMWallCube::pointerExch(double3** pt1, double3** pt2) {
    double3* tmp;
    tmp = *pt1;
    *pt1 = *pt2;
    *pt2 = tmp;
}

void DEMWallCube::ForceExchange(Parameters* par_host) {

    pointerExch(&Dv_UpX, &Dv_pre_UpX);
    pointerExch(&Dt_UpX, &Dt_pre_UpX);
    pointerExch(&Dr_UpX, &Dr_pre_UpX);

    pointerExch(&Dv_UpY, &Dv_pre_UpY);
    pointerExch(&Dt_UpY, &Dt_pre_UpY);
    pointerExch(&Dr_UpY, &Dr_pre_UpY);

    pointerExch(&Dv_UpZ, &Dv_pre_UpZ);
    pointerExch(&Dt_UpZ, &Dt_pre_UpZ);
    pointerExch(&Dr_UpZ, &Dr_pre_UpZ);

    pointerExch(&Dv_LowX, &Dv_pre_LowX);
    pointerExch(&Dt_LowX, &Dt_pre_LowX);
    pointerExch(&Dr_LowX, &Dr_pre_LowX);

    pointerExch(&Dv_LowY, &Dv_pre_LowY);
    pointerExch(&Dt_LowY, &Dt_pre_LowY);
    pointerExch(&Dr_LowY, &Dr_pre_LowY);

    pointerExch(&Dv_LowZ, &Dv_pre_LowZ);
    pointerExch(&Dt_LowZ, &Dt_pre_LowZ);
    pointerExch(&Dr_LowZ, &Dr_pre_LowZ);

    checkCudaErrors(cudaMemset(Dv_UpX, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dt_UpX, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dr_UpX, 0.0, par_host->ParticleNum * sizeof(double3)));

    checkCudaErrors(cudaMemset(Dv_UpY, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dt_UpY, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dr_UpY, 0.0, par_host->ParticleNum * sizeof(double3)));

    checkCudaErrors(cudaMemset(Dv_UpZ, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dt_UpZ, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dr_UpZ, 0.0, par_host->ParticleNum * sizeof(double3)));

    checkCudaErrors(cudaMemset(Dv_LowX, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dt_LowX, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dr_LowX, 0.0, par_host->ParticleNum * sizeof(double3)));

    checkCudaErrors(cudaMemset(Dv_LowY, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dt_LowY, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dr_LowY, 0.0, par_host->ParticleNum * sizeof(double3)));

    checkCudaErrors(cudaMemset(Dv_LowZ, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dt_LowZ, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dr_LowZ, 0.0, par_host->ParticleNum * sizeof(double3)));
}

void DEMWallCube::ForceZero(Parameters* par_host) {
    checkCudaErrors(cudaMemset(Fc, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Tc, 0.0, par_host->ParticleNum * sizeof(double3)));
}

void DEMWallCube::AllocateMemory(Parameters* par_host, DEMWallCubeIo* wall_io) {

    checkCudaErrors(cudaMalloc((void**)&wall_device, sizeof(Cube)));

    checkCudaErrors(cudaMalloc((void**)&Pos, sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&Vel, sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&AngSpd, sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&Quat, sizeof(double4)));

    checkCudaErrors(cudaMalloc((void**)&Fc, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&Tc, par_host->ParticleNum * sizeof(double3)));

    checkCudaErrors(cudaMalloc((void**)&Dv_UpX, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&Dt_UpX, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&Dr_UpX, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&Dv_pre_UpX, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&Dt_pre_UpX, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&Dr_pre_UpX, par_host->ParticleNum * sizeof(double3)));

    checkCudaErrors(cudaMalloc((void**)&Dv_UpY, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&Dt_UpY, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&Dr_UpY, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&Dv_pre_UpY, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&Dt_pre_UpY, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&Dr_pre_UpY, par_host->ParticleNum * sizeof(double3)));

    checkCudaErrors(cudaMalloc((void**)&Dv_UpZ, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&Dt_UpZ, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&Dr_UpZ, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&Dv_pre_UpZ, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&Dt_pre_UpZ, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&Dr_pre_UpZ, par_host->ParticleNum * sizeof(double3)));

    checkCudaErrors(cudaMalloc((void**)&Dv_LowX, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&Dt_LowX, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&Dr_LowX, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&Dv_pre_LowX, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&Dt_pre_LowX, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&Dr_pre_LowX, par_host->ParticleNum * sizeof(double3)));

    checkCudaErrors(cudaMalloc((void**)&Dv_LowY, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&Dt_LowY, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&Dr_LowY, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&Dv_pre_LowY, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&Dt_pre_LowY, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&Dr_pre_LowY, par_host->ParticleNum * sizeof(double3)));

    checkCudaErrors(cudaMalloc((void**)&Dv_LowZ, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&Dt_LowZ, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&Dr_LowZ, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&Dv_pre_LowZ, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&Dt_pre_LowZ, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&Dr_pre_LowZ, par_host->ParticleNum * sizeof(double3)));

    checkCudaErrors(cudaMemset(Fc, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Tc, 0.0, par_host->ParticleNum * sizeof(double3)));

    checkCudaErrors(cudaMemset(Dv_UpX, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dt_UpX, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dr_UpX, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dv_pre_UpX, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dt_pre_UpX, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dr_pre_UpX, 0.0, par_host->ParticleNum * sizeof(double3)));

    checkCudaErrors(cudaMemset(Dv_UpY, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dt_UpY, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dr_UpY, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dv_pre_UpY, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dt_pre_UpY, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dr_pre_UpY, 0.0, par_host->ParticleNum * sizeof(double3)));

    checkCudaErrors(cudaMemset(Dv_UpZ, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dt_UpZ, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dr_UpZ, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dv_pre_UpZ, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dt_pre_UpZ, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dr_pre_UpZ, 0.0, par_host->ParticleNum * sizeof(double3)));

    checkCudaErrors(cudaMemset(Dv_LowX, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dt_LowX, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dr_LowX, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dv_pre_LowX, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dt_pre_LowX, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dr_pre_LowX, 0.0, par_host->ParticleNum * sizeof(double3)));

    checkCudaErrors(cudaMemset(Dv_LowY, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dt_LowY, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dr_LowY, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dv_pre_LowY, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dt_pre_LowY, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dr_pre_LowY, 0.0, par_host->ParticleNum * sizeof(double3)));

    checkCudaErrors(cudaMemset(Dv_LowZ, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dt_LowZ, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dr_LowZ, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dv_pre_LowZ, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dt_pre_LowZ, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(Dr_pre_LowZ, 0.0, par_host->ParticleNum * sizeof(double3)));

    checkCudaErrors(cudaMemcpy(Pos, wall_io->GetPos(), sizeof(double3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(Vel, wall_io->GetVel(), sizeof(double3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(Quat, wall_io->GetQuat(), sizeof(double4), cudaMemcpyHostToDevice));
    checkCudaErrors(
        cudaMemcpy(AngSpd, wall_io->GetAngSpd(), sizeof(double3), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(wall_device, wall_io->GetWallParameters(), sizeof(Cube),
                               cudaMemcpyHostToDevice));
}