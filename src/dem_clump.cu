#include "dem_clump.h"

DEMClumpMain::DEMClumpMain() {
    par_host = NULL;
    par_device = NULL;

    Glb = NULL;

    wall_io = NULL;
    wall = NULL;

    pene_io = NULL;
    pene = NULL;

    io = NULL;
    grid = NULL;
    force = NULL;

    clump_io = NULL;

    cube_io = NULL;
    cube = NULL;

    ClumpId = NULL;

    Pos = NULL;
    LocalPos = NULL;
    CumPos = NULL;

    Radius = NULL;
    Mass = NULL;
}

DEMClumpMain::~DEMClumpMain() {
    if (ClumpId != NULL)
        cudaFree(ClumpId);

    if (Pos != NULL)
        cudaFree(Pos);
    if (LocalPos != NULL)
        cudaFree(LocalPos);
    if (CumPos != NULL)
        cudaFree(CumPos);

    if (Radius != NULL)
        cudaFree(Radius);
}

void DEMClumpMain::AllocateGpuMemory() {
    checkCudaErrors(cudaMalloc((void**)&par_device, sizeof(Parameters)));

    checkCudaErrors(cudaMalloc((void**)&ClumpId, par_host->ParticleNum * sizeof(int)));

    checkCudaErrors(cudaMalloc((void**)&Pos, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&LocalPos, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&CumPos, par_host->ParticleNum * sizeof(double3)));
    // Initialize accumulated displacement before the first position update.
    checkCudaErrors(cudaMemset(CumPos, 0, par_host->ParticleNum * sizeof(double3)));

    checkCudaErrors(cudaMalloc((void**)&Mass, par_host->ParticleNum * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&Radius, par_host->ParticleNum * sizeof(double)));

    checkCudaErrors(cudaMalloc((void**)&totF, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&totT, par_host->ParticleNum * sizeof(double3)));
}

void DEMClumpMain::CopyDataFromHostToDevice() {
    checkCudaErrors(cudaMemcpy(par_device, par_host, sizeof(Parameters), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(Pos, io->GetPos(), par_host->ParticleNum * sizeof(double3),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(LocalPos, io->GetLocalPos(), par_host->ParticleNum * sizeof(double3),
                               cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(ClumpId, io->GetClumpId(), par_host->ParticleNum * sizeof(int),
                               cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(Mass, io->GetMass(), par_host->ParticleNum * sizeof(double),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(Radius, io->GetRadius(), par_host->ParticleNum * sizeof(double),
                               cudaMemcpyHostToDevice));
}

void DEMClumpMain::ForceZero() {
    checkCudaErrors(cudaMemset(totF, 0.0, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMemset(totT, 0.0, par_host->ParticleNum * sizeof(double3)));
}

__global__ void SumAllForces_cuk(Parameters* Par, double3* Fc, double3* Tc, double3* Fw,
                                 double3* Tw, double3* Fi, double3* Ti, double3* totF,
                                 double3* totT) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= Par->ParticleNum)
        return;

    totF[i] = Fc[i] + Fw[i] + Fi[i];
    totT[i] = Tc[i] + Tw[i] + Ti[i];

    // if(Length(totF[i])!=0) printf("totF: %f %f %f\n", totF[i].x, totF[i].y, totF[i].z);
}

void DEMClumpMain::SumAllForces(double3* Fc, double3* Tc, double3* Fw, double3* Tw, double3* Fi,
                                double3* Ti) {
    dim3 block(128, 1);
    dim3 cuda_grid_size = dim3((par_host->ParticleNum + block.x - 1) / block.x, 1);

    SumAllForces_cuk<<<cuda_grid_size, block, 0>>>(par_device, Fc, Tc, Fw, Tw, Fi, Ti, totF, totT);
    getLastCudaError("GetSumAllForces_cuk failed...     \n");
    cudaDeviceSynchronize();
}

__global__ void UpdatePos_cuk(Parameters* Par, double3* Pos, double3* CumPos,
                              const double3* LocalPos, const int* ClumpId, const double3* ClumpPos,
                              const double4* ClumpQuat) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= Par->ParticleNum)
        return;

    double DCM[3][3];
    double3 arm, cumpos, updatepos;

    int ClumpI = ClumpId[i];

    Quat2IvDCM(ClumpQuat[ClumpI], DCM);
    Map(DCM, LocalPos[i], arm);

    updatepos = ClumpPos[ClumpI] + arm;
    cumpos = CumPos[i] + updatepos - Pos[i];

    if (Length(cumpos) > Par->Thresh && Par->Update == 0) {
        atomicAdd(&Par->Update, 1);
    }

    CumPos[i] = cumpos;
    Pos[i] = updatepos;
}

void DEMClumpMain::UpdatePos(int* ClumpId, double3* ClumpPos, double4* ClumpQuat) {
    dim3 block(128, 1);
    dim3 cuda_grid_size = dim3((par_host->ParticleNum + block.x - 1) / block.x, 1);

    UpdatePos_cuk<<<cuda_grid_size, block, 0>>>(par_device, Pos, CumPos, LocalPos, ClumpId,
                                                ClumpPos, ClumpQuat);

    getLastCudaError("GetUpdatePos_cuk failed...     \n");
    cudaDeviceSynchronize();
}

void DEMClumpMain::CopyDataToIo(GlobalFlg* Glb, double3* CntFc, double3* CntTc) {
    checkCudaErrors(cudaMemcpy(io->GetPos(), Pos, par_host->ParticleNum * sizeof(double3),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(io->GetLocalPos(), LocalPos, par_host->ParticleNum * sizeof(double3),
                               cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMemcpy(io->GetClumpId(), ClumpId, par_host->ParticleNum * sizeof(int),
                               cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMemcpy(io->GetMass(), Mass, par_host->ParticleNum * sizeof(double),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(io->GetRadius(), Radius, par_host->ParticleNum * sizeof(double),
                               cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMemcpy(io->GetParticleID(), grid->GetPartFix(),
                               par_host->ParticleNum * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(io->GetSortedParticleID(), grid->GetPartIdx(),
                               par_host->ParticleNum * sizeof(int), cudaMemcpyDeviceToHost));

    if (Glb->isIntruder) {
        checkCudaErrors(cudaMemcpy(io->GetCacheFi(), CntFc,
                                   par_host->OutputInterval * sizeof(double3),
                                   cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(io->GetCacheTi(), CntTc,
                                   par_host->OutputInterval * sizeof(double3),
                                   cudaMemcpyDeviceToHost));
    }
}

void DEMClumpMain::CopyInfoToIo(size_t pitch_int, size_t pitch_double3, int* Anei,
                                double3* ChainFcn, double3* ChainCtp) {
    checkCudaErrors(cudaMemcpy2D(io->GetAnei(), sizeof(int) * par_host->BranchLmtLen, Anei,
                                 pitch_int, sizeof(int) * par_host->BranchLmtLen,
                                 par_host->ParticleNum, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy2D(io->GetChainFcn(), sizeof(double3) * par_host->BranchLmtLen,
                                 ChainFcn, pitch_double3, sizeof(double3) * par_host->BranchLmtLen,
                                 par_host->ParticleNum, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy2D(io->GetChainCtp(), sizeof(double3) * par_host->BranchLmtLen,
                                 ChainCtp, pitch_double3, sizeof(double3) * par_host->BranchLmtLen,
                                 par_host->ParticleNum, cudaMemcpyDeviceToHost));
}

void DEMClumpMain::pointerExch(double3** pt1, double3** pt2) {
    double3* tmp;
    tmp = *pt1;
    *pt1 = *pt2;
    *pt2 = tmp;
}