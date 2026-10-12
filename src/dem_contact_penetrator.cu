#include "dem_penetrator.h"
#include "dem_sphere_intruder.cuh"

#include "thrust/sort.h"
#include "thrust/device_ptr.h"
#include "thrust/scan.h"

__global__ void PrePenetrator(Parameters* par, Cylinder* cyd, double3* Pos, double* Radius,
                              double3* PenePos, double3* ContPos, double3* NVec, int* LcFlg,
                              int* part_fix) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= par->ParticleNum)
        return;

    double LatDx, ERR, norm;
    double3 RV, RVb, RVt, RVn, RVo, peneVector;
    double3 Dist;
    double DistS, DistL, Dn;
    bool enterFlag;

    enterFlag = false;
    LcFlg[i] = 0;

    if (cyd->Shape == PENETRATOR_SPHERE) {
        const SphereIntruderContact contact =
            GetSphereIntruderContact(Pos[i], Radius[i], PenePos[0], cyd->Radius, cyd->Dirct);

        if (contact.overlap > 0.0) {
            ContPos[i] = contact.surfaceArm;
            NVec[i] = contact.normalGapVector;
            LcFlg[i] = 1;

            if (cyd->WaitForContact) {
                // 多个子球可能同时接触，使用原子操作只触发一次。
                // 后续接触不会把状态2改回1。
                atomicCAS(&cyd->ContactState, 0, 1);
            }
        }
        return;
    }

    LatDx = par->CellSize_x;
    peneVector = cyd->Dirct;

    RV = Pos[i] - PenePos[0];
    ERR = Dot(RV, peneVector);

    RVn = make_double3(0.0, 0.0, 0.0);
    RVo = make_double3(0.0, 0.0, 0.0);
    if (abs(ERR) < (cyd->Height + 0.80 * LatDx)) {
        RVb = RV - ERR * peneVector;
        norm = Length(RVb);
        if (norm < (cyd->Radius + 0.80 * LatDx)) {
            if (abs(ERR) < cyd->Height) {
                RVt = ERR * peneVector;
                RVo = (1.0 / norm) * cyd->Radius * RVb + RVt;
                RVn = RV - RVo;
            } else if (norm < cyd->Radius) {
                RVt = ERR * peneVector;
                RVo = (1.0 / abs(ERR)) * cyd->Height * RVt + RVb;
                RVn = RV - RVo;
            } else {
                RVt = ERR * peneVector;
                RVo = (1.0 / norm) * cyd->Radius * RVb + (1.0 / abs(ERR)) * cyd->Height * RVt;
                RVn = RV - RVo;
            }
            enterFlag = true;
        }
    }

    if (enterFlag) {
        Dist = RVn;
        DistS = Dist.x * Dist.x + Dist.y * Dist.y + Dist.z * Dist.z;
        DistL = sqrt(DistS);
        Dn = Radius[i] - DistL;

        if (Dn > 0.0) {
            ContPos[i] = RVo;
            NVec[i] = RVn;
            LcFlg[i] = 1;
            // printf("part: %d %d\n", i, part_fix[i]);
        }
    }
}

__global__ void CreateContList_cuk(Parameters* par, int* LcFlg, int* Nsum, int* Lc) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= par->ParticleNum)
        return;

    int Nlist;

    // printf("i: %d %d\n", i, Nsum[i-1]);
    if (i == 0 && LcFlg[i] != -1) {
        Lc[0] = i;
        return;
    }

    for (int Igrid = 0; Igrid < LcFlg[i]; Igrid++) {
        Nlist = Nsum[i - 1] + Igrid;
        Lc[Nlist] = i;
        // printf("Nlist: %d %d %d\n", Nlist, i, Lw[Nlist]);
    }
}

void DEMPeneCylinder::CreateContList(Parameters* par_host, Parameters* par_device, double3* PartPos,
                                     double* PartRadius, int* part_fix) {
    int ParticleNum = par_host->ParticleNum;
    dim3 block(128, 1);
    dim3 cuda_grid_size = dim3((par_host->ParticleNum + block.x - 1) / block.x, 1);

    // checkCudaErrors(cudaMemset(Lc, -1, par_host->ParticleNum * sizeof(int)));

    PrePenetrator<<<cuda_grid_size, block, 0>>>(par_device, pene_device, PartPos, PartRadius, Pos,
                                                RVo, RVn, LcFlg, part_fix);
    cudaDeviceSynchronize();

    /*thrust::inclusive_scan(thrust::device_ptr<int>(LcFlg), thrust::device_ptr<int>(LcFlg +
    ParticleNum), thrust::device_ptr<int>(Nsum));

    CreateContList_cuk << <cuda_grid_size, block, 0 >> > (par_device, LcFlg, Nsum, Lc);
    cudaDeviceSynchronize();*/
}
