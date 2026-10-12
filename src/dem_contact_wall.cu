#include"dem_cylind_wall.h"

#include "thrust/sort.h"
#include "thrust/device_ptr.h"
#include "thrust/scan.h"

__global__ void PreWall_cuk(Parameters* par, Cylinder* cyd, double3* Pos, double* Radius,
                        double3* WallPos, double3* ContPos, double3* NVec,
                        int* LcFlg, int* part_fix)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= par->ParticleNum) return;

    double3 RV, RVb, RVt, RVn, RVo, P_cap, P_edge, P_corner, Dist;
    double3 WallVector;
    double ERR, norm, DistL, Dn, DistS;
    double d_cap, d_side, d_edge;
    bool enterFlag = false;
    int Flg1, Flg2, Flg3;

    Flg1 = 0;
    Flg2 = 0;
    Flg3 = 0;

    WallVector = cyd->Dirct;
    RV = Pos[i] - WallPos[0];
    ERR = Dot(RV, WallVector);
    RVn = make_double3(0.0, 0.0, 0.0);
    RVo = make_double3(0.0, 0.0, 0.0);
    LcFlg[i] = 0;

    RVb = RV - ERR * WallVector;
    RVt = ERR * WallVector;
    norm = Length(RVb);
    P_cap = WallPos[0] + RVt;

    d_cap = abs(ERR) - cyd->Height + Radius[i];
    d_side = norm - cyd->Radius + Radius[i];

    if (norm > 1e-6)
    {
        P_edge = WallPos[0] + ((ERR > 0) ? cyd->Height : -cyd->Height) * WallVector;
        P_edge = P_edge + (cyd->Radius / norm) * RVb;
    }
    else {
        P_edge = P_cap;
    }
    d_edge = Length(Pos[i] - P_edge);

    if (d_side >= 0 && d_cap < 0)
    {
        RVo = (cyd->Radius / norm) * RVb + RVt;
        enterFlag = true;
        Flg1 = 1;
    }
    else if (d_cap >= 0 && d_side < 0)
    {
        RVo = (cyd->Height / abs(ERR)) * RVt + RVb;
        enterFlag = true;
        Flg2 = 2;
    }
    else if (d_edge <= Radius[i]) {
        RVo = P_edge - WallPos[0];
        enterFlag = true;
        Flg3 = 3;
    }
    else if (d_side >= 0 && d_cap >= 0)
    {
        if (d_cap < d_side)
        {
            //RVo = (cyd->Height / abs(ERR)) * RVt + RVb;
            RVo = (cyd->Radius / norm) * RVb + RVt;
            enterFlag = true;
            Flg2 = 4;
        }
        else
        {
            RVo = (cyd->Height / abs(ERR)) * RVt + RVb;
            enterFlag = true;
            Flg1 = 5;
        }
    }

    if (enterFlag) {
        RVn = RV - RVo;
        DistL = Length(RVn);
        Dn = Radius[i] - DistL;

        //if(part_fix[i] == 831) printf("Wall: %d %d %f %f %f %f %f %f %f\n", part_fix[i], Flg, d_cap, d_side, d_edge, Dn, RVo.x, RVo.y, RVo.z);

        if (Dn > 0.0) {
            ContPos[i] = RVo;
            NVec[i] = RVn;
            LcFlg[i] = 1;
        }
    }
}




//__global__ void PrintVal(Parameters* par, double3* Tmp)
//{
//	int i = threadIdx.x + blockIdx.x * blockDim.x;
//	if (i >= par->ParticleNum) return;
//
//	for (i = 0; i < par->ParticleNum; i++)
//	{
//		printf("Val: %d %f %f %f\n", i, Tmp[i].x, Tmp[i].y, Tmp[i].z);
//	}
//
//	
//}
//
//void DEMWallCylinder::CreateContList(Parameters* par_host, Parameters* par_device, double3* PartPos, double* PartRadius, int* part_fix)
//{
//	int ParticleNum = par_host->ParticleNum;
//	dim3 block(128, 1);
//	dim3 cuda_grid_size = dim3((par_host->ParticleNum + block.x - 1) / block.x, 1);
//
//	//checkCudaErrors(cudaMemset(Lc, 0, par_host->ParticleNum * sizeof(int)));
//	//checkCudaErrors(cudaMemset(Nsum, 0, par_host->ParticleNum * sizeof(int)));
//
//	//PrintNsum << <1, 1 >> > (par_device, Nsum);
//	//cudaDeviceSynchronize();
//
//	//PreWall_cuk << <cuda_grid_size, block, 0 >> > (par_device, wall_device, PartPos, PartRadius, Pos, RVo, RVn, LcFlg, part_fix);
//	//cudaDeviceSynchronize();
//
//	//PrintVal << <1, 1 >> > (par_device, RVn);
//	//cudaDeviceSynchronize();
//
//	/*thrust::inclusive_scan(thrust::device_ptr<int>(LcFlg), thrust::device_ptr<int>(LcFlg + ParticleNum), thrust::device_ptr<int>(Nsum));
//
//	PrintNsum << <1, 1 >> > (par_device, Nsum);
//	cudaDeviceSynchronize();
//
//	CreateContList_Wall_cuk << <cuda_grid_size, block, 0 >> > (par_device, LcFlg, Nsum, Lc);
//	cudaDeviceSynchronize();*/
//}




