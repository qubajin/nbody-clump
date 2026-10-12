#include <math.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <cub/cub.cuh>

#include "dem_hgrid.h"

#include "thrust/sort.h"
#include "thrust/device_ptr.h"
#include "thrust/scan.h"
#include <device_launch_parameters.h>
#include <vector_functions.h>

__global__ void PrintfNsum(int* Nsum, int* Ng)
{
	for (int i = 0; i < 2; i++)
	{
		printf("Nsum: %d %d %d\n", i, Nsum[i], Ng[i]);
	}	
}


void DEMHGrid::InclusiveNsum(int ParticleNum, int* Ng, int* Nsum)
{
	thrust::inclusive_scan(thrust::device_ptr<int>(Ng), thrust::device_ptr<int>(Ng + ParticleNum), thrust::device_ptr<int>(Nsum));
	/*printf("Inclusive!!\n");
	PrintfNsum << <1, 1 >> > (Nsum, Ng);
	cudaDeviceSynchronize;*/
}

void DEMHGrid::pointerExch(int** pt1, int** pt2)
{
	int* tmp;
	tmp = *pt1;
	*pt1 = *pt2;
	*pt2 = tmp;
}

void DEMHGrid::pointerExch(double3** pt1, double3** pt2)
{
	double3* tmp;
	tmp = *pt1;
	*pt1 = *pt2;
	*pt2 = tmp;
}

void DEMHGrid::InitNList(Parameters* par_host)
{
	checkCudaErrors(cudaMemset(Ng, 0, sizeof(int) * par_host->ParticleNum));
	checkCudaErrors(cudaMemset(Ni, 0, sizeof(int) * par_host->ParticleNum));

	checkCudaErrors(cudaMemset(Lpa, -1, sizeof(int) * par_host->ParticleNum * par_host->BranchLmtLen / 2));
	checkCudaErrors(cudaMemset(Lpb, -1, sizeof(int) * par_host->ParticleNum * par_host->BranchLmtLen / 2));

}

__global__ void CreateAnei_hGrid_cuk(Parameters* par, size_t pitch, int* Ng, int* Ni, int* Anei,
									 const int* cel_beg, const int* cel_end, const int* part_cel, const int* part_idx,
									 double3* Pos, double* Radius, int* type,
									 double* cellSize, int3* cellNum, int* cellOffset) // 新增分层参数
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= par->ParticleNum) return;
	if (part_cel[index] == par->CellNum) return;

	int tmpi, target_level;
	int id1 = index;
	int my_type = type[index];
	double3 pos_i = Pos[index];
	double radius_i = Radius[index];

	// --- 逻辑改动：不仅检测本层，还要检测更高层级 ---
	// 假设 Level 0 是最粗的，Level MAX 是最细的
	// 颗粒只检测 [本层级] 和 [比自己更粗的层级]
	int loop = 0;
	for (target_level = my_type; target_level >= 0; target_level--)
	{
		double s = cellSize[target_level];
		int3 n = cellNum[target_level];
		int off = cellOffset[target_level];

		// 计算当前颗粒在目标层级网格中的坐标
		int3 celli;
		celli.x = floor((pos_i.x - par->WorldOrigin_x) / s);
		celli.y = floor((pos_i.y - par->WorldOrigin_y) / s);
		celli.z = floor((pos_i.z - par->WorldOrigin_z) / s);

		//printf("check: %d %d %d %d %d %f %f %f\n", loop, target_level, celli.x, celli.y, celli.z, pos_i.x, pos_i.y, pos_i.z);

		loop = loop + 1;

		// 搜索 27 邻域
		for (int cz = -1; cz <= 1; cz++) {
			for (int cy = -1; cy <= 1; cy++) {
				for (int cx = -1; cx <= 1; cx++) {
					int3 cellj = make_int3(celli.x + cx, celli.y + cy, celli.z + cz);

					// 越界检查
					if (cellj.x < 0 || cellj.x >= n.x || cellj.y < 0 || cellj.y >= n.y || cellj.z < 0 || cellj.z >= n.z) continue;

					int cell_idx = off + (cellj.z * n.y * n.x + cellj.y * n.x + cellj.x);

					/*if (cx == 0 && cy == 0 && cz == 0) 
						printf("inside: %d %d %d %d\n", my_type, target_level, cell_idx, off);*/

					if (cel_beg[cell_idx] != -1)
					{
						for (int j = cel_beg[cell_idx]; j < cel_end[cell_idx]; j++)
						{
							int id2 = j;

							if (id2 == index) continue;

							double3 pos_j = Pos[id2];
							double radius_j = Radius[id2];
							double3 dpos = pos_i - pos_j;

							// 碰撞检测阈值建议设为 1.05 或 1.1，1.5 可能过大
							double dist_sq = dpos.x * dpos.x + dpos.y * dpos.y + dpos.z * dpos.z;
							double contact_dist = 1.5 * (radius_i + radius_j);

							// 在 CreateAnei 碰撞判定成功后
							if (dist_sq < contact_dist * contact_dist)
							{
								// 情况 A：同层碰撞 (my_type == target_level)
								if (my_type == target_level) {
									if (id1 < id2) {
										// 这里不需要 atomic，因为只有 id1 会写 Ng[id1]
										int tmpi = Ng[id1];
										((int*)((char*)Anei + id1 * pitch))[tmpi] = id2;
										Ng[id1] = tmpi + 1;
									}
									else {
										// 这里也不需要 atomic，因为只有 id1 会写 Ni[id1]
										int tmpi = Ni[id1];
										((int*)((char*)Anei + id1 * pitch))[par->BranchLmtLen - tmpi - 1] = id2;
										Ni[id1] = tmpi + 1;
									}
								}
								// 情况 B：跨层碰撞 (小搜大，target_level < my_type)
								else {
									// 小颗粒 id1 发现了大颗粒 id2
									// 1. 存入小颗粒自己的列表 (id1 一定大于 id2，所以存入 Ni)
									int tmpi = Ni[id1];
									((int*)((char*)Anei + id1 * pitch))[par->BranchLmtLen - tmpi - 1] = id2;
									Ni[id1] = tmpi + 1;

									// 2. 代替大颗粒 id2，把小颗粒 id1 存入大颗粒的 Ng 中
									// 关键：此时可能有多个小颗粒同时搜到同一个大颗粒 id2，必须 atomic!
									int tmpj = atomicAdd(&Ng[id2], 1);
									int* row_Anei_j = (int*)((char*)Anei + id2 * pitch);
									row_Anei_j[tmpj] = id1;
								}
							}
						}
					}
				}
			}
		}
	}
}

__global__ void CreateContList_hGrid_cuk(Parameters* par, int* Ng, int* Ni, int* Nsum, int* Anei, int* Aref, int* Lpa, int* Lpb, size_t pitch)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= par->ParticleNum) return;

	int j, Nlist, Cmax;
	Cmax = par->BranchLmtLen;

	int* row_Anei_i = (int*)((char*)Anei + i * pitch);
	int* row_Aref_i = (int*)((char*)Aref + i * pitch);

	for (int Igrid = 0; Igrid < Ng[i]; Igrid++)
	{
		// 计算全局接触索引 Nlist
		Nlist = (i == 0) ? Igrid : Nsum[i - 1] + Igrid;

		row_Aref_i[Igrid] = Nlist;
		j = row_Anei_i[Igrid];

		Lpa[Nlist] = i;
		Lpb[Nlist] = j;

		// --- 同步对方的 Aref ---
		int* row_Anei_j = (int*)((char*)Anei + j * pitch);
		int* row_Aref_j = (int*)((char*)Aref + j * pitch);
		int Ni_j = Ni[j]; // 提前读取，减少内存访问

		for (int k = 0; k < Ni_j; k++)
		{
			// 注意：Ni 是从 Anei 的末尾倒着存的
			if (i == row_Anei_j[Cmax - 1 - k])
			{
				row_Aref_j[Cmax - 1 - k] = Nlist;
				break; // 找到就退出，这是分层网格效率的关键
			}
		}
	}
}

__global__ void InheritContHistory_hGrid_cuk(int cp, int* Lpa, int* Lpb, int* Lpb_pre, int* part_idx, int* part_idx_pre, int* Nsum_pre,
											 double3* Dv, double3* Dv_pre, double3* Dt, double3* Dt_pre, double3* Dr, double3* Dr_pre)
{
	int Nlist = blockIdx.x * blockDim.x + threadIdx.x;
	if (Lpa[Nlist] == -1 || Nlist >= cp) return;

	//printf("Nlist: %d %d %d %d\n", Nlist, Lpa[Nlist], Lpb[Nlist], Lpb_pre[Nlist]);

	int i, j, i_pre, j_pre, Nlist_pre;

	i = Lpa[Nlist];
	j = Lpb[Nlist];
	i_pre = part_idx_pre[part_idx[i]];
	j_pre = part_idx_pre[part_idx[j]];

	if (j_pre > i_pre)
	{
		for (Nlist_pre = Nsum_pre[i_pre - 1]; Nlist_pre < Nsum_pre[i_pre]; Nlist_pre++)
		{
			//printf("1: %d %d %d\n", j_pre, Nlist_pre, Lpb_pre[Nlist_pre]);
			if (j_pre == Lpb_pre[Nlist_pre])
			{
				Dv[Nlist] = Dv_pre[Nlist_pre];
				Dt[Nlist] = Dt_pre[Nlist_pre];
				Dr[Nlist] = Dr_pre[Nlist_pre];
				//printf("j_pre > i_pre Nlist: %d %d %d %d\n", i, j, i_pre, j_pre);
				//printf("j_pre > i_pre Nlist: %d %d %d %f %f %f\n", Nlist, Lpa[Nlist], Lpb[Nlist], Dt[Nlist].x, Dt[Nlist].y, Dt[Nlist].z);
			}
		}
	}
	else if (j_pre < i_pre)
	{
		for (Nlist_pre = Nsum_pre[j_pre - 1]; Nlist_pre < Nsum_pre[j_pre]; Nlist_pre++)
		{
			//printf("2: %d %d %d\n", i_pre, Nlist_pre, Lpb_pre[Nlist_pre]);
			if (i_pre == Lpb_pre[Nlist_pre])
			{
				Dv[Nlist] = -1.0 * Dv_pre[Nlist_pre];
				Dt[Nlist] = -1.0 * Dt_pre[Nlist_pre];
				Dr[Nlist] = -1.0 * Dr_pre[Nlist_pre];
				//printf("j_pre > i_pre Nlist: %d %d %d %d\n", i, j, i_pre, j_pre);
				//printf("j_pre < i_pre Nlist: %d %d %d %f %f %f\n", Nlist, Lpa[Nlist], Lpb[Nlist], Dt[Nlist].x, Dt[Nlist].y, Dt[Nlist].z);

			}
		}
	}
}



void DEMHGrid::CreateContList(Parameters* par_host, Parameters* par_device, double3* sortedPos, double* sortedRadius, int* sortedType,
							  double3* Dv, double3* Dv_pre, double3* Dt, double3* Dt_pre, double3* Dr, double3* Dr_pre)
{
	//checkCudaErrors(cudaMemcpyToSymbol(par, par_device, sizeof(Parameters), 0, cudaMemcpyDeviceToDevice));

	dim3 block(128, 1);
	dim3 cuda_grid_size = dim3((par_host->ParticleNum + block.x - 1) / block.x, 1);

	pointerExch(&Lpb, &Lpb_pre);
	pointerExch(&Nsum, &Nsum_pre);
	pointerExch(&Dv, &Dv_pre);
	pointerExch(&Dt, &Dt_pre);
	pointerExch(&Dr, &Dr_pre);

	InitNList(par_host);

	//
	CreateAnei_hGrid_cuk << <cuda_grid_size, block, 0 >> > (par_device, pitch, Ng, Ni, Anei, cell_beg, cell_end, part_cel, part_idx, sortedPos, sortedRadius,
		                                                    sortedType, cellSize, cellNum, cellOffset);
	getLastCudaError("CreateAnei_cuk failed...     \n");
	cudaDeviceSynchronize();

	/*(Parameters* par, size_t pitch, int* Ng, int* Ni, int* Anei,
									 const int* cel_beg, const int* cel_end, const int* part_idx,
									 double3* Pos, double* Radius, int* type,
									 double* cellSize, int3* cellNum, int* cellOffset)*/

	InclusiveNsum(par_host->ParticleNum, Ng, Nsum);
	cudaDeviceSynchronize();

	CreateContList_hGrid_cuk << <cuda_grid_size, block, 0 >> > (par_device, Ng, Ni, Nsum, Anei, Aref, Lpa, Lpb, pitch);
	getLastCudaError("CreateContList_cuk failed...     \n");
	cudaDeviceSynchronize();

	//PrintGridInfo();

	cuda_grid_size.x = (par_host->ParticleNum * (par_host->BranchLmtLen / 2) + block.x - 1) / block.x;

	InheritContHistory_hGrid_cuk << <cuda_grid_size, block, 0 >> > (par_host->ParticleNum * (par_host->BranchLmtLen / 2), Lpa, Lpb, Lpb_pre, part_idx, part_idx_pre, Nsum_pre,
																	Dv, Dv_pre, Dt, Dt_pre, Dr, Dr_pre);
	getLastCudaError("InheritContHistory_cuk failed...     \n");
	cudaDeviceSynchronize();

	pointerExch(&Dv, &Dv_pre);
	pointerExch(&Dt, &Dt_pre);
	pointerExch(&Dr, &Dr_pre);

	checkCudaErrors(cudaMemset(Dv, 0.0, par_host->ParticleNum * par_host->BranchLmtLen / 2 * sizeof(double3)));
	checkCudaErrors(cudaMemset(Dt, 0.0, par_host->ParticleNum * par_host->BranchLmtLen / 2 * sizeof(double3)));
	checkCudaErrors(cudaMemset(Dr, 0.0, par_host->ParticleNum * par_host->BranchLmtLen / 2 * sizeof(double3)));
}
