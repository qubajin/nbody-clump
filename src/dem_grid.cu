#include <math.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <cstdlib>

#include "dem_grid.h"

#include "thrust/sort.h"
#include "thrust/device_ptr.h"
#include "thrust/scan.h"
#include <device_launch_parameters.h>
#include <vector_functions.h>

// Set by GPU threads when either neighbor list exceeds its capacity.
__device__ int nbody_neighbor_overflow = 0;

// static __constant__ Parameters par;

DEMGrid::DEMGrid() {
    buffer_storage_bytes = 0;
    pitch = 0;

    buffer = NULL;

    part_cel = NULL;
    part_idx = NULL;
    cell_beg = NULL;
    cell_end = NULL;

    part_fix = NULL;
    part_idx_pre = NULL;

    part_cel_cache = NULL;
    part_idx_cache = NULL;
    part_fix_cache = NULL;

    Ng = NULL;
    Ni = NULL;
    Nsum = NULL;

    Nsum_pre = NULL;

    Lpa = NULL;
    Lpb = NULL;

    Lbox = NULL;

    Lpb_pre = NULL;

    Anei = NULL;
    Aref = NULL;
}

DEMGrid::~DEMGrid() {
    if (part_cel != NULL)
        cudaFree(part_cel);
    if (part_idx != NULL)
        cudaFree(part_idx);
    if (cell_beg != NULL)
        cudaFree(cell_beg);
    if (cell_end != NULL)
        cudaFree(cell_end);

    if (part_fix != NULL)
        cudaFree(part_fix);
    if (part_idx_pre != NULL)
        cudaFree(part_idx_pre);

    if (part_cel_cache != NULL)
        cudaFree(part_cel_cache);
    if (part_idx_cache != NULL)
        cudaFree(part_idx_cache);
    if (part_fix_cache != NULL)
        cudaFree(part_fix_cache);

    if (Ng != NULL)
        cudaFree(Ng);
    if (Ni != NULL)
        cudaFree(Ni);
    if (Nsum != NULL)
        cudaFree(Nsum);

    if (Nsum_pre != NULL)
        cudaFree(Nsum_pre);

    if (Lpa != NULL)
        cudaFree(Lpa);
    if (Lpb != NULL)
        cudaFree(Lpb);

    if (Lbox != NULL)
        cudaFree(Lbox);

    if (Lpb_pre != NULL)
        cudaFree(Lpb_pre);

    if (Anei != NULL)
        cudaFree(Anei);
    if (Aref != NULL)
        cudaFree(Aref);
}

__global__ void FindBegEndInCell_cuk(Parameters* par, int* cell_beg, int* cell_end, int* part_cel) {
    // ��̬�����ڴ��СӦΪ (blockDim.x + 1) * sizeof(int)
    extern __shared__ int shared_cells[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int celli = -1; // Ĭ����Чֵ

    // 1. ��ȫ�ؼ������ݵ������ڴ�
    if (i < par->ParticleNum) {
        celli = part_cel[i];
        shared_cells[threadIdx.x + 1] = celli;

        // ���� Block �ĵ�һ���̣߳���ȡȫ���ڴ���ǰһ�����ӵ� CellIndex
        if (threadIdx.x == 0) {
            if (i > 0)
                shared_cells[0] = part_cel[i - 1];
            else
                shared_cells[0] = -1; // �� 0 ������û��ǰ��
        }
    }

    // ����ͬ����ȷ�� shared_cells[threadIdx.x] ���ڵ�ǰ�߳��ǿ��õ�
    __syncthreads();

    // 2. ֻ������Ч���ӷ�Χ�ڲŽ���д�����
    if (i < par->ParticleNum) {
        // �ؼ�������ȷ�� celli û�г��� cell_beg/cell_end �ķ��䷶Χ
        // ��ֹ GetParticleCellIndex_cuk �����������Խ��
        if (celli >= 0 && celli < par->CellNum) {
            // �����ǰ���ӵ� Cell ��ǰһ����ͬ��˵������ Cell �Ŀ�ʼ
            if (i == 0 || celli != shared_cells[threadIdx.x]) {
                cell_beg[celli] = i;

                // ͬʱ��������ǵ� 0 �����ӣ�˵����Ҳ��ǰһ�� Cell �Ľ���
                if (i > 0) {
                    int prev_cell = shared_cells[threadIdx.x];
                    if (prev_cell >= 0 && prev_cell < par->CellNum) {
                        cell_end[prev_cell] = i;
                    }
                }
            }

            // �������һ�����ӵ��������������Ȼ�������� Cell �Ľ���
            if (i == par->ParticleNum - 1) {
                cell_end[celli] = i + 1;
            }
        }
    }
}

__global__ void GetParticleCellIndex_cuk(Parameters* par, int* part_cel, int* part_idx,
                                         int* part_idx_pre, double3* pos) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < par->ParticleNum) {
        double3 pos_i = pos[i];

        // 1. ��ʼ������ֵ
        part_idx[i] = i;
        part_idx_pre[i] = i;

        // 2. ��Χ��飨����΢С�� epsilon
        // �ݲ��ֹ��Ե������
        if (pos_i.z >= par->BoxUpLmt_z || pos_i.y >= par->BoxUpLmt_y ||
            pos_i.x >= par->BoxUpLmt_x || pos_i.z < par->BoxLowLmt_z ||
            pos_i.y < par->BoxLowLmt_y || pos_i.x < par->BoxLowLmt_x) {
            // ��������ܳ��߽磬�������Ϊ���һ��������Ͱ������
            part_cel[i] = par->CellNum - 1;
            return;
        }

        // 3. ������������
        int3 celli;
        // ʹ�� fmax/fmin ȷ��������� celli ������Ϊ�������
        celli.x = (int)floor((pos_i.x - par->WorldOrigin_x) / par->CellSize_x);
        celli.y = (int)floor((pos_i.y - par->WorldOrigin_y) / par->CellSize_y);
        celli.z = (int)floor((pos_i.z - par->WorldOrigin_z) / par->CellSize_z);

        // 4. �ؼ������������������� [0, GridDim - 1] ֮��
        // ��һ���ܳ��׷�ֹ���� cell_idx
        // �������ɸ�����޴����
        celli.x = max(0, min(par->GridDim_x - 1, celli.x));
        celli.y = max(0, min(par->GridDim_y - 1, celli.y));
        celli.z = max(0, min(par->GridDim_z - 1, celli.z));

        // 5. ����һά����
        int cell_idx =
            celli.z * par->GridDim_y * par->GridDim_x + celli.y * par->GridDim_x + celli.x;

        // ���һ�ζ��׼��
        if (cell_idx < 0)
            cell_idx = 0;
        if (cell_idx >= par->CellNum)
            cell_idx = par->CellNum - 1;

        part_cel[i] = cell_idx;
    }
}

__global__ void GetParticleFixIndex_cuk(Parameters* par, int* part_fix) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < par->ParticleNum) {
        part_fix[i] = i;
        // printf("%d %d\n", par->ParticleNum, part_fix[i]);
    }
}

void DEMGrid::GridConfig(Parameters* par_host) {
    checkCudaErrors(cudaMalloc((void**)&part_cel, par_host->ParticleNum * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&part_idx, par_host->ParticleNum * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&cell_beg, par_host->CellNum * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&cell_end, par_host->CellNum * sizeof(int)));

    checkCudaErrors(cudaMalloc((void**)&part_fix, par_host->ParticleNum * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&part_idx_pre, par_host->ParticleNum * sizeof(int)));

    checkCudaErrors(cudaMalloc((void**)&part_cel_cache, par_host->ParticleNum * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&part_idx_cache, par_host->ParticleNum * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&part_fix_cache, par_host->ParticleNum * sizeof(int)));

    checkCudaErrors(cudaMalloc((void**)&Ng, par_host->ParticleNum * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&Ni, par_host->ParticleNum * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&Nsum, par_host->ParticleNum * sizeof(int)));
    checkCudaErrors(cudaMemset(Nsum, 0, sizeof(int) * par_host->ParticleNum));

    checkCudaErrors(cudaMalloc((void**)&Nsum_pre, par_host->ParticleNum * sizeof(int)));
    checkCudaErrors(cudaMemset(Nsum_pre, 0, sizeof(int) * par_host->ParticleNum)); // Nsum_pre��ʼ��

    checkCudaErrors(
        cudaMalloc((void**)&Lpa, par_host->ParticleNum * par_host->BranchLmtLen / 2 * sizeof(int)));
    checkCudaErrors(
        cudaMalloc((void**)&Lpb, par_host->ParticleNum * par_host->BranchLmtLen / 2 * sizeof(int)));
    checkCudaErrors(cudaMemset(Lpb, -1,
                               sizeof(int) * par_host->ParticleNum * par_host->BranchLmtLen / 2));

    checkCudaErrors(cudaMalloc((void**)&Lbox,
                               par_host->ParticleNum * par_host->BranchLmtLen / 2 * sizeof(int)));

    checkCudaErrors(cudaMalloc((void**)&Lpb_pre,
                               par_host->ParticleNum * par_host->BranchLmtLen / 2 * sizeof(int)));
    checkCudaErrors(cudaMemset(Lpb_pre, -1,
                               sizeof(int) * par_host->ParticleNum * par_host->BranchLmtLen /
                                   2)); // Lpb_pre��ʼ��

    checkCudaErrors(cudaMallocPitch(&Anei, &pitch, par_host->BranchLmtLen * sizeof(int),
                                    par_host->ParticleNum));
    checkCudaErrors(cudaMallocPitch(&Aref, &pitch, par_host->BranchLmtLen * sizeof(int),
                                    par_host->ParticleNum));

    checkCudaErrors(cudaMalloc((void**)&Cache_Int, par_host->ParticleNum * sizeof(int)));

    checkCudaErrors(cudaMalloc((void**)&Cache_Double, par_host->ParticleNum * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&Cache_Double3, par_host->ParticleNum * sizeof(double3)));
    checkCudaErrors(cudaMalloc((void**)&Cache_Double4, par_host->ParticleNum * sizeof(double4)));
}

void DEMGrid::InitPartFixIdx(Parameters* par_host, Parameters* par_device) {
    dim3 block(128, 1);
    dim3 cuda_grid_size = dim3((par_host->ParticleNum + block.x - 1) / block.x, 1);

    // checkCudaErrors(cudaMemcpyToSymbolAsync(par, par_device, sizeof(Parameters), 0,
    // cudaMemcpyDeviceToDevice));

    GetParticleFixIndex_cuk<<<cuda_grid_size, block, 0>>>(par_device, part_fix);
    getLastCudaError("GetParticleFixIndex_cuk failed...     \n");
    cudaDeviceSynchronize();
}

void DEMGrid::GridZero(Parameters* par_host) {
    checkCudaErrors(cudaMemset(
        Lbox, 0, sizeof(int) * par_host->ParticleNum * par_host->BranchLmtLen / 2)); // Lpb_pre��ʼ��
}

void DEMGrid::SortParticleByCellIndex(Parameters* par_host) {
    size_t temp_storage_bytes = 0;

    checkCudaErrors(cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, part_cel,
                                                    part_cel_cache, part_idx_pre, part_idx_cache,
                                                    par_host->ParticleNum));

    if (temp_storage_bytes > buffer_storage_bytes) {
        buffer_storage_bytes = temp_storage_bytes;
        if (buffer != NULL)
            cudaFree(buffer);
        checkCudaErrors(cudaMalloc(&buffer, buffer_storage_bytes));
    }

    checkCudaErrors(cub::DeviceRadixSort::SortPairs(buffer, temp_storage_bytes, part_cel,
                                                    part_cel_cache, part_idx_pre, part_idx_cache,
                                                    par_host->ParticleNum));
    checkCudaErrors(cub::DeviceRadixSort::SortPairs(buffer, temp_storage_bytes, part_cel,
                                                    part_cel_cache, part_fix, part_fix_cache,
                                                    par_host->ParticleNum));

    checkCudaErrors(cudaMemcpyAsync(part_cel, part_cel_cache, par_host->ParticleNum * sizeof(int),
                                    cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyAsync(part_idx_pre, part_idx_cache,
                                    par_host->ParticleNum * sizeof(int), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyAsync(part_fix, part_fix_cache, par_host->ParticleNum * sizeof(int),
                                    cudaMemcpyDeviceToDevice));
}

__global__ void PrintPartInfo_cuk(int* part_fix, int* part_idx, int* part_cel) {
    for (int i = 0; i < 2; i++) {
        if (part_fix[i] == 2)
            printf("%d %d %d\n", part_fix[i], part_idx[i], part_cel[i]);
    }
}

__global__ void PrintLpbInfo_cuk(int* Lpb, int* Lpb_pre) {
    for (int i = 0; i < 4; i++) {
        printf("PrintLpbInfo: %d %d\n", Lpb[i], Lpb_pre[i]);
    }
    printf("==================================\n");
}

__global__ void PrintNum_cuk(int* Lpa, int* Lpb) {
    for (int i = 0; i < 4; i++) {
        // if(Lpa[i]!=-1) printf("Nsum: %d %d %d\n", i, Lpa[i], Lpb[i]);
        printf("Nsum: %d %d %d\n", i, Lpa[i], Lpb[i]);
    }
    // printf("==================================\n");
}

void DEMGrid::PrintGridInfo() {
    // PrintPartInfo_cuk << <1, 1 >> > (part_fix, part_idx, part_cel);
    // PrintLpbInfo_cuk << <1, 1 >> > (Lpb, Lpb_pre);
    PrintNum_cuk<<<1, 1>>>(Lpa, Lpb);
    cudaDeviceSynchronize();
}

__global__ void SortArray_int_cuk(Parameters* par, int* part_idx_pre, int* Val, int* sortedVal) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= par->ParticleNum)
        return;

    int index = part_idx_pre[i];

    // printf("sort: %d %d %f %f\n", i, index, sortedVal[i], Val[index]);

    sortedVal[i] = Val[index];

    // Val[i] = sortedVal[i];
}

void DEMGrid::SortArray(Parameters* par_host, Parameters* par_device, int* Val) {
    dim3 block(128, 1);
    dim3 cuda_grid_size = dim3((par_host->ParticleNum + block.x - 1) / block.x, 1);

    checkCudaErrors(
        cudaMemcpy(Cache_Int, Val, par_host->ParticleNum * sizeof(int), cudaMemcpyDeviceToDevice));

    SortArray_int_cuk<<<cuda_grid_size, block>>>(par_device, part_idx_pre, Cache_Int, Val);
    getLastCudaError("SortArray_double_cuk failed...     \n");
    cudaDeviceSynchronize();
}

__global__ void SortArray_double_cuk(Parameters* par, int* part_idx_pre, double* Val,
                                     double* sortedVal) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= par->ParticleNum)
        return;

    int index = part_idx_pre[i];

    // printf("sort: %d %d %f %f\n", i, index, sortedVal[i], Val[index]);

    sortedVal[i] = Val[index];

    // Val[i] = sortedVal[i];
}

void DEMGrid::SortArray(Parameters* par_host, Parameters* par_device, double* Val) {
    dim3 block(128, 1);
    dim3 cuda_grid_size = dim3((par_host->ParticleNum + block.x - 1) / block.x, 1);

    checkCudaErrors(cudaMemcpy(Cache_Double, Val, par_host->ParticleNum * sizeof(double),
                               cudaMemcpyDeviceToDevice));

    SortArray_double_cuk<<<cuda_grid_size, block>>>(par_device, part_idx_pre, Cache_Double, Val);
    getLastCudaError("SortArray_double_cuk failed...     \n");
    cudaDeviceSynchronize();
}

__global__ void SortArray_double3_cuk(Parameters* par, int* part_idx_pre, double3* Val,
                                      double3* sortedVal) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= par->ParticleNum)
        return;

    int index = part_idx_pre[i];

    sortedVal[i] = Val[index];
}

void DEMGrid::SortArray(Parameters* par_host, Parameters* par_device, double3* Val) {
    dim3 block(128, 1);
    dim3 cuda_grid_size = dim3((par_host->ParticleNum + block.x - 1) / block.x, 1);

    checkCudaErrors(cudaMemcpy(Cache_Double3, Val, par_host->ParticleNum * sizeof(double3),
                               cudaMemcpyDeviceToDevice));

    SortArray_double3_cuk<<<cuda_grid_size, block>>>(par_device, part_idx_pre, Cache_Double3, Val);
    getLastCudaError("SortArray_double3_cuk failed...     \n");
    cudaDeviceSynchronize();
}

__global__ void SortArray_double4_cuk(Parameters* par, int* part_idx_pre, double4* Val,
                                      double4* sortedVal) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= par->ParticleNum)
        return;

    int index = part_idx_pre[i];

    sortedVal[i] = Val[index];
}

void DEMGrid::SortArray(Parameters* par_host, Parameters* par_device, double4* Val) {
    dim3 block(128, 1);
    dim3 cuda_grid_size = dim3((par_host->ParticleNum + block.x - 1) / block.x, 1);

    checkCudaErrors(cudaMemcpy(Cache_Double4, Val, par_host->ParticleNum * sizeof(double4),
                               cudaMemcpyDeviceToDevice));

    SortArray_double4_cuk<<<cuda_grid_size, block>>>(par_device, part_idx_pre, Cache_Double4, Val);
    getLastCudaError("SortArray_double3_cuk failed...     \n");
    cudaDeviceSynchronize();
}

void DEMGrid::InitNList(Parameters* par_host) {
    checkCudaErrors(cudaMemset(Ng, 0, sizeof(int) * par_host->ParticleNum));
    checkCudaErrors(cudaMemset(Ni, 0, sizeof(int) * par_host->ParticleNum));

    checkCudaErrors(
        cudaMemset(Lpa, -1, sizeof(int) * par_host->ParticleNum * par_host->BranchLmtLen / 2));
    checkCudaErrors(
        cudaMemset(Lpb, -1, sizeof(int) * par_host->ParticleNum * par_host->BranchLmtLen / 2));
    // Mark unused neighbor slots as invalid before rebuilding the list.
    checkCudaErrors(cudaMemset2D(Anei, pitch, 0xFF, par_host->BranchLmtLen * sizeof(int),
                                 par_host->ParticleNum));
}

void DEMGrid::pointerExch(int** pt1, int** pt2) {
    int* tmp;
    tmp = *pt1;
    *pt1 = *pt2;
    *pt2 = tmp;
}

void DEMGrid::pointerExch(double3** pt1, double3** pt2) {
    double3* tmp;
    tmp = *pt1;
    *pt1 = *pt2;
    *pt2 = tmp;
}

__global__ void CreateAnei_cuk(Parameters* par, size_t pitch, int* Ng, int* Ni, int* Anei,
                               const int* cel_beg, const int* cel_end, const int* part_cel,
                               const int* part_idx, double3* Pos, double* Radius) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= par->ParticleNum)
        return;

    // ����Ͱ��������
    if (part_cel[index] == par->CellNum - 1)
        return;

    double3 pos_i = Pos[index];
    double radius_i = Radius[index];

    // ���㵱ǰ�������ڵ����� 3D ����
    int3 celli;
    celli.x = floor((pos_i.x - par->WorldOrigin_x) * par->i_CellSize_x);
    celli.y = floor((pos_i.y - par->WorldOrigin_y) * par->i_CellSize_y);
    celli.z = floor((pos_i.z - par->WorldOrigin_z) * par->i_CellSize_z);

    // ��ȡ��ǰ������ Anei �е����׵�ַ (ʹ�� char*
    // ǿ�ư��ֽ�ƫ��)
    int* row_Anei = (int*)((char*)Anei + index * pitch);

    for (int cz = -1; cz <= 1; cz++) {
        for (int cy = -1; cy <= 1; cy++) {
            for (int cx = -1; cx <= 1; cx++) {

                int3 cellj = make_int3(celli.x + cx, celli.y + cy, celli.z + cz);

                // 1. ����߽��飺��ֹ cell_idx ��������������ֵ��±���
                if (cellj.x < 0 || cellj.x >= par->GridDim_x || cellj.y < 0 ||
                    cellj.y >= par->GridDim_y || cellj.z < 0 || cellj.z >= par->GridDim_z)
                    continue;

                int cell_idx = (cellj.z * par->GridDim_y + cellj.y) * par->GridDim_x + cellj.x;

                int start = cel_beg[cell_idx];
                if (start != -1) {
                    int end = cel_end[cell_idx];
                    for (int j = start; j < end; j++) {
                        if (j == index)
                            continue;

                        double3 pos_j = Pos[j];
                        double radius_j = Radius[j];

                        // �������
                        double dx = pos_i.x - pos_j.x;
                        double dy = pos_i.y - pos_j.y;
                        double dz = pos_i.z - pos_j.z;
                        double dist_sq = dx * dx + dy * dy + dz * dz;
                        double limit_dist = 1.50 * (radius_i + radius_j);

                        if (dist_sq < limit_dist * limit_dist) {
                            // 2. �ھ��б������� + Printf ����
                            if (index < j) {
                                int count = Ng[index];
                                if (count < par->BranchLmtLen / 2) {
                                    row_Anei[count] = j;
                                    Ng[index] = count + 1;
                                } else {
                                    // ���ڷ�������ĵ�һ���̴߳�ӡ�������ǧ��������־ˢ��
                                    if (count == par->BranchLmtLen / 2) {
                                        atomicExch(&nbody_neighbor_overflow, 1);
                                    }
                                }
                            } else {
                                int count = Ni[index];
                                if (count < par->BranchLmtLen / 2) {
                                    // ע�⣺����һ��Ҫȷ���±겻Ϊ����
                                    int write_pos = par->BranchLmtLen - 1 - count;
                                    row_Anei[write_pos] = j;
                                    Ni[index] = count + 1;
                                } else {
                                    if (count == par->BranchLmtLen / 2) {
                                        atomicExch(&nbody_neighbor_overflow, 1);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

//
void DEMGrid::InclusiveNsum(int ParticleNum, int* Ng, int* Nsum) {
    thrust::inclusive_scan(thrust::device_ptr<int>(Ng), thrust::device_ptr<int>(Ng + ParticleNum),
                           thrust::device_ptr<int>(Nsum));
    // printf("Inclusive!!\n");
}

__global__ void CreateContList_cuk(Parameters* par, int* Ng, int* Ni, int* Nsum, int* Anei,
                                   int* Aref, int* Lpa, int* Lpb, size_t pitch) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= par->ParticleNum)
        return;

    int j, Nlist, Cmax;
    int *row_Anei, *row_Anei2, *row_Aref, *row_Aref2;

    Cmax = par->BranchLmtLen;

    row_Anei = (int*)((char*)Anei + i * pitch);
    row_Aref = (int*)((char*)Aref + i * pitch);

    for (int Igrid = 0; Igrid < Ng[i]; Igrid++) {
        if (i == 0) {
            Nlist = Igrid;
        } else {
            Nlist = Nsum[i - 1] + Igrid;
        }

        row_Aref[Igrid] = Nlist;

        j = row_Anei[Igrid];

        Lpa[Nlist] = i;
        Lpb[Nlist] = j;

        // if(i==0) printf("%d %d %d %d %d\n", Nlist, Nsum[i - 1], Igrid, Lpa[Nlist], Lpb[Nlist]);

        for (int k = 0; k < Ni[j]; k++) {
            row_Anei2 = (int*)((char*)Anei + j * pitch);

            if (i == row_Anei2[Cmax - 1 - k]) {
                // if (i == 20 && j == 21) printf("grid: %d %d\n", Cmax - 1 - k, Nlist);
                row_Aref2 = (int*)((char*)Aref + j * pitch);
                row_Aref2[Cmax - 1 - k] = Nlist;
            }
        }
    }
}

__global__ void InheritContHistory_cuk(int cp, int* Lpa, int* Lpb, int* Lpb_pre, int* part_idx,
                                       int* part_idx_pre, int* Nsum_pre, double3* Dv,
                                       double3* Dv_pre, double3* Dt, double3* Dt_pre, double3* Dr,
                                       double3* Dr_pre) {
    int Nlist = blockIdx.x * blockDim.x + threadIdx.x;
    if (Nlist >= cp || Lpa[Nlist] == -1)
        return;

    // printf("Nlist: %d %d %d %d\n", Nlist, Lpa[Nlist], Lpb[Nlist], Lpb_pre[Nlist]);

    int i, j, i_pre, j_pre, Nlist_pre;

    i = Lpa[Nlist];
    j = Lpb[Nlist];
    i_pre = part_idx_pre[part_idx[i]];
    j_pre = part_idx_pre[part_idx[j]];

    if (j_pre > i_pre) {
        const int history_begin = (i_pre == 0) ? 0 : Nsum_pre[i_pre - 1];
        for (Nlist_pre = history_begin; Nlist_pre < Nsum_pre[i_pre]; Nlist_pre++) {
            // printf("1: %d %d %d\n", j_pre, Nlist_pre, Lpb_pre[Nlist_pre]);
            if (j_pre == Lpb_pre[Nlist_pre]) {
                Dv[Nlist] = Dv_pre[Nlist_pre];
                Dt[Nlist] = Dt_pre[Nlist_pre];
                Dr[Nlist] = Dr_pre[Nlist_pre];
                // printf("j_pre > i_pre Nlist: %d %d %d %d\n", i, j, i_pre, j_pre);
                // printf("j_pre > i_pre Nlist: %d %d %d %f %f %f\n", Nlist, Lpa[Nlist], Lpb[Nlist],
                // Dt[Nlist].x, Dt[Nlist].y, Dt[Nlist].z);
            }
        }
    } else if (j_pre < i_pre) {
        const int history_begin = (j_pre == 0) ? 0 : Nsum_pre[j_pre - 1];
        for (Nlist_pre = history_begin; Nlist_pre < Nsum_pre[j_pre]; Nlist_pre++) {
            // printf("2: %d %d %d\n", i_pre, Nlist_pre, Lpb_pre[Nlist_pre]);
            if (i_pre == Lpb_pre[Nlist_pre]) {
                Dv[Nlist] = -1.0 * Dv_pre[Nlist_pre];
                Dt[Nlist] = -1.0 * Dt_pre[Nlist_pre];
                Dr[Nlist] = -1.0 * Dr_pre[Nlist_pre];
                // printf("j_pre > i_pre Nlist: %d %d %d %d\n", i, j, i_pre, j_pre);
                // printf("j_pre < i_pre Nlist: %d %d %d %f %f %f\n", Nlist, Lpa[Nlist], Lpb[Nlist],
                // Dt[Nlist].x, Dt[Nlist].y, Dt[Nlist].z);
            }
        }
    }
}

void DEMGrid::NeighborSearch(Parameters* par_host, Parameters* par_device, double3* pos) {
    // checkCudaErrors(cudaMemcpyToSymbol(par, par_device, sizeof(Parameters), 0,
    // cudaMemcpyDeviceToDevice));

    dim3 block(128, 1);
    dim3 cuda_grid_size = dim3((par_host->ParticleNum + block.x - 1) / block.x, 1);

    GetParticleCellIndex_cuk<<<cuda_grid_size, block, 0>>>(par_device, part_cel, part_idx,
                                                           part_idx_pre, pos);
    getLastCudaError("GetParticleCellIndex_cuk failed...     \n");
    cudaDeviceSynchronize();

    // PrintPartInfo_cuk << <1, 1 >> > (part_fix, part_idx_pre, part_cel);
    // cudaDeviceSynchronize();

    SortParticleByCellIndex(par_host);
    cudaDeviceSynchronize();

    /*PrintPartInfo_cuk << <1, 1 >> > (part_fix, part_idx_pre, part_cel);
    cudaDeviceSynchronize();*/

    checkCudaErrors(cudaMemsetAsync(cell_beg, -1, par_host->CellNum * sizeof(int)));
    checkCudaErrors(cudaMemsetAsync(cell_end, -1, par_host->CellNum * sizeof(int)));
    cudaDeviceSynchronize();

    FindBegEndInCell_cuk<<<cuda_grid_size, block, (block.x + 1) * sizeof(int)>>>(
        par_device, cell_beg, cell_end, part_cel);
    getLastCudaError("FindBegEndInCell_cuk failed...     \n");
    cudaDeviceSynchronize();
}

void DEMGrid::CreateContList(Parameters* par_host, Parameters* par_device, double3* sortedPos,
                             double* sortedRadius, double3* Dv, double3* Dv_pre, double3* Dt,
                             double3* Dt_pre, double3* Dr, double3* Dr_pre) {
    // checkCudaErrors(cudaMemcpyToSymbol(par, par_device, sizeof(Parameters), 0,
    // cudaMemcpyDeviceToDevice));

    dim3 block(128, 1);
    dim3 cuda_grid_size = dim3((par_host->ParticleNum + block.x - 1) / block.x, 1);

    pointerExch(&Lpb, &Lpb_pre);
    pointerExch(&Nsum, &Nsum_pre);
    pointerExch(&Dv, &Dv_pre);
    pointerExch(&Dt, &Dt_pre);
    pointerExch(&Dr, &Dr_pre);

    InitNList(par_host);

    //
    // Reset the overflow flag before each neighbor-list construction.
    int overflow = 0;
    checkCudaErrors(cudaMemcpyToSymbol(nbody_neighbor_overflow, &overflow, sizeof(overflow)));

    CreateAnei_cuk<<<cuda_grid_size, block, 0>>>(par_device, pitch, Ng, Ni, Anei, cell_beg,
                                                 cell_end, part_cel, part_idx, sortedPos,
                                                 sortedRadius);

    getLastCudaError("CreateAnei_cuk failed...\n");
    checkCudaErrors(cudaDeviceSynchronize());

    // Reject an incomplete neighbor list before contact-force calculation.
    checkCudaErrors(cudaMemcpyFromSymbol(&overflow, nbody_neighbor_overflow, sizeof(overflow)));

    if (overflow != 0) {
        std::fprintf(stderr,
                     "FATAL: Neighbor list overflow (BranchLmtLen=%d). "
                     "Simulation stopped before using the incomplete list. "
                     "Review neighbor capacity and GPU memory before rerunning.\n",
                     par_host->BranchLmtLen);
        std::exit(EXIT_FAILURE);
    }

    InclusiveNsum(par_host->ParticleNum, Ng, Nsum);
    cudaDeviceSynchronize();

    CreateContList_cuk<<<cuda_grid_size, block, 0>>>(par_device, Ng, Ni, Nsum, Anei, Aref, Lpa, Lpb,
                                                     pitch);
    getLastCudaError("CreateContList_cuk failed...     \n");
    cudaDeviceSynchronize();

    // PrintGridInfo();

    cuda_grid_size.x =
        (par_host->ParticleNum * (par_host->BranchLmtLen / 2) + block.x - 1) / block.x;

    InheritContHistory_cuk<<<cuda_grid_size, block, 0>>>(
        par_host->ParticleNum * (par_host->BranchLmtLen / 2), Lpa, Lpb, Lpb_pre, part_idx,
        part_idx_pre, Nsum_pre, Dv, Dv_pre, Dt, Dt_pre, Dr, Dr_pre);
    getLastCudaError("InheritContHistory_cuk failed...     \n");
    checkCudaErrors(cudaDeviceSynchronize());

    pointerExch(&Dv, &Dv_pre);
    pointerExch(&Dt, &Dt_pre);
    pointerExch(&Dr, &Dr_pre);

    checkCudaErrors(
        cudaMemset(Dv, 0.0, par_host->ParticleNum * par_host->BranchLmtLen / 2 * sizeof(double3)));
    checkCudaErrors(
        cudaMemset(Dt, 0.0, par_host->ParticleNum * par_host->BranchLmtLen / 2 * sizeof(double3)));
    checkCudaErrors(
        cudaMemset(Dr, 0.0, par_host->ParticleNum * par_host->BranchLmtLen / 2 * sizeof(double3)));
}