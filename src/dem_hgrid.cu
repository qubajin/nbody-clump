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

DEMHGrid::DEMHGrid()
{
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

	cellSize = NULL;
	cellNum = NULL;
	cellOffset = NULL;
}

DEMHGrid::~DEMHGrid()
{
	if (part_cel != NULL) cudaFree(part_cel);
	if (part_idx != NULL) cudaFree(part_idx);
	if (cell_beg != NULL) cudaFree(cell_beg);
	if (cell_end != NULL) cudaFree(cell_end);

	if (part_fix != NULL) cudaFree(part_fix);
	if (part_idx_pre != NULL) cudaFree(part_idx_pre);

	if (part_cel_cache != NULL) cudaFree(part_cel_cache);
	if (part_idx_cache != NULL) cudaFree(part_idx_cache);
	if (part_fix_cache != NULL) cudaFree(part_fix_cache);

	if (Ng != NULL) cudaFree(Ng);
	if (Ni != NULL) cudaFree(Ni);
	if (Nsum != NULL) cudaFree(Nsum);

	if (Nsum_pre != NULL) cudaFree(Nsum_pre);

	if (Lpa != NULL) cudaFree(Lpa);
	if (Lpb != NULL) cudaFree(Lpb);

	if (Lbox != NULL) cudaFree(Lbox);

	if (Lpb_pre != NULL) cudaFree(Lpb_pre);

	if (Anei != NULL) cudaFree(Anei);
	if (Aref != NULL) cudaFree(Aref);

	if (cellNum != NULL) cudaFree(cellNum);
	if (cellSize != NULL) cudaFree(cellSize);
	if (cellOffset != NULL) cudaFree(cellOffset);
}

void DEMHGrid::GridConfig(Parameters* par_host)
{
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

	checkCudaErrors(cudaMalloc((void**)&Nsum_pre, par_host->ParticleNum * sizeof(int)));
	checkCudaErrors(cudaMemset(Nsum_pre, 0, sizeof(int) * par_host->ParticleNum));       //Nsum_pre初始化

	checkCudaErrors(cudaMalloc((void**)&Lpa, par_host->ParticleNum * par_host->BranchLmtLen / 2 * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&Lpb, par_host->ParticleNum * par_host->BranchLmtLen / 2 * sizeof(int)));

	checkCudaErrors(cudaMalloc((void**)&Lbox, par_host->ParticleNum * par_host->BranchLmtLen / 2 * sizeof(int)));

	checkCudaErrors(cudaMalloc((void**)&Lpb_pre, par_host->ParticleNum * par_host->BranchLmtLen / 2 * sizeof(int)));
	checkCudaErrors(cudaMemset(Lpb_pre, -1, sizeof(int) * par_host->ParticleNum * par_host->BranchLmtLen / 2));       //Lpb_pre初始化

	checkCudaErrors(cudaMallocPitch(&Anei, &pitch, par_host->BranchLmtLen * sizeof(int), par_host->ParticleNum));
	checkCudaErrors(cudaMallocPitch(&Aref, &pitch, par_host->BranchLmtLen * sizeof(int), par_host->ParticleNum));

	checkCudaErrors(cudaMalloc((void**)&Cache_Int, par_host->ParticleNum * sizeof(int)));

	checkCudaErrors(cudaMalloc((void**)&Cache_Double, par_host->ParticleNum * sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&Cache_Double3, par_host->ParticleNum * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&Cache_Double4, par_host->ParticleNum * sizeof(double4)));

	checkCudaErrors(cudaMalloc((void**)&cellSize, par_host->GridLevels * sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&cellNum, par_host->GridLevels * sizeof(int3)));
	checkCudaErrors(cudaMalloc((void**)&cellOffset, par_host->GridLevels * sizeof(int)));
}

__global__ void PrintHGrid_cuk(double* cellSize, int3* cellNum, int* cellOffset)
{
	for (int i = 0; i < 2; i++)
	{
		printf("cell: %f %d %d %d %d\n", cellSize[i], cellNum[i].x, cellNum[i].y, cellNum[i].z, cellOffset[i]);
	}
}

void DEMHGrid::CopyDataFromHostToDevice(Parameters* par_host, DEMSphereIo* io)
{
	checkCudaErrors(cudaMemcpy(cellSize, io->GetCellSize(), par_host->GridLevels * sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(cellNum, io->GetCellNum(), par_host->GridLevels * sizeof(int3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(cellOffset, io->GetCellOffset(), par_host->GridLevels * sizeof(int), cudaMemcpyHostToDevice));

	/*PrintHGrid_cuk << <1, 1 >> > (cellSize, cellNum, cellOffset);
	getLastCudaError("PrintHGrid_cuk failed...     \n");
	cudaDeviceSynchronize();*/

}

__global__ void GetParticleFixIndex_hGrid_cuk(Parameters* par, int* part_fix)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < par->ParticleNum)
	{
		part_fix[i] = i;
		//printf("GetParticleFixIndex_hGrid_cuk: %d %d\n", par->ParticleNum, part_fix[i]);
	}
}

void DEMHGrid::InitPartFixIdx(Parameters* par_host, Parameters* par_device)
{
	dim3 block(128, 1);
	dim3 cuda_grid_size = dim3((par_host->ParticleNum + block.x - 1) / block.x, 1);

	//checkCudaErrors(cudaMemcpyToSymbolAsync(par, par_device, sizeof(Parameters), 0, cudaMemcpyDeviceToDevice));

	GetParticleFixIndex_hGrid_cuk << <cuda_grid_size, block, 0 >> > (par_device, part_fix);
	getLastCudaError("GetParticleFixIndex_cuk failed...     \n");
	cudaDeviceSynchronize();
}

__global__ void FindBegEndInCell_hGrid_cuk(Parameters* par, int* cell_beg, int* cell_end, int* part_cel)
{
	extern __shared__ int shared_cells[];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int celli;

	if (i < par->ParticleNum)
	{
		celli = part_cel[i];
		shared_cells[threadIdx.x + 1] = celli;

		// First thread in this block
		if (i > 0 && threadIdx.x == 0) shared_cells[0] = part_cel[i - 1];
	}

	__syncthreads();

	if (i < par->ParticleNum)
	{
		if (i == 0 || celli != shared_cells[threadIdx.x])
		{
			cell_beg[celli] = i;
			if (i > 0) cell_end[shared_cells[threadIdx.x]] = i;
		}

		if (i == par->ParticleNum - 1) cell_end[celli] = i + 1;
	}

}

__global__ void GetParticleCellIndex_hGrid_cuk(Parameters* par, int* part_cel, int* part_idx, int* part_idx_pre, 
											   double* cellSize, int3* cellNum, int* cellOffset, double3* pos, int* type)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < par->ParticleNum)
	{
		double3 pos_i = pos[i];
		int t = type[i]; // 获取该颗粒所属层级

		if (pos_i.z > par->BoxUpLmt_z || pos_i.y > par->BoxUpLmt_y || pos_i.x > par->BoxUpLmt_x 
		|| pos_i.z < par->BoxLowLmt_z || pos_i.y < par->BoxLowLmt_y || pos_i.x < par->BoxLowLmt_x)
		{
			part_idx[i] = i;
			part_idx_pre[i] = i;
			part_cel[i] = par->CellNum;
			//printf("i: %d\n", i);
			return;
		}

		double s = cellSize[t];
		int3 n = cellNum[t];

		int3 celli;
		celli.x = floor((pos_i.x - par->WorldOrigin_x) / s);
		celli.y = floor((pos_i.y - par->WorldOrigin_y) / s);
		celli.z = floor((pos_i.z - par->WorldOrigin_z) / s);

		// 关键修改：使用该层的 nx, ny 计算局部索引，再加上该层的 offset
		int cell_idx = cellOffset[t] + (celli.z * n.y * n.x + celli.y * n.x + celli.x);

		part_idx[i] = i;
		part_idx_pre[i] = i;
		part_cel[i] = cell_idx;

		//printf("gridcheck: %d %d %d\n", part_idx[i], part_idx[i], part_cel[i]);
	}
}

void DEMHGrid::SortParticleByCellIndex(Parameters* par_host)
{
	size_t temp_storage_bytes = 0;

	checkCudaErrors(cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, part_cel, part_cel_cache, part_idx_pre, part_idx_cache, par_host->ParticleNum));

	if (temp_storage_bytes > buffer_storage_bytes)
	{
		buffer_storage_bytes = temp_storage_bytes;
		if (buffer != NULL) cudaFree(buffer);
		checkCudaErrors(cudaMalloc(&buffer, buffer_storage_bytes));
	}

	checkCudaErrors(cub::DeviceRadixSort::SortPairs(buffer, temp_storage_bytes, part_cel, part_cel_cache, part_idx_pre, part_idx_cache, par_host->ParticleNum));
	checkCudaErrors(cub::DeviceRadixSort::SortPairs(buffer, temp_storage_bytes, part_cel, part_cel_cache, part_fix, part_fix_cache, par_host->ParticleNum));

	checkCudaErrors(cudaMemcpyAsync(part_cel, part_cel_cache, par_host->ParticleNum * sizeof(int), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpyAsync(part_idx_pre, part_idx_cache, par_host->ParticleNum * sizeof(int), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpyAsync(part_fix, part_fix_cache, par_host->ParticleNum * sizeof(int), cudaMemcpyDeviceToDevice));

}


void DEMHGrid::NeighborSearch(Parameters* par_host, Parameters* par_device, double3* pos, int* type)
{
	dim3 block(128, 1);
	dim3 cuda_grid_size = dim3((par_host->ParticleNum + block.x - 1) / block.x, 1);

	GetParticleCellIndex_hGrid_cuk << <cuda_grid_size, block, 0 >> > (par_device, part_cel, part_idx, part_idx_pre, cellSize, cellNum, cellOffset, pos, type);
	getLastCudaError("GetParticleCellIndex_cuk failed...     \n");
	cudaDeviceSynchronize();

	//PrintPartInfo_cuk << <1, 1 >> > (part_fix, part_idx_pre, part_cel);
	//cudaDeviceSynchronize();

	SortParticleByCellIndex(par_host);
	cudaDeviceSynchronize();

	/*PrintPartInfo_cuk << <1, 1 >> > (part_fix, part_idx_pre, part_cel);
	cudaDeviceSynchronize();*/

	checkCudaErrors(cudaMemsetAsync(cell_beg, -1, par_host->CellNum * sizeof(int)));
	cudaDeviceSynchronize();

	FindBegEndInCell_hGrid_cuk << < cuda_grid_size, block, (block.x + 1) * sizeof(int) >> > (par_device, cell_beg, cell_end, part_cel);
	getLastCudaError("FindBegEndInCell_cuk failed...     \n");
	cudaDeviceSynchronize();

}


__global__ void SortArray_int_hGrid_cuk(Parameters* par, int* part_idx_pre, int* Val, int* sortedVal)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= par->ParticleNum) return;

	int index = part_idx_pre[i];

	//printf("sort: %d %d %f %f\n", i, index, sortedVal[i], Val[index]);

	sortedVal[i] = Val[index];

	//Val[i] = sortedVal[i];
}


void DEMHGrid::SortArray(Parameters* par_host, Parameters* par_device, int* Val)
{
	dim3 block(128, 1);
	dim3 cuda_grid_size = dim3((par_host->ParticleNum + block.x - 1) / block.x, 1);

	checkCudaErrors(cudaMemcpy(Cache_Int, Val, par_host->ParticleNum * sizeof(int), cudaMemcpyDeviceToDevice));

	SortArray_int_hGrid_cuk << <block, cuda_grid_size >> > (par_device, part_idx_pre, Cache_Int, Val);
	getLastCudaError("SortArray_double_cuk failed...     \n");
	cudaDeviceSynchronize();
}



__global__ void SortArray_double_hGrid_cuk(Parameters* par, int* part_idx_pre, double* Val, double* sortedVal)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= par->ParticleNum) return;

	int index = part_idx_pre[i];

	//printf("sort: %d %d %f %f\n", i, index, sortedVal[i], Val[index]);

	sortedVal[i] = Val[index];

	//Val[i] = sortedVal[i];
}


void DEMHGrid::SortArray(Parameters* par_host, Parameters* par_device, double* Val)
{
	dim3 block(128, 1);
	dim3 cuda_grid_size = dim3((par_host->ParticleNum + block.x - 1) / block.x, 1);

	checkCudaErrors(cudaMemcpy(Cache_Double, Val, par_host->ParticleNum * sizeof(double), cudaMemcpyDeviceToDevice));

	SortArray_double_hGrid_cuk << <block, cuda_grid_size >> > (par_device, part_idx_pre, Cache_Double, Val);
	getLastCudaError("SortArray_double_cuk failed...     \n");
	cudaDeviceSynchronize();
}

__global__ void SortArray_double3_hGrid_cuk(Parameters* par, int* part_idx_pre, double3* Val, double3* sortedVal)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= par->ParticleNum) return;

	int index = part_idx_pre[i];

	sortedVal[i] = Val[index];

}


void DEMHGrid::SortArray(Parameters* par_host, Parameters* par_device, double3* Val)
{
	dim3 block(128, 1);
	dim3 cuda_grid_size = dim3((par_host->ParticleNum + block.x - 1) / block.x, 1);

	checkCudaErrors(cudaMemcpy(Cache_Double3, Val, par_host->ParticleNum * sizeof(double3), cudaMemcpyDeviceToDevice));

	SortArray_double3_hGrid_cuk << <block, cuda_grid_size >> > (par_device, part_idx_pre, Cache_Double3, Val);
	getLastCudaError("SortArray_double3_cuk failed...     \n");
	cudaDeviceSynchronize();
}

__global__ void SortArray_double4_hGrid_cuk(Parameters* par, int* part_idx_pre, double4* Val, double4* sortedVal)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= par->ParticleNum) return;

	int index = part_idx_pre[i];

	sortedVal[i] = Val[index];
}


void DEMHGrid::SortArray(Parameters* par_host, Parameters* par_device, double4* Val)
{
	dim3 block(128, 1);
	dim3 cuda_grid_size = dim3((par_host->ParticleNum + block.x - 1) / block.x, 1);

	checkCudaErrors(cudaMemcpy(Cache_Double4, Val, par_host->ParticleNum * sizeof(double4), cudaMemcpyDeviceToDevice));

	SortArray_double4_hGrid_cuk << <block, cuda_grid_size >> > (par_device, part_idx_pre, Cache_Double4, Val);
	getLastCudaError("SortArray_double3_cuk failed...     \n");
	cudaDeviceSynchronize();
}


