#pragma once
#ifndef _DEM_GRID_H_
#define _DEM_GRID_H_

#include"type.h"
#include"dem_parameters.h"

#include<cuda_runtime.h>

class DEMGrid
{
private:

	size_t buffer_storage_bytes;
	size_t pitch;

	void* buffer;

	int* part_cel;
	int* part_idx;
	int* cell_beg;
	int* cell_end;

	int* part_fix;
	int* part_idx_pre;

	int* part_fix_cache;
	int* part_cel_cache;
	int* part_idx_cache;

	int* Ng;
	int* Ni;
	int* Nsum;

	int* Nsum_pre;

	int* Lpa;
	int* Lpb;

	int* Lpb_pre;

	int* Lbox;

	int* Anei;
	int* Aref;

	int* Cache_Int;
	double* Cache_Double;
	double3* Cache_Double3;
	double4* Cache_Double4;

	void SortParticleByCellIndex(Parameters* par_host);

	void PrintGridInfo();

	void InclusiveNsum(int ParticleNum, int* Ng, int* Nsum);

	void InitNList(Parameters* par_host);

	void pointerExch(int** pt1, int** pt2);

	void pointerExch(double3** pt1, double3** pt2);

public:

	DEMGrid();

	~DEMGrid();

	size_t GetPitch() { return pitch; }
	int* GetPartCel() { return part_cel; }
	int* GetPartIdx() { return part_idx; }
	int* GetPartFix() { return part_fix; }
	int* GetCellBeg() { return cell_beg; }
	int* GetCellEnd() { return cell_end; }

	int* GetPartIdxPre() { return part_idx_pre; }

	int* GetNg() { return Ng; }
	int* GetNi() { return Ni; }
	int* GetNsum() { return Nsum; }
	int* GetNsumPre() { return Nsum_pre; }

	int* GetLpa() { return Lpa; }
	int* GetLpb() { return Lpb; }
	int* GetLpbPre() { return Lpb_pre; }
	int* GetLbox() { return Lbox; }
	int* GetAnei() { return Anei; }
	int* GetAref() { return Aref; }

	void InitPartFixIdx(Parameters* par_host, Parameters* par_device);

	void GridConfig(Parameters* par_host);

	void NeighborSearch(Parameters* par_host, Parameters* par_device, double3* pos);

	void GridZero(Parameters* par_host);

	void SortArray(Parameters* par_host, Parameters* par_device, int* Val);

	void SortArray(Parameters* par_host, Parameters* par_device, double* Val);

	void SortArray(Parameters* par_host, Parameters* par_device, double3* Val);

	void SortArray(Parameters* par_host, Parameters* par_device, double4* Val);

	void CreateContList(Parameters* par_host, Parameters* par_device, double3* sortedPos, double* sortedRadius,
						double3* Dv, double3* Dv_pre, double3* Dt, double3* Dt_pre, double3* Dr, double3* Dr_pre);

};



#endif // !_DEM_GRID_H_


