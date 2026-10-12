#pragma once
#ifndef _DEM_CLUMPSPHERE_H_
#define _DEM_CLUMPSPHERE_H_


#include<iostream>
#include<sstream>
#include<fstream>
#include<iomanip>
#include<cmath>
#include<ctime>
#include<vector>
#include<string>
#include<cuda_runtime.h>
#include<boost/variant.hpp>
#include<boost/algorithm/string.hpp>
#include<boost/filesystem.hpp>
#include<helper_cuda.h>
#include<chrono>
#include<thrust/extrema.h>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>      // 必须包含此头文件
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h> // 显式指定执行策略


#include"type.h"
#include"dem_parameters.h"
#include"dem_clumpsphere_io.h"


class DEMClumpSphere
{
private:

	double3* Pos;
	double3* Vel;
	double3* AngSpd;
	double4* Quat;

	double4* Property;

	double3* Fc;
	double3* Tc;

	thrust::device_vector<int> d_Permutation;   // 排序索引映射 
	thrust::device_vector<int> d_SortedKeys;    // 排序后的 ClumpId 序列
	thrust::device_vector<int> d_UniqueKeys;    // 规约后的唯一 ID 列表
	thrust::device_vector<double3> d_CompactBuf; // 规约中间缓冲
	thrust::device_vector<double3> d_TorqueContrib; // 每帧计算的力矩贡献

	bool is_initialized = false; // 标记是否已完成排序初始化


public:

	DEMClumpSphere();
	~DEMClumpSphere();

	double3* GetPos() { return Pos; }
	double3* GetVel() { return Vel; }
	double3* GetAngSpd() { return AngSpd; }
	double4* GetQuat() { return Quat; }

	double4* GetProperty() { return Property; }

	double3* GetFc() { return Fc; }
	double3* GetTc() { return Tc; }

	void AllocateGpuMemory(Parameters* Par);

	void CopyDataFromHostToDevice(Parameters* Par, DEMClumpSphereIo* clump_io);

	void CopyDataToIo(Parameters* Par, DEMClumpSphereIo* clump_io);

	void ForceZero(Parameters* Par);

	void SumForceOnClump(Parameters* par_host, Parameters* par_device, int* ClumpId, double3* SphPos, double3* LocalPos, double3* SphFc, double3* SphTc);

	void Integrate_Before(Parameters* par_host, Parameters* par_device, double3* totF, double3* totT);

	void Integrate_After(Parameters* par_host, Parameters* par_device, double3* totF, double3* totT);
};




#endif
