#include"dem_clumpsphere.h"

DEMClumpSphere::DEMClumpSphere()
{
	Pos = NULL;
	Vel = NULL;
	Quat = NULL;
	AngSpd = NULL;

	Property = NULL;

	Fc = NULL;
	Tc = NULL;
}

DEMClumpSphere::~DEMClumpSphere()
{
	if (Pos != NULL) cudaFree(Pos);
	if (Vel != NULL) cudaFree(Vel);
	if (AngSpd != NULL) cudaFree(AngSpd);
	if (Quat != NULL) cudaFree(Quat);

	if (Fc != NULL)cudaFree(Fc);

	if (Tc != NULL) cudaFree(Tc);
}

void DEMClumpSphere::AllocateGpuMemory(Parameters* Par)
{
	checkCudaErrors(cudaMalloc((void**)&Pos, Par->ClumpNum * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&Vel, Par->ClumpNum * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&AngSpd, Par->ClumpNum * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&Quat, Par->ClumpNum * sizeof(double4)));
	checkCudaErrors(cudaMalloc((void**)&Property, Par->ClumpNum * sizeof(double4)));

	checkCudaErrors(cudaMalloc((void**)&Fc, Par->ClumpNum * sizeof(double3)));
	checkCudaErrors(cudaMalloc((void**)&Tc, Par->ClumpNum * sizeof(double3))); 
	// Initialize clump forces and torques before the first integration step.
	checkCudaErrors(cudaMemset(Fc, 0, Par->ClumpNum * sizeof(double3)));
	checkCudaErrors(cudaMemset(Tc, 0, Par->ClumpNum * sizeof(double3)));

	int n = Par->ParticleNum;

	d_Permutation.resize(n);
	d_SortedKeys.resize(n);
	d_UniqueKeys.resize(n);
	d_CompactBuf.resize(n);
	d_TorqueContrib.resize(n);

	is_initialized = false;
}

void DEMClumpSphere::CopyDataFromHostToDevice(Parameters* Par, DEMClumpSphereIo* clump_io)
{
	checkCudaErrors(cudaMemcpy(Pos, clump_io->GetPos(), Par->ClumpNum * sizeof(double3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(Vel, clump_io->GetVel(), Par->ClumpNum * sizeof(double3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(AngSpd, clump_io->GetAngSpd(), Par->ClumpNum * sizeof(double3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(Quat, clump_io->GetQuat(), Par->ClumpNum * sizeof(double4), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(Property, clump_io->GetProperty(), Par->ClumpNum * sizeof(double4), cudaMemcpyHostToDevice));
}

void DEMClumpSphere::CopyDataToIo(Parameters* Par, DEMClumpSphereIo* clump_io)
{
	checkCudaErrors(cudaMemcpy(clump_io->GetPos(), Pos, Par->ClumpNum * sizeof(double3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(clump_io->GetVel(), Vel, Par->ClumpNum * sizeof(double3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(clump_io->GetAngSpd(), AngSpd, Par->ClumpNum * sizeof(double3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(clump_io->GetQuat(), Quat, Par->ClumpNum * sizeof(double4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(clump_io->GetProperty(), Property, Par->ClumpNum * sizeof(double4), cudaMemcpyDeviceToHost));
}

void DEMClumpSphere::ForceZero(Parameters* Par)
{
	checkCudaErrors(cudaMemset(Fc, 0.0, Par->ClumpNum * sizeof(double3)));
	checkCudaErrors(cudaMemset(Tc, 0.0, Par->ClumpNum * sizeof(double3)));
}

struct double3_add {
	__host__ __device__
		double3 operator()(const double3& a, const double3& b) const {
		return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
	}
};


__global__ void ComputeTorqueFromWorldPos_Kernel(Parameters* Par, const int* ClumpId, const double3* SphPos, const double3* ClumpPos, 
												 const double3* SphFc, const double3* SphTc, double3* TorqueContrib)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= Par->ParticleNum) return;

	int cid = ClumpId[i];
	if (cid < 0) return;

	// 1. ֱ��ͨ��������������õ����� r
	double3 ps = SphPos[i];
	double3 pc = ClumpPos[cid];
	double3 r = make_double3(ps.x - pc.x, ps.y - pc.y, ps.z - pc.z);

	// 2. T_total = T_sph + r x F_sph
	double3 fs = SphFc[i];
	double3 ts = SphTc[i];

	TorqueContrib[i].x = ts.x + (r.y * fs.z - r.z * fs.y);
	TorqueContrib[i].y = ts.y + (r.z * fs.x - r.x * fs.z);
	TorqueContrib[i].z = ts.z + (r.x * fs.y - r.y * fs.x);
}


void DEMClumpSphere::SumForceOnClump(Parameters* par_host, Parameters* par_device, int* ClumpId, double3* SphPos, double3* LocalPos, double3* SphFc, double3* SphTc)
{
	int n = par_host->ParticleNum;
	int nClp = par_host->ClumpNum;

	// --- 1. ��װԭʼָ�� ---
	thrust::device_ptr<int> id_ptr(ClumpId);
	thrust::device_ptr<double3> sph_fc_ptr(SphFc);
	thrust::device_ptr<double3> clp_fc_ptr(Fc);
	thrust::device_ptr<double3> clp_tc_ptr(Tc);

	// --- 2. ׼����֡�� Key �� ���� ---
	// ÿһ֡��Ҫ���¿������µ� ClumpId �� ����
	thrust::copy(thrust::device, id_ptr, id_ptr + n, d_SortedKeys.begin());
	thrust::sequence(thrust::device, d_Permutation.begin(), d_Permutation.end());

	// --- 3. �������ع��� ---
	int threads = 256;
	int blocks = (n + threads - 1) / threads;
	ComputeTorqueFromWorldPos_Kernel << <blocks, threads >> > (par_device, ClumpId, SphPos, Pos, SphFc, SphTc,
															   thrust::raw_pointer_cast(d_TorqueContrib.data()));

	// --- 4. ʵʱ���� (��Ϊ ClumpId ����) ---
	// �������ʱ��һ�������ڲ�ʹ��ԭ�Ӳ�����������Ǳ����
	thrust::sort_by_key(thrust::device, d_SortedKeys.begin(), d_SortedKeys.end(), d_Permutation.begin());


	// �������� Fc
	auto fc_it = thrust::make_permutation_iterator(sph_fc_ptr, d_Permutation.begin());
	auto res_fc = thrust::reduce_by_key(thrust::device, d_SortedKeys.begin(), d_SortedKeys.end(),
										fc_it, d_UniqueKeys.begin(), d_CompactBuf.begin(), thrust::equal_to<int>(), double3_add());

	int numActive = res_fc.first - d_UniqueKeys.begin();
	thrust::scatter(thrust::device, d_CompactBuf.begin(), d_CompactBuf.begin() + numActive, d_UniqueKeys.begin(), clp_fc_ptr);

	// ���������� Tc
	auto tc_it = thrust::make_permutation_iterator(d_TorqueContrib.begin(), d_Permutation.begin());
	thrust::reduce_by_key(thrust::device, d_SortedKeys.begin(), d_SortedKeys.end(), tc_it,
						  d_UniqueKeys.begin(), d_CompactBuf.begin(), thrust::equal_to<int>(), double3_add());

	thrust::scatter(thrust::device, d_CompactBuf.begin(), d_CompactBuf.begin() + numActive, d_UniqueKeys.begin(), clp_tc_ptr);
}



