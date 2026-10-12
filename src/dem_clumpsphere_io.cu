#include"dem_clumpsphere_io.h"
#include "dem_binary_io.h"

DEMClumpSphereIo::DEMClumpSphereIo()
{
	iniconfile = "ch.00000000.bt";

	Pos = NULL;
	Vel = NULL;
	AngSpd = NULL;
	Quat = NULL;
	Property = NULL;
}

DEMClumpSphereIo::~DEMClumpSphereIo()
{
	if (Pos != NULL) cudaFreeHost(Pos);
	if (Vel != NULL) cudaFreeHost(Vel);
	if (AngSpd != NULL) cudaFreeHost(AngSpd);
	if (Quat != NULL) cudaFreeHost(Quat);
	if (Property != NULL) cudaFreeHost(Property);
}

void DEMClumpSphereIo::LoadPhase(Parameters* Par)
{
	cout << "========================= Load Phase =========================" << endl;

	double tmp;

	cudaHostAlloc((void**)&Pos, Par->ClumpNum * sizeof(double3), cudaHostAllocDefault);
	cudaHostAlloc((void**)&Vel, Par->ClumpNum * sizeof(double3), cudaHostAllocDefault);
	cudaHostAlloc((void**)&AngSpd, Par->ClumpNum * sizeof(double3), cudaHostAllocDefault);
	cudaHostAlloc((void**)&Quat, Par->ClumpNum * sizeof(double4), cudaHostAllocDefault);


	cudaHostAlloc((void**)&Property, Par->ClumpNum * sizeof(double4), cudaHostAllocDefault);

	string dat_name = iniconfile;
	ifstream input_stream;
	input_stream.open(dat_name.c_str());

	if (input_stream)
	{
		for (int i = 0; i < Par->ClumpNum; i++)
		{
			input_stream >> tmp >> Pos[i].x >> Pos[i].y >> Pos[i].z >> Vel[i].x >> Vel[i].y >> Vel[i].z >> \
								   Quat[i].x >> Quat[i].y >> Quat[i].z >> Quat[i].w >> AngSpd[i].x >> AngSpd[i].y >> AngSpd[i].z >> \
								   Property[i].x >> Property[i].y >> Property[i].z >> Property[i].w;

			//cout << "pos: " << i << " " << Pos[i].x << " " << Pos[i].y << " " << Pos[i].z  << endl;
		}
		input_stream.close();
	}
	else
	{
		cout << "Error! Cannot open " << dat_name << "!" << endl;
		exit(EXIT_FAILURE);
	}

	cout << "========================= Finish loading particle data =========================" << endl;

}

void DEMClumpSphereIo::SaveClumps_Bin(int ii, Parameters* Par)
{
    dem_binary::SaveClumps(dem_binary::FileName("./OutClump", "ch", ii),
                          Par->ClumpNum, Pos, Vel, Quat, AngSpd, Property);
}

void DEMClumpSphereIo::SaveClumps_Bin(int ii, Parameters* Par, double3* IndPos,
                                   double3* IndVel, double4* IndQuat, double3* IndAngSpd)
{
    dem_binary::SaveClumps(dem_binary::FileName("./OutClump", "ch", ii),
                          Par->ClumpNum, Pos, Vel, Quat, AngSpd, Property,
                          IndPos, IndVel, IndQuat, IndAngSpd);
}

// Legacy entry points retain their zero-based iteration argument.
void DEMClumpSphereIo::SaveClumps(int ii, Parameters* Par)
{
    SaveClumps_Bin(dem_binary::NextStep(ii), Par);
}

void DEMClumpSphereIo::SaveClumps(int ii, Parameters* Par, double3* IndPos,
                               double3* IndVel, double4* IndQuat, double3* IndAngSpd)
{
    SaveClumps_Bin(dem_binary::NextStep(ii), Par, IndPos, IndVel, IndQuat, IndAngSpd);
}
