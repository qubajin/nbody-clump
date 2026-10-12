#include"dem_cube_wall_io.h"

DEMWallCubeIo::DEMWallCubeIo()
{
	wall_host = NULL;

	Pos = NULL;
	Vel = NULL;
	AngSpd = NULL;
	Quat = NULL;
}

DEMWallCubeIo::~DEMWallCubeIo()
{
	if (Pos != NULL) cudaFree(Pos);
	if (Vel != NULL) cudaFree(Vel);
	if (AngSpd != NULL) cudaFree(AngSpd);
	if (Quat != NULL) cudaFree(Quat);
}

void DEMWallCubeIo::LoadInfo()
{
	string Filename;
	FILE* infile;

	int Nvs, Nfs;
	double tmp;

	Filename = "./Rocks/";
	Filename.append("CubeWall.bt");

	const char* file = Filename.c_str();
	infile = fopen(file, "r");

	
	Pos = new double3;
	Vel = new double3;
	Quat = new double4;
	AngSpd = new double3;

	wall_host = new Cube;

	fscanf(infile, "%lf%lf%lf", &wall_host->SideLength.x, &wall_host->SideLength.y, &wall_host->SideLength.z);

	fscanf(infile, "%lf%lf%lf", &Pos[0].x, &Pos[0].y, &Pos[0].z);
	fscanf(infile, "%lf%lf%lf", &Vel[0].x, &Vel[0].y, &Vel[0].z);
	fscanf(infile, "%lf%lf%lf%lf", &Quat[0].x, &Quat[0].y, &Quat[0].z, &Quat[0].w);
	fscanf(infile, "%lf%lf%lf", &AngSpd[0].x, &AngSpd[0].y, &AngSpd[0].z);

	//cout << wall_host->Radius << " " << wall_host->Height << endl;

	fclose(infile);
}