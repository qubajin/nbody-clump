#include"dem_cylind_wall_io.h"

DEMWallCylinderIo::DEMWallCylinderIo()
{
	wall_host = NULL;
	Pos = NULL;
	Vel = NULL;
	AngSpd = NULL;
	Quat = NULL;
}

DEMWallCylinderIo::~DEMWallCylinderIo()
{
	if (Pos != NULL) cudaFree(Pos);
	if (Vel != NULL) cudaFree(Vel);
	if (AngSpd != NULL) cudaFree(AngSpd);
	if (Quat != NULL) cudaFree(Quat);
}

void DEMWallCylinderIo::LoadInfo()
{
	string Filename;
	FILE* infile;

	int Nvs, Nfs;
	double tmp;

	Filename = "./Rocks/";
	Filename.append("CylindWall.bt");

	const char* file = Filename.c_str();
	infile = fopen(file, "r");

	wall_host = new Cylinder;

	Pos = new double3;
	Vel = new double3;
	Quat = new double4;
	AngSpd = new double3;

	fscanf(infile, "%lf%lf", &wall_host->Radius, &wall_host->Height);
	fscanf(infile, "%lf%lf%lf", &wall_host->Dirct.x, &wall_host->Dirct.y, &wall_host->Dirct.z);

	fscanf(infile, "%lf%lf%lf%lf", &wall_host->Property.x, &wall_host->Property.y, &wall_host->Property.z, &wall_host->Property.w);

	fscanf(infile, "%lf%lf%lf", &Pos[0].x, &Pos[0].y, &Pos[0].z);
	fscanf(infile, "%lf%lf%lf", &Vel[0].x, &Vel[0].y, &Vel[0].z);
	fscanf(infile, "%lf%lf%lf%lf", &Quat[0].x, &Quat[0].y, &Quat[0].z, &Quat[0].w);
	fscanf(infile, "%lf%lf%lf", &AngSpd[0].x, &AngSpd[0].y, &AngSpd[0].z);

	//cout << "Penetrator: " << Vel[0].x << " " << Vel[0].y << " " << Vel[0].z << endl;

	//cout << wall_host->Radius << " " << wall_host->Height << endl;

	fclose(infile);
}