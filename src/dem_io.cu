#include "dem_io.h"
#include "dem_binary_io.h"

DEMSphereIo::DEMSphereIo()
{
	bodyfilefolder = "./Rocks/";
	iniconfile = "ph.00000000.bt";

	Par = NULL;
	Glbflg = NULL;

	Pos = NULL;
	Vel = NULL;
	AngSpd = NULL;
	Quat = NULL;
	Radius = NULL;

	ParticleID = NULL;

	Property = NULL;

	CacheFi = NULL;
	CacheTi = NULL;

}

DEMSphereIo::~DEMSphereIo()
{
	if (Pos != NULL) cudaFreeHost(Pos);
	if (Vel != NULL) cudaFreeHost(Vel);
	if (AngSpd != NULL) cudaFreeHost(AngSpd);
	if (Quat != NULL) cudaFreeHost(Quat);

	if (ParticleID != NULL) cudaFreeHost(ParticleID);

	if (Property != NULL) cudaFreeHost(Property);

	if (CacheFi != NULL) cudaFreeHost(CacheFi);
	if (CacheTi != NULL) cudaFreeHost(CacheTi);

}

void DEMSphereIo::LoadParameters()
{
	cout << "Load Parameters ..." << endl;

	const char* parafile = "world.par";

	ifstream infile;
	string line, core, name, value;

	int numchar, id, n;
	int flag = 0;

	infile.open(parafile); //"world.par"

	Par = new Parameters;

	Glbflg = new GlobalFlg;

	while (getline(infile, line))
	{
		numchar = line.length();
		for (n = 0; n < numchar; n++) if (line[n] == '#') break;
		core.assign(line, 0, n);

		core.erase(std::remove(core.begin(), core.end(), '\t'), core.end());

		core.erase(std::remove(core.begin(), core.end(), ' '), core.end());
		//
		if (!core.empty())
		{
			//
			numchar = core.length();
			id = core.find('=');
			assert((id > 0) && (id + 1 < numchar));
			name.assign(core, 0, id);

			//cout<<name<<endl;

			n = numchar - id - 1;
			assert(n > 0);
			value.assign(core, id + 1, n);

			//
			if (name == "NumberBody") { Par->ParticleNum = atof(value.c_str());         flag++; }
			else if (name == "BodyFileDir") { bodyfilefolder = value;                flag++; }
			else if (name == "IniConFile") { iniconfile = value;                    flag++; }
			else if (name == "BranchLmtLen") { Par->BranchLmtLen = atof(value.c_str());                flag++; }
			//
			else if (name == "StepSize") { Par->StepSize = atof(value.c_str());        flag++; }
			else if (name == "EndStep") { Par->EndStep = atof(value.c_str());         flag++; }
			else if (name == "StartStep") { Par->StartStep = atof(value.c_str());       flag++; }
			else if (name == "OutputInterval") { Par->OutputInterval = atof(value.c_str());  flag++; }
			//
			//Particle Parameters
			else if (name == "Ymod") { Par->Ymod = atof(value.c_str());            flag++; }
			else if (name == "Prat") { Par->Prat = atof(value.c_str());            flag++; }
			else if (name == "Frcoe") { Par->Frcoe = atof(value.c_str());           flag++; }
			else if (name == "NormRes") { Par->NormRes = atof(value.c_str());         flag++; }
			else if (name == "Frvel") { Par->Frvel = atof(value.c_str());           flag++; }
			else if (name == "MU_T") { Par->MU_T = atof(value.c_str());     flag++; }
			else if (name == "MU_R") { Par->MU_R = atof(value.c_str());     flag++; }
			else if (name == "Beta") { Par->Beta = atof(value.c_str());     flag++; }
			else if (name == "Cohesion") { Par->Cohesion = atof(value.c_str());     flag++; }
			//Grid
			else if (name == "Thresh") { Par->Thresh = atof(value.c_str());           flag++; }
			else if (name == "GridLevels") { Par->GridLevels = atof(value.c_str());     flag++; }
			//
			else if (name == "ClumpNum") { Par->ClumpNum = atof(value.c_str());     flag++; }
			//
			else if (name == "Gravity") { Par->Gravity = atof(value.c_str());     flag++; }
			// Boundary Parameters
			else if (name == "Update") { Par->Update = atof(value.c_str());     flag++; }
			else if (name == "WorldSize_x") { Par->WorldSize_x = atof(value.c_str());     flag++; }
			else if (name == "WorldSize_y") { Par->WorldSize_y = atof(value.c_str());     flag++; }
			else if (name == "WorldSize_z") { Par->WorldSize_z = atof(value.c_str());     flag++; }
			else if (name == "WorldOrigin_x") { Par->WorldOrigin_x = atof(value.c_str());   flag++; }
			else if (name == "WorldOrigin_y") { Par->WorldOrigin_y = atof(value.c_str());   flag++; }
			else if (name == "WorldOrigin_z") { Par->WorldOrigin_z = atof(value.c_str());   flag++; }
			else if (name == "BoxLowLmt_x") { Par->BoxLowLmt_x = atof(value.c_str());     flag++; }
			else if (name == "BoxLowLmt_y") { Par->BoxLowLmt_y = atof(value.c_str());     flag++; }
			else if (name == "BoxLowLmt_z") { Par->BoxLowLmt_z = atof(value.c_str());     flag++; }
			else if (name == "BoxUpLmt_x") { Par->BoxUpLmt_x = atof(value.c_str());   flag++; }
			else if (name == "BoxUpLmt_y") { Par->BoxUpLmt_y = atof(value.c_str());   flag++; }
			else if (name == "BoxUpLmt_z") { Par->BoxUpLmt_z = atof(value.c_str());   flag++; }
			//
			else if (name == "isIntruder") { Glbflg->isIntruder = atof(value.c_str());   flag++; }
			else if (name == "isSelfGravity") { Glbflg->isSelfGravity = atof(value.c_str());   flag++; }
			else if (name == "isWallForce") { Glbflg->isWallForce = atof(value.c_str());   flag++; }
			else if (name == "isQuaternion") { Glbflg->isQuaternion = atof(value.c_str());   flag++; }
			//
			else { cout << "Unrecoganized item " << name << "!" << endl; exit(1); }
			//
		}
	}
	infile.close();

	if (flag != NumParameter)
	{
		cout << flag << endl;
		cout << "Important parameter setting missing!" << endl;
		exit(1);
	}

	double lnCOR = log(Par->NormRes);

	Par->Damping = 2.0 * sqrt(5.0 / 6.0) * lnCOR / sqrt(lnCOR * lnCOR + PI * PI);
	//Par->Damping = sqrt(5.0) * lnCOR / sqrt(lnCOR * lnCOR + PI * PI);

	cout << "========================= Finish loading parameters =========================" << endl;
}


void DEMSphereIo::LoadPhase()
{
	cout << "========================= Load Phase =========================" << endl;

	double tmp;

	cudaHostAlloc((void**)&Pos, Par->ParticleNum * sizeof(double3), cudaHostAllocDefault);
	cudaHostAlloc((void**)&Vel, Par->ParticleNum * sizeof(double3), cudaHostAllocDefault);
	cudaHostAlloc((void**)&AngSpd, Par->ParticleNum * sizeof(double3), cudaHostAllocDefault);
	cudaHostAlloc((void**)&Quat, Par->ParticleNum * sizeof(double4), cudaHostAllocDefault);


	cudaHostAlloc((void**)&Radius, Par->ParticleNum * sizeof(double), cudaHostAllocDefault);
	cudaHostAlloc((void**)&Property, Par->ParticleNum * sizeof(double4), cudaHostAllocDefault);


	cudaHostAlloc((void**)&ParticleID, Par->ParticleNum * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**)&sortedParticleID, Par->ParticleNum * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**)&Type, Par->ParticleNum * sizeof(int), cudaHostAllocDefault);

	string dat_name = iniconfile;
	ifstream input_stream;
	input_stream.open(dat_name.c_str());

	if (input_stream)
	{
		for (int i = 0; i < Par->ParticleNum; i++)
		{
			input_stream >> tmp >> tmp >> Type[i] >> Pos[i].x >> Pos[i].y >> Pos[i].z >> Vel[i].x >> Vel[i].y >> Vel[i].z >> \
				Quat[i].x >> Quat[i].y >> Quat[i].z >> Quat[i].w >> AngSpd[i].x >> AngSpd[i].y >> AngSpd[i].z >> \
				Property[i].x >> Property[i].y >> Property[i].z >> Property[i].w >> Radius[i];

			//cout << "pos: "<< i << " " << Type[i] << " " << Pos[i].x << " " << Pos[i].y << " " << Pos[i].z << endl;

			ParticleID[i] = i;

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

void DEMSphereIo::LoadGridLevels()
{
	cudaHostAlloc((void**)&cellSize, Par->GridLevels * sizeof(double), cudaHostAllocDefault);
	cudaHostAlloc((void**)&cellNum, Par->GridLevels * sizeof(int3), cudaHostAllocDefault);
	cudaHostAlloc((void**)&cellOffset, Par->GridLevels * sizeof(int), cudaHostAllocDefault);

	cout << "========================= Load HGrid =========================" << endl;

	string dat_name = "grid.bt";
	ifstream input_stream;
	input_stream.open(dat_name.c_str());
	int totCellNum = 0;
	int tmp;

	if (input_stream)
	{
		for (int i = 0; i < Par->GridLevels; i++)
		{
			input_stream >> tmp >> cellSize[i] >> cellNum[i].x >> cellNum[i].y >> cellNum[i].z >> cellOffset[i] >> tmp;
			totCellNum = totCellNum + cellNum[i].x * cellNum[i].y * cellNum[i].z;

			cout << "HGrid:" << " " << i << " " << cellSize[i] << " " << cellNum[i].x << " " << cellNum[i].y << " " << cellNum[i].z << " " << cellOffset[i] << endl;
		}
		input_stream.close();

	}
	else
	{
		cout << "Error! Cannot open " << dat_name << "!" << endl;
		exit(EXIT_FAILURE);
	}

	Par->CellNum = totCellNum;

	cout << "Configuration of HGrid ..." << endl;
	cout << "WorldOrigin:" << " " << Par->WorldOrigin_x << " " << Par->WorldOrigin_y << " " << Par->WorldOrigin_z << endl;
	cout << "WorldSize:" << " " << Par->WorldSize_x << " " << Par->WorldSize_y << " " << Par->WorldSize_z << endl;
	cout << "BoxUpLmt:" << " " << Par->BoxUpLmt_x << " " << Par->BoxUpLmt_y << " " << Par->BoxUpLmt_z << endl;
	cout << "BoxLowLmt:" << " " << Par->BoxLowLmt_x << " " << Par->BoxLowLmt_y << " " << Par->BoxLowLmt_z << endl;
	cout << "totCellNum:" << " " << Par->CellNum << endl;
	

	
	cout << "========================= Finish loading hgrid data =========================" << endl;
}

void DEMSphereIo::SetMeshSize()
{
	double MeshSize = 0.0;

	for (int i = 0; i < Par->ParticleNum; i++)
	{
		if (Radius[i] > MeshSize)
		{
			MeshSize = Radius[i];
		}
	}

	MeshSize = 3.0 * MeshSize;

	//×ÐÏ¸Õå×ÃÒ»ÏÂ
	Par->CellSize_x = MeshSize;
	Par->CellSize_y = MeshSize;
	Par->CellSize_z = MeshSize;

	Par->i_CellSize_x = 1.0 / MeshSize;
	Par->i_CellSize_y = 1.0 / MeshSize;
	Par->i_CellSize_z = 1.0 / MeshSize;

	Par->WorldOrigin_x = Par->WorldOrigin_x - Par->CellSize_x;
	Par->WorldOrigin_y = Par->WorldOrigin_y - Par->CellSize_y;
	Par->WorldOrigin_z = Par->WorldOrigin_z - Par->CellSize_z;

	Par->WorldSize_x = Par->WorldSize_x + 2 * Par->CellSize_x;
	Par->WorldSize_y = Par->WorldSize_y + 2 * Par->CellSize_y;
	Par->WorldSize_z = Par->WorldSize_z + 2 * Par->CellSize_z;

	Par->GridDim_x = ceil(Par->WorldSize_x / Par->CellSize_x);
	Par->GridDim_y = ceil(Par->WorldSize_y / Par->CellSize_y);
	Par->GridDim_z = ceil(Par->WorldSize_z / Par->CellSize_z);

	Par->CellNum = Par->GridDim_x * Par->GridDim_y * Par->GridDim_z;

	cout << "Configuration of Grid ..." << endl;
	cout << "CellSize:" << " " << Par->CellSize_x << " " << Par->CellSize_y << " " << Par->CellSize_z << endl;
	cout << "WorldOrigin:" << " " << Par->WorldOrigin_x << " " << Par->WorldOrigin_y << " " << Par->WorldOrigin_z << endl;
	cout << "WorldSize:" << " " << Par->WorldSize_x << " " << Par->WorldSize_y << " " << Par->WorldSize_z << endl;
	cout << "CellSize:" << " " << Par->CellSize_x << " " << Par->CellSize_y << " " << Par->CellSize_z << endl;
	cout << "GridDim:" << " " << Par->GridDim_x << " " << Par->GridDim_y << " " << Par->GridDim_z << endl;
	cout << "BoxUpLmt:" << " " << Par->BoxUpLmt_x << " " << Par->BoxUpLmt_y << " " << Par->BoxUpLmt_z << endl;
	cout << "BoxLowLmt:" << " " << Par->BoxLowLmt_x << " " << Par->BoxLowLmt_y << " " << Par->BoxLowLmt_z << endl;

}


void  DEMSphereIo::AllocateCpuMemory()
{
	cudaHostAlloc((void**)&CacheFi, Par->OutputInterval * sizeof(double3), cudaHostAllocDefault);
	cudaHostAlloc((void**)&CacheTi, Par->OutputInterval * sizeof(double3), cudaHostAllocDefault);

	Anei = (int*)malloc(sizeof(int) * Par->BranchLmtLen * Par->ParticleNum);
	ChainFcn = (double3*)malloc(sizeof(double3) * Par->BranchLmtLen * Par->ParticleNum);
	ChainCtp = (double3*)malloc(sizeof(double3) * Par->BranchLmtLen * Par->ParticleNum);

	memset(Anei, 0, Par->BranchLmtLen * Par->ParticleNum * sizeof(int));
	memset(ChainFcn, 0.0, Par->BranchLmtLen * Par->ParticleNum * sizeof(double3));
	memset(ChainCtp, 0.0, Par->BranchLmtLen * Par->ParticleNum * sizeof(double3));
}

void DEMSphereIo::SaveParticles_Bin(int ii)
{
    const int count = Par->ParticleNum;
    const std::size_t n = dem_binary::Count(count);
    dem_binary::Writer file(dem_binary::FileName("./OutPart", "ph", ii));
    file.Scalar(count);
    file.Write(sortedParticleID, n);
    file.Write(ParticleID, n);
    file.Write(Type, n);
    file.Write(Pos, n);
    file.Write(Vel, n);
    file.Write(Quat, n);
    file.Write(AngSpd, n);
    file.Write(Property, n);
    file.Write(Radius, n);
    file.Close();
}

void DEMSphereIo::SaveParticles_Bin(int ii, double3* IndPos, double3* IndVel,
                                  double4* IndQuat, double3* IndAngSpd)
{
    // Keep ph exactly compatible with the sphere-only layout provided by the user.
    // The extra intruder has its own one-entry file using the clump field layout.
    SaveParticles_Bin(ii);
    dem_binary::SaveClumps(dem_binary::FileName("./OutPart", "intruder", ii),
                          0, NULL, NULL, NULL, NULL, NULL,
                          IndPos, IndVel, IndQuat, IndAngSpd);
    dem_binary::SaveForceHistory(ii, Par->OutputInterval, CacheFi, CacheTi);
}

void DEMSphereIo::SaveContInfo_Bin(int ii)
{
    dem_binary::SaveContacts(ii, Par->ParticleNum, Par->BranchLmtLen,
                             GetAnei(), GetChainFcn(), GetChainCtp());
}

// Legacy entry points retain their zero-based iteration argument.
void DEMSphereIo::SaveParticles(int ii)
{
    SaveParticles_Bin(dem_binary::NextStep(ii));
}

void DEMSphereIo::SaveParticles(int ii, double3* IndPos, double3* IndVel,
                              double4* IndQuat, double3* IndAngSpd)
{
    SaveParticles_Bin(dem_binary::NextStep(ii), IndPos, IndVel, IndQuat, IndAngSpd);
}

void DEMSphereIo::SaveContInfo(int ii)
{
    SaveContInfo_Bin(dem_binary::NextStep(ii));
}
