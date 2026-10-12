#include"dem.h"

void DEMmain::RunSimulation()
{
	int count, count2;
	int NumThresh;

	count = 0;
	count2 = 0;

	io = new DEMSphereIo;

	io->LoadParameters();

	io->LoadPhase();

	io->SetMeshSize();

	io->AllocateCpuMemory();

	par_host = io->GetParameters();

	Glb = io->GetGlobalFlag();

	AllocateGpuMemory();

	CopyDataFromHostToDevice();

	//
	grid = new DEMGrid();

	grid->GridConfig(par_host);

	grid->InitPartFixIdx(par_host, par_device);

	//
	force = new DEMForce();

	force->ForceConfig(par_host, Glb);

	force->ForceZero(par_host, Glb);

	//
	wall_io = new DEMWallCylinderIo;

	wall_io->LoadInfo();

	wall = new DEMWallCylinder;

	wall->AllocateMemory(par_host, wall_io);

	//
	pene_io = new DEMPeneCylinderIo;

	pene_io->LoadInfo();

	pene = new DEMPeneCylinder;

	pene->AllocateMemory(par_host, pene_io);


	// Create File Folder
	std::string path("./OutPart/");
	if (!boost::filesystem::exists(path))
	{
		std::cout << "Create results folder: " << path << std::endl;
		boost::filesystem::create_directory(path);
	}

	//if (Glb->isIntruder)
	//{
	//	path = "./OutFrce/";
	//	if (!boost::filesystem::exists(path))
	//	{
	//		std::cout << "Create results folder: " << path << std::endl;
	//		boost::filesystem::create_directory(path);
	//	}
	//}

	for (int ii = par_host->StartStep; ii < par_host->EndStep; ii++)
	{
		//cout << ii << endl;

		force->Integrate_Before(par_host, par_device, Pos, Vel, AngSpd, CumPos, Radius, Property,
								wall->GetFc(), wall->GetTc(), pene->GetFc(), pene->GetTc(), grid->GetPartFix());

		force->Attitude(par_host, par_device, AngSpd, Quat, Property, wall->GetTc(), pene->GetTc());

		if (Glb->isIntruder)
		{
			pene->Integrate_Before(par_device);
		}
			
		checkCudaErrors(cudaMemcpy(par_host, par_device, sizeof(Parameters), cudaMemcpyDeviceToHost));

		if (par_host->Update != 0 || ii == par_host->StartStep)
		{
			//cout << "Update!!" << endl;

			grid->NeighborSearch(par_host, par_device, Pos);

			grid->SortArray(par_host, par_device, Pos);
			grid->SortArray(par_host, par_device, Vel);
			grid->SortArray(par_host, par_device, AngSpd);
			grid->SortArray(par_host, par_device, Quat);
			grid->SortArray(par_host, par_device, Radius);
			grid->SortArray(par_host, par_device, Property);

			grid->SortArray(par_host, par_device, Type);

			grid->SortArray(par_host, par_device, wall->GetDvCap());
			grid->SortArray(par_host, par_device, wall->GetDtCap());
			grid->SortArray(par_host, par_device, wall->GetDrCap());

			grid->SortArray(par_host, par_device, wall->GetDvUp());
			grid->SortArray(par_host, par_device, wall->GetDtUp());
			grid->SortArray(par_host, par_device, wall->GetDrUp());

			grid->SortArray(par_host, par_device, wall->GetDvSide());
			grid->SortArray(par_host, par_device, wall->GetDtSide());
			grid->SortArray(par_host, par_device, wall->GetDrSide());

			if (Glb->isIntruder)
			{
				grid->SortArray(par_host, par_device, pene->GetDv());
				grid->SortArray(par_host, par_device, pene->GetDt());
				grid->SortArray(par_host, par_device, pene->GetDr());
			}

			grid->CreateContList(par_host, par_device, Pos, Radius, force->GetDv(), force->GetDvPre(), force->GetDt(), 
								 force->GetDtPre(), force->GetDr(), force->GetDrPre());
			
			par_host->Update = 0;

			checkCudaErrors(cudaMemset(CumPos, 0, par_host->ParticleNum * sizeof(double3)));

			checkCudaErrors(cudaMemcpy(par_device, par_host, sizeof(Parameters), cudaMemcpyHostToDevice));

		}

		force->ForceExchange(par_host);

		wall->ForceExchange(par_host);

		if (Glb->isIntruder)
		{
			pene->ForceExchange(par_host);
		}

		force->ForceZero(par_host, Glb);

		wall->ForceZero(par_host);

		force->CalcContactForce(par_host, par_device, grid->GetLpa(), grid->GetLpb(), Pos, Vel, AngSpd, Quat, 
								Property, Radius, grid->GetAref(), grid->GetNg(), grid->GetNi(), grid->GetPitch(), grid->GetPartFix());

		wall->CalcContactForce(par_host, par_device, Pos, Vel, AngSpd, Radius, Property);

		if (Glb->isIntruder)
		{
			pene->ForceZero(par_host);

			pene->CreateContList(par_host, par_device, Pos, Radius, grid->GetPartFix());

			pene->CalcContactForce(par_host, par_device, Pos, Vel, AngSpd, Radius, Property);

			pene->SumForce(par_host, pene->GetFc(), pene->GetTotFc());

			pene->SumForce(par_host, pene->GetTc(), pene->GetTotTc());

			pene->Integrate_After(count, par_device);

		}

		force->Integrate_After(par_host, par_device, Vel, AngSpd, Property, wall->GetFc(), wall->GetTc(), pene->GetFc(), pene->GetTc());

		count++;
		if (count == par_host->OutputInterval)
		{
			printf("StepSize = %d\n", ii + 1);
			CopyDataToIo(pene->GetCacheFc(), pene->GetCacheTc());
			if (Glb->isIntruder)
			{
				pene->CopyDataToIo(pene_io);
				io->SaveParticles_Bin(ii + 1, pene_io->GetPos(), pene_io->GetVel(), pene_io->GetQuat(), pene_io->GetAngSpd());
			}
			else
			{
				io->SaveParticles_Bin(ii + 1);
			}
			count = 0;
		}

		/*if (Glb->isIntruder)
		{
			count2++;

			NumThresh = 2 * par_host->OutputInterval;

			if (count2 == NumThresh)
			{
				force->CalcFcnChain(par_host, par_device, grid->GetAref(), grid->GetNg(), grid->GetNi(), grid->GetPitch());
				CopyInfoToIo(grid->GetPitch(), force->GetPitch(), grid->GetAnei(), force->GetChainFcn(), force->GetChainCtp());
				io->SaveContInfo_Bin(ii + 1);
				count2 = 0;
			}
		}*/

	}


}