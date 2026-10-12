#include"dem.h"

void DEMmain::RunSimulation_HGrid()
{
	int count, count2 = 0, NumThresh;

	count = 0;

	io = new DEMSphereIo;

	io->LoadParameters();

	io->LoadPhase();

	io->LoadGridLevels();

	io->AllocateCpuMemory();

	par_host = io->GetParameters();

	Glb = io->GetGlobalFlag();

	AllocateGpuMemory();

	CopyDataFromHostToDevice();

	//
	hgrid = new DEMHGrid();

	hgrid->GridConfig(par_host);

	hgrid->CopyDataFromHostToDevice(par_host, io);

	hgrid->InitPartFixIdx(par_host, par_device);

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

	for (int ii = par_host->StartStep; ii < par_host->EndStep; ii++)
	{
		//cout << ii << endl;

		force->Integrate_Before(par_host, par_device, Pos, Vel, AngSpd, CumPos, Radius, Property,
								wall->GetFc(), wall->GetTc(), pene->GetFc(), pene->GetTc(), hgrid->GetPartFix());

		force->Attitude(par_host, par_device, AngSpd, Quat, Property, wall->GetTc(), pene->GetTc());

		if (Glb->isIntruder)
		{
			pene->Integrate_Before(par_device);
		}

		checkCudaErrors(cudaMemcpy(par_host, par_device, sizeof(Parameters), cudaMemcpyDeviceToHost));

		if (par_host->Update != 0 || ii == par_host->StartStep)
		{
			//cout << "Update!!" << endl;

			hgrid->NeighborSearch(par_host, par_device, Pos, Type);

			hgrid->SortArray(par_host, par_device, Pos);
			hgrid->SortArray(par_host, par_device, Vel);
			hgrid->SortArray(par_host, par_device, AngSpd);
			hgrid->SortArray(par_host, par_device, Quat);
			hgrid->SortArray(par_host, par_device, Radius);
			hgrid->SortArray(par_host, par_device, Property);

			hgrid->SortArray(par_host, par_device, Type);

			hgrid->SortArray(par_host, par_device, wall->GetDvCap());
			hgrid->SortArray(par_host, par_device, wall->GetDtCap());
			hgrid->SortArray(par_host, par_device, wall->GetDrCap());

			hgrid->SortArray(par_host, par_device, wall->GetDvUp());
			hgrid->SortArray(par_host, par_device, wall->GetDtUp());
			hgrid->SortArray(par_host, par_device, wall->GetDrUp());

			hgrid->SortArray(par_host, par_device, wall->GetDvSide());
			hgrid->SortArray(par_host, par_device, wall->GetDtSide());
			hgrid->SortArray(par_host, par_device, wall->GetDrSide());

			if (Glb->isIntruder)
			{
				hgrid->SortArray(par_host, par_device, pene->GetDv());
				hgrid->SortArray(par_host, par_device, pene->GetDt());
				hgrid->SortArray(par_host, par_device, pene->GetDr());
			}

			hgrid->CreateContList(par_host, par_device, Pos, Radius, Type, force->GetDv(), force->GetDvPre(), force->GetDt(),
								  force->GetDtPre(), force->GetDr(), force->GetDrPre());

			par_host->Update = 0;

			checkCudaErrors(cudaMemset(CumPos, 0, par_host->ParticleNum * sizeof(double3)));

			checkCudaErrors(cudaMemcpy(par_device, par_host, sizeof(Parameters), cudaMemcpyHostToDevice));

		}
		else
		{
			force->ForceExchange(par_host);
		}
	
		wall->ForceExchange(par_host);

		if (Glb->isIntruder)
		{
			pene->ForceExchange(par_host);
		}

		force->ForceZero(par_host, Glb);

		wall->ForceZero(par_host);

		force->CalcContactForce(par_host, par_device, hgrid->GetLpa(), hgrid->GetLpb(), Pos, Vel, AngSpd, Quat,
								Property, Radius, hgrid->GetAref(), hgrid->GetNg(), hgrid->GetNi(), hgrid->GetPitch(), hgrid->GetPartFix());

		wall->CalcContactForce(par_host, par_device, Pos, Vel, AngSpd, Radius, Property);

		if (Glb->isIntruder)
		{
			pene->ForceZero(par_host);

			pene->CreateContList(par_host, par_device, Pos, Radius, hgrid->GetPartFix());

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
			CopyDataToIo_HGrid(pene->GetCacheFc(), pene->GetCacheTc());
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
				force->CalcFcnChain(par_host, par_device, hgrid->GetAref(), hgrid->GetNg(), hgrid->GetNi(), hgrid->GetPitch());
				CopyInfoToIo(hgrid->GetPitch(), force->GetPitch(), hgrid->GetAnei(), force->GetChainFcn(), force->GetChainCtp());
				io->SaveContInfo_Bin(ii + 1);
				count2 = 0;
			}
		}*/

	}

}