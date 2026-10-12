#include "dem_clump.h"

void DEMClumpMain::RunSimulationClump() {
    int count, count2, NumThresh;

    count = 0;

    count2 = 0;

    io = new DEMSubSphereIo;

    io->LoadParameters();

    io->LoadPhase();

    io->SetMeshSize();

    io->AllocateCpuMemory();

    par_host = io->GetParameters();

    Glb = io->GetGlobalFlag();

    AllocateGpuMemory();

    CopyDataFromHostToDevice();

    NumThresh = 2 * par_host->OutputInterval;

    //
    clump_io = new DEMClumpSphereIo;

    clump_io->LoadPhase(par_host);

    clump = new DEMClumpSphere;

    clump->AllocateGpuMemory(par_host);

    clump->CopyDataFromHostToDevice(par_host, clump_io);

    //
    grid = new DEMGrid();

    grid->GridConfig(par_host);

    grid->InitPartFixIdx(par_host, par_device);

    //
    force = new DEMForce();

    force->ForceConfig(par_host, Glb);

    force->ForceZero(par_host, Glb);

    //
    cube_io = new DEMWallCubeIo;

    cube_io->LoadInfo();

    cube = new DEMWallCube;

    cube->AllocateMemory(par_host, cube_io);

    //
    pene_io = new DEMPeneCylinderIo;

    pene_io->LoadInfo();

    pene = new DEMPeneCylinder;

    pene->AllocateMemory(par_host, pene_io);

    // Create File Folders
    if (Glb->isIntruder) {
        std::vector<std::string> folders = {"./OutPart/", "./OutForce/", "./OutClump/",
                                            "./OutCnt/"};
        // std::vector<std::string> folders = { "./OutPart/", "./OutForce/", "./OutClump/"};
        for (const auto& path : folders) {
            if (!boost::filesystem::exists(path)) {
                std::cout << "Create results folder: " << path << std::endl;
                boost::filesystem::create_directory(path);
            }
        }
    } else {
        std::vector<std::string> folders = {"./OutPart/", "./OutClump/", "./OutCnt/"};

        for (const auto& path : folders) {
            if (!boost::filesystem::exists(path)) {
                std::cout << "Create results folder: " << path << std::endl;
                boost::filesystem::create_directory(path);
            }
        }
    }

    // Store the geometry used for this result set.  The MATLAB renderer reads
    // this file before the mutable Rocks/Penetrator.bt configuration, which
    // prevents old cylinder results from being displayed as a sphere (or vice
    // versa) after the next configuration change.
    if (Glb->isIntruder) {
        Cylinder* peneParameters = pene_io->GetPeneParameters();
        ofstream shapeFile("./OutClump/PenetratorShape.bt");
        if (peneParameters->Shape == PENETRATOR_SPHERE) {
            shapeFile << "sphere " << peneParameters->Radius << endl;
        } else {
            shapeFile << "cylinder " << peneParameters->Radius << " " << peneParameters->Height
                      << endl;
        }
    }

    for (int ii = par_host->StartStep; ii < par_host->EndStep; ii++) {
        // cout << ii << endl;

        clump->Integrate_Before(par_host, par_device, clump->GetFc(), clump->GetTc());

        UpdatePos(ClumpId, clump->GetPos(), clump->GetQuat());

        if (Glb->isIntruder) {
            pene->Integrate_Before(par_device);
        }

        checkCudaErrors(
            cudaMemcpy(par_host, par_device, sizeof(Parameters), cudaMemcpyDeviceToHost));

        if (par_host->Update != 0 || ii == par_host->StartStep) {

            grid->NeighborSearch(par_host, par_device, Pos);

            grid->SortArray(par_host, par_device, Pos);
            grid->SortArray(par_host, par_device, LocalPos);

            grid->SortArray(par_host, par_device, Radius);
            grid->SortArray(par_host, par_device, Mass);

            grid->SortArray(par_host, par_device, ClumpId);

            grid->SortArray(par_host, par_device, cube->GetDvUpX());
            grid->SortArray(par_host, par_device, cube->GetDtUpX());
            grid->SortArray(par_host, par_device, cube->GetDrUpX());

            grid->SortArray(par_host, par_device, cube->GetDvUpY());
            grid->SortArray(par_host, par_device, cube->GetDtUpY());
            grid->SortArray(par_host, par_device, cube->GetDrUpY());

            grid->SortArray(par_host, par_device, cube->GetDvUpZ());
            grid->SortArray(par_host, par_device, cube->GetDtUpZ());
            grid->SortArray(par_host, par_device, cube->GetDrUpZ());

            grid->SortArray(par_host, par_device, cube->GetDvLowX());
            grid->SortArray(par_host, par_device, cube->GetDtLowX());
            grid->SortArray(par_host, par_device, cube->GetDrLowX());

            grid->SortArray(par_host, par_device, cube->GetDvLowY());
            grid->SortArray(par_host, par_device, cube->GetDtLowY());
            grid->SortArray(par_host, par_device, cube->GetDrLowY());

            grid->SortArray(par_host, par_device, cube->GetDvLowZ());
            grid->SortArray(par_host, par_device, cube->GetDtLowZ());
            grid->SortArray(par_host, par_device, cube->GetDrLowZ());

            if (Glb->isIntruder) {
                grid->SortArray(par_host, par_device, pene->GetDv());
                grid->SortArray(par_host, par_device, pene->GetDt());
                grid->SortArray(par_host, par_device, pene->GetDr());
            }

            grid->CreateContList(par_host, par_device, Pos, Radius, force->GetDv(),
                                 force->GetDvPre(), force->GetDt(), force->GetDtPre(),
                                 force->GetDr(), force->GetDrPre());

            par_host->Update = 0;

            checkCudaErrors(cudaMemset(CumPos, 0, par_host->ParticleNum * sizeof(double3)));

            checkCudaErrors(
                cudaMemcpy(par_device, par_host, sizeof(Parameters), cudaMemcpyHostToDevice));

        } else {
            force->ForceExchange(par_host);
        }

        cube->ForceExchange(par_host);

        if (Glb->isIntruder) {
            pene->ForceExchange(par_host);
        }

        force->ForceZero(par_host, Glb);

        cube->ForceZero(par_host);

        force->CalcContactForce_Clump(
            par_host, par_device, grid->GetLpa(), grid->GetLpb(), ClumpId, Pos, LocalPos, Mass,
            Radius, clump->GetVel(), clump->GetAngSpd(), clump->GetQuat(), grid->GetAref(),
            grid->GetNg(), grid->GetNi(), grid->GetPitch(), grid->GetPartFix());

        if (Glb->isWallForce) {
            cube->CalcContactForce_Clump(par_host, par_device, ClumpId, Pos, LocalPos, Radius, Mass,
                                         clump->GetVel(), clump->GetAngSpd(), clump->GetQuat());
        }

        if (Glb->isIntruder) {
            pene->ForceZero(par_host);

            pene->CreateContList(par_host, par_device, Pos, Radius, grid->GetPartFix());

            pene->CalcContactForce_Clump(par_host, par_device, ClumpId, Pos, LocalPos, Radius, Mass,
                                         clump->GetVel(), clump->GetAngSpd(), clump->GetQuat());

            pene->SumForce(par_host, pene->GetFc(), pene->GetTotFc());

            // Torque on the penetrator must be accumulated about the penetrator
            // centre, rather than reusing the torque acting on each particle.
            pene->SumForce(par_host, pene->GetPeneTc(), pene->GetTotTc());

            pene->Integrate_After(count, par_device);
        }

        ForceZero();

        SumAllForces(force->GetFc(), force->GetTc(), cube->GetFc(), cube->GetTc(), pene->GetFc(),
                     pene->GetTc());

        clump->SumForceOnClump(par_host, par_device, ClumpId, Pos, LocalPos, totF, totT);

        clump->Integrate_After(par_host, par_device, clump->GetFc(), clump->GetTc());

        count++;
        if (count == par_host->OutputInterval) {
            printf("StepSize = %d\n", ii + 1);
            CopyDataToIo(Glb, pene->GetCacheFc(), pene->GetCacheTc());
            clump->CopyDataToIo(par_host, clump_io);

            if (Glb->isIntruder) {
                pene->CopyDataToIo(par_host, pene_io, io);
                io->SaveParticles_Bin(ii + 1, Glb);
                clump_io->SaveClumps_Bin(ii + 1, par_host, pene_io->GetPos(), pene_io->GetVel(),
                                         pene_io->GetQuat(), pene_io->GetAngSpd());
            } else {
                clump->CopyDataToIo(par_host, clump_io);
                io->SaveParticles_Bin(ii + 1, Glb);
                clump_io->SaveClumps_Bin(ii + 1, par_host);
            }

            count = 0;
        }

        // Save particle-particle force chains in both settling and intrusion runs.
        count2++;

        if (count2 == NumThresh) {
            force->CalcFcnChain(par_host, par_device, grid->GetAref(), grid->GetNg(), grid->GetNi(),
                                grid->GetPitch());

            CopyInfoToIo(grid->GetPitch(), force->GetPitch(), grid->GetAnei(), force->GetChainFcn(),
                         force->GetChainCtp());

            io->SaveContInfo_Bin(ii + 1);
            count2 = 0;
        }
    }
}
