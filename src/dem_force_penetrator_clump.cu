#include"dem_penetrator.h"
#include"dem_sphere_intruder.cuh"

__global__ void ForcePene_clump_cuk(Parameters* par, int* ClumpId, double3* Pos, double3* LocalPos, double* Radius, double* Mass,
                                    double3* ClumpVel, double3* ClumpAngSpd, double4* ClumpQuat,
                                    Cylinder* cyd, double3* PenePos, double3* PeneVel, double3* PeneAngSpd,                             
                                    double3* Dv_pre, double3* Dv, double3* Dt_pre, double3* Dt, double3* Dr_pre, double3* Dr,
                                    int* LcFlg, double3* RVo, double3* RVn, double3* Fc, double3* Tc, double3* peneTc, 
                                    double3* Fcn, double3* Fct, double3* Tcn, double3* Tct)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int cp = par->ParticleNum;

    if (i >= cp) return;

    if (LcFlg[i] == 0) return;

    int clumpI;
    double3 Dist, DistU, Vrel, Vrot, Vtot, Vnor, Vtan;
    double3 normal_force, tangential_force, tangential_history;
    double3 rolling_moment, rolling_history;
    double3 twisting_moment, twisting_history;
    double3 cohesive_force;
    double3 Ds, Dtheta, DthetaR, DthetaT, H, Mr, Mt;
    double3 Vwall, Arm, tmpT;

    double DistS, DistL, DistR, normal_forceL, tangential_forceL, tangential_historyL;
    double rolling_momentL, rolling_historyL, twisting_momentL, twisting_historyL;
    double Ap, Rij, Mij, Kn, Cn, Ks, Cs, Kr, Cr, Kt, Ct;
    double Dn, DsL, DthetaL, DthetaRL, DthetaTL, ERR, DCM[3][3];

    double Ymod = par->Ymod;
	SphereIntruderContact sphereContact;

    clumpI = ClumpId[i];

    Dist = RVn[i];
    H = Dv_pre[i];
    Mr = Dr_pre[i];
    Mt = Dt_pre[i];

    DistS = Dist.x * Dist.x + Dist.y * Dist.y + Dist.z * Dist.z;
    DistL = sqrt(DistS);
    Dn = Radius[i] - DistL;
	if (cyd->Shape == PENETRATOR_SPHERE)
	{
		sphereContact = GetSphereIntruderContact(
			Pos[i], Radius[i], PenePos[0], cyd->Radius, cyd->Dirct);
		Dist = sphereContact.normalGapVector;
		DistL = sphereContact.particleArm;
		Dn = sphereContact.overlap;
	}

    if (Dn > 0.0)
    {
		if (cyd->Shape == PENETRATOR_SPHERE)
		{
			DistU = sphereContact.normal;
		}
		else
		{
			DistR = 1.0 / DistL;
			DistU = DistR * Dist;
		}
        Ap = DistL;
        Rij = Radius[i];
		if (cyd->Shape == PENETRATOR_SPHERE)
		{
			Rij = Radius[i] * cyd->Radius / (Radius[i] + cyd->Radius);
		}
        Mij = Mass[i];
        if (cyd->MotionMode == PENETRATOR_GRAVITY)
        {
            Mij = Mass[i] * cyd->Property.w / (Mass[i] + cyd->Property.w);
        }
        Kn = 2.0 * Ymod * sqrt(Rij * Dn) / (3.0 * (1.0 - par->Prat * par->Prat));
        Cn = par->Damping * sqrt(Kn * Mij);

        Ks = 2.0 * Ymod / (1.0 + par->Prat) / (2.0 - par->Prat) * sqrt(Rij * Dn);
        Cs = par->Damping * sqrt(Ks * Mij);

        Kr = 0.25 * Kn * (par->Beta * Rij) * (par->Beta * Rij);
        Cr = 0.25 * Cn * (par->Beta * Rij) * (par->Beta * Rij);
        Kt = 0.50 * Ks * (par->Beta * Rij) * (par->Beta * Rij);
        Ct = 0.50 * Cs * (par->Beta * Rij) * (par->Beta * Rij);

        Quat2IvDCM(ClumpQuat[clumpI], DCM);
        Map(DCM, LocalPos[i], Arm);
        Arm = -1.0 * Arm + Ap * DistU;

        Vwall = Cross(PeneAngSpd[0], RVo[i]);
        Vrel = ClumpVel[clumpI] - PeneVel[0];
        Vrot = Cross(Arm, ClumpAngSpd[clumpI]) - Vwall;
        Vtot = Vrel + Vrot;

        //printf("Vwall: %f %f %f %f %f %f %f %f %f\n", Vwall.x, Vwall.y, Vwall.z, Vrot.x, Vrot.y, Vrot.z, Vtot.x, Vtot.y, Vtot.z);

        ERR = Dot(DistU, Vtot);
        Vnor = ERR * DistU;
        Vtan = Vtot - Vnor;

        normal_force = Kn * Dn * DistU + Cn * Vnor;
        normal_forceL = Length(normal_force);

        Ds = par->StepSize * Vtan;
        DsL = Length(Ds);

        tangential_force = -Ks * Ds + Cs * Vtan + H;
        tangential_forceL = Length(tangential_force);

        if (tangential_forceL > normal_forceL * par->Frcoe)
        {
            tangential_force = par->Frcoe * normal_forceL * (1.0 / tangential_forceL) * tangential_force;
            tangential_history = tangential_force;
        }
        else
        {
            tangential_history = tangential_force - Cs * Vtan;
        }

		cohesive_force = make_double3(0.0, 0.0, 0.0);

        Fc[i] = normal_force + tangential_force;
        Tc[i] = -Ap * Cross(DistU, tangential_force);
        peneTc[i] = -1.0 * Cross(RVo[i], (normal_force + tangential_force + cohesive_force));

        Fcn[i] = normal_force;
        Fct[i] = tangential_force;

        ////printf("Fc: %d %f %f %f\n", i, Fc[i].x, Fc[i].y, Fc[i].z);

        Dtheta = par->StepSize * (ClumpAngSpd[clumpI] - PeneAngSpd[0]);
        DthetaL = Dot(Dtheta, DistU);

        DthetaT = DthetaL * DistU;
        DthetaTL = Length(DthetaT);

        DthetaR = Dtheta - DthetaT;
        DthetaRL = Length(DthetaR);

        rolling_moment = Kr * DthetaR + Mr;
        rolling_momentL = Length(rolling_moment);

        if (rolling_momentL > 0.525 * par->Beta * Rij * normal_forceL)
        {
            rolling_moment = 0.525 * par->Beta * Rij * normal_forceL * (1.0 / rolling_momentL) * rolling_moment;
            rolling_history = rolling_moment;
        }
        else
        {
            rolling_history = rolling_moment;
            rolling_moment = rolling_moment - Cr * (1.0 / par->StepSize) * DthetaR;
        }

        twisting_moment = Kt * DthetaT + Mt;
        twisting_momentL = Length(twisting_moment);

        if (twisting_momentL > 0.65 * par->Frcoe * par->Beta * Rij * normal_forceL)
        {
            twisting_moment = 0.65 * par->Frcoe * par->Beta * Rij * normal_forceL * (1.0 / twisting_momentL) * twisting_moment;
            twisting_history = twisting_moment;
        }
        else
        {
            twisting_history = twisting_moment;
            twisting_moment = twisting_moment - Ct * (1.0 / par->StepSize) * DthetaT;
        }

        Tc[i] = Tc[i] - rolling_moment - twisting_moment;
        peneTc[i] = peneTc[i] + rolling_moment + twisting_moment;

        Dv[i] = tangential_history;
        Dt[i] = twisting_history;
        Dr[i] = rolling_history;

        Tcn[i] = Cross(normal_force, Arm);
        Tct[i] = Cross(tangential_force, Arm) - rolling_moment - twisting_moment;

        //printf("Fcn: %f %f %f\n", Fcn[i].x, Fcn[i].y, Fcn[i].z);

    }
}







void DEMPeneCylinder::CalcContactForce_Clump(Parameters* par_host, Parameters* par_device, int* ClumpId, double3* PartPos, double3* PartLocalPos, double* PartRadius, double* PartMass,
											 double3* ClumpVel, double3* ClumpAngSpd, double4* ClumpQuat)
{
    int cp = par_host->ParticleNum;

    dim3 block(128, 1);
    dim3 cuda_grid_size((cp + block.x - 1) / block.x, 1);

    ForcePene_clump_cuk << <cuda_grid_size, block >> > (par_device, ClumpId, PartPos, PartLocalPos, PartRadius, PartMass,
                                                        ClumpVel,ClumpAngSpd,ClumpQuat,
                                                        pene_device, Pos, Vel, AngSpd,
                                                        Dv_pre, Dv, Dt_pre, Dt, Dr_pre, Dr,
                                                        LcFlg, RVo, RVn, Fc, Tc, peneTc,
                                                        Fcn, Fct, Tcn, Tct);
    cudaDeviceSynchronize();
}

