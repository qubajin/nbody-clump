#include "dem_penetrator.h"

__global__ void IntegratePene_Before_cuk(Parameters* par, Cylinder* pene, double3* Pos,
                                         double3* Vel, double3* AngSpd, double4* Quat, double3* F,
                                         double3* T) {
    double Dt, norm, mass;
    double3 peneF, peneT, inertia;
    double4 Qdot, Qdotdot;
    double3 AngSpdB, AngSpdDotB;
    double DCM[3][3], IvDCM[3][3];

    // Quat2DCM(Quat[0], DCM);
    // Quat2IvDCM(Quat[0], IvDCM);

    // mass = pene->Property.w;
    // inertia = make_double3(pene->Property.x, pene->Property.y, pene->Property.z);

    // peneF = (1.0 / mass) * F[0];
    // Map(IvDCM, T[0], peneT);

    // Map(DCM, AngSpd[0], AngSpdB);

    // AngSpdDotB.x = (peneT.x - (inertia.z - inertia.y) * AngSpdB.y * AngSpdB.z) / inertia.x;
    // AngSpdDotB.y = (peneT.y - (inertia.x - inertia.z) * AngSpdB.z * AngSpdB.x) / inertia.y;
    // AngSpdDotB.z = (peneT.z - (inertia.y - inertia.x) * AngSpdB.x * AngSpdB.y) / inertia.z;

    Dt = par->StepSize;

    if (pene->MotionMode == PENETRATOR_GRAVITY &&
        (!pene->WaitForContact || pene->ContactState != 0)) {
        mass = pene->Property.w;
        // Fc is the force exerted by the penetrator on the particles,
        // so the penetrator receives the equal and opposite reaction.
        peneF = (-1.0 / mass) * F[0];
        peneF.z = peneF.z + par->Gravity;
        Pos[0] = Pos[0] + Dt * Vel[0] + 0.5 * Dt * Dt * peneF;
        Vel[0] = Vel[0] + 0.5 * Dt * peneF;
    } else {
        // Kinematic mode: contact forces cannot change the prescribed speed.
        Pos[0] = Pos[0] + Dt * Vel[0];
    }

    /*Qdot.x = ( AngSpdB.x * Quat[0].w - AngSpdB.y * Quat[0].z + AngSpdB.z * Quat[0].y) * 0.5;
    Qdot.y = ( AngSpdB.x * Quat[0].z + AngSpdB.y * Quat[0].w - AngSpdB.z * Quat[0].x) * 0.5;
    Qdot.z = (-AngSpdB.x * Quat[0].y + AngSpdB.y * Quat[0].x + AngSpdB.z * Quat[0].w) * 0.5;
    Qdot.w = (-AngSpdB.x * Quat[0].x - AngSpdB.y * Quat[0].y - AngSpdB.z * Quat[0].z) * 0.5;

    Qdotdot.x = ( AngSpdDotB.x * Quat[0].w - AngSpdDotB.y * Quat[0].z + AngSpdDotB.z * Quat[0].y) *
    0.5 + ( AngSpd[0].x * Qdot.w - AngSpd[0].y * Qdot.z + AngSpd[0].z * Qdot.y) * 0.5; Qdotdot.y = (
    AngSpdDotB.x * Quat[0].z + AngSpdDotB.y * Quat[0].w - AngSpdDotB.z * Quat[0].x) * 0.5 + (
    AngSpd[0].x * Qdot.z + AngSpd[0].y * Qdot.w - AngSpd[0].z * Qdot.x) * 0.5; Qdotdot.z =
    (-AngSpdDotB.x * Quat[0].y + AngSpdDotB.y * Quat[0].x + AngSpdDotB.z * Quat[0].w) * 0.5 +
    (-AngSpd[0].x * Qdot.y + AngSpd[0].y * Qdot.x + AngSpd[0].z * Qdot.w) * 0.5; Qdotdot.w =
    (-AngSpdDotB.x * Quat[0].x - AngSpdDotB.y * Quat[0].y - AngSpdDotB.z * Quat[0].z) * 0.5 +
    (-AngSpd[0].x * Qdot.x - AngSpd[0].y * Qdot.y - AngSpd[0].z * Qdot.z) * 0.5;

    Quat[0] = Quat[0] + Dt * Qdot + 0.5 * Dt * Dt * Qdotdot;

    norm = sqrt(Quat[0].x * Quat[0].x + Quat[0].y * Quat[0].y + Quat[0].z * Quat[0].z + Quat[0].w *
    Quat[0].w);

    Quat[0] = (1 / norm) * Quat[0];

    AngSpdB = AngSpdB + 0.5 * Dt * AngSpdDotB;
    pene->Dirct = make_double3(DCM[0][2], DCM[1][2], DCM[2][2]);

    Map(IvDCM, AngSpdB, AngSpd[0]);*/
}

void DEMPeneCylinder::Integrate_Before(Parameters* par_device) {
    // IntegratePene_Before_cuk(Parameters * par, Cylinder * pene_device, double3 * Pos, double3 *
    // Vel, double3 * AngSpd, double4 * Quat, double3 * AngSpdDotB, double3 * F, double3 * T,
    // double4 * Property)
    IntegratePene_Before_cuk<<<1, 1>>>(par_device, pene_device, Pos, Vel, AngSpd, Quat, totFc,
                                       totTc);
    cudaDeviceSynchronize();
}

__global__ void Integrate_After_cuk(int count, Parameters* par, Cylinder* pene, double3* Vel,
                                    double3* AngSpd, double4* Quat, double3* F, double3* T,
                                    double3* CacheFc, double3* CacheTc) {
    double Dt, mass;
    double3 peneF, reactionForce;
    // double IvDCM[3][3];
    if (pene->WaitForContact && pene->ContactState == 1) {
        printf("First contact, pre-kick velocity: "
               "vx=%.12e vy=%.12e vz=%.12e m/s\n",
               Vel[0].x, Vel[0].y, Vel[0].z);

        pene->ContactState = 2;
    }

    if (pene->MotionMode == PENETRATOR_GRAVITY &&
        (!pene->WaitForContact || pene->ContactState != 0)) {
        Dt = par->StepSize;
        mass = pene->Property.w;
        reactionForce = -1.0 * F[0];
        peneF = (1.0 / mass) * reactionForce;
        peneF.z = peneF.z + par->Gravity;
        Vel[0] = Vel[0] + 0.5 * Dt * peneF;
    } else {
        reactionForce = -1.0 * F[0];
    }
    // AngSpdB = AngSpdB + 0.5 * Dt * AngSpdDotB;
    // Quat2IvDCM(Quat[0], IvDCM);
    // Map(IvDCM, AngSpdB, AngSpd[0]);

    // Store the contact reaction acting on the penetrator.  Gravity is kept
    // separate so the reported force can be compared directly with a load cell.
    CacheFc[count] = reactionForce;
    CacheTc[count] = T[0];

    // if (Length(F[0]) != 0) printf("integrate: %d %f %f %f\n", count, CacheFc[count].x,
    // CacheFc[count].y, CacheFc[count].z);
}

void DEMPeneCylinder::Integrate_After(int count, Parameters* par_device) {
    Integrate_After_cuk<<<1, 1>>>(count, par_device, pene_device, Vel, AngSpd, Quat, totFc, totTc,
                                  CacheFc, CacheTc);
}
