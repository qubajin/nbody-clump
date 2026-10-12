#pragma once
#include <vector_types.h>

#ifndef _DEM_PARAMETERS_H_
#define _DEM_PARAMETERS_H_

struct
{
	int ParticleNum;
	int CellNum;
	int BranchLmtLen;

	//Grid
	unsigned int Update;

	unsigned int GridDim_x;
	unsigned int GridDim_y;
	unsigned int GridDim_z;

	double CellSize_x;
	double CellSize_y;
	double CellSize_z;

	double i_CellSize_x;
	double i_CellSize_y;
	double i_CellSize_z;

	double WorldOrigin_x;
	double WorldOrigin_y;
	double WorldOrigin_z;

	double WorldSize_x;
	double WorldSize_y;
	double WorldSize_z;

	double BoxUpLmt_x;
	double BoxUpLmt_y;
	double BoxUpLmt_z;

	double BoxLowLmt_x;
	double BoxLowLmt_y;
	double BoxLowLmt_z;

	//ODE
	int EndStep;
	int StartStep;
	int OutputInterval;
	double StepSize;

	//MECH
	double Ymod;        /* i: Young's Module                           */
	double Prat;        /* i: Poisson's Ratio                          */
	double Frvel;       /* i: Friction Regularisation Velocity         */
	double Frcoe;       /* i: Friction Coefficient                     */
	double NormRes;     /* i: Restitution                              */	
	double Damping;
	double Gravity;
	double Beta;
	double MU_T;
	double MU_R;
	double Cohesion;

	double Thresh;       
	double GridLevels;

	int ClumpNum;

}typedef Parameters;


struct
{
	bool isSelfGravity;
	bool isIntruder;
	bool isWallForce;
	bool isQuaternion;

}typedef GlobalFlg;


struct
{
	double4 Property;

	double Radius;
	double Height;
	double3 Dirct;
    // Shape and motion fields used by the existing penetrator implementation.
    int Shape;
    int MotionMode;
    int WaitForContact;
    int ContactState;

}typedef Cylinder;

enum PenetratorShape { PENETRATOR_CYLINDER = 0, PENETRATOR_SPHERE = 1 };
enum PenetratorMotionMode { PENETRATOR_CONSTANT_VELOCITY = 0, PENETRATOR_GRAVITY = 1 };

struct
{
	double3 SideLength;

}typedef Cube;


#endif


