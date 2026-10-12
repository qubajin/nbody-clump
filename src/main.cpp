#include"dem_clump.h"
#include "dem.h"

int main()
{

	DEMClumpMain dem;
	dem.RunSimulationClump();

	//DEMmain dem;
	//dem.RunSimulation2();

	//RunSimulation 圆柱体边界
	//RunSimulation2 立方体边界

	cout << "Final!!" << endl;
}