#pragma once
#ifndef _DEM_SPHERE_INTRUDER_CUH_
#define _DEM_SPHERE_INTRUDER_CUH_

#include <cuda_runtime.h>
#include <math.h>

#include "type.h"

struct SphereIntruderContact
{
	double3 normal;
	double3 surfaceArm;
	double3 normalGapVector;
	double gap;
	double overlap;
	double particleArm;
};

static __inline__ __host__ __device__ SphereIntruderContact
GetSphereIntruderContact(const double3 particlePos,
	const double particleRadius,
	const double3 intruderPos,
	const double intruderRadius,
	const double3 fallbackDirection)
{
	SphereIntruderContact contact;
	const double3 centreVector = particlePos - intruderPos;
	const double centreDistance = Length(centreVector);

	if (centreDistance > 1.0e-15)
	{
		contact.normal = (1.0 / centreDistance) * centreVector;
	}
	else
	{
		contact.normal = Norm(fallbackDirection);
		if (Length(contact.normal) < 1.0e-15)
		{
			contact.normal = make_double3(0.0, 0.0, 1.0);
		}
	}

	contact.surfaceArm = intruderRadius * contact.normal;
	contact.gap = centreDistance - intruderRadius;
	contact.overlap = particleRadius - contact.gap;
	contact.particleArm = contact.gap;
	contact.normalGapVector = contact.particleArm * contact.normal;

	return contact;
}

#endif
