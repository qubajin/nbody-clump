#pragma once
#ifndef _TYPE_H_
#define _TYPE_H_

#include<cuda_runtime.h>
#include<math.h>

#include<stdio.h>

static __inline__ __host__ __device__ int3 operator + (const int3& v1, const int3& v2)
{
	int3 v;
	v.x = v1.x + v2.x;
	v.y = v1.y + v2.y;
	v.z = v1.z + v2.z;

	return v;
}

static __inline__ __host__ __device__ double3 operator + (const double3& v1, const double3& v2)
{
	double3 v;
	v.x = v1.x + v2.x;
	v.y = v1.y + v2.y;
	v.z = v1.z + v2.z;

	return v;
}

static __inline__ __host__ __device__ double3 operator - (const double3& v1, const double3& v2)
{
	double3 v;
	v.x = v1.x - v2.x;
	v.y = v1.y - v2.y;
	v.z = v1.z - v2.z;

	return v;
}

static __inline__ __host__ __device__ double3 operator * (const double s, const double3 u)
{
	double3 v;

	v.x = s * u.x;
	v.y = s * u.y;
	v.z = s * u.z;

	return v;
}

static __inline__ __host__ __device__ double3 operator * (const double3 u, const double3 v)
{
	double3 w;

	w.x = u.x * v.x;
	w.y = u.y * v.y;
	w.z = u.z * v.z;

	return w;
}

static __inline__ __host__ __device__ double3 operator / (const double3& v1, const double3& v2)
{
	double3 v;
	v.x = v1.x / v2.x;
	v.y = v1.y / v2.y;
	v.z = v1.z / v2.z;

	return v;
}

static __inline__ __host__ __device__ double3 Norm(const double3 v1)
{
	double3 v;
	double tmp = sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);

	if (tmp > 1e-20)
	{
		v.x = v1.x / tmp;
		v.y = v1.y / tmp;
		v.z = v1.z / tmp;
	}
	else
	{
		v = make_double3(0.0, 0.0, 0.0);
	}

	return v;
}

static __inline__ __host__ __device__ double Dot(const double3 v1, const double3 v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

static __inline__ __host__ __device__ double3 Cross(const double3 v1, const double3 v2)
{
	double3 w;

	double a, b, c;

	a = v1.y * v2.z - v1.z * v2.y;
	b = v1.z * v2.x - v1.x * v2.z;
	c = v1.x * v2.y - v1.y * v2.x;

	w = make_double3(a, b, c);

	return w;

}

static __inline__ __host__ __device__ double Length(const double3 v1)
{
	double tmp = v1.x * v1.x + v1.y * v1.y + v1.z * v1.z;

	if (tmp > 1e-32)
	{
		tmp = sqrt(tmp);
	}
	else
	{
		tmp = 0.0;
	}

	return tmp;
}

static __inline__ __host__ __device__ double Length(const double v)
{
	double tmp;

	if (v > 1e-32)
	{
		tmp = sqrt(v);
	}
	else
	{
		tmp = 0.0;
	}

	return tmp;
}


//-------------------------------------------
//-------------------------------------------
//-------------------------------------------
static __inline__ __host__ __device__ double4 operator + (const double4& v1, const double4& v2)
{
	double4 v;
	v.x = v1.x + v2.x;
	v.y = v1.y + v2.y;
	v.z = v1.z + v2.z;
	v.w = v1.w + v2.w;

	return v;
}

static __inline__ __host__ __device__ double4 operator - (const double4& v1, const double4& v2)
{
	double4 v;
	v.x = v1.x - v2.x;
	v.y = v1.y - v2.y;
	v.z = v1.z - v2.z;
	v.w = v1.w - v2.w;

	return v;
}

static __inline__ __host__ __device__ double4 operator * (const double4& v1, const double4& v2)
{
	double4 v;
	v.x = v1.x * v2.x;
	v.y = v1.y * v2.y;
	v.z = v1.z * v2.z;
	v.w = v1.w * v2.w;

	return v;
}

static __inline__ __host__ __device__ double4 operator * (const double s, const double4 u)
{
	double4 v;
	v.x = s * u.x;
	v.y = s * u.y;
	v.z = s * u.z;
	v.w = s * u.w;

	return v;
}

static __inline__ __host__ __device__ double4 operator / (const double4& v1, const double4& v2)
{
	double4 v;
	v.x = v1.x / v2.x;
	v.y = v1.y / v2.y;
	v.z = v1.z / v2.z;
	v.w = v1.w / v2.w;

	return v;
}

static __inline__ __host__ __device__ double4 Norm(const double4& v1)
{
	double4 v;
	double tmp = sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z + v1.w * v1.w);

	if (tmp > 1e-32)
	{
		v.x = v1.x / tmp;
		v.y = v1.y / tmp;
		v.z = v1.z / tmp;
		v.w = v1.w / tmp;
	}
	else
	{
		v = make_double4(0.0, 0.0, 0.0, 0.0);
	}

	return v;
}

void  __inline__ __host__ __device__ Quat2DCM(const double4 q, double t_MatI[3][3])  //Cnb
{
	
	t_MatI[0][0] = q.x * q.x - q.y * q.y - q.z * q.z + q.w * q.w;
	t_MatI[1][0] = 2.0 * (q.x * q.y - q.z * q.w);
	t_MatI[2][0] = 2.0 * (q.x * q.z + q.y * q.w);
	t_MatI[0][1] = 2.0 * (q.x * q.y + q.z * q.w);
	t_MatI[1][1] = -q.x * q.x + q.y * q.y - q.z * q.z + q.w * q.w;
	t_MatI[2][1] = 2.0 * (q.y * q.z - q.x * q.w);
	t_MatI[0][2] = 2.0 * (q.x * q.z - q.y * q.w);
	t_MatI[1][2] = 2.0 * (q.y * q.z + q.x * q.w);
	t_MatI[2][2] = -q.x * q.x - q.y * q.y + q.z * q.z + q.w * q.w;

}

void  __inline__ __host__ __device__ Quat2IvDCM(const double4 q, double t_MatB[3][3])   //Cbn
{

	t_MatB[0][0] = q.x * q.x - q.y * q.y - q.z * q.z + q.w * q.w;
	t_MatB[1][0] = 2.0 * (q.x * q.y + q.z * q.w);
	t_MatB[2][0] = 2.0 * (q.x * q.z - q.y * q.w);
	t_MatB[0][1] = 2.0 * (q.x * q.y - q.z * q.w);
	t_MatB[1][1] = -q.x * q.x + q.y * q.y - q.z * q.z + q.w * q.w;
	t_MatB[2][1] = 2.0 * (q.y * q.z + q.x * q.w);
	t_MatB[0][2] = 2.0 * (q.x * q.z + q.y * q.w);
	t_MatB[1][2] = 2.0 * (q.y * q.z - q.x * q.w);
	t_MatB[2][2] = -q.x * q.x - q.y * q.y + q.z * q.z + q.w * q.w;
}

void __inline__ __host__ __device__ DCM2Quat(const double m[3][3], double4 q)
{
	double t;
	if (m[2][2] < 0)
	{
		if (m[0][0] > m[1][1])
		{
			t = 1 + m[0][0] - m[1][1] - m[2][2];
			q.x = t;
			q.y = m[0][1] + m[1][0];
			q.z = m[2][0] + m[0][2];
			q.w = m[1][2] - m[2][1];
		}
		else
		{
			t = 1 - m[0][0] + m[1][1] - m[2][2];
			q.x = m[0][1] + m[1][0];
			q.y = t;
			q.z = m[1][2] + m[2][1];
			q.w = m[2][0] - m[0][2];
		}
	}
	else
	{
		if (m[0][0] < -m[1][1])
		{
			t = 1 - m[0][0] - m[1][1] + m[2][2];
			q.x = m[2][0] - m[0][2];
			q.y = m[1][2] + m[2][1];
			q.z = t;
			q.w = m[0][1] - m[1][0];
		}
		else
		{
			t = 1 + m[0][0] + m[1][1] + m[2][2];
			q.x = m[1][2] - m[2][1];
			q.y = m[2][0] - m[0][2];
			q.z = m[0][1] - m[1][0];
			q.w = t;
		}
	}
	q = 0.5 / sqrt(t) * q;
}

void  __inline__ __host__ __device__ Map(const double m[3][3], const double3 u, double3& v)
{
	double a, b, c;

	a = m[0][0] * u.x + m[0][1] * u.y + m[0][2] * u.z;
	b = m[1][0] * u.x + m[1][1] * u.y + m[1][2] * u.z;
	c = m[2][0] * u.x + m[2][1] * u.y + m[2][2] * u.z;

	v = make_double3(a, b, c);
}

void __inline__ __host__ __device__ Rx(double phi, double m[3][3])
{
	double a = (1 - (phi * phi / 4)) / (1 + (phi * phi / 4));  //cos(phi)
	double b = phi / (1 + (phi * phi / 4));  //sin(phi)

	m[0][0] = 1;
	m[0][1] = 0;
	m[0][2] = 0;
	m[1][0] = 0;
	m[1][1] = a;
	m[1][2] = -b;
	m[2][0] = 0;
	m[2][1] = b;
	m[2][2] = a;

}

void __inline__ __host__ __device__ Ry(double phi, double m[3][3])
{
	double a = (1 - (phi * phi / 4)) / (1 + (phi * phi / 4));  //cos(phi)
	double b = phi / (1 + (phi * phi / 4));  //sin(phi)

	m[0][0] = a;
	m[0][1] = 0;
	m[0][2] = b;
	m[1][0] = 0;
	m[1][1] = 1;
	m[1][2] = 0;
	m[2][0] = -b;
	m[2][1] = 0;
	m[2][2] = a;
}

void __inline__ __host__ __device__ Rz(double phi, double m[3][3])
{
	double a = (1 - (phi * phi / 4)) / (1 + (phi * phi / 4));  //cos(phi)
	double b = phi / (1 + (phi * phi / 4));  //sin(phi)

	m[0][0] = a;
	m[0][1] = -b;
	m[0][2] = 0;
	m[1][0] = b;
	m[1][1] = a;
	m[1][2] = 0;
	m[2][0] = 0;
	m[2][1] = 0;
	m[2][2] = 1;
}

void __inline__ __host__ __device__ matrixTranspose(double a[3][3])
{
	double tmp;

	tmp = a[0][1];
	a[0][1] = a[1][0];
	a[1][0] = tmp;

	tmp = a[0][2];
	a[0][2] = a[2][0];
	a[2][0] = tmp;

	tmp = a[1][2];
	a[1][2] = a[2][1];
	a[2][1] = tmp;
}

void __inline__ __host__ __device__ matrixMultiply(const double a[3][3], const double b[3][3], double c[3][3])
{
	c[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0];
	c[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1];
	c[0][2] = a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2];
	c[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0];
	c[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1];
	c[1][2] = a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2];
	c[2][0] = a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0];
	c[2][1] = a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1];
	c[2][2] = a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2];
}


#define Max2(a,b) (((a) > (b)) ? (a) : (b))
#define Min2(a,b) (((a) < (b)) ? (a) : (b))

__inline__ __host__ __device__ double Max3(double a, double b, double c)
{
	return a > b ? (a > c ? a : c) : (b > c ? b : c);
}

__inline__ __host__ __device__ double Min3(double a, double b, double c)
{
	return a < b ? (a < c ? a : c) : (b < c ? b : c);
}


#endif // !_TYPE_H_



