/*
    LA: linear algebra C++ interface library
    Copyright (C) 2020 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "vecmat3.h"
#ifndef QUAT_NO_RANDOM 
#include "la_random.h"
#endif

namespace LA_Vecmat3 {


//http://en.wikipedia.org/wiki/Fast_inverse_square_root
float fast_sqrtinv(float number )
{
 long i;
 float x2, y;
 const float threehalfs = 1.5F;

 x2 = number * 0.5F;
 y  = number;
 i  = * ( long * ) &y;                       // evil floating point bit level hacking
 i  = 0x5f3759df - ( i >> 1 );               // what the fuck? 
 y  = * ( float * ) &i;
 y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
 y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

 return y;
}


template<>
Vec3<float> & Vec3<float>::fast_normalize(void) {*this *= fast_sqrtinv(normsqr()); return *this;};

template<typename T>
Vec3<T>& Vec3<T>::fast_normalize(void) {normalize(); return *this;};


template<typename T>
Mat3<T> Vec3<T>::outer(const Vec3<T> &rhs) const
{
Mat3<T> m;
m[0][0]=q[0]*rhs.q[0];
m[0][1]=q[0]*rhs.q[1];
m[0][2]=q[0]*rhs.q[2];

m[1][0]=q[1]*rhs.q[0];
m[1][1]=q[1]*rhs.q[1];
m[1][2]=q[1]*rhs.q[2];

m[2][0]=q[2]*rhs.q[0];
m[2][1]=q[2]*rhs.q[1];
m[2][2]=q[2]*rhs.q[2];

return m;
}

template<typename T>
void Vec3<T>::addouter(Mat3<T> &m, const Vec3<T> &rhs,  const T weight) const
{
m[0][0]+=weight* q[0]*rhs.q[0];
m[0][1]+=weight* q[0]*rhs.q[1];
m[0][2]+=weight* q[0]*rhs.q[2];

m[1][0]+=weight* q[1]*rhs.q[0];
m[1][1]+=weight* q[1]*rhs.q[1];
m[1][2]+=weight* q[1]*rhs.q[2];

m[2][0]+=weight* q[2]*rhs.q[0];
m[2][1]+=weight* q[2]*rhs.q[1];
m[2][2]+=weight* q[2]*rhs.q[2];

}


template<typename T>
void Vec3<T>::inertia(Mat3<T> &m, const T weight) const
{
T r2 = q[0]*q[0]+q[1]*q[1]+q[2]*q[2];
T tmp;

m[0][0] -= weight*(q[0]*q[0]-r2);

tmp= weight*q[0]*q[1];
m[0][1] -= tmp; 
m[1][0] -= tmp;

tmp = weight*q[0]*q[2];
m[0][2] -=  tmp;
m[2][0] -= tmp;

m[1][1] -= weight*(q[1]*q[1]-r2);

tmp =  weight*q[1]*q[2];
m[1][2] -= tmp;
m[2][1] -= tmp;

m[2][2] -= weight*(q[2]*q[2]-r2);
}



template<typename T>
const Vec3<T> Vec3<T>::operator*(const Mat3<T> &rhs) const
                {
                Vec3<T> r;
                r[0] = q[0]*rhs.q[0][0] + q[1]*rhs.q[1][0] + q[2]*rhs.q[2][0];
                r[1] = q[0]*rhs.q[0][1] + q[1]*rhs.q[1][1] + q[2]*rhs.q[2][1];
                r[2] = q[0]*rhs.q[0][2] + q[1]*rhs.q[1][2] + q[2]*rhs.q[2][2];
                return r;
                }; 

template<typename T>
const Vec3<T> Vec3<T>::timesT(const Mat3<T> &rhs) const
                { 
                Vec3<T> r;
                r[0] = q[0]*rhs.q[0][0] + q[1]*rhs.q[0][1] + q[2]*rhs.q[0][2];
                r[1] = q[0]*rhs.q[1][0] + q[1]*rhs.q[1][1] + q[2]*rhs.q[1][2];
                r[2] = q[0]*rhs.q[2][0] + q[1]*rhs.q[2][1] + q[2]*rhs.q[2][2];
                return r;
                }; 




template<typename T>
Mat3<T>& Mat3<T>::operator+=(const Mat3 &rhs) 
{
q[0][0]+=rhs.q[0][0];q[0][1]+=rhs.q[0][1];q[0][2]+=rhs.q[0][2]; q[1][0]+=rhs.q[1][0];q[1][1]+=rhs.q[1][1];q[1][2]+=rhs.q[1][2]; q[2][0]+=rhs.q[2][0];q[2][1]+=rhs.q[2][1];q[2][2]+=rhs.q[2][2]; 
return *this;
}

template<typename T>
Mat3<T>& Mat3<T>::operator-=(const Mat3 &rhs) 
{
q[0][0]-=rhs.q[0][0];q[0][1]-=rhs.q[0][1];q[0][2]-=rhs.q[0][2]; q[1][0]-=rhs.q[1][0];q[1][1]-=rhs.q[1][1];q[1][2]-=rhs.q[1][2]; q[2][0]-=rhs.q[2][0];q[2][1]-=rhs.q[2][1];q[2][2]-=rhs.q[2][2]; 
return *this;
}


template<typename T>
const Mat3<T> Mat3<T>::operator*(const Mat3 &rhs) const
                {
                Mat3<T> r;
                r[0][0]= q[0][0]*rhs.q[0][0] + q[0][1]*rhs.q[1][0] + q[0][2]*rhs.q[2][0];
                r[0][1]= q[0][0]*rhs.q[0][1] + q[0][1]*rhs.q[1][1] + q[0][2]*rhs.q[2][1];
                r[0][2]= q[0][0]*rhs.q[0][2] + q[0][1]*rhs.q[1][2] + q[0][2]*rhs.q[2][2];
                r[1][0]= q[1][0]*rhs.q[0][0] + q[1][1]*rhs.q[1][0] + q[1][2]*rhs.q[2][0];
                r[1][1]= q[1][0]*rhs.q[0][1] + q[1][1]*rhs.q[1][1] + q[1][2]*rhs.q[2][1];
                r[1][2]= q[1][0]*rhs.q[0][2] + q[1][1]*rhs.q[1][2] + q[1][2]*rhs.q[2][2];
                r[2][0]= q[2][0]*rhs.q[0][0] + q[2][1]*rhs.q[1][0] + q[2][2]*rhs.q[2][0];
                r[2][1]= q[2][0]*rhs.q[0][1] + q[2][1]*rhs.q[1][1] + q[2][2]*rhs.q[2][1];
                r[2][2]= q[2][0]*rhs.q[0][2] + q[2][1]*rhs.q[1][2] + q[2][2]*rhs.q[2][2];
                return r;
                }  

template<typename T>
const Mat3<T> Mat3<T>::Ttimes(const Mat3 &rhs) const
                {
                Mat3<T> r;
                r[0][0]= q[0][0]*rhs.q[0][0] + q[1][0]*rhs.q[1][0] + q[2][0]*rhs.q[2][0];
                r[0][1]= q[0][0]*rhs.q[0][1] + q[1][0]*rhs.q[1][1] + q[2][0]*rhs.q[2][1];
                r[0][2]= q[0][0]*rhs.q[0][2] + q[1][0]*rhs.q[1][2] + q[2][0]*rhs.q[2][2];
                r[1][0]= q[0][1]*rhs.q[0][0] + q[1][1]*rhs.q[1][0] + q[2][1]*rhs.q[2][0];
                r[1][1]= q[0][1]*rhs.q[0][1] + q[1][1]*rhs.q[1][1] + q[2][1]*rhs.q[2][1];
                r[1][2]= q[0][1]*rhs.q[0][2] + q[1][1]*rhs.q[1][2] + q[2][1]*rhs.q[2][2];
                r[2][0]= q[0][2]*rhs.q[0][0] + q[1][2]*rhs.q[1][0] + q[2][2]*rhs.q[2][0];
                r[2][1]= q[0][2]*rhs.q[0][1] + q[1][2]*rhs.q[1][1] + q[2][2]*rhs.q[2][1];
                r[2][2]= q[0][2]*rhs.q[0][2] + q[1][2]*rhs.q[1][2] + q[2][2]*rhs.q[2][2];
                return r;
                }


template<typename T>
const Mat3<T> Mat3<T>::timesT(const Mat3 &rhs) const
                {
                Mat3<T> r;
                r[0][0]= q[0][0]*rhs.q[0][0] + q[0][1]*rhs.q[0][1] + q[0][2]*rhs.q[0][2];
                r[0][1]= q[0][0]*rhs.q[1][0] + q[0][1]*rhs.q[1][1] + q[0][2]*rhs.q[1][2];
                r[0][2]= q[0][0]*rhs.q[2][0] + q[0][1]*rhs.q[2][1] + q[0][2]*rhs.q[2][2];
                r[1][0]= q[1][0]*rhs.q[0][0] + q[1][1]*rhs.q[0][1] + q[1][2]*rhs.q[0][2];
                r[1][1]= q[1][0]*rhs.q[1][0] + q[1][1]*rhs.q[1][1] + q[1][2]*rhs.q[1][2];
                r[1][2]= q[1][0]*rhs.q[2][0] + q[1][1]*rhs.q[2][1] + q[1][2]*rhs.q[2][2];
                r[2][0]= q[2][0]*rhs.q[0][0] + q[2][1]*rhs.q[0][1] + q[2][2]*rhs.q[0][2];
                r[2][1]= q[2][0]*rhs.q[1][0] + q[2][1]*rhs.q[1][1] + q[2][2]*rhs.q[1][2];
                r[2][2]= q[2][0]*rhs.q[2][0] + q[2][1]*rhs.q[2][1] + q[2][2]*rhs.q[2][2];
                return r;
                }

template<typename T>
const Mat3<T> Mat3<T>::TtimesT(const Mat3 &rhs) const
                {
                Mat3<T> r;
                r[0][0]= q[0][0]*rhs.q[0][0] + q[1][0]*rhs.q[0][1] + q[2][0]*rhs.q[0][2];
                r[0][1]= q[0][0]*rhs.q[1][0] + q[1][0]*rhs.q[1][1] + q[2][0]*rhs.q[1][2];
                r[0][2]= q[0][0]*rhs.q[2][0] + q[1][0]*rhs.q[2][1] + q[2][0]*rhs.q[2][2];
                r[1][0]= q[0][1]*rhs.q[0][0] + q[1][1]*rhs.q[0][1] + q[2][1]*rhs.q[0][2];
                r[1][1]= q[0][1]*rhs.q[1][0] + q[1][1]*rhs.q[1][1] + q[2][1]*rhs.q[1][2];
                r[1][2]= q[0][1]*rhs.q[2][0] + q[1][1]*rhs.q[2][1] + q[2][1]*rhs.q[2][2];
                r[2][0]= q[0][2]*rhs.q[0][0] + q[1][2]*rhs.q[0][1] + q[2][2]*rhs.q[0][2];
                r[2][1]= q[0][2]*rhs.q[1][0] + q[1][2]*rhs.q[1][1] + q[2][2]*rhs.q[1][2];
                r[2][2]= q[0][2]*rhs.q[2][0] + q[1][2]*rhs.q[2][1] + q[2][2]*rhs.q[2][2];
                return r;
                }




template<typename T>
T Mat3<T>::determinant() const 
{
return q[0][0]*(q[2][2]*q[1][1]-q[2][1]*q[1][2])-q[1][0]*(q[2][2]*q[0][1]-q[2][1]*q[0][2])+q[2][0]*(q[1][2]*q[0][1]-q[1][1]*q[0][2]);
}

template<typename T>
void Mat3<T>::transposeme() 
{T t; t=q[0][1]; q[0][1]=q[1][0]; q[1][0]=t; t=q[0][2]; q[0][2]=q[2][0]; q[2][0]=t; t=q[1][2]; q[1][2]=q[2][1]; q[2][1]=t;};

template<typename T>
const Mat3<T> Mat3<T>::inverse(T *det) const
                {
                Mat3<T> r;
                r[0][0]=  q[2][2]*q[1][1]-q[2][1]*q[1][2];
                r[0][1]= -q[2][2]*q[0][1]+q[2][1]*q[0][2];
                r[0][2]=  q[1][2]*q[0][1]-q[1][1]*q[0][2];
                r[1][0]= -q[2][2]*q[1][0]+q[2][0]*q[1][2];
                r[1][1]=  q[2][2]*q[0][0]-q[2][0]*q[0][2];
                r[1][2]= -q[1][2]*q[0][0]+q[1][0]*q[0][2];
                r[2][0]=  q[2][1]*q[1][0]-q[2][0]*q[1][1];
                r[2][1]= -q[2][1]*q[0][0]+q[2][0]*q[0][1];
                r[2][2]=  q[1][1]*q[0][0]-q[1][0]*q[0][1];
		T d=determinant();
		if(det) *det=d;
                return r/d;
                }

template<typename T>
const Vec3<T> Mat3<T>::linear_solve(const  Vec3<T> &rhs, T *det) const
                {
                Vec3<T> r;
                r[0]= ( q[2][2]*q[1][1]-q[2][1]*q[1][2]) * rhs[0]
                      +( -q[2][2]*q[0][1]+q[2][1]*q[0][2]) * rhs[1]
                      +( +q[1][2]*q[0][1]-q[1][1]*q[0][2]) * rhs[2];
                r[1]= (-q[2][2]*q[1][0]+q[2][0]*q[1][2]) * rhs[0]
                      +( +q[2][2]*q[0][0]-q[2][0]*q[0][2])  * rhs[1]
                      +( -q[1][2]*q[0][0]+q[1][0]*q[0][2]) * rhs[2];
                r[2]= ( q[2][1]*q[1][0]-q[2][0]*q[1][1]) * rhs[0]
                      +( -q[2][1]*q[0][0]+q[2][0]*q[0][1]) *rhs[1]
                      +( q[1][1]*q[0][0]-q[1][0]*q[0][1]) * rhs[2];
		T d=determinant();
		if(det) *det=d;
                return r/d;
                }

template<typename T>
const Vec3<T> Mat3<T>::operator*(const Vec3<T> &rhs) const
                {
                Vec3<T> r;
                r[0] = q[0][0]*rhs.q[0] + q[0][1]*rhs.q[1] + q[0][2]*rhs.q[2];
                r[1] = q[1][0]*rhs.q[0] + q[1][1]*rhs.q[1] + q[1][2]*rhs.q[2];
                r[2] = q[2][0]*rhs.q[0] + q[2][1]*rhs.q[1] + q[2][2]*rhs.q[2];
                return r;
                }

template<typename T>
const Vec3<T> Mat3<T>::Ttimes(const Vec3<T> &rhs) const
                {
                Vec3<T> r;
                r[0] = q[0][0]*rhs.q[0] + q[1][0]*rhs.q[1] + q[2][0]*rhs.q[2];
                r[1] = q[0][1]*rhs.q[0] + q[1][1]*rhs.q[1] + q[2][1]*rhs.q[2];
                r[2] = q[0][2]*rhs.q[0] + q[1][2]*rhs.q[1] + q[2][2]*rhs.q[2];
                return r;
                }





//cf. https://en.wikipedia.org/wiki/Euler_angles and NASA paper cited therein
template<typename T>
void euler2rotmat(const T *eul, Mat3<T> &a, const char *type, bool transpose, bool direction, bool reverse)
{
T c2=cos(eul[1]); 
T s2=sin(eul[1]);
T c1=cos(eul[reverse?2:0]);
T s1=sin(eul[reverse?2:0]);
T c3=cos(eul[reverse?0:2]);
T s3=sin(eul[reverse?0:2]);

if(direction) {s1= -s1; s2= -s2; s3= -s3;}

switch(Euler_case(type[0],type[1],type[2]))
{
case Euler_case('x','z','x'):
	{
	a[0][0]= c2;
	a[0][1]= -c3*s2;
	a[0][2]= s2*s3;
	a[1][0]= c1*s2;
	a[1][1]= c1*c2*c3-s1*s3;
	a[1][2]= -c3*s1-c1*c2*s3; 
	a[2][0]= s1*s2; 
	a[2][1]= c1*s3+c2*c3*s1; 
	a[2][2]= c1*c3-c2*s1*s3;
	}
	break;

case Euler_case('x','y','x'):
	{
	a[0][0]= c2;
	a[0][1]= s2*s3;
	a[0][2]= c3*s2;
	a[1][0]= s1*s2;
	a[1][1]= c1*c3-c2*s1*s3;
	a[1][2]=  -c1*s3-c2*c3*s1;
	a[2][0]=  -c1*s2;
	a[2][1]=  c3*s1+c1*c2*s3;
	a[2][2]=  c1*c2*c3-s1*s3;
	}
	break;

case Euler_case('y','x','y'):
        {
        a[0][0]= c1*c3-c2*s1*s3;
        a[0][1]= s1*s2;
        a[0][2]= c1*s3+c2*c3*s1;
        a[1][0]= s2*s3;
        a[1][1]= c2;
        a[1][2]= -c3*s2;
        a[2][0]= -c3*s1-c1*c2*s3;
        a[2][1]= c1*s2;
        a[2][2]= c1*c2*c3-s1*s3;
        }
	break;

case Euler_case('y','z','y'):
        {
        a[0][0]= c1*c2*c3-s1*s3;
        a[0][1]= -c1*s2;
        a[0][2]= c3*s1+c1*c2*s3;
        a[1][0]= c3*s2;
        a[1][1]= c2;
        a[1][2]= s2*s3;
        a[2][0]= -c1*s3;
        a[2][1]= s1*s2;
        a[2][2]= c1*c3-c2*s1*s3;
        }
	break;

case Euler_case('z','y','z'):
        {
        a[0][0]= c1*c2*c3-s1*s3;
        a[0][1]= -c3*s1-c1*c2*s3;
        a[0][2]= c1*s2;
        a[1][0]= c1*s3+c2*c3*s1;
        a[1][1]= c1*c3-c2*s1*s3;
        a[1][2]= s1*s2;
        a[2][0]= -c3*s2;
        a[2][1]= s2*s3;
        a[2][2]= c2;
        }
	break;

case Euler_case('z','x','z'):
        {
        a[0][0]= c1*c3-c2*s1*s3;
        a[0][1]= -c1*s3-c2*c3*s1;
        a[0][2]= s1*s2;
        a[1][0]= c3*s1+c1*c2*s3;
        a[1][1]= c1*c2*c3-s1*s3;
        a[1][2]= -c1*s2;
        a[2][0]= s2*s3;
        a[2][1]= c3*s2;
        a[2][2]= c2;
        }
	break;

case Euler_case('x','z','y'):
        {
        a[0][0]= c2*c3;
        a[0][1]= -s2;
        a[0][2]= c2*s3;
        a[1][0]= s1*s3+c1*c3*s2;
        a[1][1]= c1*c2;
        a[1][2]= c1*s2*s3-c3*s1;
        a[2][0]= c3*s1*s2-c1*s3;
        a[2][1]= c2*s1;
        a[2][2]= c1*c3+s1*s2*s3;
        }
	break;

case Euler_case('x','y','z'):
        {
        a[0][0]= c2*c3;
        a[0][1]= -c2*s3;
        a[0][2]= s2;
        a[1][0]= c1*s3+c3*s1*s2;
        a[1][1]= c1*c3-s1*s2*s3;
        a[1][2]= -c2*s1;
        a[2][0]= s1*s3-c1*c3*s2;
        a[2][1]= c3*s1+c1*s2*s3;
        a[2][2]= c1*c2;
        }
	break;

case Euler_case('y','x','z'):
        {
        a[0][0]= c1*c3+s1*s2*s3;
        a[0][1]= c3*s1*s2-c1*s3;
        a[0][2]= c2*s1;
        a[1][0]= c2*s3;
        a[1][1]= c2*c3;
        a[1][2]= -s2;
        a[2][0]= c1*s2*s3-c3*s1;
        a[2][1]= c1*c3*s2+s1*s3;
        a[2][2]= c1*c2;
        }
	break;

case Euler_case('y','z','x'):
        {
        a[0][0]= c1*c2;
        a[0][1]= s1*s3-c1*c3*s2;
        a[0][2]= c3*s1+c1*s2*s3;
        a[1][0]= s2;
        a[1][1]= c2*c3;
        a[1][2]= -c2*s3;
        a[2][0]= -c2*s1;
        a[2][1]= c1*s3+c3*s1*s2;
        a[2][2]= c1*c3-s1*s2*s3;
        }
	break;

case Euler_case('z','y','x'):
        {
        a[0][0]= c1*c2;
        a[0][1]= c1*s2*s3-c3*s1;
        a[0][2]= s1*s2+c1*c3*s2;
        a[1][0]= c2*s1;
        a[1][1]= c1*c3+s1*s2*s3;
        a[1][2]= c3*s1*s2-c1*s3;
        a[2][0]= -s2;
        a[2][1]= c2*s3;
        a[2][2]= c2*c3;
        }
	break;

case Euler_case('z','x','y'):
        {
        a[0][0]= c1*c3-s1*s2*s3;
        a[0][1]= -c2*s1;
        a[0][2]= c1*s3+c3*s1*s2;
        a[1][0]= c3*s1+c1*s2*s3;
        a[1][1]= c1*c2;
        a[1][2]= s1*s3-c1*c3*s2;
        a[2][0]= -c2*s3;
        a[2][1]= s2;
        a[2][2]= c2*c3;
        }
	break;
}//switch

if(transpose) a.transposeme();
}


template<typename T>
void rotmat2euler(T *eul, const Mat3<T> &a, const char *type, bool transpose, bool direction, bool reverse)
{
T m11=a[0][0];
T m22=a[1][1];
T m33=a[2][2];
T m12=transpose?a[1][0]:a[0][1];
T m21=transpose?a[0][1]:a[1][0];
T m13=transpose?a[2][0]:a[0][2];
T m31=transpose?a[0][2]:a[2][0];
T m23=transpose?a[2][1]:a[1][2];
T m32=transpose?a[1][2]:a[2][1];

switch(Euler_case(type[0],type[1],type[2]))
{

case Euler_case('x','z','x'):
	{
        eul[0]=atan2(m31,m21);
        eul[1]=atan2(sqrt(1-m11*m11),m11);
        eul[2]=atan2(m13,-m12);
	}
	break;

case Euler_case('x','y','x'):
	{
	eul[0]=atan2(m21,-m31);
	eul[1]=atan2(sqrt(1-m11*m11),m11);
	eul[2]=atan2(m12,m13);
	}
	break;

case Euler_case('y','x','y'):
        {
        eul[0]=atan2(m12,m32);
        eul[1]=atan2(sqrt(1-m22*m22),m22);
        eul[2]=atan2(m21,-m23);
        }
	break;

case Euler_case('y','z','y'):
        {
        eul[0]=atan2(m32,-m12);
        eul[1]=atan2(sqrt(1-m22*m22),m22);
        eul[2]=atan2(m23,m21);
        }
	break;

case Euler_case('z','y','z'):
        {
        eul[0]=atan2(m23,m13);
        eul[1]=atan2(sqrt(1-m33*m33),m33);
        eul[2]=atan2(m32,-m31);
        }
	break;

case Euler_case('z','x','z'):
        {
        eul[0]=atan2(m13,-m23);
        eul[1]=atan2(sqrt(1-m33*m33),m33);
        eul[2]=atan2(m31,m32);
        }
	break;

case Euler_case('x','z','y'):
        {
	eul[0]=atan2(m32,m22);
	eul[1]=atan2(-m12,sqrt(1-m12*m12));
	eul[2]=atan2(m13,m11);
        }
	break;

case Euler_case('x','y','z'):
        {
	eul[0]=atan2(-m23,m33);
	eul[1]=atan2(m13,sqrt(1-m13*m13));
	eul[2]=atan2(-m12,m11);
        }
	break;

case Euler_case('y','x','z'):
        {
        eul[0]=atan2(m31,m33);
        eul[1]=atan2(-m23,sqrt(1-m23*m23));
        eul[2]=atan2(m21,m22);
        }
	break;

case Euler_case('y','z','x'):
        {
        eul[0]=atan2(-m31,m11);
        eul[1]=atan2(m21,sqrt(1-m21*m21));
        eul[2]=atan2(-m23,m22);
        }
	break;

case Euler_case('z','y','x'):
        {
        eul[0]=atan2(m21,m11);
        eul[1]=atan2(-m31,sqrt(1-m31*m31));
        eul[2]=atan2(m32,m33);
        }
	break;

case Euler_case('z','x','y'):
        {
        eul[0]=atan2(-m12,m22);
        eul[1]=atan2(m32,sqrt(1-m32*m32));
        eul[2]=atan2(-m31,m33);
        }
	break;

}//switch

if(reverse)
	{
	T t=eul[0]; eul[0]=eul[2]; eul[2]=t;
	}
if(direction) 
	{
	eul[0] *= (T)-1;
	eul[1] *= (T)-1;
	eul[2] *= (T)-1;
	}
}


//stream I/O
#ifndef AVOID_STDSTREAM
template <typename T>
std::istream& operator>>(std::istream  &s, Vec3<T> &x)
{
s >> x.q[0];
s >> x.q[1];
s >> x.q[2];
return s;
}

template <typename T>
std::ostream& operator<<(std::ostream &s, const Vec3<T> &x) {
s << x.q[0]<<" ";
s << x.q[1]<<" ";
s << x.q[2];
return s;
}

template <typename T>
std::istream& operator>>(std::istream  &s, Mat3<T> &x)
{
s >> x.q[0][0];
s >> x.q[0][1];
s >> x.q[0][2];
s >> x.q[1][0];
s >> x.q[1][1];
s >> x.q[1][2];
s >> x.q[2][0];
s >> x.q[2][1];
s >> x.q[2][2];
return s;
}

template <typename T>
std::ostream& operator<<(std::ostream &s, const Mat3<T> &x) {
s << x.q[0][0]<<" "<< x.q[0][1]<<" " << x.q[0][2]<<std::endl;
s << x.q[1][0]<<" "<< x.q[1][1]<<" " << x.q[1][2]<<std::endl;
s << x.q[2][0]<<" "<< x.q[2][1]<<" " << x.q[2][2]<<std::endl;
return s;
}
#endif


template <typename T>
void Mat3<T>::symmetrize()
{
T tmp=(q[0][1]+q[1][0])/2;
q[0][1]=q[1][0]=tmp;
tmp=(q[0][2]+q[2][0])/2;
q[0][2]=q[2][0]=tmp;
tmp=(q[2][1]+q[1][2])/2;
q[2][1]=q[1][2]=tmp;
}


#ifndef QUAT_NO_RANDOM
template <>
void Vec3<double>::randomize(const double x)
{
for(int i=0;i<3;++i)  q[i]=x*RANDDOUBLESIGNED();
}


template <>
void Mat3<double>::randomize(const double x, const bool symmetric)
{
if(symmetric)  for(int i=0;i<3;++i) for(int j=0; j<=i; ++j) q[j][i]=q[i][j]=x*RANDDOUBLESIGNED();
else for(int i=0;i<3;++i) for(int j=0; j<3; ++j) q[i][j]=x*RANDDOUBLESIGNED();
}

#endif



#if 0
//the Kopp's method is numerically unstable if the matrix is already close to diagonal
//for example repeated diagonalization of inertia tensor and performing a body rotation
//yields nonsense on the second run
//I switched to Jacobi
//
//eigensolver for 3x3 matrix by Joachim Kopp - analytic formula version,
//might be unstable for ill-conditioned ones, then use other methods
//cf. arxiv physics 0610206v3
//
//// Numerical diagonalization of 3x3 matrcies
//// Copyright (C) 2006  Joachim Kopp
//// ----------------------------------------------------------------------------
//// This library is free software; you can redistribute it and/or
//// modify it under the terms of the GNU Lesser General Public
//// License as published by the Free Software Foundation; either
//// version 2.1 of the License, or (at your option) any later version.
////
//// This library is distributed in the hope that it will be useful,
//// but WITHOUT ANY WARRANTY; without even the implied warranty of
//// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//// Lesser General Public License for more details.
////
//// You should have received a copy of the GNU Lesser General Public
//// License along with this library; if not, write to the Free Software
//// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
//// ----------------------------------------------------------------------------
//

//numeric_limits not available on some crosscompilers for small MCUs
#ifdef QUAT_NO_DOUBLE
#define NO_NUMERIC_LIMITS
#endif

#define M_SQRT3    1.73205080756887729352744634151   // sqrt(3)
#define SQR(x)      ((x)*(x))                        // x^2
//
template <typename T>
static void eival_sym(const  Mat3<T> &q, Vec3<T> &w, const bool sortdown) 
{
T  m, c1, c0;
  
  // Determine coefficients of characteristic poynomial. We write
  //       | a   d   f  |
  //  A =  | d*  b   e  |
  //       | f*  e*  c  |
  T de = q[0][1] * q[1][2];                                    // d * e
  T dd = SQR(q[0][1]);                                         // d^2
  T ee = SQR(q[1][2]);                                         // e^2
  T ff = SQR(q[0][2]);                                         // f^2
  m  = q[0][0] + q[1][1] + q[2][2];
  c1 = (q[0][0]*q[1][1] + q[0][0]*q[2][2] + q[1][1]*q[2][2])        // a*b + a*c + b*c - d^2 - e^2 - f^2
          - (dd + ee + ff);
  c0 = q[2][2]*dd + q[0][0]*ee + q[1][1]*ff - q[0][0]*q[1][1]*q[2][2]
            - 2.0 * q[0][2]*de;                                     // c*d^2 + a*e^2 + b*f^2 - a*b*c - 2*f*d*e)

  T p, sqrt_p, Q, c, s, phi;
  p = SQR(m) - 3.0*c1;
  Q = m*(p - (3.0/2.0)*c1) - (27.0/2.0)*c0;
  sqrt_p = sqrt(abs(p));

  phi = 27.0 * ( 0.25*SQR(c1)*(p - c1) + c0*(Q + 27.0/4.0*c0));
  phi = (1.0/3.0) * atan2(sqrt(abs(phi)), Q);
  
  c = sqrt_p*cos(phi);
  s = (1.0/M_SQRT3)*sqrt_p*sin(phi);

  w[1]  = (1.0/3.0)*(m - c);
  w[2]  = w[1] + s;
  w[0]  = w[1] + c;
  w[1] -= s;

if(sortdown)
	{
        if(w[0]<w[1]) {T tmp=w[0]; w[0]=w[1]; w[1]=tmp;}
        if(w[0]<w[2]) {T tmp=w[0]; w[0]=w[2]; w[2]=tmp;}
        if(w[1]<w[2]) {T tmp=w[1]; w[1]=w[2]; w[2]=tmp;}
	}
else
//sort in ascending order
	{
	if(w[0]>w[1]) {T tmp=w[0]; w[0]=w[1]; w[1]=tmp;}
	if(w[0]>w[2]) {T tmp=w[0]; w[0]=w[2]; w[2]=tmp;}
	if(w[1]>w[2]) {T tmp=w[1]; w[1]=w[2]; w[2]=tmp;}
	}
}

// Calculates the eigenvalues and normalized eigenvectors of a symmetric 3x3
// matrix A using Cardano's method for the eigenvalues and an analytical
// method based on vector cross products for the eigenvectors.
// Only the diagonal and upper triangular parts of A need to contain meaningful
// values. However, all of A may be used as temporary storage and may hence be
// destroyed.
// ----------------------------------------------------------------------------
// Parameters:
//   A: The symmetric input matrix
//   Q: Storage buffer for eigenvectors
//   w: Storage buffer for eigenvalues
// ----------------------------------------------------------------------------
// Return value:
//   0: Success
//  -1: Error
// ----------------------------------------------------------------------------
// Dependencies:
//   dsyevc3()
// ----------------------------------------------------------------------------
// Version history:
//   v1.1 (12 Mar 2012): Removed access to lower triangualr part of A
//     (according to the documentation, only the upper triangular part needs
//     to be filled)
//   v1.0: First released version
// ----------------------------------------------------------------------------
template <typename T>
bool Mat3<T>::eivec_sym(Vec3<T> &w, Mat3 &v, const bool sortdown) const
{
  T norm;          // Squared norm or inverse norm of current eigenvector
  T n0, n1;        // Norm of first and second columns of A
  T n0tmp, n1tmp;  // "Templates" for the calculation of n0/n1 - saves a few FLOPS
  T thresh;        // Small number used as threshold for floating point comparisons
  T error;         // Estimated maximum roundoff error in some steps
  T wmax;          // The eigenvalue of maximum modulus
  T f, t;          // Intermediate storage
  int i, j;             // Loop counters

  // Calculate eigenvalues
  eival_sym(*this, w,sortdown);
Mat3<T> A(*this); //scratch copy

  wmax = abs(w[0]);
  if ((t=abs(w[1])) > wmax)
    wmax = t;
  if ((t=abs(w[2])) > wmax)
    wmax = t;
  thresh = SQR(8.0 * DBL_EPSILON  * wmax);

  // Prepare calculation of eigenvectors
  n0tmp   = SQR(A[0][1]) + SQR(A[0][2]);
  n1tmp   = SQR(A[0][1]) + SQR(A[1][2]);
  v[0][1] = A[0][1]*A[1][2] - A[0][2]*A[1][1];
  v[1][1] = A[0][2]*A[0][1] - A[1][2]*A[0][0];
  v[2][1] = SQR(A[0][1]);

  // Calculate first eigenvector by the formula
  //   v[0] = (A - w[0]).e1 x (A - w[0]).e2
  A[0][0] -= w[0];
  A[1][1] -= w[0];
  v[0][0] = v[0][1] + A[0][2]*w[0];
  v[1][0] = v[1][1] + A[1][2]*w[0];
  v[2][0] = A[0][0]*A[1][1] - v[2][1];
  norm    = SQR(v[0][0]) + SQR(v[1][0]) + SQR(v[2][0]);
  n0      = n0tmp + SQR(A[0][0]);
  n1      = n1tmp + SQR(A[1][1]);
  error   = n0 * n1;
  
  if (n0 <= thresh)         // If the first column is zero, then (1,0,0) is an eigenvector
  {
    v[0][0] = 1.0;
    v[1][0] = 0.0;
    v[2][0] = 0.0;
  }
  else if (n1 <= thresh)    // If the second column is zero, then (0,1,0) is an eigenvector
  {
    v[0][0] = 0.0;
    v[1][0] = 1.0;
    v[2][0] = 0.0;
  }
  else if (norm < SQR(64.0 * DBL_EPSILON) * error)
  {                         // If angle between A[0] and A[1] is too small, don't use
    t = SQR(A[0][1]);       // cross product, but calculate v ~ (1, -A0/A1, 0)
    f = -A[0][0] / A[0][1];
    if (SQR(A[1][1]) > t)
    {
      t = SQR(A[1][1]);
      f = -A[0][1] / A[1][1];
    }
    if (SQR(A[1][2]) > t)
      f = -A[0][2] / A[1][2];
    norm    = 1.0/sqrt(1 + SQR(f));
    v[0][0] = norm;
    v[1][0] = f * norm;
    v[2][0] = 0.0;
  }
  else                      // This is the standard branch
  {
    norm = sqrt(1.0 / norm);
    for (j=0; j < 3; j++)
      v[j][0] = v[j][0] * norm;
  }

  
  // Prepare calculation of second eigenvector
  t = w[0] - w[1];
  if (abs(t) > 8.0 * DBL_EPSILON * wmax)
  {
    // For non-degenerate eigenvalue, calculate second eigenvector by the formula
    //   v[1] = (A - w[1]).e1 x (A - w[1]).e2
    A[0][0] += t;
    A[1][1] += t;
    v[0][1]  = v[0][1] + A[0][2]*w[1];
    v[1][1]  = v[1][1] + A[1][2]*w[1];
    v[2][1]  = A[0][0]*A[1][1] - v[2][1];
    norm     = SQR(v[0][1]) + SQR(v[1][1]) + SQR(v[2][1]);
    n0       = n0tmp + SQR(A[0][0]);
    n1       = n1tmp + SQR(A[1][1]);
    error    = n0 * n1;
 
    if (n0 <= thresh)       // If the first column is zero, then (1,0,0) is an eigenvector
    {
      v[0][1] = 1.0;
      v[1][1] = 0.0;
      v[2][1] = 0.0;
    }
    else if (n1 <= thresh)  // If the second column is zero, then (0,1,0) is an eigenvector
    {
      v[0][1] = 0.0;
      v[1][1] = 1.0;
      v[2][1] = 0.0;
    }
    else if (norm < SQR(64.0 * DBL_EPSILON) * error)
    {                       // If angle between A[0] and A[1] is too small, don't use
      t = SQR(A[0][1]);     // cross product, but calculate v ~ (1, -A0/A1, 0)
      f = -A[0][0] / A[0][1];
      if (SQR(A[1][1]) > t)
      {
        t = SQR(A[1][1]);
        f = -A[0][1] / A[1][1];
      }
      if (SQR(A[1][2]) > t)
        f = -A[0][2] / A[1][2];
      norm    = 1.0/sqrt(1 + SQR(f));
      v[0][1] = norm;
      v[1][1] = f * norm;
      v[2][1] = 0.0;
    }
    else
    {
      norm = sqrt(1.0 / norm);
      for (j=0; j < 3; j++)
        v[j][1] = v[j][1] * norm;
    }
  }
  else
  {
    // For degenerate eigenvalue, calculate second eigenvector according to
    //   v[1] = v[0] x (A - w[1]).e[i]
    //   
    // This would really get to complicated if we could not assume all of A to
    // contain meaningful values.
    A[1][0]  = A[0][1];
    A[2][0]  = A[0][2];
    A[2][1]  = A[1][2];
    A[0][0] += w[0];
    A[1][1] += w[0];
    for (i=0; i < 3; i++)
    {
      A[i][i] -= w[1];
      n0       = SQR(A[0][i]) + SQR(A[1][i]) + SQR(A[2][i]);
      if (n0 > thresh)
      {
        v[0][1]  = v[1][0]*A[2][i] - v[2][0]*A[1][i];
        v[1][1]  = v[2][0]*A[0][i] - v[0][0]*A[2][i];
        v[2][1]  = v[0][0]*A[1][i] - v[1][0]*A[0][i];
        norm     = SQR(v[0][1]) + SQR(v[1][1]) + SQR(v[2][1]);
        if (norm > SQR(256.0 * DBL_EPSILON) * n0) // Accept cross product only if the angle between
        {                                         // the two vectors was not too small
          norm = sqrt(1.0 / norm);
          for (j=0; j < 3; j++)
            v[j][1] = v[j][1] * norm;
          break;
        }
      }
    }
    
    if (i == 3)    // This means that any vector orthogonal to v[0] is an EV.
    {
      for (j=0; j < 3; j++)
        if (v[j][0] != 0.0)                                   // Find nonzero element of v[0] ...
        {                                                     // ... and swap it with the next one
          norm          = 1.0 / sqrt(SQR(v[j][0]) + SQR(v[(j+1)%3][0]));
          v[j][1]       = v[(j+1)%3][0] * norm;
          v[(j+1)%3][1] = -v[j][0] * norm;
          v[(j+2)%3][1] = 0.0;
          break;
        }
    }
  }
      
  
  // Calculate third eigenvector according to
  //   v[2] = v[0] x v[1]
  v[0][2] = v[1][0]*v[2][1] - v[2][0]*v[1][1];
  v[1][2] = v[2][0]*v[0][1] - v[0][0]*v[2][1];
  v[2][2] = v[0][0]*v[1][1] - v[1][0]*v[0][1];

return 0;
}

#undef SQR
//end Kopp's eigensolver for 3x3 matrix
/////////////////////////////////////////////////////////////////////////////////////////////
#else


//begin Jacobi's  eigensolver for 3x3 matrix

#define ROTATE(a,i,j,k,l) g=a[i][j];h=a[k][l];a[i][j]=g-s*(h+g*tau); a[k][l]=h+s*(g-h*tau);
#define SWAP(i,j) {T tmp; tmp=d[i]; d[i]=d[j]; d[j]=tmp; for(int ii=0; ii<n; ++ii) {tmp=v[ii][i]; v[ii][i]=v[ii][j]; v[ii][j]=tmp;}}

template <typename T>
bool Mat3<T>::eivec_sym(Vec3<T> &d, Mat3 &v, const bool sortdown) const
{
Mat3<T> a(*this);//will be overwritten but we are const
const int n=3;
int j,iq,ip,i;
T tresh,theta,tau,t,sm,s,h,g,c;

Vec3<T> b,z;
	for (ip=0;ip<n;ip++) {
		for (iq=0;iq<n;iq++) v[ip][iq]=0.0;
		v[ip][ip]=1.0;
	}
	for (ip=0;ip<n;ip++) {
		b[ip]=d[ip]=a[ip][ip];
		z[ip]=0.0;
	}
	for (i=1;i<=50;i++) {
		sm=0.0;
		for (ip=0;ip<n-1;ip++) {
			for (iq=ip+1;iq<n;iq++)
				sm += abs(a[ip][iq]);
		}
		if (sm == 0.0)  
			{
			if(sortdown)
				{
			        if(d[0]<d[1]) SWAP(0,1)
			        if(d[0]<d[2]) SWAP(0,2)
			        if(d[1]<d[2]) SWAP(1,2)
				}
			else
				{
				if(d[0]>d[1]) SWAP(0,1)
				if(d[0]>d[2]) SWAP(0,2)
				if(d[1]>d[2]) SWAP(1,2)
				}
			return 0; 
			}
		if (i < 4)
			tresh=0.2*sm/(n*n);
		else
			tresh=0.0;
		for (ip=0;ip<n-1;ip++) {
			for (iq=ip+1;iq<n;iq++) {
				g=100.0*abs(a[ip][iq]);
				if (i > 4 && abs(d[ip])+g == abs(d[ip])
					&& abs(d[iq])+g == abs(d[iq]))
					a[ip][iq]=0.0;
				else if (abs(a[ip][iq]) > tresh) {
					h=d[iq]-d[ip];
					if (abs(h)+g == abs(h))
						t=(a[ip][iq])/h;
					else {
						theta=0.5*h/(a[ip][iq]);
						t=1.0/(abs(theta)+sqrt(1.0+theta*theta));
						if (theta < 0.0) t = -t;
					}
					c=1.0/sqrt(1+t*t);
					s=t*c;
					tau=s/(1.0+c);
					h=t*a[ip][iq];
					z[ip] -= h;
					z[iq] += h;
					d[ip] -= h;
					d[iq] += h;
					a[ip][iq]=0.0;
					for (j=0;j<=ip-1;j++) {
						ROTATE(a,j,ip,j,iq)
					}
					for (j=ip+1;j<=iq-1;j++) {
						ROTATE(a,ip,j,j,iq)
					}
					for (j=iq+1;j<n;j++) {
						ROTATE(a,ip,j,iq,j)
					}
					for (j=0;j<n;j++) {
						ROTATE(v,j,ip,j,iq)
					}
				}
			}
		}
		for (ip=0;ip<n;ip++) {
			b[ip] += z[ip];
			d[ip]=b[ip];
			z[ip]=0.0;
		}
	}
return 1;
}

#undef ROTATE
#undef SWAP

#endif
//end Jacobi's eigensolver for 3x3 matrix
/////////////////////////////////////////////////////////////////////////////////////////////






template<typename T>
T Mat3<T>::norm(const T scalar) const
{
        T sum(0);
        for(int i=0; i<3; i++)
                for(int j=0; j<3; j++) {
                        T tmp = q[i][j];
                        if(i == j) tmp -= scalar;
                        sum += tmp*tmp;
                }
return sqrt(sum);
}

//efficient SVD for 3x3 matrices cf. Mc.Adams et al. UWM technical report 1690 (2011)
//


template<typename T> 
static inline T ABS(const T a)
{
return a<0?-a:a;
}

template<typename T> 
static inline T MAX(const T a, const T b)
{
return a>b ? a : b;
}



template<typename T>
void QRGivensQuaternion(T a1, T a2, T &ch, T &sh)
{
    // a1 = pivot point on diagonal
    // a2 = lower triangular entry we want to annihilate
    T rho = sqrt(a1*a1 + a2*a2);

    sh = rho > DBL_EPSILON ? a2 : 0;
    ch = ABS(a1) + MAX(rho,DBL_EPSILON);
    if(a1 < 0)
	{
	T tmp=sh;
	sh=ch;
	ch=tmp;
	}
    T w = (T)1/sqrt(ch*ch+sh*sh);
    ch *= w;
    sh *= w;
}


template<typename T>
void Mat3<T>::qrd(Mat3 &Q, Mat3 &R)
{

T ch1,sh1,ch2,sh2,ch3,sh3;
T a,b;
    
// first givens rotation (ch,0,0,sh)
QRGivensQuaternion(q[0][0],q[1][0],ch1,sh1);
a=1-2*sh1*sh1;
b=2*ch1*sh1;
// apply B = Q' * B
R[0][0]= a*q[0][0]+b*q[1][0];  R[0][1]= a*q[0][1]+b*q[1][1];  R[0][2]= a*q[0][2]+b*q[1][2];
R[1][0]= -b*q[0][0]+a*q[1][0]; R[1][1]= -b*q[0][1]+a*q[1][1]; R[1][2]= -b*q[0][2]+a*q[1][2];
R[2][0]= q[2][0];          R[2][1]= q[2][1];          R[2][2]= q[2][2];
  
// second givens rotation (ch,0,-sh,0)
QRGivensQuaternion(R[0][0],R[2][0],ch2,sh2);
a= 1-2*sh2*sh2;
b= 2*ch2*sh2;
// apply B = Q' * B;
q[0][0]= a*R[0][0]+b*R[2][0];  q[0][1]= a*R[0][1]+b*R[2][1];  q[0][2]= a*R[0][2]+b*R[2][2];
q[1][0]= R[1][0];           q[1][1]= R[1][1];           q[1][2]= R[1][2];
q[2][0]= -b*R[0][0]+a*R[2][0]; q[2][1]= -b*R[0][1]+a*R[2][1]; q[2][2]= -b*R[0][2]+a*R[2][2];

// third givens rotation (ch,sh,0,0)
QRGivensQuaternion(q[1][1],q[2][1],ch3,sh3);
a= 1-2*sh3*sh3;
b= 2*ch3*sh3;
// R is now set to desired value
R[0][0]= q[0][0];             R[0][1]= q[0][1];           R[0][2]= q[0][2];
R[1][0]= a*q[1][0]+b*q[2][0];     R[1][1]= a*q[1][1]+b*q[2][1];   R[1][2]= a*q[1][2]+b*q[2][2];
R[2][0]= -b*q[1][0]+a*q[2][0];    R[2][1]= -b*q[1][1]+a*q[2][1];  R[2][2]= -b*q[1][2]+a*q[2][2];

// construct the cumulative rotation Q=Q1 * Q2 * Q3
// the number of floating point operations for three quaternion multiplications
// is more or less comparable to the explicit form of the joined matrix.
// certainly more memory-efficient!
T sh12= sh1*sh1;
T sh22= sh2*sh2;
T sh32= sh3*sh3;

Q[0][0]= (-1+2*sh12)*(-1+2*sh22); 
Q[0][1]= 4*ch2*ch3*(-1+2*sh12)*sh2*sh3+2*ch1*sh1*(-1+2*sh32); 
Q[0][2]= 4*ch1*ch3*sh1*sh3-2*ch2*(-1+2*sh12)*sh2*(-1+2*sh32);

Q[1][0]= 2*ch1*sh1*(1-2*sh22); 
Q[1][1]= -8*ch1*ch2*ch3*sh1*sh2*sh3+(-1+2*sh12)*(-1+2*sh32); 
Q[1][2]= -2*ch3*sh3+4*sh1*(ch3*sh1*sh3+ch1*ch2*sh2*(-1+2*sh32));
    
Q[2][0]= 2*ch2*sh2; 
Q[2][1]= 2*ch3*(1-2*sh22)*sh3; 
Q[2][2]= (-1+2*sh22)*(-1+2*sh32);
}


template<typename T>
void Mat3<T>::svd(Mat3 &u, Vec3<T> &w, Mat3 &v, bool proper_rotations) const
{
Mat3 ata= this->Ttimes(*this);
Vec3<T> ww; //actually not needed squares of singular values
ata.eivec_sym(ww,v,true);
Mat3 uw = (*this)*v;
Mat3 r;
uw.qrd(u,r);
w[0]= r[0][0];
w[1]= r[1][1];
w[2]= r[2][2];

if(proper_rotations) return;
if(w[2]<0)
	{
	w[2] = -w[2];
	u[0][2] = -u[0][2];
	u[1][2] = -u[1][2];
	u[2][2] = -u[2][2];
	}
}


template<typename T>
void  Mat3<T>::diagmultl(const Vec3<T> &rhs)
{
for(int i=0; i<3; ++i) for(int j=0; j<3; ++j) q[i][j] *= rhs[i];
}

template<typename T>
void  Mat3<T>::diagmultr(const Vec3<T> &rhs)
{
for(int i=0; i<3; ++i) for(int j=0; j<3; ++j) q[i][j] *= rhs[j];
}


template<typename T>
const Mat3<T>   Mat3<T>::svdinverse(const T thr) const
{
Mat3 u,v;
Vec3<T> w;
svd(u,w,v);
for(int i=0; i<3; ++i) w[i] = (w[i]<thr)? 0 : 1/w[i];
v.diagmultr(w);
return v.timesT(u);
}


//cf. https://en.wikipedia.org/wiki/3D_projection
template<typename T>
void perspective(T *proj_xy, const Vec3<T> &point, const Mat3<T> &rot_angle, const Vec3<T> &camera,  const Vec3<T> &plane_to_camera)
{
Vec3<T> d=rot_angle*point-camera;
T scale = plane_to_camera[2]/d[2];
for(int i=0; i<2; ++i) proj_xy[i]= scale*d[i] + plane_to_camera[i];
}


//force instantization
#define INSTANTIZE(T) \
template class Vec3<T>; \
template class Mat3<T>; \
template void euler2rotmat(const T *eul, Mat3<T> &a, const char *type, bool transpose=0, bool direction=0, bool reverse=0); \
template void rotmat2euler(T *eul, const Mat3<T> &a, const char *type, bool transpose=0, bool direction=0, bool reverse=0); \
template void  perspective(T *proj_xy, const Vec3<T> &point, const Mat3<T> &rot_angle, const Vec3<T> &camera,  const Vec3<T> &plane_to_camera); \


#ifndef AVOID_STDSTREAM
#define INSTANTIZE2(T) \
template std::istream& operator>>(std::istream  &s, Vec3<T> &x); \
template std::ostream& operator<<(std::ostream &s, const Vec3<T> &x); \
template std::istream& operator>>(std::istream  &s, Mat3<T> &x); \
template std::ostream& operator<<(std::ostream &s, const Mat3<T> &x); \

#endif



INSTANTIZE(float)
#ifndef QUAT_NO_DOUBLE
INSTANTIZE(double)
#endif

#ifndef AVOID_STDSTREAM
INSTANTIZE2(float)
#ifndef QUAT_NO_DOUBLE
INSTANTIZE2(double)
#endif
#endif

}//namespace
