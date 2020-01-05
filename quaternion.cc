/*
    LA: linear algebra C++ interface library
    Copyright (C) 2008-2020 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>

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

#include "quaternion.h"

using namespace LA_Quaternion;


//do not replicate this code in each object file, therefore not in .h
//and instantize the templates for the types needed


template<typename T>
Quaternion<T> Quaternion<T>::operator*(const Quaternion<T> &rhs) const 
{
return Quaternion<T>
	(
	q[0]*rhs.q[0]-q[1]*rhs.q[1]-q[2]*rhs.q[2]-q[3]*rhs.q[3],
	q[0]*rhs.q[1]+q[1]*rhs.q[0]+q[2]*rhs.q[3]-q[3]*rhs.q[2],
	q[0]*rhs.q[2]+q[2]*rhs.q[0]+q[3]*rhs.q[1]-q[1]*rhs.q[3],
	q[0]*rhs.q[3]+q[3]*rhs.q[0]+q[1]*rhs.q[2]-q[2]*rhs.q[1]
	);
};	

template<typename T>
Quaternion<T> Quaternion<T>::times_vec3(const T *rhs) const
{
return Quaternion<T>
        (
       -q[1]*rhs[0]-q[2]*rhs[1]-q[3]*rhs[2],
        q[0]*rhs[0]+q[2]*rhs[2]-q[3]*rhs[1],
        q[0]*rhs[1]+q[3]*rhs[0]-q[1]*rhs[2],
        q[0]*rhs[2]+q[1]*rhs[1]-q[2]*rhs[0]
        );
};

template<typename T>
Quaternion<T> Quaternion<T>::vec3_times(const T *lhs) const
{
return Quaternion<T>
        (
       -lhs[0]*q[1]-lhs[1]*q[2]-lhs[2]*q[3],
        lhs[0]*q[0]+lhs[1]*q[3]-lhs[2]*q[2],
        lhs[1]*q[0]+lhs[2]*q[1]-lhs[0]*q[3],
        lhs[2]*q[0]+lhs[0]*q[2]-lhs[1]*q[1]
        );
};




//basically the same code as in normquat2rotmat, but avoiding extra storage
template<typename T>
void Quaternion<T>::rotate(T *to, const T *from, Quaternion<T> *grad) const 
{
//some subexpression eliminations
{
T q00= q[0]*q[0];
T q11= q[1]*q[1];
T q22= q[2]*q[2];
T q33= q[3]*q[3];

to[0] = (q00+q11-q22-q33) * from[0];
to[1] = (q00+q22-q11-q33) * from[1];
to[2] = (q00+q33-q11-q22) * from[2];
}

T q01= q[0]*q[1];
T q02= q[0]*q[2];
T q03= q[0]*q[3];
T q12= q[1]*q[2];
T q13= q[1]*q[3];
T q23= q[2]*q[3];

T f0=from[0]+from[0];
T f1=from[1]+from[1];
T f2=from[2]+from[2];

to[0] += (q12+q03)*f1;
to[0] += (q13-q02)*f2;
to[1] += (q12-q03)*f0;
to[1] += (q23+q01)*f2;
to[2] += (q13+q02)*f0;
to[2] += (q23-q01)*f1;

/*
to[0] = (2*q[0]*q[0]-1+2*q[1]*q[1]) * from[0] +
         2*(q[1]*q[2]+q[0]*q[3]) *    from[1] +
         2*(q[1]*q[3]-q[0]*q[2]) *    from[2];
to[1] =  2*(q[1]*q[2]-q[0]*q[3]) *    from[0] +
        (2*q[0]*q[0]-1+2*q[2]*q[2]) * from[1] +
         2*(q[2]*q[3]+q[0]*q[1]) *    from[2];
to[2] =  2*(q[1]*q[3]+q[0]*q[2])    * from[0] +
         2*(q[2]*q[3]-q[0]*q[1])    * from[1] +
        (2*q[0]*q[0]-1+2*q[3]*q[3]) * from[2];
*/

if(grad)
	{
	grad[0][0]=  q[0]*f0 + q[3]*f1 - q[2]*f2;
	grad[0][1]=  q[1]*f0 + q[2]*f1 + q[3]*f2;
	grad[0][2]= -q[2]*f0 + q[1]*f1 - q[0]*f2;
	grad[0][3]= -q[3]*f0 + q[0]*f1 + q[1]*f2;

        grad[1][0]= -q[3]*f0 + q[0]*f1 + q[1]*f2;
        grad[1][1]=  q[2]*f0 - q[1]*f1 + q[0]*f2;
        grad[1][2]=  q[1]*f0 + q[2]*f1 + q[3]*2;
        grad[1][3]= -q[0]*f0 - q[3]*f1 + q[2]*f2;

        grad[2][0]=  q[2]*f0 - q[1]*f1 + q[0]*f2;
        grad[2][1]=  q[3]*f0 - q[0]*f1 - q[1]*f2;
        grad[2][2]=  q[0]*f0 + q[3]*f1 - q[2]*f2;
        grad[2][3]=  q[1]*f0 + q[2]*f1 + q[3]*f2;

	}
}


template<typename T>
Quaternion<T> Quaternion<T>::rotateby(const Quaternion<T> &rhs)
{
//return rhs.inverse() * *this * rhs; //inefficient reference implementation
Quaternion<T> r;
r[0]=0;
rhs.rotate(&r[1],&q[1]);
return r;
}

template<typename T>
Quaternion<T> Quaternion<T>::rotate_match(T *to, const T *from, const T *match) const
{
Quaternion<T> grad[3];
Quaternion<T>::rotate(to, from, grad);
Quaternion<T> derivative;
derivative =  grad[0] * (to[0]-match[0]);
derivative += grad[1] * (to[1]-match[1]);
derivative += grad[2] * (to[2]-match[2]);
return derivative;
}


//optionally skip this for microcontrollers if not needed
//note that C++ standard headers should use float version of the functions for T=float
template<typename T>
void Quaternion<T>::normquat2euler(T *e) const
{
e[0]= atan2(2*q[1]*q[2]-2*q[0]*q[3],2*q[0]*q[0]+2*q[1]*q[1]-1);
e[1]= -asin(2*q[1]*q[3]+2*q[0]*q[2]);
e[2]= atan2(2*q[2]*q[3]-2*q[0]*q[1],2*q[0]*q[0]+2*q[3]*q[3]-1);
}

template<typename T>
void Quaternion<T>::axis2normquat(const T *axis, const T &angle)
{
T a = (T).5*angle;
q[0]=cos(a);
T s=sin(a);
q[1]=axis[0]*s;
q[2]=axis[1]*s;
q[3]=axis[2]*s;
}

template<typename T>
void Quaternion<T>::normquat2axis(T *axis,  T &angle) const
{
T s = sqrt(q[1]*q[1] + q[2]*q[2] +q[3]*q[3]);
angle = 2*atan2(s,q[0]);
s= 1/s;
axis[0]= q[1]*s;
axis[1]= q[2]*s;
axis[2]= q[3]*s;
}



//force instantization
#define INSTANTIZE(T) \
template class Quaternion<T>; \


INSTANTIZE(float)
#ifndef QUAT_NO_DOUBLE
INSTANTIZE(double)
#endif


