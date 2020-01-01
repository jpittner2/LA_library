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


//do not replicate this code in each object file, therefore not in .h
//and instantize the templates for the types needed


template<typename T>
const Quaternion<T> Quaternion<T>::operator*(const Quaternion<T> &rhs) const 
{
return Quaternion<T>
	(
	this->q[0]*rhs.q[0]-this->q[1]*rhs.q[1]-this->q[2]*rhs.q[2]-this->q[3]*rhs.q[3],
	this->q[0]*rhs.q[1]+this->q[1]*rhs.q[0]+this->q[2]*rhs.q[3]-this->q[3]*rhs.q[2],
	this->q[0]*rhs.q[2]+this->q[2]*rhs.q[0]+this->q[3]*rhs.q[1]-this->q[1]*rhs.q[3],
	this->q[0]*rhs.q[3]+this->q[3]*rhs.q[0]+this->q[1]*rhs.q[2]-this->q[2]*rhs.q[1]
	);
};	


//basically the same code as in normquat2rotmat, but avoiding extra storage
template<typename T>
void Quaternion<T>::rotate(T *to, const T *from) const 
{
to[0] = (2*q[0]*q[0]-1+2*q[1]*q[1]) * from[0] +
         2*(q[1]*q[2]+q[0]*q[3]) *    from[1] +
         2*(q[1]*q[3]-q[0]*q[2]) *    from[2];
to[1] =  2*(q[1]*q[2]-q[0]*q[3]) *    from[0] +
        (2*q[0]*q[0]-1+2*q[2]*q[2]) * from[1] +
         2*(q[2]*q[3]+q[0]*q[1]) *    from[2];
to[2] =  2*(q[1]*q[3]+q[0]*q[2])    * from[0] +
         2*(q[2]*q[3]-q[0]*q[1])    * from[1] +
        (2*q[0]*q[0]-1+2*q[3]*q[3]) * from[2];
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


//optionally skip this for microcontrollers if not needed
//note that C++ standard headers should use float version of the functions for T=float
#ifndef AVOID_GONIOM_FUNC
template<typename T>
void  normquat2euler(const Quaternion<T> &q, T *e)
{
e[0]= atan2(2*q[1]*q[2]-2*q[0]*q[3],2*q[0]*q[0]+2*q[1]*q[1]-1);
e[1]= -asin(2*q[1]*q[3]+2*q[0]*q[2]);
e[2]= atan2(2*q[2]*q[3]-2*q[0]*q[1],2*q[0]*q[0]+2*q[3]*q[3]-1);
}

template<typename T>
void axis2normquat(const T *axis, const T &angle, Quaternion<T> &q)
{
T a = (T).5*angle;
q[0]=cos(a);
T s=sin(a);
q[1]=axis[0]*s;
q[2]=axis[1]*s;
q[3]=axis[2]*s;
}

template<typename T>
void normquat2axis(const Quaternion<T> &q, T *axis,  T &angle)
{
T s = sqrt(q[1]*q[1] + q[2]*q[2] +q[3]*q[3]);
angle = 2*atan2(s,q[0]);
s= 1/s;
axis[0]= q[1]*s;
axis[1]= q[2]*s;
axis[2]= q[3]*s;
}
#endif



//force instantization
#define INSTANTIZE(T) \
template class Quaternion<T>; \

#define INSTANTIZE2(T) \
template void normquat2euler(const Quaternion<T> &q, T *e); \
template void axis2normquat(const T *axis, const T &angle, Quaternion<T> &q); \
template void normquat2axis(const Quaternion<T> &q, T *axis,  T &angle); \


INSTANTIZE(float)
#ifndef QUAT_NO_DOUBLE
INSTANTIZE(double)
#endif

#ifndef AVOID_GONIOM_FUNC
INSTANTIZE2(float)
#ifndef QUAT_NO_DOUBLE
INSTANTIZE2(double)
#endif
#endif

