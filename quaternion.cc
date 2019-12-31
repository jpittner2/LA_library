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


//optionally skip this for microcontrollers if not needed
//note that C++ standard headers should use float version of the functions for T=float
#ifndef AVOID_GONIOM_FUNC
template<typename T>
void  normquat2euler(const Quaternion<T> &q, T (&e)[3])
{
e[0]= atan2(2*q[1]*q[2]-2*q[0]*q[3],2*q[0]*q[0]+2*q[1]*q[1]-1);
e[1]= -asin(2*q[1]*q[3]+2*q[0]*q[2]);
e[2]= atan2(2*q[2]*q[3]-2*q[0]*q[1],2*q[0]*q[0]+2*q[3]*q[3]-1);
}
#endif



//force instantization
#define INSTANTIZE(T) \
template class Quaternion<T>; \
template void normquat2euler(const Quaternion<T> &q, T (&e)[3]); \

INSTANTIZE(float)
#ifndef QUAT_NO_DOUBLE
INSTANTIZE(double)
#endif
