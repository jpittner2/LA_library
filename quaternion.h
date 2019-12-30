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
#ifndef _QUATERNION_H_
#define _QUATERNION_H_

#include <stdlib.h>
#include <iostream>

template <typename T>
class Quaternion 
	{
public:
	//just plain old data
	T q[4];
	//
	Quaternion(void) {};
	Quaternion(const T x, const T u=0, const T v=0, const T w=0) {q[0]=x; q[1]=u; q[2]=v; q[3]=w;}; //quaternion from real

	//compiler generates default copy constructor and assignment operator
	
	//formal indexing
	const T operator[](const int i) const {return this->q[i];};
	T operator[](const int i) {return this->q[i];};

	//operations of quaternions with scalars
	Quaternion& operator=(const T x) {q[0]=x; memset(&q[1],0,3*sizeof(T)); return *this;};  //quaternion from real 
	Quaternion& operator+=(const T rhs) {this->q[0]+=rhs; return *this;};
	Quaternion& operator-=(const T rhs) {this->q[0]-=rhs; return *this;};
	Quaternion& operator*=(const T rhs) {this->q[0]*=rhs; this->q[1]*=rhs; this->q[2]*=rhs; this->q[3]*=rhs; return *this;};
	Quaternion& operator/=(const T rhs) {return *this *= ((T)1/rhs);};
	const Quaternion operator+(const T rhs) const {return Quaternion(*this) += rhs;};
        const Quaternion operator-(const T rhs) const {return Quaternion(*this) -= rhs;};
        const Quaternion operator*(const T rhs) const {return Quaternion(*this) *= rhs;};
        const Quaternion operator/(const T rhs) const {return Quaternion(*this) /= rhs;};

	//quaternion algebra
	Quaternion& operator+=(const Quaternion &rhs) {this->q[0]+=rhs.q[0];this->q[1]+=rhs.q[1];this->q[2]+=rhs.q[2];this->q[3]+=rhs.q[3]; return *this;};
	Quaternion& operator-=(const Quaternion &rhs) {this->q[0]-=rhs.q[0];this->q[1]-=rhs.q[1];this->q[2]-=rhs.q[2];this->q[3]-=rhs.q[3]; return *this;};
	const Quaternion operator+(const Quaternion &rhs) const {return Quaternion(*this) += rhs;};
	const Quaternion operator-(const Quaternion &rhs) const {return Quaternion(*this) -= rhs;};
	const Quaternion operator*(const Quaternion &rhs) const 
		{
		return Quaternion
			(
			this->q[0]*rhs.q[0]-this->q[1]*rhs.q[1]-this->q[2]*rhs.q[2]-this->q[3]*rhs.q[3],
			this->q[0]*rhs.q[1]+this->q[1]*rhs.q[0]+this->q[2]*rhs.q[3]-this->q[3]*rhs.q[2],
			this->q[0]*rhs.q[2]+this->q[2]*rhs.q[0]+this->q[3]*rhs.q[1]-this->q[1]*rhs.q[3],
			this->q[0]*rhs.q[3]+this->q[3]*rhs.q[0]+this->q[1]*rhs.q[2]-this->q[2]*rhs.q[1]
			);
		};	
	Quaternion& conjugateme(void) {q[1] = -q[1]; q[2] = -q[2]; q[3] = -q[3]; return *this;}
	Quaternion conjugate(void) const {return Quaternion(*this).conjugateme();}
	T normsqr(void) const {return q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];};
	T norm(void) const;
	Quaternion inverse(void) const {return Quaternion(*this).conjugateme()/this->normsqr();};
	const Quaternion operator/(const Quaternion &rhs) const {return *this * rhs.inverse();};
	};

	
//stream I/O
template <typename T>
std::istream& operator>>(std::istream  &s, Quaternion<T> &x)
{
s >> x.q[0];
s >> x.q[1];
s >> x.q[2];
s >> x.q[3];
return s;
}

template <typename T>
std::ostream& operator<<(std::ostream &s, const Quaternion<T> &x) {
s << x.q[0]<<" ";
s << x.q[1]<<" ";
s << x.q[2]<<" ";
s << x.q[3];
return s;
}



#endif /* _QUATERNION_H_ */

