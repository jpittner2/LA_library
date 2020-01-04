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

//this header defines simple classes for 3-dimensional vectors and matrices to describe rotations etc.
//the class is compatible with functions in quaternion.h used for SO(3) parametrization
//it should be compilable separately from LA as well as being a part of LA

#ifndef _VECMAT3_H_
#define _VECMAT3_H_

#include <stdlib.h>
#include <iostream>
#include <cstring>
#include <math.h>


template <typename T>
class Vec3
	{
public:
	//just plain old data
	T q[3];
	//
	Vec3(void) {};
	Vec3(const T x, const T u=0, const T v=0) {q[0]=x; q[1]=u; q[2]=v;}; //Vec3 from real(s)
	explicit Vec3(const T* x) {memcpy(q,x,3*sizeof(T));} 

	//compiler generates default copy constructor and assignment operator
	
	//formal indexing
	const T operator[](const int i) const {return this->q[i];};
	T& operator[](const int i) {return this->q[i];};

	//operations of Vec3s with scalars
	Vec3& operator*=(const T rhs) {this->q[0]*=rhs; this->q[1]*=rhs; this->q[2]*=rhs; return *this;};
	Vec3& operator/=(const T rhs) {return *this *= ((T)1/rhs);};
        const Vec3 operator*(const T rhs) const {return Vec3(*this) *= rhs;};
        const Vec3 operator/(const T rhs) const {return Vec3(*this) /= rhs;};

	//Vec3 algebra
	const Vec3 operator-() const {Vec3 r(*this); r.q[0]= -r.q[0]; r.q[1]= -r.q[1]; r.q[2]= -r.q[2]; return r;}; //unary minus
	Vec3& operator+=(const Vec3 &rhs) {this->q[0]+=rhs.q[0];this->q[1]+=rhs.q[1];this->q[2]+=rhs.q[2]; return *this;};
	Vec3& operator-=(const Vec3 &rhs) {this->q[0]-=rhs.q[0];this->q[1]-=rhs.q[1];this->q[2]-=rhs.q[2]; return *this;};
	const Vec3 operator+(const Vec3 &rhs) const {return Vec3(*this) += rhs;};
	const Vec3 operator-(const Vec3 &rhs) const {return Vec3(*this) -= rhs;};
	const Vec3 operator*(const Vec3 &rhs) const {Vec3 x; x[0] = q[1]*rhs.q[2]-q[2]*rhs.q[1]; x[1] = q[2]*rhs.q[0]-q[0]*rhs.q[2]; x[2] = q[0]*rhs.q[1]-q[1]*rhs.q[0]; return x;}; //vector product
	T dot(const Vec3 &rhs) const  {return q[0]*rhs.q[0] + q[1]*rhs.q[1] + q[2]*rhs.q[2];};
	T normsqr(void) const {return dot(*this);};
	T norm(void) const {return sqrt(this->normsqr());};
	Vec3& normalize(void) {*this /= this->norm(); return *this;};
	};

	
//stream I/O
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


#endif /* _VECMAT3_H_ */

