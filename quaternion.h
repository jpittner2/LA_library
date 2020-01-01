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

//this actually should be compilable separately from LA as well as being a part of LA

#ifndef _QUATERNION_H_
#define _QUATERNION_H_

#include <stdlib.h>
#include <iostream>
#include <complex>
#include <cstring>
#include <math.h>


template <typename T>
class Quaternion 
	{
public:
	//just plain old data
	T q[4];
	//
	Quaternion(void) {};
	Quaternion(const T x, const T u=0, const T v=0, const T w=0) {q[0]=x; q[1]=u; q[2]=v; q[3]=w;}; //quaternion from real(s)
	Quaternion(const std::complex<T> &rhs) {q[0]=rhs.real(); q[1]=rhs.imag(); q[2]=0; q[3]=0;} //quaternion from complex
	explicit Quaternion(const T* x, const int shift=1) {q[0]=0; memcpy(q+shift,x,(4-shift)*sizeof(T));} //for shift=1 quaternion from xyz vector

	//compiler generates default copy constructor and assignment operator
	
	//formal indexing
	const T operator[](const int i) const {return this->q[i];};
	T& operator[](const int i) {return this->q[i];};

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
	const Quaternion operator-() const {Quaternion r(*this); r.q[0]= -r.q[0]; r.q[1]= -r.q[1]; r.q[2]= -r.q[2]; r.q[3]= -r.q[3]; return r;}; //unary minus
	Quaternion& operator+=(const Quaternion &rhs) {this->q[0]+=rhs.q[0];this->q[1]+=rhs.q[1];this->q[2]+=rhs.q[2];this->q[3]+=rhs.q[3]; return *this;};
	Quaternion& operator-=(const Quaternion &rhs) {this->q[0]-=rhs.q[0];this->q[1]-=rhs.q[1];this->q[2]-=rhs.q[2];this->q[3]-=rhs.q[3]; return *this;};
	const Quaternion operator+(const Quaternion &rhs) const {return Quaternion(*this) += rhs;};
	const Quaternion operator-(const Quaternion &rhs) const {return Quaternion(*this) -= rhs;};
	const Quaternion operator*(const Quaternion &rhs) const;
	Quaternion& conjugateme(void) {q[1] = -q[1]; q[2] = -q[2]; q[3] = -q[3]; return *this;}
	Quaternion conjugate(void) const {return Quaternion(*this).conjugateme();}
	T normsqr(void) const {return q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];};
	T norm(void) const {return sqrt(this->normsqr());};
	Quaternion& normalize(bool unique_sign=false) {*this /= this->norm(); if(unique_sign && q[0]<0) *this *= (T)-1; return *this;};
	Quaternion inverse(void) const {return Quaternion(*this).conjugateme()/this->normsqr();};
	const Quaternion operator/(const Quaternion &rhs) const {return *this * rhs.inverse();};
	Quaternion rotateby(const Quaternion &rhs, bool rhs_is_normalized=true); //conjugation-rotation of this by rhs
	void rotate(T *rhs) const; //rotate xyz vector by *this
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


//"euler" or Tait-Bryan angles [corresponding to meul -r -T xyz -d -t -R]
template<typename T>
void  normquat2euler(const Quaternion<T> &q, T *);


//the following must be in .h due to the generic M type which is unspecified and can be any type providing [][], either plain C matrix or LA matrix

//conversion between quanternions and rotation matrices
//
template<typename T, typename M>
void normquat2rotmat(const Quaternion<T> &q, M &a)
{
a[0][0] = 2*q[0]*q[0]-1+2*q[1]*q[1];
a[0][1] = 2*(q[1]*q[2]+q[0]*q[3]);
a[0][2] = 2*(q[1]*q[3]-q[0]*q[2]);
a[1][0] = 2*(q[1]*q[2]-q[0]*q[3]);
a[1][1] = 2*q[0]*q[0]-1+2*q[2]*q[2];
a[1][2] = 2*(q[2]*q[3]+q[0]*q[1]);
a[2][0] = 2*(q[1]*q[3]+q[0]*q[2]);
a[2][1] = 2*(q[2]*q[3]-q[0]*q[1]);
a[2][2] = 2*q[0]*q[0]-1+2*q[3]*q[3];
}

template<typename T, typename M>
void quat2rotmat(Quaternion<T> q, M &a, const bool already_normalized=false)
{
if(!already_normalized) q.normalize();
normquat2rotmat(q,a);
}


//normalized quaternion from rotation matrix
//convention compatible with the paper on MEMS sensors by Sebastian O.H. Madgwick
//the rotation matrix correcponds to transpose of (4) in Sarabandi and Thomas paper
//where the method is described

template<typename T, typename M>
void rotmat2normquat(const M &a, Quaternion<T> &q)
{
T tr= a[0][0]+a[1][1]+a[2][2];
if(tr>=0)
        {
        q[0] = (T).5*sqrt((T)1. +tr);
        q[1] = (T).5*sqrt((T)1. +a[0][0]-a[1][1]-a[2][2]);
        q[2] = (T).5*sqrt((T)1. -a[0][0]+a[1][1]-a[2][2]);
        q[3] = (T).5*sqrt((T)1. -a[0][0]-a[1][1]+a[2][2]);
        }
else
        {
        T a12p = a[0][1]+a[1][0];
        T a12m = a[0][1]-a[1][0];
        T a13p = a[0][2]+a[2][0];
        T a13m = a[0][2]-a[2][0];
        T a23p = a[1][2]+a[2][1];
        T a23m = a[1][2]-a[2][1];

        q[0] = (T).5*sqrt((a23m*a23m+a13m*a13m+a12m*a12m)/((T)3.-tr));
        q[1] = (T).5*sqrt((a23m*a23m+a12p*a12p+a13p*a13p)/((T)3.-a[0][0]+a[1][1]+a[2][2]));
        q[2] = (T).5*sqrt((a13m*a13m+a12p*a12p+a23p*a23p)/((T)3.+a[0][0]-a[1][1]+a[2][2]));
        q[3] = (T).5*sqrt((a12m*a12m+a13p*a13p+a23p*a23p)/((T)3.+a[0][0]+a[1][1]-a[2][2]));
        }

if(a[1][2]-a[2][1]<0) q[1] = -q[1];
if(a[2][0]-a[0][2]<0) q[2] = -q[2];
if(a[0][1]-a[1][0]<0) q[3] = -q[3];
}


//rotation about unit vector axis through an angle to a normalized quaternion
#ifndef AVOID_GONIOM_FUNC
template<typename T>
void axis2normquat(const T *axis, const T &angle, Quaternion<T> &q);

template<typename T>
void normquat2axis(const Quaternion<T> &q, T *axis,  T &angle);
#endif



#endif /* _QUATERNION_H_ */

