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
#ifndef AVOID_STDSTREAM
#include <iostream>
#endif
#include <string.h>
#include <math.h>
#include <stdio.h>

namespace LA_Vecmat3 {

//forward declaration
template <typename T> class Mat3;


template <typename T>
class Vec3
	{
	friend class Mat3<T>;
public:
	//just plain old data
	T q[3];
	//
	Vec3(void) {};
	Vec3(const T x, const T u=0, const T v=0) {q[0]=x; q[1]=u; q[2]=v;}; //Vec3 from real(s)
	Vec3(const T* x) {memcpy(q,x,3*sizeof(T));} 
	
	//get pointer to data transparently
	inline operator const T*() const {return q;};
	inline operator T*() {return q;};

	//compiler generates default copy constructor and assignment operator
	
	//formal indexing
	inline const T operator[](const int i) const {return q[i];};
	inline T& operator[](const int i) {return q[i];};

	//operations of Vec3s with scalars
	Vec3& operator*=(const T rhs) {q[0]*=rhs; q[1]*=rhs; q[2]*=rhs; return *this;};
	Vec3& operator/=(const T rhs) {return *this *= ((T)1/rhs);};
        const Vec3 operator*(const T rhs) const {return Vec3(*this) *= rhs;};
        const Vec3 operator/(const T rhs) const {return Vec3(*this) /= rhs;};

	//Vec3 algebra
	const Vec3 operator-() const {Vec3 r(*this); r.q[0]= -r.q[0]; r.q[1]= -r.q[1]; r.q[2]= -r.q[2]; return r;}; //unary minus
	Vec3& operator+=(const Vec3 &rhs) {q[0]+=rhs.q[0];q[1]+=rhs.q[1];q[2]+=rhs.q[2]; return *this;};
	Vec3& operator-=(const Vec3 &rhs) {q[0]-=rhs.q[0];q[1]-=rhs.q[1];q[2]-=rhs.q[2]; return *this;};
	const Vec3 operator+(const Vec3 &rhs) const {return Vec3(*this) += rhs;};
	const Vec3 operator-(const Vec3 &rhs) const {return Vec3(*this) -= rhs;};
	const Vec3 operator*(const Vec3 &rhs) const {Vec3 x; x[0] = q[1]*rhs.q[2]-q[2]*rhs.q[1]; x[1] = q[2]*rhs.q[0]-q[0]*rhs.q[2]; x[2] = q[0]*rhs.q[1]-q[1]*rhs.q[0]; return x;}; //vector product
	T dot(const Vec3 &rhs) const  {return q[0]*rhs.q[0] + q[1]*rhs.q[1] + q[2]*rhs.q[2];};
	T normsqr(void) const {return dot(*this);};
	T norm(void) const {return sqrt(normsqr());};
	Vec3& normalize(void) {*this /= norm(); return *this;};
	const Vec3 operator*(const Mat3<T> &rhs) const
                {
                Vec3 r;
                r[0] = q[0]*rhs.q[0][0] + q[1]*rhs.q[1][0] + q[2]*rhs.q[2][0];
                r[1] = q[0]*rhs.q[0][1] + q[1]*rhs.q[1][1] + q[2]*rhs.q[2][1];
                r[2] = q[0]*rhs.q[0][2] + q[1]*rhs.q[1][2] + q[2]*rhs.q[2][2];
                return r;
                }; //matrix times vector
	//C-style IO
	void fprintf(FILE *f, const char *format) const {::fprintf(f,format,q[0],q[1],q[2]);};
        void sprintf(char *f, const char *format) const {::sprintf(f,format,q[0],q[1],q[2]);};
	int fscanf(FILE *f, const char *format) const {return ::fscanf(f,format,q[0],q[1],q[2]);};
        int sscanf(char *f, const char *format) const {return ::sscanf(f,format,q[0],q[1],q[2]);};
};


template <typename T>
class Mat3
	{
	friend class Vec3<T>;
public:
	//just plain old data
	T q[3][3];
	//
	Mat3(void) {};
	Mat3(const T x) {memset(q,0,9*sizeof(T)); q[0][0]=q[1][1]=q[2][2]=x;}; //scalar matrix
	Mat3(const T* x) {memcpy(q,x,9*sizeof(T));} 
	
	//get pointer to data transparently
	inline operator const T*() const {return q;};
	inline operator T*() {return q;};

	//compiler generates default copy constructor and assignment operator
	
	//formal indexing
	inline const T* operator[](const int i) const {return q[i];};
	inline T* operator[](const int i) {return q[i];};
	inline const T operator()(const int i, const int j) const {return q[i][j];};
	inline T& operator()(const int i, const int j) {return q[i][j];};
	

	//operations of Mat3s with scalars
	Mat3& operator+=(const T rhs) {q[0][0]+=rhs; q[1][1]+=rhs; q[2][2]+=rhs; return *this;};
	Mat3& operator-=(const T rhs) {q[0][0]-=rhs; q[1][1]-=rhs; q[2][2]-=rhs; return *this;};
	const Mat3 operator+(const T rhs) const {return Mat3(*this) += rhs;};
	const Mat3 operator-(const T rhs) const {return Mat3(*this) -= rhs;};
	Mat3& operator*=(const T rhs) {q[0][0]*=rhs; q[0][1]*=rhs; q[0][2]*=rhs; q[1][0]*=rhs; q[1][1]*=rhs; q[1][2]*=rhs; q[2][0]*=rhs; q[2][1]*=rhs; q[2][2]*=rhs; return *this;};
	Mat3& operator/=(const T rhs) {return *this *= ((T)1/rhs);};
        const Mat3 operator*(const T rhs) const {return Mat3(*this) *= rhs;};
        const Mat3 operator/(const T rhs) const {return Mat3(*this) /= rhs;};

	//Mat3 algebra
	const Mat3 operator-() const {return *this * (T)-1;}; //unary minus
	Mat3& operator+=(const Mat3 &rhs) {q[0][0]+=rhs.q[0][0];q[0][1]+=rhs.q[0][1];q[0][2]+=rhs.q[0][2]; q[1][0]+=rhs.q[1][0];q[1][1]+=rhs.q[1][1];q[1][2]+=rhs.q[1][2]; q[2][0]+=rhs.q[2][0];q[2][1]+=rhs.q[2][1];q[2][2]+=rhs.q[2][2]; return *this;};
	Mat3& operator-=(const Mat3 &rhs) {q[0][0]-=rhs.q[0][0];q[0][1]-=rhs.q[0][1];q[0][2]-=rhs.q[0][2]; q[1][0]-=rhs.q[1][0];q[1][1]-=rhs.q[1][1];q[1][2]-=rhs.q[1][2]; q[2][0]-=rhs.q[2][0];q[2][1]-=rhs.q[2][1];q[2][2]-=rhs.q[2][2]; return *this;};
	const Mat3 operator+(const Mat3 &rhs) const {return Mat3(*this) += rhs;};
	const Mat3 operator-(const Mat3 &rhs) const {return Mat3(*this) -= rhs;};
	const Mat3 operator*(const Mat3 &rhs) const
		{
		Mat3 r;
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
		};  //matrix product
	const Vec3<T> operator*(const Vec3<T> &rhs) const
		{
		Vec3<T> r;
		r[0] = q[0][0]*rhs.q[0] + q[0][1]*rhs.q[1] + q[0][2]*rhs.q[2];
		r[1] = q[1][0]*rhs.q[0] + q[1][1]*rhs.q[1] + q[1][2]*rhs.q[2];
		r[2] = q[2][0]*rhs.q[0] + q[2][1]*rhs.q[1] + q[2][2]*rhs.q[2];
		return r;
		}; //matrix times vector
	T trace() const  {return q[0][0]+q[1][1]+q[2][2];};
	T determinant() const {return q[0][0]*(q[2][2]*q[1][1]-q[2][1]*q[1][2])-q[1][0]*(q[2][2]*q[0][1]-q[2][1]*q[0][2])+q[2][0]*(q[1][2]*q[0][1]-q[1][1]*q[0][2]); };//determinant
	void transposeme() {T t; t=q[0][1]; q[0][1]=q[1][0]; q[1][0]=t; t=q[0][2]; q[0][2]=q[2][0]; q[2][0]=t; t=q[1][2]; q[1][2]=q[2][1]; q[2][1]=t;};
	const Mat3 transpose() const {Mat3 r(*this); r.transposeme(); return r;};
	const Mat3 inverse() const
		{
		Mat3 r;
		r[0][0]=  q[2][2]*q[1][1]-q[2][1]*q[1][2];
		r[0][1]= -q[2][2]*q[0][1]+q[2][1]*q[0][2];
		r[0][2]=  q[1][2]*q[0][1]-q[1][1]*q[0][2];
		r[1][0]= -q[2][2]*q[1][0]+q[2][0]*q[1][1];
		r[1][1]=  q[2][2]*q[0][0]-q[2][0]*q[0][2];
		r[1][2]= -q[1][2]*q[0][0]+q[1][0]*q[0][2];
		r[2][0]=  q[2][1]*q[1][0]-q[2][0]*q[1][1];
		r[2][1]= -q[2][1]*q[0][0]+q[2][0]*q[0][1];
		r[2][2]=  q[1][1]*q[0][0]-q[1][0]*q[0][1];
		return r/determinant();
		};
	//C-style IO
        void fprintf(FILE *f, const char *format) const {::fprintf(f,format,q[0][0],q[0][1],q[0][2]); ::fprintf(f,format,q[1][0],q[1][1],q[1][2]); ::fprintf(f,format,q[2][0],q[2][1],q[2][2]);};
        int fscanf(FILE *f, const char *format) const {return ::fscanf(f,format,q[0][0],q[0][1],q[0][2]) + ::fscanf(f,format,q[1][0],q[1][1],q[1][2]) + ::fscanf(f,format,q[2][0],q[2][1],q[2][2]);};

};


	
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

}//namespace
#endif /* _VECMAT3_H_ */

