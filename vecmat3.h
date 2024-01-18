/*
    LA: linear algebra C++ interface library
    Copyright (C) 2020-2021 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>

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

//this header defines simple classes for 3-dimensional REAL-valued vectors and matrices to describe rotations etc.
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

#ifdef NO_NUMERIC_LIMITS                     
#define DBL_EPSILON 1.19209290e-07f
#else 
#define  DBL_EPSILON std::numeric_limits<T>::epsilon()
#endif



float fast_sqrtinv(float);

//forward declaration
template <typename T> class Mat3;


template <typename T>
class Vec3
	{
	friend class Mat3<T>;
public:
	//just plain old data
	T q[3];
	T (&elements())[3] {return q;};
        const T (&elements()const)[3]  {return q;};

	//
	Vec3(void) {};
	Vec3(const T x, const T u=0, const T v=0) {q[0]=x; q[1]=u; q[2]=v;}; //Vec3 from real(s)
	Vec3(const T* x) {memcpy(q,x,3*sizeof(T));} 
	Vec3(const T (&a)[3]) {memcpy(q,a,3*sizeof(T));};
	
	//get pointer to data transparently
	inline operator const T*() const {return q;};
	inline operator T*() {return q;};

	//compiler generates default copy constructor and assignment operator
	
	//formal indexing
	inline const T& operator[](const int i) const {return q[i];};
	inline T& operator[](const int i) {return q[i];};

	//operations of Vec3s with scalars
	void clear() {memset(q,0,3*sizeof(T));}
	Vec3& operator*=(const T rhs) {q[0]*=rhs; q[1]*=rhs; q[2]*=rhs; return *this;};
	Vec3& operator/=(const T rhs) {return *this *= ((T)1/rhs);};
        const Vec3 operator*(const T rhs) const {return Vec3(*this) *= rhs;};
        const Vec3 operator/(const T rhs) const {return Vec3(*this) /= rhs;};
	T sum() const {return q[0]+q[1]+q[2];};
	T asum() const {return abs(q[0])+abs(q[1])+abs(q[2]);};
	T sumsqr() const {return q[0]*q[0]+q[1]*q[1]+q[2]*q[2];};
	T prod() const {return q[0]*q[1]*q[2];};

	//Vec3 algebra
	const Vec3 operator-() const {Vec3 r(*this); r.q[0]= -r.q[0]; r.q[1]= -r.q[1]; r.q[2]= -r.q[2]; return r;}; //unary minus
	Vec3& operator+=(const Vec3 &rhs) {q[0]+=rhs.q[0];q[1]+=rhs.q[1];q[2]+=rhs.q[2]; return *this;};
	Vec3& operator-=(const Vec3 &rhs) {q[0]-=rhs.q[0];q[1]-=rhs.q[1];q[2]-=rhs.q[2]; return *this;};
	const Vec3 operator+(const Vec3 &rhs) const {return Vec3(*this) += rhs;};
	const Vec3 operator-(const Vec3 &rhs) const {return Vec3(*this) -= rhs;};
	const Vec3 operator*(const Vec3 &rhs) const {Vec3 x; x[0] = q[1]*rhs.q[2]-q[2]*rhs.q[1]; x[1] = q[2]*rhs.q[0]-q[0]*rhs.q[2]; x[2] = q[0]*rhs.q[1]-q[1]*rhs.q[0]; return x;}; //vector product
	T dot(const Vec3 &rhs) const  {return q[0]*rhs.q[0] + q[1]*rhs.q[1] + q[2]*rhs.q[2];};
	const Vec3 elementwise_product(const Vec3 &rhs) const {Vec3 x; x[0]=q[0]*rhs.q[0];  x[1]=q[1]*rhs.q[1];  x[2]=q[2]*rhs.q[2]; return x;};
	T normsqr(void) const {return dot(*this);};
	T norm(void) const {return sqrt(normsqr());};
	Vec3& normalize(void) {*this /= norm(); return *this;};
	Vec3& fast_normalize(void);
	const Vec3 operator*(const Mat3<T> &rhs) const;
	const Vec3 timesT(const Mat3<T> &rhs) const; //with transpose
	Mat3<T> outer(const Vec3 &rhs) const; //tensor product
	void addouter(Mat3<T> &m, const Vec3 &rhs, const T weight) const; //tensor product
	void inertia(Mat3<T> &itensor, const T weight) const; //contribution to inertia tensor
	void randomize(const T x);

	//C-style IO
	int fprintf(FILE *f, const char *format) const {return ::fprintf(f,format,q[0],q[1],q[2]);};
        int sprintf(char *f, const char *format) const {return ::sprintf(f,format,q[0],q[1],q[2]);};
	int fscanf(FILE *f, const char *format) const {return ::fscanf(f,format,q[0],q[1],q[2]);};
        int sscanf(char *f, const char *format) const {return ::sscanf(f,format,q[0],q[1],q[2]);};
};


template <typename T>
inline T  hypot3(const Vec3<T> &c, const Vec3<T> &d) {return((c-d).norm());}


template <typename T>
class Mat3
	{
	friend class Vec3<T>;
public:
	//just plain old data
	T q[3][3];
	//
	T (&elements())[3][3] {return q;};
	const T (&elements()const)[3][3]  {return q;};
	Mat3(void) {};
	Mat3(const T (&a)[3][3]) {memcpy(q,a,3*3*sizeof(T));}
	Mat3(const T x) {memset(q,0,9*sizeof(T)); q[0][0]=q[1][1]=q[2][2]=x;}; //scalar matrix
	Mat3& operator=(const T &x) {memset(q,0,9*sizeof(T)); q[0][0]=q[1][1]=q[2][2]=x; return *this;}; //scalar matrix
	void indentity() {*this = (T)1;};
	Mat3(const T* x) {memcpy(q,x,9*sizeof(T));} 
	Mat3(const T x00, const T x01,const T x02,const T x10,const T x11,const T x12,const T x20,const T x21,const T x22) 
		{q[0][0]=x00; q[0][1]=x01; q[0][2]=x02; q[1][0]=x10; q[1][1]=x11; q[1][2]=x12; q[2][0]=x20; q[2][1]=x21; q[2][2]=x22;};
	
	//get pointer to data transparently
	inline operator const T*() const {return &q[0][0];};
	inline operator T*() {return &q[0][0];};

	//compiler generates default copy constructor and assignment operator
	
	//formal indexing
	inline const T* operator[](const int i) const {return q[i];};
	inline T* operator[](const int i) {return q[i];};
	inline const T& operator()(const int i, const int j) const {return q[i][j];};
	inline T& operator()(const int i, const int j) {return q[i][j];};
	

	//operations of Mat3s with scalars
	void clear() {memset(&q[0][0],0,9*sizeof(T));}
	Mat3& operator+=(const T rhs) {q[0][0]+=rhs; q[1][1]+=rhs; q[2][2]+=rhs; return *this;};
	Mat3& operator-=(const T rhs) {q[0][0]-=rhs; q[1][1]-=rhs; q[2][2]-=rhs; return *this;};
	const Mat3 operator+(const T rhs) const {return Mat3(*this) += rhs;};
	const Mat3 operator-(const T rhs) const {return Mat3(*this) -= rhs;};
	Mat3& operator*=(const T rhs) {q[0][0]*=rhs; q[0][1]*=rhs; q[0][2]*=rhs; q[1][0]*=rhs; q[1][1]*=rhs; q[1][2]*=rhs; q[2][0]*=rhs; q[2][1]*=rhs; q[2][2]*=rhs; return *this;};
	Mat3& operator/=(const T rhs) {return *this *= ((T)1/rhs);};
        const Mat3 operator*(const T rhs) const {return Mat3(*this) *= rhs;};
        const Mat3 operator/(const T rhs) const {return Mat3(*this) /= rhs;};

	void randomize(const T x, const bool symmetric=false);

	//Mat3 algebra
	const Mat3 operator-() const {return *this * (T)-1;}; //unary minus
	Mat3& operator+=(const Mat3 &rhs);
	Mat3& operator-=(const Mat3 &rhs);
	const Mat3 operator+(const Mat3 &rhs) const {return Mat3(*this) += rhs;};
	const Mat3 operator-(const Mat3 &rhs) const {return Mat3(*this) -= rhs;};
	const Mat3 operator*(const Mat3 &rhs) const; //matrix product
	const Mat3 timesT(const Mat3 &rhs) const; //matrix product with transpose
	const Mat3 Ttimes(const Mat3 &rhs) const; //matrix product with transpose
	const Mat3 TtimesT(const Mat3 &rhs) const; //matrix product with transpose
	const Vec3<T> operator*(const Vec3<T> &rhs) const; //matrix times vector
	const Vec3<T> Ttimes(const Vec3<T> &rhs) const; //matrix times vector with transpose
	T trace() const  {return q[0][0]+q[1][1]+q[2][2];};
	T determinant() const;
	void transposeme();
	const Mat3 transpose() const {Mat3 r(*this); r.transposeme(); return r;};
	const Mat3 inverse(T *det = NULL) const;
	const Vec3<T> linear_solve(const  Vec3<T> &rhs, T *det = NULL) const;
	//C-style IO
        int fprintf(FILE *f, const char *format) const {int n= ::fprintf(f,format,q[0][0],q[0][1],q[0][2]); n+=::fprintf(f,format,q[1][0],q[1][1],q[1][2]); n+=::fprintf(f,format,q[2][0],q[2][1],q[2][2]); return n;};
        int fscanf(FILE *f, const char *format) const {return ::fscanf(f,format,q[0][0],q[0][1],q[0][2]) + ::fscanf(f,format,q[1][0],q[1][1],q[1][2]) + ::fscanf(f,format,q[2][0],q[2][1],q[2][2]);};
	void symmetrize(); //average offdiagonal elements
	void eival_sym(Vec3<T> &w, const bool sortdown=false) const; //only for real symmetric matrix, symmetry is not checked
	void eivec_sym(Vec3<T> &w, Mat3 &v, const bool sortdown=false) const; //only for real symmetric matrix, symmetry is not checked
	T norm(const T scalar = 0) const;
	void qrd(Mat3 &q, Mat3 &r); //not const, destroys the matrix
	void svd(Mat3 &u, Vec3<T> &w, Mat3 &v, bool proper_rotations=false) const;  //if proper_rotations = true, singular value can be negative but u and v are proper rotations
	void  diagmultl(const Vec3<T> &rhs);
	void  diagmultr(const Vec3<T> &rhs);
	const Mat3 svdinverse(const T thr=1000*DBL_EPSILON) const;
};



//stream I/O
#ifndef AVOID_STDSTREAM
template <typename T>
std::istream& operator>>(std::istream  &s, Vec3<T> &x);

template <typename T>
std::ostream& operator<<(std::ostream &s, const Vec3<T> &x);

template <typename T>
std::istream& operator>>(std::istream  &s, Mat3<T> &x);

template <typename T>
std::ostream& operator<<(std::ostream &s, const Mat3<T> &x);
#endif

//euler angles to rotation matrices cf. https://en.wikipedia.org/wiki/Euler_angles and NASA paper cited therein

#define Euler_case(a,b,c) (((a)-'x')*9+((b)-'x')*3+((c)-'x'))

template<typename T>
void euler2rotmat(const T *eul, Mat3<T> &a, const char *type, bool transpose=0, bool direction=0, bool reverse=0);

template<typename T>
void rotmat2euler(T *eul, const Mat3<T> &a, const char *type, bool transpose=0, bool direction=0, bool reverse=0);

template<typename T>
void perspective(T *proj_xy, const Vec3<T> &point, const Mat3<T> &rot_angle, const Vec3<T> &camera,  const Vec3<T> &plane_to_camera);


}//namespace

using namespace  LA_Vecmat3;
namespace LA {
//forward declaration, needed of this file is used separately from the rest of LA
template<typename T>
class LA_traits;

template<typename T>
class LA_traits<Vec3<T> >
{
public:
       static bool is_plaindata() {return true;};
       static void copyonwrite(Vec3<T>& x) {};
       typedef T normtype;
};

template<typename T>
class LA_traits<Mat3<T> >
{
public:
       static bool is_plaindata() {return true;};
       static void copyonwrite(Mat3<T>& x) {};
       typedef T normtype;
};


}
#endif /* _VECMAT3_H_ */

