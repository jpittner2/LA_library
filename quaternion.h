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

//this actually should be compilable separately from LA as well as being a part of LA

#ifndef _QUATERNION_H_
#define _QUATERNION_H_


#ifndef AVOID_STDSTREAM
#include <iostream>
#endif
#include <stdio.h>
#include <string.h>
#include <complex>
#include <math.h>

namespace LA_Quaternion {

template <typename T>
class Quaternion 
	{
public:
	//just plain old data
	T q[4];
	//methods
	Quaternion(void) {};
	Quaternion(const T x, const T u=0, const T v=0, const T w=0) {q[0]=x; q[1]=u; q[2]=v; q[3]=w;}; //quaternion from real(s)
	Quaternion(const std::complex<T> &rhs) {q[0]=rhs.real(); q[1]=rhs.imag(); q[2]=0; q[3]=0;} //quaternion from complex
	explicit Quaternion(const T* x, const int shift=1) {q[0]=0; memcpy(q+shift,x,(4-shift)*sizeof(T));} //for shift=1 quaternion from xyz vector

	//compiler generates default copy constructor and assignment operator
	
	//formal indexing
	inline const T operator[](const int i) const {return q[i];};
	inline T& operator[](const int i) {return q[i];};

	//get pointer to data transparently
        inline operator const T*() const {return q;};
        inline operator T*() {return q;};

	//operations of quaternions with scalars
	void clear() {memset(q,0,4*sizeof(T));}
	Quaternion& operator=(const T x) {q[0]=x; memset(&q[1],0,3*sizeof(T)); return *this;};  //quaternion from real 
	Quaternion& operator+=(const T rhs) {q[0]+=rhs; return *this;};
	Quaternion& operator-=(const T rhs) {q[0]-=rhs; return *this;};
	Quaternion& operator*=(const T rhs) {q[0]*=rhs; q[1]*=rhs; q[2]*=rhs; q[3]*=rhs; return *this;};
	Quaternion& operator/=(const T rhs) {return *this *= ((T)1/rhs);};
	const Quaternion operator+(const T rhs) const {return Quaternion(*this) += rhs;};
        const Quaternion operator-(const T rhs) const {return Quaternion(*this) -= rhs;};
        const Quaternion operator*(const T rhs) const {return Quaternion(*this) *= rhs;};
        const Quaternion operator/(const T rhs) const {return Quaternion(*this) /= rhs;};

	//quaternion algebra
	const Quaternion operator-() const {Quaternion r(*this); r.q[0]= -r.q[0]; r.q[1]= -r.q[1]; r.q[2]= -r.q[2]; r.q[3]= -r.q[3]; return r;}; //unary minus
	Quaternion& operator+=(const Quaternion &rhs) {q[0]+=rhs.q[0];q[1]+=rhs.q[1];q[2]+=rhs.q[2];q[3]+=rhs.q[3]; return *this;};
	Quaternion& operator-=(const Quaternion &rhs) {q[0]-=rhs.q[0];q[1]-=rhs.q[1];q[2]-=rhs.q[2];q[3]-=rhs.q[3]; return *this;};
	Quaternion operator+(const Quaternion &rhs) const {return Quaternion(*this) += rhs;};
	Quaternion operator-(const Quaternion &rhs) const {return Quaternion(*this) -= rhs;};
	Quaternion operator*(const Quaternion &rhs) const; //regular product
	Quaternion times_vec3(const T *rhs) const; //save flops for quaternions representing vectors
	Quaternion vec3_times(const T *rhs) const; //save flops for quaternions representing vectors
	
	Quaternion& conjugateme(void) {q[1] = -q[1]; q[2] = -q[2]; q[3] = -q[3]; return *this;}
	Quaternion conjugate(void) const {return Quaternion(*this).conjugateme();}
	T dot(const Quaternion &rhs) const  {return q[0]*rhs.q[0] + q[1]*rhs.q[1] + q[2]*rhs.q[2] + q[3]*rhs.q[3];};
	T normsqr(void) const {return dot(*this);};
	T norm(void) const {return sqrt(normsqr());};
	Quaternion& fast_normalize(bool unique_sign=false); //using quick 1/sqrt for floats
	Quaternion& normalize(T *getnorm=NULL, bool unique_sign=false);
	Quaternion inverse(void) const {return Quaternion(*this).conjugateme()/normsqr();};
	const Quaternion operator/(const Quaternion &rhs) const {return *this * rhs.inverse();};
	Quaternion rotateby(const Quaternion &rhs); //conjugation-rotation of this by NORMALIZED  rhs
	void rotate(T *to, const T *from, Quaternion<T> *grad=NULL) const; //rotate xyz vector by NORMALIZED *this
	Quaternion rotate_match(T *to, const T *from, const T *match) const; //gradient of quaternion rotation which should "to" = "from" transformed by *this to match a given vector by gradient descent
	Quaternion commutator(const Quaternion &rhs) const {return *this * rhs - rhs * *this;}; //could be made more efficient
	Quaternion anticommutator(const Quaternion &rhs) const {return *this * rhs + rhs * *this;}; //could be made more efficient
	T geodesic_distance(const Quaternion &rhs) const {T t=dot(rhs); return acos(2*t*t-1);}; //length of great arc between two quaternions on the S3 hypersphere

	//some conversions (for all 12 cases of euler angles go via rotation matrices), cf. also the 1977 NASA paper
	void normquat2eulerzyx(T *eul) const;  //corresponds to [meul -r -T xyz -d -t -R] or  euler2rotmat(...,"xyz",true,true,true)
	inline void eulerzyx2normquat(const T *eul) {euler2normquat(eul,"zyx");};  
	void normquat2euler(T *eul, const char *type) const; 
	void euler2normquat(const T *eul, const char *type);
	void axis2normquat(const T *axis, const T &angle);
	void normquat2axis(T *axis,  T &angle) const;
	

	//C-style IO
	int fprintf(FILE *f, const char *format) const {return ::fprintf(f,format,q[0],q[1],q[2],q[3]);};
	int sprintf(char *f, const char *format) const {return ::sprintf(f,format,q[0],q[1],q[2],q[3]);};
	int fscanf(FILE *f, const char *format) const {return ::fscanf(f,format,q[0],q[1],q[2],q[3]);};
        int sscanf(char *f, const char *format) const {return ::sscanf(f,format,q[0],q[1],q[2],q[3]);};
	};

	
//stream I/O ... cannot be moved to .cc, since we do e.g. Quaternion<NRMat<double>>>
#ifndef AVOID_STDSTREAM

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
std::ostream& operator<<(std::ostream &s, const Quaternion<T> &x)
{
s << x.q[0]<<" ";
s << x.q[1]<<" ";
s << x.q[2]<<" ";
s << x.q[3];
return s;
}
#endif




//the following must be in .h due to the generic M type which is unspecified and can be any type providing [][], either plain C matrix, Mat3 class, or std::matrix or LA matrix NRMat
//maybe we go via T* and recast it to T (*)[3] and move this to .cc to avoid replication of the code in multiple object files?

//conversion from normalized quaternion to SU(2) matrix (+/- q yields different SU(2) element)
template<typename T, typename M>
void normquat2su2mat(const Quaternion<T> &q, M &a)
{
a[0][0] = std::complex<T>(q[0],q[1]);
a[0][1] = std::complex<T>(q[2],q[3]);
a[1][0] = std::complex<T>(-q[2],q[3]);
a[1][1] = std::complex<T>(q[0],-q[1]);
}


//use transpose option to match nasa paper definition
//conversion between quanternions and rotation matrices (+/- q yields the same rotation)
//
template<typename T, typename M>
void normquat2rotmat(const Quaternion<T> &q, M &a, bool transpose=false)
{
//some explicit common subexpression optimizations
{
T q00= q[0]*q[0];
T q11= q[1]*q[1];
T q22= q[2]*q[2];
T q33= q[3]*q[3];

a[0][0] = q00+q11-q22-q33;
a[1][1] = q00+q22-q11-q33;
a[2][2] = q00+q33-q11-q22;
}

T q01= q[0]*q[1];
T q02= q[0]*q[2];
T q03= q[0]*q[3];
T q12= q[1]*q[2];
T q13= q[1]*q[3];
T q23= q[2]*q[3];

if(transpose) //effectively sign change of the temporaries
	{
	q01 = -q01;
	q02 = -q03;
	q03 = -q02;
	}
a[0][1] = 2*(q12+q03);
a[1][0] = 2*(q12-q03);
a[0][2] = 2*(q13-q02);
a[2][0] = 2*(q13+q02);
a[1][2] = 2*(q23+q01);
a[2][1] = 2*(q23-q01);
}


//transpose option to match nasa
template<typename T, typename M>
void quat2rotmat(Quaternion<T> q, M &a,  bool transpose=false, const bool already_normalized=false)
{
if(!already_normalized) q.normalize();
normquat2rotmat(q,a,transpose);
}


//use transpose option to match nasa
//derivative of the rotation matrix by quaternion elements
template<typename T, typename M>
void normquat2rotmatder(const Quaternion<T> &q, Quaternion<M> &a, bool transpose=false)
{
//some explicit common subexpression optimizations
T q0= q[0]+q[0];
T q1= q[1]+q[1];
T q2= q[2]+q[2];
T q3= q[3]+q[3];

a[0][0][0]=  q0;
a[0][0][1]=  q3;
a[0][0][2]= -q2;
a[0][1][0]= -q3;
a[0][1][1]=  q0;
a[0][1][2]=  q1;
a[0][2][0]=  q2;
a[0][2][1]= -q1;
a[0][2][2]=  q0;

a[1][0][0]=  q1;
a[1][0][1]=  q2;
a[1][0][2]=  q3;
a[1][1][0]=  q2;
a[1][1][1]= -q1;
a[1][1][2]=  q0;
a[1][2][0]=  q3;
a[1][2][1]= -q0;
a[1][2][2]= -q1;

a[2][0][0]= -q2;
a[2][0][1]=  q1;
a[2][0][2]= -q0;
a[2][1][0]=  q1;
a[2][1][1]=  q2;
a[2][1][2]=  q3;
a[2][2][0]=  q0;
a[2][2][1]=  q3;
a[2][2][2]= -q2;

a[3][0][0]= -q3;
a[3][0][1]=  q0;
a[3][0][2]=  q1;
a[3][1][0]= -q0;
a[3][1][1]= -q3;
a[3][1][2]=  q2;
a[3][2][0]=  q1;
a[3][2][1]=  q2;
a[3][2][2]=  q3;

if(transpose)
	{
	a[0].transposeme();
	a[1].transposeme();
	a[2].transposeme();
	a[3].transposeme();
	}
}


//normalized quaternion from rotation matrix
//convention compatible with the paper on MEMS sensors by Sebastian O.H. Madgwick
//the rotation matrix correcponds to transpose of (4) in Sarabandi and Thomas paper or the NASA paper
//where the method is described

template<typename T, typename M>
void rotmat2normquat(const M &a, Quaternion<T> &q, bool transpose=false)
{
T tr= a[0][0]+a[1][1]+a[2][2];
T a12m = transpose? a[1][0]-a[0][1] : a[0][1]-a[1][0];
T a13m = transpose? a[2][0]-a[0][2] : a[0][2]-a[2][0];
T a23m = transpose? a[2][1]-a[1][2] : a[1][2]-a[2][1];
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
        T a13p = a[0][2]+a[2][0];
        T a23p = a[1][2]+a[2][1];

        q[0] = (T).5*sqrt((a23m*a23m+a13m*a13m+a12m*a12m)/((T)3.-tr));
        q[1] = (T).5*sqrt((a23m*a23m+a12p*a12p+a13p*a13p)/((T)3.-a[0][0]+a[1][1]+a[2][2]));
        q[2] = (T).5*sqrt((a13m*a13m+a12p*a12p+a23p*a23p)/((T)3.+a[0][0]-a[1][1]+a[2][2]));
        q[3] = (T).5*sqrt((a12m*a12m+a13p*a13p+a23p*a23p)/((T)3.+a[0][0]+a[1][1]-a[2][2]));
        }

if(a23m<0) q[1] = -q[1];
if(a13m>0) q[2] = -q[2];
if(a12m<0) q[3] = -q[3];
}



//Quaternion Functions - cf. https://en.wikipedia.org/wiki/Quaternion 


template<typename T>
Quaternion<T> exp(const Quaternion<T> &x);


//NOTE: log(exp(x))  need not be always = x ... log is not unique!
//NOTE2: log(x*y) != log(y*x) != log(x)+log(y)
template<typename T>
Quaternion<T> log(const Quaternion<T> &x); 


template<typename T>
Quaternion<T> pow(const Quaternion<T> &x, const T &y);



} //namespace

#endif /* _QUATERNION_H_ */

