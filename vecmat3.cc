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
void Vec3<T>::inertia(Mat3<T> &m, const T weight) const
{
T r2 = q[0]*q[0]+q[1]*q[1]+q[2]*q[2];

m[0][0] -= weight*(q[0]*q[0]-r2);
m[0][1] -= weight*q[0]*q[1];
m[0][2] -= weight*q[0]*q[2];

m[1][0] -= weight*q[1]*q[0];
m[1][1] -= weight*(q[1]*q[1]-r2);
m[1][2] -= weight*q[1]*q[2];

m[2][0] -= weight*q[2]*q[0];
m[2][1] -= weight*q[2]*q[1];
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
T Mat3<T>::determinant() const 
{
return q[0][0]*(q[2][2]*q[1][1]-q[2][1]*q[1][2])-q[1][0]*(q[2][2]*q[0][1]-q[2][1]*q[0][2])+q[2][0]*(q[1][2]*q[0][1]-q[1][1]*q[0][2]);
}

template<typename T>
void Mat3<T>::transposeme() 
{T t; t=q[0][1]; q[0][1]=q[1][0]; q[1][0]=t; t=q[0][2]; q[0][2]=q[2][0]; q[2][0]=t; t=q[1][2]; q[1][2]=q[2][1]; q[2][1]=t;};

template<typename T>
const Mat3<T> Mat3<T>::inverse() const
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
                return r/determinant();
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
#define DBL_EPSILON 1.19209290e-07f
#else
#define  DBL_EPSILON std::numeric_limits<T>::epsilon()
#endif

#define M_SQRT3    1.73205080756887729352744634151   // sqrt(3)
#define SQR(x)      ((x)*(x))                        // x^2
//
template <typename T>
void Mat3<T>::eival_sym(Vec3<T> &w) const
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

  T p, sqrt_p, q, c, s, phi;
  p = SQR(m) - 3.0*c1;
  q = m*(p - (3.0/2.0)*c1) - (27.0/2.0)*c0;
  sqrt_p = sqrt(abs(p));

  phi = 27.0 * ( 0.25*SQR(c1)*(p - c1) + c0*(q + 27.0/4.0*c0));
  phi = (1.0/3.0) * atan2(sqrt(abs(phi)), q);
  
  c = sqrt_p*cos(phi);
  s = (1.0/M_SQRT3)*sqrt_p*sin(phi);

  w[1]  = (1.0/3.0)*(m - c);
  w[2]  = w[1] + s;
  w[0]  = w[1] + c;
  w[1] -= s;

//sort in ascending order
if(w[0]>w[1]) {T tmp=w[0]; w[0]=w[1]; w[1]=tmp;}
if(w[0]>w[2]) {T tmp=w[0]; w[0]=w[2]; w[2]=tmp;}
if(w[1]>w[2]) {T tmp=w[1]; w[1]=w[2]; w[2]=tmp;}
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
void Mat3<T>::eivec_sym(Vec3<T> &w, Mat3 &v) const
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
  eival_sym(w);
Mat3<T> A(*this); //scratch copy

  wmax = fabs(w[0]);
  if ((t=fabs(w[1])) > wmax)
    wmax = t;
  if ((t=fabs(w[2])) > wmax)
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
  if (fabs(t) > 8.0 * DBL_EPSILON * wmax)
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

}

#undef SQR
//end eigensolver for 3x3 matrix
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

//force instantization
#define INSTANTIZE(T) \
template class Vec3<T>; \
template class Mat3<T>; \
template void euler2rotmat(const T *eul, Mat3<T> &a, const char *type, bool transpose=0, bool direction=0, bool reverse=0); \
template void rotmat2euler(T *eul, const Mat3<T> &a, const char *type, bool transpose=0, bool direction=0, bool reverse=0); \


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
