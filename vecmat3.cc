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

#include "vecmat3.h"

using namespace LA_Vecmat3;

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
void LA_Vecmat3::euler2rotmat(const T *eul, Mat3<T> &a, const char *type, bool transpose, bool direction, bool reverse)
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
void LA_Vecmat3::rotmat2euler(T *eul, const Mat3<T> &a, const char *type, bool transpose, bool direction, bool reverse)
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
std::istream& LA_Vecmat3::operator>>(std::istream  &s, Vec3<T> &x)
{
s >> x.q[0];
s >> x.q[1];
s >> x.q[2];
return s;
}

template <typename T>
std::ostream& LA_Vecmat3::operator<<(std::ostream &s, const Vec3<T> &x) {
s << x.q[0]<<" ";
s << x.q[1]<<" ";
s << x.q[2];
return s;
}

template <typename T>
std::istream& LA_Vecmat3::operator>>(std::istream  &s, Mat3<T> &x)
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
std::ostream& LA_Vecmat3::operator<<(std::ostream &s, const Mat3<T> &x) {
s << x.q[0][0]<<" "<< x.q[0][1]<<" " << x.q[0][2]<<std::endl;
s << x.q[1][0]<<" "<< x.q[1][1]<<" " << x.q[1][2]<<std::endl;
s << x.q[2][0]<<" "<< x.q[2][1]<<" " << x.q[2][2]<<std::endl;
return s;
}
#endif




//force instantization
#define INSTANTIZE(T) \
template class Vec3<T>; \
template class Mat3<T>; \
template void LA_Vecmat3::euler2rotmat(const T *eul, Mat3<T> &a, const char *type, bool transpose=0, bool direction=0, bool reverse=0); \
template void LA_Vecmat3::rotmat2euler(T *eul, const Mat3<T> &a, const char *type, bool transpose=0, bool direction=0, bool reverse=0); \


#ifndef AVOID_STDSTREAM
#define INSTANTIZE2(T) \
template std::istream& LA_Vecmat3::operator>>(std::istream  &s, Vec3<T> &x); \
template std::ostream& LA_Vecmat3::operator<<(std::ostream &s, const Vec3<T> &x); \
template std::istream& LA_Vecmat3::operator>>(std::istream  &s, Mat3<T> &x); \
template std::ostream& LA_Vecmat3::operator<<(std::ostream &s, const Mat3<T> &x); \

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
