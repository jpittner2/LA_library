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

#include "quaternion.h"
#include "vecmat3.h"

using namespace LA_Vecmat3;

namespace LA_Quaternion {

//do not replicate this code in each object file, therefore not in .h
//and instantize the templates for the types needed


template<typename T>
Quaternion<T>& Quaternion<T>::normalize(T *getnorm, bool unique_sign) 
{
T nn=norm(); 
if(getnorm) *getnorm=nn; 
if(unique_sign && q[0]<0)  nn= -nn;
*this /= nn; 
return *this;
};

template<>
Quaternion<float> & Quaternion<float>::fast_normalize(bool unique_sign)
{
float nn=fast_sqrtinv(normsqr()); 
if(unique_sign && q[0]<0)  nn= -nn;       
*this *= nn;
return *this;
}

template<typename T>
Quaternion<T>& Quaternion<T>::fast_normalize(bool unique_sign)  
{
return normalize(NULL,unique_sign);
};





template<typename T>
Quaternion<T> Quaternion<T>::operator*(const Quaternion<T> &rhs) const 
{
return Quaternion<T>
	(
	q[0]*rhs.q[0]-q[1]*rhs.q[1]-q[2]*rhs.q[2]-q[3]*rhs.q[3],
	q[0]*rhs.q[1]+q[1]*rhs.q[0]+q[2]*rhs.q[3]-q[3]*rhs.q[2],
	q[0]*rhs.q[2]+q[2]*rhs.q[0]+q[3]*rhs.q[1]-q[1]*rhs.q[3],
	q[0]*rhs.q[3]+q[3]*rhs.q[0]+q[1]*rhs.q[2]-q[2]*rhs.q[1]
	);
};	

template<typename T>
Quaternion<T> Quaternion<T>::times_vec3(const T *rhs) const
{
return Quaternion<T>
        (
       -q[1]*rhs[0]-q[2]*rhs[1]-q[3]*rhs[2],
        q[0]*rhs[0]+q[2]*rhs[2]-q[3]*rhs[1],
        q[0]*rhs[1]+q[3]*rhs[0]-q[1]*rhs[2],
        q[0]*rhs[2]+q[1]*rhs[1]-q[2]*rhs[0]
        );
};

template<typename T>
Quaternion<T> Quaternion<T>::vec3_times(const T *lhs) const
{
return Quaternion<T>
        (
       -lhs[0]*q[1]-lhs[1]*q[2]-lhs[2]*q[3],
        lhs[0]*q[0]+lhs[1]*q[3]-lhs[2]*q[2],
        lhs[1]*q[0]+lhs[2]*q[1]-lhs[0]*q[3],
        lhs[2]*q[0]+lhs[0]*q[2]-lhs[1]*q[1]
        );
};




//basically the same code as in normquat2rotmat, but avoiding extra storage
template<typename T>
void Quaternion<T>::rotate(T *to, const T *from, Quaternion<T> *grad) const 
{
//some subexpression eliminations
{
T q00= q[0]*q[0];
T q11= q[1]*q[1];
T q22= q[2]*q[2];
T q33= q[3]*q[3];

to[0] = (q00+q11-q22-q33) * from[0];
to[1] = (q00+q22-q11-q33) * from[1];
to[2] = (q00+q33-q11-q22) * from[2];
}


T q01= q[0]*q[1];
T q02= q[0]*q[2];
T q03= q[0]*q[3];
T q12= q[1]*q[2];
T q13= q[1]*q[3];
T q23= q[2]*q[3];

T f0=from[0]+from[0];
T f1=from[1]+from[1];
T f2=from[2]+from[2];

to[0] += (q12+q03)*f1;
to[0] += (q13-q02)*f2;
to[1] += (q12-q03)*f0;
to[1] += (q23+q01)*f2;
to[2] += (q13+q02)*f0;
to[2] += (q23-q01)*f1;

/*
to[0] = (2*q[0]*q[0]-1+2*q[1]*q[1]) * from[0] +
         2*(q[1]*q[2]+q[0]*q[3]) *    from[1] +
         2*(q[1]*q[3]-q[0]*q[2]) *    from[2];
to[1] =  2*(q[1]*q[2]-q[0]*q[3]) *    from[0] +
        (2*q[0]*q[0]-1+2*q[2]*q[2]) * from[1] +
         2*(q[2]*q[3]+q[0]*q[1]) *    from[2];
to[2] =  2*(q[1]*q[3]+q[0]*q[2])    * from[0] +
         2*(q[2]*q[3]-q[0]*q[1])    * from[1] +
        (2*q[0]*q[0]-1+2*q[3]*q[3]) * from[2];
*/

if(grad)
	{
	grad[0][0]=  q[0]*f0 + q[3]*f1 - q[2]*f2;
	grad[0][1]=  q[1]*f0 + q[2]*f1 + q[3]*f2;
	grad[0][2]= -q[2]*f0 + q[1]*f1 - q[0]*f2;
	grad[0][3]= -q[3]*f0 + q[0]*f1 + q[1]*f2;

        grad[1][0]= -q[3]*f0 + q[0]*f1 + q[1]*f2;
        grad[1][1]=  q[2]*f0 - q[1]*f1 + q[0]*f2;
        grad[1][2]=  q[1]*f0 + q[2]*f1 + q[3]*f2;
        grad[1][3]= -q[0]*f0 - q[3]*f1 + q[2]*f2;

        grad[2][0]=  q[2]*f0 - q[1]*f1 + q[0]*f2;
        grad[2][1]=  q[3]*f0 - q[0]*f1 - q[1]*f2;
        grad[2][2]=  q[0]*f0 + q[3]*f1 - q[2]*f2;
        grad[2][3]=  q[1]*f0 + q[2]*f1 + q[3]*f2;

	}
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

template<typename T>
Quaternion<T> Quaternion<T>::rotate_match(T *to, const T *from, const T *match) const
{
Quaternion<T> grad[3];
Quaternion<T>::rotate(to, from, grad);
Quaternion<T> derivative;
derivative =  grad[0] * (to[0]-match[0]);
derivative += grad[1] * (to[1]-match[1]);
derivative += grad[2] * (to[2]-match[2]);
return derivative;
}



//optionally skip this for microcontrollers if not needed
//note that C++ standard headers automatically use float versions  of the goniometric functions for T=float
template<typename T>
void Quaternion<T>::normquat2eulerzyx(T *e) const
{
e[0]= atan2(2*q[1]*q[2]-2*q[0]*q[3],2*q[0]*q[0]+2*q[1]*q[1]-1);
e[1]= -asin(2*q[1]*q[3]+2*q[0]*q[2]);
e[2]= atan2(2*q[2]*q[3]-2*q[0]*q[1],2*q[0]*q[0]+2*q[3]*q[3]-1);
}


template<typename T>
void Quaternion<T>::normquat2euler(T *eul, const char *type) const
{
Mat3<T> rotmat;
normquat2rotmat(*this,rotmat,false);
rotmat2euler(eul,rotmat,type,false,false,false);
}

//could be done more efficiently since not all rotmat element s are needed in each case, but this is lazy implementation
template<typename T>
void Quaternion<T>::euler2normquat(const T *e, const char *type)
{
T s1=sin(e[0]*(T)0.5);
T s2=sin(e[1]*(T)0.5);
T s3=sin(e[2]*(T)0.5);
T c1=cos(e[0]*(T)0.5);
T c2=cos(e[1]*(T)0.5);
T c3=cos(e[2]*(T)0.5);

switch(Euler_case(type[0],type[1],type[2]))
{

case Euler_case('x','z','x'):
	{
        q[0] = c2*(c1*c3-s1*s3);
        q[1] = c2*(s1*c3+c1*s3);
        q[2] = -s2*(s1*c3-c1*s3);
        q[3] = s2*(c1*c3+s1*s3);
	}
	break;

case Euler_case('x','y','x'):
	{
        q[0] = c2*(c1*c3-s1*s3);
        q[1] = c2*(s1*c3+c1*s3);
        q[2] = s2*(c1*c3+s1*s3);
        q[3] = s2*(s1*c3-c1*s3);
	}
	break;

case Euler_case('y','x','y'):
        {
        q[0] = c2*(c1*c3-s1*s3);
        q[1] = s2*(c1*c3+s1*s3);
        q[2] = c2*(s1*c3+c1*s3);
        q[3] = -s2*(s1*c3-c1*s3);
        }
	break;

case Euler_case('y','z','y'):
        {
        q[0] = c2*(c1*c3-s1*s3);
        q[1] = s2*(s1*c3-c1*s3);
        q[2] = c2*(s1*c3+c1*s3);
        q[3] = s2*(c1*c3+s1*s3);
        }
	break;

case Euler_case('z','y','z'):
        {
        q[0] = c2*(c1*c3-s1*s3);
        q[1] = -s2*(s1*c3-c1*s3);
        q[2] = s2*(c1*c3+s1*s3);
        q[3] = c2*(s1*c3+c1*s3);
        }
	break;

case Euler_case('z','x','z'):
        {
        q[0] = c2*(c1*c3-s1*s3);
        q[1] = s2*(c1*c3+s1*s3);
        q[2] = s2*(s1*c3-c1*s3);
        q[3] = c2*(s1*c3+c1*s3);
        }
	break;

case Euler_case('x','z','y'):
        {
        q[0] = s1*s2*s3 +c1*c2*c3;
        q[1] = s1*c2*c3 - c1*s2*s3;
        q[2] = -s1*s2*c3+c1*c2*s3;
        q[3] = s1*c2*s3 +c1*s2*c3;
        }
	break;

case Euler_case('x','y','z'):
        {
	q[0] = -s1*s2*s3 + c1*c2*c3;
	q[1] = s1*c2*c3 +c1*s2*s3;
	q[2] = -s1*c2*s3 + c1*s2*c3;
	q[3] = s1*s2*c3 + c1*c2*s3;
        }
	break;

case Euler_case('y','x','z'):
        {
        q[0] = s1*s2*s3 + c1*c2*c3;
        q[1] = s1*c2*s3 + c1*s2*c3;
        q[2] = s1*c2*c3 - c1*s2*s3;
        q[3] = -s1*s2*c3 + c1*c2*s3;
        }
	break;

case Euler_case('y','z','x'):
        {
        q[0] = -s1*s2*s3 + c1*c2*c3;
        q[1] = s1*s2*c3 + c1*c2*s3;
        q[2] = s1*c2*c3 + c1*s2*s3;
        q[3] = -s1*c2*s3 + c1*s2*c3;
        }
	break;

case Euler_case('z','y','x'):
        {
	q[0] = c1*c2*c3 + s1*s2*s3;
	q[1] = s1*s2*c3 - c1*c2*s3;
	q[2] = -s1*c2*s3 - c1*s2*c3;
	q[3] = -s1*c2*c3 + c1*s2*s3;
        }
	break;

case Euler_case('z','x','y'):
        {
        q[0] = -s1*s2*s3 + c1*c2*c3;
        q[1] = -s1*c2*s3 + c1*s2*c3;
        q[2] = s1*s2*c3 +  c1*c2*s3;
        q[3] = s1*c2*c3 + c1*s2*s3;
        }
	break;
}//switch
}


//
template<typename T>
void Quaternion<T>::axis2normquat(const T *axis, const T &angle)
{
T a = ((T).5)*angle;
q[0]=cos(a);
T s=sin(a);
q[1]=axis[0]*s;
q[2]=axis[1]*s;
q[3]=axis[2]*s;
}

template<typename T>
void Quaternion<T>::normquat2axis(T *axis,  T &angle) const
{
T s = sqrt(q[1]*q[1] + q[2]*q[2] +q[3]*q[3]);
angle = 2*atan2(s,q[0]);
s= 1/s;
axis[0]= q[1]*s;
axis[1]= q[2]*s;
axis[2]= q[3]*s;
}



template<typename T>
Quaternion<T> exp(const Quaternion<T> &x) 
{
Quaternion<T> r;
T vnorm = sqrt(x[1]*x[1]+x[2]*x[2]+x[3]*x[3]);
r[0] = cos(vnorm);
vnorm = sin(vnorm)/vnorm;
r[1] = x[1] * vnorm;
r[2] = x[2] * vnorm;
r[3] = x[3] * vnorm;
r*= ::exp(x[0]);
return r;
}


//NOTE: log(exp(x))  need not be always = x ... log is not unique!
//NOTE2: log(x*y) != log(y*x) != log(x)+log(y)
template<typename T>
Quaternion<T> log(const Quaternion<T> &x) 
{
Quaternion<T> r;
T vnorm = x[1]*x[1]+x[2]*x[2]+x[3]*x[3];
T xnorm = vnorm + x[0]*x[0];
vnorm = sqrt(vnorm);
xnorm = sqrt(xnorm);
r[0] = ::log(xnorm);
T tmp = acos(x[0]/xnorm)/vnorm;
r[1] = x[1] * tmp;
r[2] = x[2] * tmp;
r[3] = x[3] * tmp;
return r;
}


template<typename T>
Quaternion<T> pow(const Quaternion<T> &x, const T &y)
{
Quaternion<T> r;
T vnorm = x[1]*x[1]+x[2]*x[2]+x[3]*x[3];
T xnorm = vnorm + x[0]*x[0];
vnorm = sqrt(vnorm);
xnorm = sqrt(xnorm);
T phi = acos(x[0]/xnorm);
r[0] = cos(y*phi);
T tmp = sin(y*phi)/vnorm;
r[1] = x[1] * tmp;
r[2] = x[2] * tmp;
r[3] = x[3] * tmp;
r *= ::pow(xnorm,y);
return r;
}


#ifndef QUAT_NO_RANDOM
//method to efficiently generate a uniformly random point on a unit 4-sphere by G. Marsaglia, Ann. Math. Stat. 43, 645 (1972)
template<typename T>
void Quaternion<T>::random_rotation()
{
T  s1,s2,s;
do
	{
	q[0]= RANDDOUBLESIGNED();
	q[1]= RANDDOUBLESIGNED();
	s1 = q[0]*q[0] + q[1]*q[1];
	}
	while(s1>1);
do
        {
        q[2]= RANDDOUBLESIGNED();
        q[3]= RANDDOUBLESIGNED();
        s2 = q[2]*q[2] + q[3]*q[3];
        }
        while(s2>1);
s = sqrt((1-s1)/s2);
q[2] *= s;
q[3] *= s;
}
#endif

//force instantization
#define INSTANTIZE(T) \
template class Quaternion<T>; \
template Quaternion<T> pow(const Quaternion<T> &x, const T &y); \
template Quaternion<T> log(const Quaternion<T> &x); \
template Quaternion<T> exp(const Quaternion<T> &x); \



INSTANTIZE(float)
#ifndef QUAT_NO_DOUBLE
INSTANTIZE(double)
#endif

} //namespace
