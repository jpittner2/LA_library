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
#include <ctype.h>

using namespace LA_Vecmat3;

//cf. https://en.wikipedia.org/wiki/Euler_angles
template<typename T>
void euler2rotmat(const Vec3<T> &eul, Mat3<T> a, char *type, bool transpose, bool direction, bool reverse)
{
T c2=cos(eul[1]); 
T s2=sin(eul[1]);
T c1=cos(eul[reverse?2:0]);
T s1=sin(eul[reverse?2:0]);
T c3=cos(eul[reverse?0:2]);
T s3=sin(eul[reverse?0:2]);

if(direction) {s1= -s1; s2= -s2; s3= -s3;}

if(tolower(type[0])=='x' && tolower(type[1])=='z'  && tolower(type[2])=='x')
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

if(tolower(type[0])=='x' && tolower(type[1])=='y'  && tolower(type[2])=='x')
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

if(tolower(type[0])=='y' && tolower(type[1])=='x'  && tolower(type[2])=='y')
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

if(tolower(type[0])=='y' && tolower(type[1])=='z'  && tolower(type[2])=='y')
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

if(tolower(type[0])=='z' && tolower(type[1])=='y'  && tolower(type[2])=='z')
        {
        a[0][0]= c1*c2*c3-s1*s3;
        a[0][1]= -c3*s1;
        a[0][2]= c1*s2;
        a[1][0]= c1*s3+c2*c3*s1;
        a[1][1]= c1*c3-c2*s1*s3;
        a[1][2]= s1*s2;
        a[2][0]= -c3*s2;
        a[2][1]= s2*s3;
        a[2][2]= c2;
        }

if(tolower(type[0])=='z' && tolower(type[1])=='x'  && tolower(type[2])=='z')
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

//improper angles Tait-Bryan

if(tolower(type[0])=='x' && tolower(type[1])=='z'  && tolower(type[2])=='y')
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

if(tolower(type[0])=='x' && tolower(type[1])=='y'  && tolower(type[2])=='z')
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

if(tolower(type[0])=='y' && tolower(type[1])=='x'  && tolower(type[2])=='z')
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

if(tolower(type[0])=='y' && tolower(type[1])=='z'  && tolower(type[2])=='x')
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

if(tolower(type[0])=='z' && tolower(type[1])=='y'  && tolower(type[2])=='x')
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

if(tolower(type[0])=='z' && tolower(type[1])=='x'  && tolower(type[2])=='y')
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

if(transpose) a.transposeme();
}

