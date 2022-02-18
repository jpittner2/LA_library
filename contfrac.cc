/*
    LA: linear algebra C++ interface library
    Copyright (C) 2022 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>

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

#include "contfrac.h"
#include <stdio.h>
#include <string.h>
#include <math.h>


namespace LA {

template <typename T>
ContFrac<T>::ContFrac(double x, const int n, const T thres) : NRVec<T>(n+1)
{
for(int i=0; i<=n; ++i)
	{
	NRVec<T>::v[i]=floor(x);
	x -= NRVec<T>::v[i];
	double y= 1./x;
	if(x==0. || (thres && fabs(y)>thres)) {resize(i,true); return;}
	x=y;
	}
}


//we have to recursively first determine length and then allocate and fill the values during recursion unwinding
template <typename T>
static void cf_helper(ContFrac<T> *me, T p, T q, int level)
{
T div=p/q;
{
T rem=p%q;
if(rem) cf_helper(me,q,rem,level+1);
else me->resize(level);
}
(*me)[level]=div;
}

template <typename T>
ContFrac<T>::ContFrac(const T p, const T q) : NRVec<T>()
{
cf_helper<T>(this,p,q,0);
}


template <typename T>
ContFrac<T> ContFrac<T>::reciprocal() const
{
int n=this->length();
if((*this)[0] == 0)
	{
	ContFrac<T> r(n-1);
	for(int i=1; i<=n; ++i) r[i-1] = (*this)[i];
	return r;
	}
else
	{
	ContFrac<T> r(n+1);
	r[0]=0;
	for(int i=0; i<=n; ++i) r[i+1] = (*this)[i];
	return r;
	}
}


template <typename T>
void ContFrac<T>::convergent(T *p, T*q, const int trunc) const
{
int top=this->length();
if(trunc != -1) top=trunc;
NRVec<T> hh(top+3),kk(top+3);
T *h= &hh[2]; 
T *k= &kk[2];
//start for recurrent relations
h[-2]=k[-1]=0;
h[-1]=k[-2]=1;
for(int i=0; i<=top; ++i)
	{
	if(i>0 && (*this)[i]==0) //terminate by 0 which means infinity if not canonically shortened
		{
		*p=h[i-1];
		*q=k[i-1];
		return;
		}
	h[i] = (*this)[i]*h[i-1] + h[i-2];
	k[i] = (*this)[i]*k[i-1] + k[i-2];
	}
*p=h[top];
*q=k[top];
}

template <typename T>
double ContFrac<T>::value(const int trunc) const
{
T p,q;
convergent(&p,&q,trunc);
double x=p;
x/=q;
return x;
}


template <typename T>
void ContFrac<T>::canonicalize()
{
int n=this->length();
if(n==0) return;
if((*this)[n]==1) {(*this)[n]=0; ++(*this)[n-1];} //avoid deepest 1/1
for(int i=1; i<=n; ++i) //truncate if possible
	{
	if((*this)[i]==0) //convention for infinity
		{
		resize(i-1,true);
		return;
		}
	}
}


/***************************************************************************//**
 * forced instantization in the corresponding object file
 ******************************************************************************/
template class ContFrac<int>;
template class ContFrac<unsigned int>;
template class ContFrac<long>;
template class ContFrac<unsigned long>;
template class ContFrac<long long>;
template class ContFrac<unsigned long long>;


#define INSTANTIZE(T) \



INSTANTIZE(int)
INSTANTIZE(unsigned int)



}//namespace
