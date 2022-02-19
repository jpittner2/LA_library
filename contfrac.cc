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
#include "permutation.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <list>


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
this->copyonwrite();
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


template <typename T>
ContFrac<T> Homographic<T>::value(const ContFrac<T>&x) const
{
Homographic<T> h(*this);

std::list<T> l;
typename ContFrac<T>::iterator px=x.begin();

do //scan all input
	{
	//digest next input term
	Homographic<T> hnew;
	if(px==x.end()|| px!=x.begin()&& *px==0) //input is infinity
		{
		hnew.x[0][0]=hnew.x[0][1]=h.x[0][1];
                hnew.x[1][0]=hnew.x[1][1]=h.x[1][1];
		}
	else
		{
		hnew.x[0][0]=h.x[0][1];
		hnew.x[1][0]=h.x[1][1];
		hnew.x[0][1]=h.x[0][0]+h.x[0][1]* *px;
        	hnew.x[1][1]=h.x[1][0]+h.x[1][1]* *px;
		}

	//std::cout<<"hnew\n"<< hnew.x[0][0]<<" "<< hnew.x[0][1]<<"\n"<< hnew.x[1][0]<<" "<< hnew.x[1][1]<<"\n";

	//check if we are ready to output
	bool inf=0;	
	T q0,q1;
	if(hnew.x[1][0]==0) inf=1; else q0=hnew.x[0][0]/hnew.x[1][0];
	if(hnew.x[1][1]==0) inf=1; else q1=hnew.x[0][1]/hnew.x[1][1];
	while(!inf && q0==q1) //ready to output
		{
		l.push_back(q0);
		//std::cout <<"out "<<q0<<std::endl;	
		Homographic<T> hnew2;
		hnew2.x[0][0]=hnew.x[1][0];	
		hnew2.x[0][1]=hnew.x[1][1];	
		hnew2.x[1][0]=hnew.x[0][0]-hnew.x[1][0]*q0;
		hnew2.x[1][1]=hnew.x[0][1]-hnew.x[1][1]*q0;
		//std::cout<<"hnew2\n"<< hnew2.x[0][0]<<" "<< hnew2.x[0][1]<<"\n"<< hnew2.x[1][0]<<" "<< hnew2.x[1][1]<<"\n";
		hnew=hnew2;
		inf=0;
		if(hnew.x[1][0]==0) inf=1; else q0=hnew.x[0][0]/hnew.x[1][0];
        	if(hnew.x[1][1]==0) inf=1; else q1=hnew.x[0][1]/hnew.x[1][1];
		}
	//termination
	if(hnew.x[1][0]==0&&hnew.x[1][1]==0) //terminate
		{
		//std::cout<<"terminate at "<<i<<"\n";
		if(px!=x.end()) laerror("unexpected termination in Homographic::value");
		break;
		}
	h=hnew;
	++px;
	} while(px!=x.beyondend());
//std::cout <<"size= "<<l.size()<<std::endl;
return ContFrac<T>(l);
}


template <typename T>
void Rational<T>::simplify()
{
if(den<0)
	{
	num= -num;
	den= -den;
	}
T g=gcd(num,den); 
if(g>1) 
	{
	num/=g; 
	den/=g;
	}
}

template <typename T>
Rational<T> & Rational<T>::operator*=(const T  &rhs)
{
T r=rhs;
T g=gcd(r,den);
if(g>1)
        {
        r/=g;
        den/=g;
        }
num*=r;
return *this;
}


template <typename T>
Rational<T> & Rational<T>::operator/=(const T  &rhs)
{
T r=rhs;
T g=gcd(r,num);
if(g>1)
        {
        r/=g;
        num/=g;
        }
den*=r;
return *this;
}


template <typename T>
Rational<T>  Rational<T>::operator+(const Rational &rhs) const
{
Rational r;
r.den = lcm(den,rhs.den);
r.num = num*(r.den/den) + rhs.num*(r.den/rhs.den);
r.simplify();
return r;
}

template <typename T>
Rational<T>  Rational<T>::operator-(const Rational &rhs) const
{
Rational r;
r.den = lcm(den,rhs.den);
r.num = num*(r.den/den) - rhs.num*(r.den/rhs.den);
r.simplify();
return r;
}


template <typename T>
Rational<T> & Rational<T>::operator*=(const Rational &rhs)
{
Rational r(rhs);
T g;
g=gcd(num,r.den);
if(g>1)
	{
	num/=g;
	r.den/=g;
	}
g=gcd(den,r.num);
if(g>1)
        {
        den/=g;
        r.num/=g;
        }
num*=r.num;
den*=r.den;
return *this;
}




/***************************************************************************//**
 * forced instantization in the corresponding object file
 ******************************************************************************/
template class Rational<int>;
template class Rational<unsigned int>;
template class Rational<long>;
template class Rational<unsigned long>;
template class Rational<long long>;
template class Rational<unsigned long long>;

template class ContFrac<int>;
template class ContFrac<unsigned int>;
template class ContFrac<long>;
template class ContFrac<unsigned long>;
template class ContFrac<long long>;
template class ContFrac<unsigned long long>;

template class Homographic<int>;
template class Homographic<unsigned int>;
template class Homographic<long>;
template class Homographic<unsigned long>;
template class Homographic<long long>;
template class Homographic<unsigned long long>;

template class BiHomographic<int>;
template class BiHomographic<unsigned int>;
template class BiHomographic<long>;
template class BiHomographic<unsigned long>;
template class BiHomographic<long long>;
template class BiHomographic<unsigned long long>;




#define INSTANTIZE(T) \



INSTANTIZE(int)
INSTANTIZE(unsigned int)



}//namespace
