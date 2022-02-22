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
if(rem) 
	{
	if(rem<0) {--div; rem+=q;} //prevent negative a_i i>0
	cf_helper(me,q,rem,level+1);
	}
else me->resize(level);
}
(*me)[level]=div;
}

template <typename T>
ContFrac<T>::ContFrac(T p, T q) : NRVec<T>()
{
if(q<0) {p= -p; q= -q;}
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


//compare assuming they are canonical
template <typename T>
T ContFrac<T>::compare(const ContFrac<T> &rhs) const
{
int l=length();
if(rhs.length()<l) l=rhs.length();
for(int i=0; i<=l; ++i)
	{
	T d=(*this)[i]-rhs[i];
	if(d) return (i&1)? -d :d;
	}
if(length()==rhs.length()) return 0;
else if(length()<rhs.length()) return (length()&1) ? 1 : -1;
else return (rhs.length()&1) ? -1 : 1;
}


template <typename T>
void ContFrac<T>::canonicalize()
{
int n=this->length();
if(n==0) return;
if(n>0 && (*this)[1]<0) //handle negative a_i i>0
	{
	for(int i=0; i<=n; ++i) (*this[i]) = -(*this[i]);
	*this = -(*this);
	}
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
Homographic<T> Homographic<T>::input(const T &z, const bool inf) const
{
Homographic<T> hnew;
if(inf) //effective infinity, end of input
                {
                hnew.v[0][0]=hnew.v[0][1]=v[0][1];
                hnew.v[1][0]=hnew.v[1][1]=v[1][1];
                }
else
                {
                hnew.v[0][0]=v[0][1];
                hnew.v[1][0]=v[1][1];
                hnew.v[0][1]=v[0][0]+v[0][1]* z;
                hnew.v[1][1]=v[1][0]+v[1][1]* z;
                }
return hnew;
}

template <typename T>
Homographic<T> Homographic<T>::output(const T &z) const
{
Homographic<T> hnew;
                hnew.v[0][0]=v[1][0];   
                hnew.v[0][1]=v[1][1];   
                hnew.v[1][0]=v[0][0]-v[1][0]*z;
                hnew.v[1][1]=v[0][1]-v[1][1]*z;
return hnew;
}


template <typename T>
bool Homographic<T>::outputready(T &z,bool first) const
{
bool inf=0;
T q0,q1;
if(v[1][0]==0) inf=1; else q0=v[0][0]/v[1][0];
if(v[1][1]==0) inf=1; else q1=v[0][1]/v[1][1];
if(!inf && q0==q1) 
	{
	z=q0; 
	if(first && q0<0) --z; //prevent negative a1 etc.
	return true;
	}
return false;
}

template <typename T>
bool Homographic<T>::terminate() const
{
return v[1][0]==0&&v[1][1]==0;
}


template <typename T>
ContFrac<T> Homographic<T>::value(const ContFrac<T>&x) const
{
Homographic<T> h(*this);

std::list<T> l;
bool first=true;
for(typename ContFrac<T>::iterator px=x.begin(); px!=x.beyondend(); ++px)
	{
	//digest next input term
	h=h.input(*px,px==x.end()|| px!=x.begin()&& *px==0);

	//output as much as possible
	T out;
	while(h.outputready(out,first))
		{
		l.push_back(out);
		h=h.output(out);
		first=false;
		}

	//terminate if exhausted
	if(h.terminate())
		{
		if(px!=x.end()) laerror("unexpected termination in Homographic::value");
		break;
		}
	}
if(l.back()==1)  //simplify by removing a trailing 1
	{
	l.pop_back();
	l.back()+=1;
	}
return ContFrac<T>(l);
}



template <typename T>
BiHomographic<T> BiHomographic<T>::inputx(const T &x, const bool inf) const
{
BiHomographic<T> hnew;
for(int i=0; i<2; ++i)
	{
	hnew.v[i][0][0]= v[i][0][1];
	hnew.v[i][0][1]= inf?v[i][0][1] : v[i][0][0]+v[i][0][1]*x;
	hnew.v[i][1][0]= v[i][1][1];
	hnew.v[i][1][1]= inf?v[i][1][1] : v[i][1][0]+v[i][1][1]*x;
	}
return hnew;
}


template <typename T>
BiHomographic<T> BiHomographic<T>::inputy(const T &y, const bool inf) const
{
BiHomographic<T> hnew;
for(int i=0; i<2; ++i)
        {
        hnew.v[i][0][0]= v[i][1][0];
        hnew.v[i][0][1]= v[i][1][1];
        hnew.v[i][1][0]= inf?v[i][1][0] : v[i][0][0]+v[i][1][0]*y;
        hnew.v[i][1][1]= inf?v[i][1][1] : v[i][0][1]+v[i][1][1]*y;
        }
return hnew;
}


template <typename T>
BiHomographic<T> BiHomographic<T>::output(const T &z) const
{
BiHomographic<T> hnew;
for(int i=0; i<2; ++i) for(int j=0; j<2; ++j)
	{
	hnew.v[0][i][j]= v[1][i][j];
	hnew.v[1][i][j]= v[0][i][j] - v[1][i][j]*z;
	}
return hnew;
}

template <typename T>
int BiHomographic<T>::inputselect() const
{
if(v[1][0][0]==0)
	{
	if(v[1][0][1]==0) return 1;
	else return 0;
	}
if(v[1][0][1]==0) return 0;
if(v[1][1][0]==0) return 1;
if(MYABS(v[0][0][1]/v[1][0][1] - v[0][0][0]/v[1][0][0]) > MYABS(v[0][1][0]/v[1][1][0] - v[0][0][0]/v[1][0][0])) return 0;
return 1;
}


template <typename T>
bool BiHomographic<T>::outputready(T &z,bool first) const
{
T q[2][2];
for(int i=0; i<2; ++i) for(int j=0; j<2; ++j)
        {
	if(v[1][i][j]==0) return false;
	else q[i][j]=v[0][i][j]/v[1][i][j];
	if(q[i][j]!=q[0][0])  return false;
	}
z=q[0][0]; 
if(first && z<0) --z;
return true;
}


template <typename T>
bool BiHomographic<T>::terminate() const
{
return  v[1][0][0]==0&&v[1][0][1]==0&&v[1][1][0]==0&&v[1][1][1]==0;
}


template <typename T>
ContFrac<T>  BiHomographic<T>::value(const ContFrac<T>&x, const ContFrac<T>&y) const
{
BiHomographic<T> h(*this);

std::list<T> l;
typename ContFrac<T>::iterator px=x.begin();
typename ContFrac<T>::iterator py=y.begin();

bool first=true;
do
	{
	//select next input term
	int which;
	if(px==x.beyondend()) which=1;
	else if(py==y.beyondend()) which=0;
	else which = h.inputselect();

	if(which) {h=h.inputy(*py,py==y.end()|| py!=y.begin()&& *py==0); ++py;}
	else {h=h.inputx(*px,px==x.end()|| px!=x.begin()&& *px==0); ++px;}

	//output as much as possible
	T out;
	while(h.outputready(out,first))
		{
		l.push_back(out);
		h=h.output(out);
		first=false;
		}

	//terminate if exhausted
	if(h.terminate())
		{
		if(px!=x.end()&&px!=x.beyondend() || py!=y.end()&&py!=y.beyondend()) laerror("unexpected termination in Homographic::value");
		break;
		}
	}
	while(px!=x.beyondend() || py!=y.beyondend());

if(l.back()==1)  //simplify by removing a trailing 1
        {
        l.pop_back();
        l.back()+=1;
        }

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
if(MYABS(g)>1)
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
if(MYABS(g)>1)
        {
        r/=g;
        num/=g;
        }
den*=r;
return *this;
}


//try avoiding overflows at the cost of speed
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
if(MYABS(g)>1)
	{
	num/=g;
	r.den/=g;
	}
g=gcd(den,r.num);
if(MYABS(g)>1)
        {
        den/=g;
        r.num/=g;
        }
num*=r.num;
den*=r.den;
if(den<0) {den= -den; num= -num;}
return *this;
}


//unary -
template <typename T>
ContFrac<T> ContFrac<T>::operator-() const
{
int l=length();
if(l==0)
	{
	ContFrac<T> r(0);
	r[0]= -(*this)[0];
	return r;
	}
if((*this)[1]!=1)
	{
	ContFrac<T> r(l+1);
	r[0]= -(*this)[0]-1;
	r[1]= 1;
	r[2]= (*this)[1]-1;
	for(int i=2; i<=l; ++i) r[i+1] = (*this)[i];
	return r;
	}
else //a_1-1 == 0
	{
	if(l==1) //we have trailing 0, actually the input was not canonical
		{
		ContFrac<T> r(0);
        	r[0]= -(*this)[0]-1;
		return r;
		}
	else
		{
		ContFrac<T> r(l-1);
		r[0]= -(*this)[0]-1;
		r[1]= 1+(*this)[2];
		for(int i=3; i<=l; ++i) r[i-1] = (*this)[i];
		return r;
		}
	}
}



/***************************************************************************//**
 * forced instantization in the corresponding object file
 ******************************************************************************/
template class Rational<int>;
template class Rational<long>;
template class Rational<long long>;

template class ContFrac<int>;
template class ContFrac<long>;
template class ContFrac<long long>;

template class Homographic<int>;
template class Homographic<long>;
template class Homographic<long long>;

template class BiHomographic<int>;
template class BiHomographic<long>;
template class BiHomographic<long long>;




#define INSTANTIZE(T) \



INSTANTIZE(int)
INSTANTIZE(unsigned int)



}//namespace
