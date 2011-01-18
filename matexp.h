/*
    LA: linear algebra C++ interface library
    Copyright (C) 2008 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>

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
#ifndef _MATEXP_H_
#define _MATEXP_H_
//general routine for polynomial of a matrix, tuned to minimize the number
//of matrix-matrix multiplications on cost of additions and memory
// the polynom and exp routines will work on any type, for which traits class
// is defined containing definition of an element type, norm and axpy operation
#include "la_traits.h"
#include "laerror.h"
#include <math.h>

namespace LA {

template<class T,class R>
const T polynom0(const T &x, const NRVec<R> &c)
{
int order=c.size()-1;
T z,y;

//trivial reference implementation by horner scheme
if(order==0) {y=x; y=c[0];} //to avoid the problem: we do not know the size of the matrix to contruct a scalar one
else
	{
	int i;
	z=x*c[order];
	for(i=order-1; i>=0; i--)
		{
		//std::cerr<<"TEST polynom0 "<<i<<'\n';
		if(i<order-1) {LA_traits<T>::deallocate(z); z=y*x;} //for large matrices avoid storing 4 ones simultaneously
		LA_traits<T>::deallocate(y); y=z+c[i];
		}
	}

return y;
}



//algorithm which minimazes number of multiplications, at the cost of storage
template<class T,class R>
const T polynom(const T &x, const NRVec<R> &c)
{
int n=c.size()-1;
int i,j,k,m=0,t;

if(n<=4) return polynom0(x,c); //here the horner scheme is optimal

//first find m which minimizes the number of multiplications
j=10*n;
for(i=2;i<=n+1;i++)
    {	
    t=i-2+2*(n/i)-(n%i)?0:1;
    if(t<j)
	{
	j=t;
	m=i;
	}
    }


//allocate array for powers up to m
T *xpows = new T[m];
xpows[0]=x;
for(i=1;i<m;i++) xpows[i]=xpows[i-1]*x;


//run the summation loop
T r,s,f;
k= -1;
for(i=0; i<=n/m;i++)
	{
	for(j=0;j<m;j++)
		{
		k++;
		if(k>n) break;
		if(j==0) {
			  if(i==0) s=x; /*just to get the dimensions of the matrix*/ 
			  s=c[k]; /*create diagonal matrix*/
			  }
		else  
			LA_traits<T>::axpy(s,xpows[j-1],c[k]); //general  s+=xpows[j-1]*c[k]; but more efficient for matrices
		}

	if(i==0) {r=s; f=xpows[m-1];}
	else
		{
		r+= s*f;
		f=f*xpows[m-1];
		}
	}
 
delete[] xpows;
return r;
}


//for general objects
template<class T>
const T ncommutator ( const T &x, const T &y, int nest=1, const bool right=1)
{
T z;
if(right) {z=x; while(--nest>=0) z=z*y-y*z;}
else {z=y; while(--nest>=0) z=x*z-z*x;}
return z;
}

template<class T>
const T nanticommutator ( const T &x, const T &y, int nest=1, const bool right=1)
{
T z;
if(right) {z=x; while(--nest>=0) z=z*y+y*z;}
else {z=y; while(--nest>=0) z=x*z+z*x;}
return z;
}

//general BCH expansion (can be written more efficiently in a specialization for matrices)
template<class T>
const T BCHexpansion (const T &h, const T &t, const int n, const bool verbose=0)\
{
T result=h;
double factor=1.;
T z=h;
for(int i=1; i<=n; ++i)
	{
	factor/=i;
	z= z*t-t*z;
	if(verbose) std::cerr << "BCH contribution at order "<<i<<" : "<<z.norm()*factor<<std::endl;
	result+= z*factor; 
	}
return result;
}


template<class T>
const T ipow( const T &x, int i)
{
if(i<0) laerror("negative exponent in ipow");
if(i==0) {T r=x; r=(typename LA_traits<T>::elementtype)1; return r;}//trick for matrix dimension
if(i==1) return x;
T y,z;
z=x;
while(!(i&1))
	{
	z = z*z;
	i >>= 1;
	}
y=z; 
while((i >>= 1)/*!=0*/)
                {
                z = z*z;
                if(i&1) y = y*z;
                }
return y;
}

inline int nextpow2(const double n)
{
const double log2=std::log(2.);
if(n<=.75) return 0; //try to keep the taylor expansion short
if(n<=1.) return 1;
return int(std::ceil(std::log(n)/log2-std::log(.75)));
}

//should better be computed by mathematica to have accurate last digits, perhaps chebyshev instead, see exp in glibc
//is shared also for sine and cosine now
static const double exptaylor[]={
1.,
1.,
0.5,
0.1666666666666666666666,
0.0416666666666666666666,
0.0083333333333333333333,
0.0013888888888888888888,
0.00019841269841269841253,
2.4801587301587301566e-05,
2.7557319223985892511e-06,
2.7557319223985888276e-07,
2.5052108385441720224e-08,
2.0876756987868100187e-09,
1.6059043836821613341e-10,
1.1470745597729724507e-11,
7.6471637318198164055e-13,
4.7794773323873852534e-14,
2.8114572543455205981e-15,
1.5619206968586225271e-16,
8.2206352466243294955e-18,
4.1103176233121648441e-19,
1.9572941063391262595e-20,
0.};


//S is element type of T, but T may be any user-defined
template<class T, class C, class S>
NRVec<C> exp_aux(const T &x, int &power, int maxpower, int maxtaylor, S prescale)
{

double mnorm= x.norm() * std::abs(prescale);
power=nextpow2(mnorm); 
if(maxpower>=0 && power>maxpower) power=maxpower;
double scale=std::exp(-std::log(2.)*power);


//find how long taylor expansion will be necessary
const double precision=1e-14; //further decreasing brings nothing
double s,t;
s=mnorm*scale;
int n=0;
t=1.;
do	{
	n++;
	t*=s;
	}
while(t*exptaylor[n]>precision);//taylor 0 will terminate in any case

if(maxtaylor>=0 && n>maxtaylor) n=maxtaylor; //useful e.g. if the matrix is nilpotent in order n+1 as the CC T operator for n electrons


int i; //adjust the coefficients in order to avoid scaling the argument
NRVec<C> taylor2(n+1);
for(i=0,t=1.;i<=n;i++)
	{
	taylor2[i]=exptaylor[i]*t;
	t*=scale;
	}
//std::cout <<"TEST power, scale "<<power<<" "<<scale<<std::endl; 
//std::cout <<"TEST taylor2 "<<taylor2<<std::endl;
return taylor2;
}



template<class T, class C, class S>
void sincos_aux(NRVec<C> &si, NRVec<C> &co, const T &x, int &power,int maxpower, int maxtaylor, const S prescale)
{
double mnorm= x.norm() * std::abs(prescale);
power=nextpow2(mnorm); 
if(maxpower>=0 && power>maxpower) power=maxpower;
double scale=std::exp(-std::log(2.)*power);

//find how long taylor expansion will be necessary
const double precision=1e-14; //further decreasing brings nothing
double s,t;
s=mnorm*scale;
int n=0;
t=1.;
do	{
	n++;
	t*=s;
	}
while(t*exptaylor[n]>precision);//taylor 0 will terminate in any case

if(maxtaylor>=0 && n>maxtaylor) n=maxtaylor; //useful e.g. if the matrix is nilpotent in order n+1 as the CC T operator for n electrons
if((n&1)==0) ++n; //force it to be odd to have same length in sine and cosine
si.resize((n+1)/2);
co.resize((n+1)/2);

int i; //adjust the coefficients in order to avoid scaling the argument
for(i=0,t=1.;i<=n;i++)
	{
	if(i&1) si[i>>1] = exptaylor[i]* (i&2?-t:t);
	else 	co[i>>1] = exptaylor[i]* (i&2?-t:t);
	t*=scale;
	}
//std::cout <<"TEST sin "<<si<<std::endl;
//std::cout <<"TEST cos "<<co<<std::endl;
}



//it seems that we do not gain anything by polynom vs polynom0, check the m-optimization!
template<class T>
const T exp(const T &x, bool horner=true, int maxpower= -1, int maxtaylor= -1 )
{
int power;

//prepare the polynom of and effectively scale T
NRVec<typename LA_traits<T>::normtype> taylor2=exp_aux<T,typename LA_traits<T>::normtype,double>(x,power,maxpower,maxtaylor,1.);

//std::cerr <<"TEST power "<<power<<std::endl;

T r= horner?polynom0(x,taylor2):polynom(x,taylor2); 
//for accuracy summing from the smallest terms up would be better, but this is more efficient for matrices

//power the result back
for(int i=0; i<power; i++) r=r*r;
return r;
}


//make exp(iH) with real H in real arithmetics
template<class T>
void sincos(T &s, T &c, const T &x, bool horner=true, int maxpower= -1, int maxtaylor= -1 )
{
int power;

NRVec<typename LA_traits<T>::normtype> taylors,taylorc;
sincos_aux<T,typename LA_traits<T>::normtype>(taylors,taylorc,x,power,maxpower,maxtaylor,1.);


//could we save something by computing both polynoms simultaneously?
{
T x2 = x*x;
s = horner?polynom0(x2,taylors):polynom(x2,taylors);
c = horner?polynom0(x2,taylorc):polynom(x2,taylorc);
}
s = s * x;

//power the results back
for(int i=0; i<power; i++)
	{
	T tmp = c*c - s*s;
	s = s*c; s *= 2.;
	c=tmp;
	}
}



//this simple implementation seems not to be numerically stable enough
//and probably not efficient either

template<class M, class V, class MEL>
void exptimesdestructive(const M &mat, V &result, V &rhs,  bool transpose, const MEL scale, int maxpower= -1, int maxtaylor= -1, bool mat_is_0=false) //uses just matrix vector multiplication
{
if(mat_is_0) {result = rhs; LA_traits<V>::copyonwrite(result); return;} //prevent returning a shallow copy of rhs
if(mat.nrows()!=mat.ncols()||(unsigned int) mat.nrows() != (unsigned int)rhs.size()) laerror("inappropriate sizes in exptimes");

int power;
//prepare the polynom of and effectively scale the matrix
NRVec<typename LA_traits<V>::normtype> taylor2=exp_aux<M,typename LA_traits<V>::normtype>(mat,power,maxpower,maxtaylor,scale);

V tmp;
bool washere=0;

for(int i=1; i<=(1<<power); ++i) //unfortunatelly, here we have to repeat it many times, unlike if the matrix is stored explicitly
	{
	washere=1;
	if(i>1) rhs=result; //apply again to the result of previous application
	else result=rhs;
	tmp=rhs; //now rhs can be used as scratch	
	result*=taylor2[0];
	for(int j=1; j<taylor2.size(); ++j)
		{
		mat.gemv(0.,rhs,transpose?'t':'n',scale,tmp);
		tmp=rhs;
		result.axpy(taylor2[j],tmp);
		}
	}

if(!washere) laerror("integer overflow due to unrealistically big power - use maxpower argument in exptimes()");

return;
}


//actually scale should be elementtype of M, but we do not have it since M can be anything user-defined
//and template paramter for it does not work due to optional arguments
//undecent solution: exptimesreal
//
template<class M, class V>
const V exptimes(const M &mat, V rhs, bool transpose=false, const typename LA_traits<V>::elementtype scale=1., int maxpower= -1, int maxtaylor= -1, bool mat_is_0=false )
{
V result;
exptimesdestructive(mat,result,rhs,transpose,scale,maxpower,maxtaylor,mat_is_0);
return result;
}

template<class M, class V>
const V exptimesreal(const M &mat, V rhs, bool transpose=false, const typename LA_traits<V>::normtype scale=1., int maxpower= -1, int maxtaylor= -1, bool mat_is_0=false )
{
V result;
exptimesdestructive(mat,result,rhs,transpose,scale,maxpower,maxtaylor,mat_is_0);
return result;
}








template<class M, class V, class S>
void sincostimes_simple(const M &mat, V &si, V &co, const V &rhs, const NRVec<typename LA_traits<V>::normtype> &taylors, const NRVec<typename LA_traits<V>::normtype> &taylorc, bool transpose, const S scale)
{
si=rhs * taylors[0];
co=rhs * taylorc[0];
V tmp=rhs;
for(int j=1; j<taylors.size(); ++j)
        {
	V tmp2(tmp.size());
	//multiply by a square of the matrix
        mat.gemv(0.,tmp2,transpose?'t':'n',scale,tmp);
	mat.gemv(0.,tmp,transpose?'t':'n',scale,tmp2);
        si.axpy(taylors[j],tmp);
        co.axpy(taylorc[j],tmp);
        }
mat.gemv(0.,tmp,transpose?'t':'n',scale,si);
si=tmp;
}

//this recursion is very inefficient, it is better to use complex exptimes!
template<class M, class V, class S>
void sincostimes_aux(const M &mat, V &si, V &co, const V &rhs, const NRVec<typename LA_traits<V>::normtype> &taylors, const NRVec<typename LA_traits<V>::normtype> &taylorc, bool transpose, const S scale, int power)
{
if(power==0) sincostimes_simple(mat,si,co,rhs,taylors,taylorc,transpose,scale);
else
	{
	V si2,co2; //no large memory allocated yet - size 0
	sincostimes_aux(mat,si2,co2,rhs,taylors,taylorc,transpose,scale,power-1);
	sincostimes_aux(mat,si,co,co2,taylors,taylorc,transpose,scale,power-1);
	V ss,cs;
	sincostimes_aux(mat,ss,cs,si2,taylors,taylorc,transpose,scale,power-1);
	co -= ss;
	si += cs;
	}
}


//inefficient, it is better to use complex exptimes!
//again scale should actually be elementtype of M which is inaccessible
template<class M, class V>
void sincostimes(const M &mat, V &si, V &co, const V &rhs,  bool transpose=false, const typename LA_traits<V>::normtype scale=1., int maxpower= -1, int maxtaylor= -1, bool mat_is_0=false) //uses just matrix vector multiplication
{
if(mat_is_0) //prevent returning a shallow copy of rhs
	{
	co = rhs; 
	LA_traits<V>::copyonwrite(co); 
	LA_traits<V>::clearme(si);
	return;
	}

if(mat.nrows()!=mat.ncols()||(unsigned int) mat.nrows() != (unsigned int)rhs.size()) laerror("inappropriate sizes in sincostimes");

//prepare the polynom of and effectively scale the matrix
int power;
NRVec<typename LA_traits<V>::normtype> taylors,taylorc;
sincos_aux<M,typename LA_traits<V>::normtype>(taylors,taylorc,mat,power,maxpower,maxtaylor,scale);
if(taylors.size()!=taylorc.size()) laerror("internal error - same size of sin and cos expansions assumed");
//the actual computation and resursive "squaring"
//std::cout <<"TEST power "<<power<<std::endl;
sincostimes_aux(mat,si,co,rhs,taylors,taylorc,transpose,scale,power);

return;
}


//@@@ power series matrix logarithm?

}//namespace
#endif
