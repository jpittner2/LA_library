/*
    LA: linear algebra C++ interface library
    Copyright (C) 2021 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>

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

#include "polynomial.h"
#include <stdio.h>
#include <string.h>


namespace LA {

template <typename T>
void Polynomial<T>::polydiv(const Polynomial &rhs, Polynomial &q, Polynomial &r) const
{
if(rhs[rhs.degree()]==(T)0) laerror("division by a polynomial with zero leading coefficient - simplify it first");
if(rhs.degree()==0) //scalar division
	{
	q= *this/rhs[0];
	r.resize(0,false);
	r[0]=0;
	return;
	}
int rdegree= rhs.degree();
int qdegree= degree()-rdegree;
if(qdegree<0)
	{
	q.resize(0,false);
	q[0]=0;
	r= *this;
	return;
	}
//general case
q.resize(qdegree,false);
r= *this; r.copyonwrite();
for(int i=degree(); i>=rdegree; --i)
	{
	T tmp= r[i]/rhs[rdegree];
	q[i-rdegree]= tmp;
	r -= rhs.shifted(i-rdegree)*tmp;
	}
r.resize(rhs.degree()-1,true);
}


template <typename T>
NRMat<T> Sylvester(const Polynomial<T> &p, const Polynomial<T> &q)
{
int nm=p.degree()+q.degree();
NRMat<T> a(nm,nm);
a.clear();
for(int i=0; i<q.degree(); ++i)
	for(int j=p.degree(); j>=0; --j)
		a(i,i+p.degree()-j)=p[j];
for(int i=0; i<p.degree(); ++i)
	for(int j=q.degree(); j>=0; --j)
		a(q.degree()+i,i+q.degree()-j)=q[j];
return a;
}

template <typename T>
NRMat<T> Polynomial<T>::companion() const
{
if((*this)[degree()]==(T)0) laerror("zero coefficient at highest degree - simplify first");
NRMat<T> a(degree(),degree());
a.clear();
for(int i=0; i<degree(); ++i) a(degree()-1,i) = -(*this)[i];
for(int i=0; i<degree()-1; ++i) a(i,i+1) = (*this)[degree()];
return a;
}

template<>
NRVec<typename LA_traits<int>::complextype> Polynomial<int>::roots() const
{
laerror("roots not implemented for integer polynomials");
return NRVec<typename LA_traits<int>::complextype>(1);
}

template <typename T>
NRVec<typename LA_traits<T>::complextype> Polynomial<T>::roots() const
{
NRMat<T> a=this->companion();
NRVec<typename LA_traits<T>::complextype> r(degree());
gdiagonalize(a,r,NULL,NULL);
return r;
}

template <typename T>
NRVec<T> Polynomial<T>::realroots(const typename LA_traits<T>::normtype thr) const
{
NRVec<typename LA_traits<T>::complextype> r = roots();
NRVec<T> rr(degree());
int nr=0;
for(int i=0; i<degree(); ++i)
	{
	if(abs(r[i].imag())<thr)
		{
		rr[nr++] = r[i].real();
		}
	}
rr.resize(nr,true);
rr.sort();
return rr;
}

template <typename T>
Polynomial<T> lagrange_interpolation(const NRVec<T> &x, const NRVec<T> &y)
{
if(x.size()!=y.size()) laerror("vectors of different length in lagrange_interpolation");
if(x.size()<1) laerror("empty vector in lagrange_interpolation");
if(x.size()==1)
        {
        Polynomial<T> p(0);
        p[0]=y[0];
        return p;
        }
int n=x.size()-1;
Polynomial<T> p(n);
p.clear();
for(int i=0; i<=n; ++i)
        {
	T prod=(T)1;
	for(int j=0; j<=n; ++j) if(j!=i) prod *= (x[i]-x[j]);
	if(prod==(T)0) laerror("repeated x-value in lagrange_interpolation");
	Polynomial<T> tmp=polyfromroots(x,i);
	p += tmp * y[i] / prod;
        }
return p;
}


template <typename T>
T Polynomial<T>::newton(const T x0, const typename LA_traits<T>::normtype thr, const int maxit) const
{
Polynomial<T> d=derivative(1);
T x=x0;
for(int i=0; i<maxit; ++i)
	{
	T v=value(*this,x);
	if(abs(v)<thr) break;
	x -= v/value(d,x);
	}

return x;
}




/***************************************************************************//**
 * forced instantization in the corresponding object file
 ******************************************************************************/
template class Polynomial<int>;
template class Polynomial<double>;
template class Polynomial<std::complex<double> >;

#define INSTANTIZE(T) \
template NRMat<T> Sylvester(const Polynomial<T> &p, const Polynomial<T> &q); \
template Polynomial<T> lagrange_interpolation(const NRVec<T> &x, const NRVec<T> &y); \


INSTANTIZE(int)
INSTANTIZE(double)
INSTANTIZE(std::complex<double>)



}//namespace
