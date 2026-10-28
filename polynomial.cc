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
#include "miscfunc.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "nonclass.h"


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
for(int i=0; i<degree(); ++i) a(degree()-1,i) = -(*this)[i]/(*this)[degree()];
for(int i=0; i<degree()-1; ++i) a(i,i+1) = (T)1; 
return a;
}

template<>
NRVec<typename LA_traits<int>::complextype> Polynomial<int>::roots() const
{
laerror("roots not implemented for integer polynomials");
return NRVec<typename LA_traits<int>::complextype>(1);
}

template<>
NRVec<typename LA_traits<unsigned int>::complextype> Polynomial<unsigned int>::roots() const
{
laerror("roots not implemented for integer polynomials");
return NRVec<typename LA_traits<unsigned int>::complextype>(1);
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
	if(MYABS(r[i].imag())<thr)
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
	if(MYABS(v)<thr) break;
	x -= v/value(d,x);
	}

return x;
}


template <typename T>
Polynomial<T> poly_gcd(const Polynomial<T> &p, const Polynomial<T> &q, const typename LA_traits<T>::normtype thr, const int d)
{
Polynomial<T> big,small;
if(p.degree() < q.degree()) {big=q; small=p;} else {big=p; small=q;}
small.simplify(thr);
if(small.degree()==0) return MYABS(small[0])<=thr ? big : small; //gcd(p,0)=p, gcd(p,const)=const
Polynomial<T> help;
do      {
        help=small;
        small= big%small;
        big=help;
	small.simplify(thr);
        }
while(!(small.degree()==0 && MYABS(small[0])<=thr) && (d<0 || small.degree()>=d)); //stop at zero remainder or below the expected degree
return big;
}


template <>
Polynomial<int> svd_gcd(const Polynomial<int> &p, const Polynomial<int> &q, const typename LA_traits<int>::normtype thr)
{
laerror("SVD gcd only for floating point numbers");
return p;
}

template <>
Polynomial<unsigned int> svd_gcd(const Polynomial<unsigned int> &p, const Polynomial<unsigned int> &q, const typename LA_traits<unsigned int>::normtype thr)
{
laerror("SVD gcd only for floating point numbers");
return p;
}



template <typename T>
Polynomial<T> svd_gcd(const Polynomial<T> &p, const Polynomial<T> &q, const typename LA_traits<T>::normtype thr)
{
Polynomial<T> big,small;
if(p.degree() < q.degree()) {big=q; small=p;} else {big=p; small=q;}
small.simplify(thr);
if(small.degree()==0) return small;
NRMat<T> s = Sylvester(p,q);
NRMat<T> u(s.nrows(),s.nrows()),v(s.ncols(),s.ncols());
NRVec<typename LA_traits<T>::normtype> w(s.nrows());
singular_decomposition(s,&u,w,&v,0);
int rank=0;
for(int i=0; i<w.size(); ++i) if(w[i]>thr*::sqrt((double)big.degree()*small.degree())) ++rank;
int d=big.degree()+small.degree()-rank;
return poly_gcd(big,small,thr,d);
}

template <typename T>
Polynomial<T> Polynomial<T>::pow(const int n) const
{
if(n<0) laerror("negative exponent in polynomial::pow");
if(n==0) {Polynomial<T> r(0); r[0]=(T)1; return r;}
return positive_power(*this,n);
}

template <typename T>
Polynomial<T> Polynomial<T>::powx(const int i) const
{
if(i<0) laerror("negative exponent in polynomial::pow");
if(i==0) {Polynomial<T> r(0); r[0]= this->sum(); return r;}
if(i==1) return *this;
Polynomial<T> r(i*this->degree());
r.clear();
for(int j=0; j<=this->degree(); ++j) r[j*i] = (*this)[j];
return r;
}

template <typename T>
void Polynomial<T>::binomial(const int n)
{
resize(n,false);
for(int i=0; i<=n; ++i) (*this)[i] = (T) binom(n,i);
}

template <typename T>
Polynomial<T>  Polynomial<T>::composition(const Polynomial &rhs) const
{
return value(*this,rhs);
}


/***************************************************************************//**
 * forced instantization in the corresponding object file
 ******************************************************************************/
template class Polynomial<int>;
template class Polynomial<unsigned int>;
template class Polynomial<double>;
template class Polynomial<std::complex<double> >;

#define INSTANTIZE(T) \
template NRMat<T> Sylvester(const Polynomial<T> &p, const Polynomial<T> &q); \
template Polynomial<T> lagrange_interpolation(const NRVec<T> &x, const NRVec<T> &y); \
template Polynomial<T> poly_gcd(const Polynomial<T> &p, const Polynomial<T> &q, const typename LA_traits<T>::normtype thr,const int d); \
template Polynomial<T> svd_gcd(const Polynomial<T> &p, const Polynomial<T> &q, const typename LA_traits<T>::normtype thr); \



INSTANTIZE(int)
INSTANTIZE(unsigned int)
INSTANTIZE(double)
INSTANTIZE(std::complex<double>)

#define INSTANTIZE2(T) \
template Polynomial<T> hermite_polynomial(int); \

INSTANTIZE2(int)
INSTANTIZE2(double)



}//namespace
