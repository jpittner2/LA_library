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


#ifndef _POLYNOMIAL_H
#define _POLYNOMIAL_H

#include "la_traits.h"
#include "vec.h"
#include "nonclass.h"
#include "matexp.h"

namespace LA {


template <typename T>
class Polynomial : public NRVec<T> {
private:
	int size() const; //prevent confusion with vector size
public:
	Polynomial():  NRVec<T>() {};
	template<int SIZE>  Polynomial(const T (&a)[SIZE]) : NRVec<T>(a) {};
	Polynomial(const NRVec<T> &v) : NRVec<T>(v) {}; //allow implicit conversion from NRVec
	Polynomial(const int n) : NRVec<T>(n+1) {};
	Polynomial(const T &a, const int n) : NRVec<T>(n+1) {NRVec<T>::clear(); (*this)[0]=a;};

	int degree() const {return NRVec<T>::size()-1;};
	void resize(const int n, const bool preserve=true) {NRVec<T>::resize(n+1,preserve);}
	
	Polynomial& operator+=(const T &a) {NOT_GPU(*this); NRVec<T>::copyonwrite(); (*this)[0]+=a; return *this;}
	Polynomial& operator-=(const T &a) {NOT_GPU(*this); NRVec<T>::copyonwrite(); (*this)[0]-=a; return *this;}
	Polynomial operator+(const T &a) const {return Polynomial(*this) += a;};
	Polynomial operator-(const T &a) const {return Polynomial(*this) -= a;};
	Polynomial operator-() const {return NRVec<T>::operator-();}
	Polynomial operator*(const T &a) const {return NRVec<T>::operator*(a);}
	Polynomial operator/(const T &a) const {return NRVec<T>::operator/(a);}
	Polynomial& operator*=(const T &a) {NRVec<T>::operator*=(a); return *this;}
	Polynomial& operator/=(const T &a) {NRVec<T>::operator/=(a); return *this;}
	
	Polynomial& operator+=(const Polynomial &rhs) 
		{
		NOT_GPU(*this); NRVec<T>::copyonwrite(); 
		if(rhs.degree()>degree()) resize(rhs.degree(),true);
		for(int i=0; i<=rhs.degree(); ++i) (*this)[i] += rhs[i];
		return *this;
		}

	Polynomial& operator-=(const Polynomial &rhs) 
                {
                NOT_GPU(*this); NRVec<T>::copyonwrite(); 
                if(rhs.degree()>degree()) resize(rhs.degree(),true);
                for(int i=0; i<=rhs.degree(); ++i) (*this)[i] -= rhs[i];
                return *this;
                }
	Polynomial operator+(const Polynomial &rhs) const {return Polynomial(*this) += rhs;};
	Polynomial operator-(const Polynomial &rhs) const {return Polynomial(*this) -= rhs;};
	Polynomial operator*(const Polynomial &rhs) const //NOTE: naive implementation! For very long polynomials FFT-based methods should be used
		{
		NOT_GPU(*this); 
		Polynomial r(degree()+rhs.degree());
		r.clear();
		for(int i=0; i<=rhs.degree(); ++i) for(int j=0; j<=degree(); ++j) r[i+j] += rhs[i]*(*this)[j];
		return r;
		};
	Polynomial& operator*=(const Polynomial &rhs) {*this = (*this)*rhs; return *this;};
	void simplify(const typename LA_traits<T>::normtype thr=0)
		{
		NOT_GPU(*this); 
		this->copyonwrite();
		int n=degree();
		while(n>0 && MYABS((*this)[n])<=thr) --n;
		resize(n,true);
		};
	void normalize() {if((*this)[degree()]==(T)0) laerror("zero coefficient at highest degree - simplify first"); *this /= (*this)[degree()];};
	Polynomial shifted(const int shift) const
		{
		if(shift==0) return *this;
		if(shift>0)
			{
			Polynomial r(degree()+shift);
			for(int i=0; i<shift; ++i) r[i]=0;
			for(int i=0; i<=degree(); ++i) r[shift+i] = (*this)[i];
			return r;
			}
		else
			{
			if(shift+degree()<0)
				{
				Polynomial r(0);
				r[0]=0;
				return r;
				}
			Polynomial r(shift+degree());
			for(int i= -shift; i<=degree(); ++i) r[shift+i] = (*this)[i];
			return r;
			}
		}
	Polynomial derivative(const int d=1) const
		{
		if(d==0) return *this;
		if(d<0) return integral(-d);
		NOT_GPU(*this); 
		int n=degree();
		if(n-d<0)
			{
			Polynomial r(0);
			r[0]=0;
			return r;
			}
		Polynomial r(n-d);
		for(int i=d; i<=n; ++i) 
			{
			int prod=1;
			for(int j=i; j>=i-d+1; --j) prod *= j;
			r[i-d] = (*this)[i]* ((T)prod);
			}
		return r;
		};
	Polynomial integral(const int d=1) const
		{
		if(d==0) return *this;
                if(d<0) return derivative(-d);
		NOT_GPU(*this); 
		int n=degree();
		Polynomial r(n+d);
		for(int i=0; i<d; ++i) r[i]=0;
		for(int i=0; i<=n; ++i) 
			{
			int prod=1;
			for(int j=i+1; j<=i+d;++j) prod *= j;
			r[i+d] = (*this)[i]/((T)prod);
			}
		return r;
		}
	Polynomial even_powers() const {int d=degree()/2; Polynomial r(d); for(int i=0; i<=degree(); i+=2) r[i/2] = (*this)[i]; return r;};
	Polynomial odd_powers() const {int d=(degree()-1)/2; Polynomial r(d); if(degree()==0) {r[0]=0; return r;} for(int i=1; i<=degree(); i+=2) r[(i-1)/2] = (*this)[i]; return r;};
	void polydiv(const Polynomial &rhs, Polynomial &q, Polynomial &r) const;
	Polynomial operator/(const Polynomial &rhs) const {Polynomial q,r; polydiv(rhs,q,r); return q;};
	Polynomial operator%(const Polynomial &rhs) const {Polynomial q,r; polydiv(rhs,q,r); return r;};
	NRMat<T> companion() const; //matrix which has this characteristic polynomial
	NRVec<typename LA_traits<T>::complextype> roots() const; //implemented for complex<double> and double only
	NRVec<T> realroots(const typename LA_traits<T>::normtype thr) const;
	T newton(const T x0, const typename LA_traits<T>::normtype thr=1e-14, const int maxit=1000) const; //solve root from the guess
	Polynomial pow(const int i) const; //integer power
	Polynomial powx(const int i) const; //substitute x^i for x in the polynomial
	void binomial(const int n); //(1+x)^n

};

//this is very general, can be used also for composition (nesting) of polynomials
//for an alternative algorithm which minimizes number of multiplications cf. also matexp.h
template <typename T, typename C>
C value(const Polynomial<T> &p, const C &x)
{
C sum(x); 
sum=0; //get matrix dimension if C is a matrix
for(int i=p.degree(); i>0; --i)
  		 {
       		 sum+= p[i];
       		 sum= sum*x; //not *= for matrices
       		 }
sum += p[0];
return sum;
}

//evaluate only even powers
template <typename T, typename C>
C even_value(const Polynomial<T> &p, const C &x)
{
C sum(x); 
sum=0; //get matrix dimension if C is a matrix
int d=p.degree();
if(d&1) --d;
C x2 = x*x;
for(int i=d; i>0; i-=2)
  		 {
       		 sum+= p[i];
       		 sum= sum*x2; //not *= for matrices
       		 }
sum += p[0];
return sum;
}

//evaluate only odd powers
template <typename T, typename C>
C odd_value(const Polynomial<T> &p, const C &x)
{
C sum(x);
sum=0; //get matrix dimension if C is a matrix
int d=p.degree();
if(d==0) return sum;
if((d&1)==0) --d;
C x2 = x*x;
for(int i=d; i>2; i-=2)
                 {
                 sum+= p[i];
                 sum= sum*x2; //not *= for matrices
                 }
sum += p[1];
sum *= x;
return sum;
}


template <typename T, typename C>
NRVec<C> values(const Polynomial<T> &p, const NRVec<C> &x)
{
NRVec<C> r(x.size());
for(int i=0; i<x.size(); ++i) r[i]=value(p,x[i]);
return r;
}


//scalar+-polynomial
template <typename T>
inline Polynomial<T> operator+(const T &a, const Polynomial<T> &rhs) {return Polynomial<T>(rhs)+=a;}
template <typename T>
inline Polynomial<T> operator-(const T &a, const Polynomial<T> &rhs) {return Polynomial<T>(rhs)-=a;}

//Sylvester matrix
template <typename T>
extern NRMat<T> Sylvester(const Polynomial<T> &p, const Polynomial<T> &q);

//polynomial from given roots
template <typename T>
Polynomial<T> polyfromroots(const NRVec<T> &roots, const int skip= -1)
{
Polynomial<T> p(0);
p[0]=(T)1;
for(int i=0; i<roots.size(); ++i) 
    if(i!=skip)
	p = p.shifted(1) - p * roots[i];
return p;
}




template <typename T>
extern Polynomial<T> lagrange_interpolation(const NRVec<T> &x, const NRVec<T> &y);


template <typename T>
extern Polynomial<T> poly_gcd(const Polynomial<T> &p, const Polynomial<T> &q, const typename LA_traits<T>::normtype thr=0, const int d= -1);

template <typename T>
extern Polynomial<T> svd_gcd(const Polynomial<T> &p, const Polynomial<T> &q, const typename LA_traits<T>::normtype thr=0);

template <typename T>
Polynomial<T> poly_lcm(const Polynomial<T> &p, const Polynomial<T> &q, const typename LA_traits<T>::normtype thr=0)
{
return p*q/poly_gcd(p,q,thr);
}


template <typename T>
Polynomial<T> hermite_polynomial(int n) //physicists definition
{
Polynomial<T> h(n);
h.clear();
h[n]=ipow((T)2,n);
for(int m=1; n-2*m>=0; m+=1)
	{
	int i=n-2*m;
	h[i] = (-h[i+2] *(i+2)*(i+1)) /(4*m);
	}
return h;
}


}//namespace
#endif
