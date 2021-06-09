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

namespace LA {

template <typename T>
class Polynomial : public NRVec<T> {
public:
	Polynomial():  NRVec<T>() {};
	Polynomial(const int n) : NRVec<T>(n+1) {};
	Polynomial(const T &a, const int n) : NRVec<T>(n+1) {NRVec<T>::clear(); (*this)[0]=a;};

	int degree() const {return NRVec<T>::size()-1;};
	void resize(const int n, const bool preserve=true) {NRVec<T>::resize(n+1,preserve);}
	
	Polynomial& operator+=(const T &a) {NOT_GPU(*this); NRVec<T>::copyonwrite(); (*this)[0]+=a; return *this;}
	Polynomial& operator-=(const T &a) {NOT_GPU(*this); NRVec<T>::copyonwrite(); (*this)[0]-=a; return *this;}
	Polynomial operator+(const T &a) const {return Polynomial(*this) += a;};
	Polynomial operator-(const T &a) const {return Polynomial(*this) -= a;};
	//operator *= and * by a scalar inherited 
	//unary- inherited
	
	Polynomial& operator+=(const Polynomial &rhs) 
		{
		NOT_GPU(*this); NRVec<T>::copyonwrite(); 
		if(rhs.degree()>degree()) resize(rhs.degree());
		for(int i=0; i<=rhs.degree(); ++i) (*this)[i] += rhs[i];
		return *this;
		}

	Polynomial& operator-=(const Polynomial &rhs) 
                {
                NOT_GPU(*this); NRVec<T>::copyonwrite(); 
                if(rhs.degree()>degree()) resize(rhs.degree());
                for(int i=0; i<=rhs.degree(); ++i) (*this)[i] -= rhs[i];
                return *this;
                }
	Polynomial operator+(const Polynomial &rhs) const {return Polynomial(*this) += rhs;};
	Polynomial operator-(const Polynomial &rhs) const {return Polynomial(*this) -= rhs;};
	Polynomial operator*(const Polynomial &rhs) const //for very long polynomials FFT should be used
		{
		Polynomial r(degree()+rhs.degree());
		r.clear();
		for(int i=0; i<=rhs.degree(); ++i) for(int j=0; j<=degree(); ++j) r[i+j] += rhs[i]*(*this)[j];
		return r;
		};

	//@@@@
	//simplify(threshold)
	//derivative,integral
	//division remainder 
	//gcd, lcm
	//roots, interpolation ... special only for real->complex - declare only and implent only template specialization  in .cc

};

//this is very general, can be used also for nesting polynomials
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

//scalar+-polynomial
template <typename T>
inline Polynomial<T> operator+(const T &a, const Polynomial<T> &rhs) {return Polynomial<T>(rhs)+=a;}
template <typename T>
inline Polynomial<T> operator-(const T &a, const Polynomial<T> &rhs) {return Polynomial<T>(rhs)-=a;}


}//namespace
#endif
