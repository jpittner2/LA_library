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


#ifndef _CONTFRAC_H
#define _CONTFRAC_H

#include "la_traits.h"
#include "vec.h"

namespace LA {

//simple finite continued fraction class
//NOTE: 0 on any position >0 means actually infinity; simplify()  shortens the vector
//presently implements just conversion to/from rationals and floats
//maybe implement arithmetic by Gosper's method cf. https://perl.plover.com/classes/cftalk/TALK

template <typename T>
class Rational {
public:
	T num;
	T den;

	Rational(const T p, const T q) : num(p),den(q) {};
};

template <typename T>
class ContFrac : public NRVec<T> {
private:
	int size() const; //prevent confusion with vector size
public:
	ContFrac():  NRVec<T>() {};
	template<int SIZE>  ContFrac(const T (&a)[SIZE]) : NRVec<T>(a) {};
	ContFrac(const NRVec<T> &v) : NRVec<T>(v) {}; //allow implicit conversion from NRVec
	ContFrac(const int n) : NRVec<T>(n+1) {};
	ContFrac(double x, const int n, const T thres=0); //might yield a non-canonical form
	ContFrac(const T p, const T q); //should yield a canonical form
	ContFrac(const Rational<T> r) : ContFrac(r.num,r.den) {};

	void canonicalize();
	void convergent(T *p, T*q, const int trunc= -1) const;
	Rational<T> rational(const int trunc= -1) const {T p,q; convergent(&p,&q,trunc); return Rational<T>(p,q);};
	double value(const int trunc= -1) const;
	ContFrac reciprocal() const;
	int length() const {return NRVec<T>::size()-1;};
	void resize(const int n, const bool preserve=true) 
		{
		int nold=length();
		NRVec<T>::resize(n+1,preserve);
		if(preserve) for(int i=nold+1; i<=n;++i) (*this)[i]=0;
		} 
	
};



}//namespace
#endif
