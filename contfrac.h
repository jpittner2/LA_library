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

//Support for rationals and a simple finite continued fraction class
//NOTE: 0 on any position >0 means actually infinity; simplify()  shortens the vector
//includes Gosper's arithmetics - cf. https://perl.plover.com/classes/cftalk/TALK
//maybe implement the self-feeding  Gosper's algorithm for sqrt(int)
//maybe do not interpret a_i=0 i>0 as termination???

template <typename T>
class ContFrac;


template <typename T>
class Rational {
public:
	T num;
	T den;

	Rational() {};
	Rational(const T p, const T q) : num(p),den(q) {};
	explicit Rational(const T (&a)[2]) :num(a[0]), den(a[1]) {};
	explicit Rational(const ContFrac<T> &cf) {cf.convergent(&num,&den);};
	void simplify();

	//basic rational arithmetics 
	Rational operator-() const {return Rational(-num,den);};
	Rational & operator+=(const T  &rhs) {num+=den*rhs; return *this;};
	Rational & operator-=(const T  &rhs) {num-=den*rhs; return *this;};
	Rational & operator*=(const T  &rhs);
	Rational & operator/=(const T  &rhs);
	Rational operator+(const T  &rhs) const {Rational r(*this); return r+=rhs;};
	Rational operator-(const T  &rhs) const {Rational r(*this); return r-=rhs;};
	Rational operator*(const T  &rhs) const {Rational r(*this); return r*=rhs;};
	Rational operator/(const T  &rhs) const {Rational r(*this); return r/=rhs;};
	Rational & operator*=(const Rational &rhs);
	Rational & operator/=(const Rational &rhs) {return (*this)*=Rational(rhs.den,rhs.num);};
	Rational operator+(const Rational  &rhs) const;
        Rational operator-(const Rational  &rhs) const;
        Rational operator*(const Rational  &rhs) const {Rational r(*this); return r*=rhs;};
        Rational operator/(const Rational  &rhs) const {Rational r(*this); return r/=rhs;};
	Rational & operator+=(const Rational &rhs) {*this = *this+rhs; return *this;};
	Rational & operator-=(const Rational &rhs) {*this = *this-rhs; return *this;};

	//combination with continued fractions
	ContFrac<T> operator+(const ContFrac<T> &rhs) const {return rhs + *this;};
	ContFrac<T> operator-(const ContFrac<T> &rhs) const {return -rhs + *this;};
	ContFrac<T> operator*(const ContFrac<T> &rhs) const {return rhs * *this;};
	ContFrac<T> operator/(const ContFrac<T> &rhs) const {return rhs.reciprocal() * *this;};


	//relational operators, relying that operator- yields a form with a positive denominator
	bool operator==(const Rational  &rhs) const {Rational t= *this-rhs; return t.num==0;};
	bool operator!=(const Rational  &rhs) const {Rational t= *this-rhs; return t.num!=0;};
	bool operator>=(const Rational  &rhs) const {Rational t= *this-rhs; return t.num>=0;};
	bool operator<=(const Rational  &rhs) const {Rational t= *this-rhs; return t.num<=0;};
	bool operator>(const Rational  &rhs) const {Rational t= *this-rhs; return t.num>0;};
        bool operator<(const Rational  &rhs) const {Rational t= *this-rhs; return t.num<0;};

};

template <typename T>
std::ostream & operator<<(std::ostream &s, const Rational<T> &x) 
{
s<<x.num<<"/"<<x.den;
return s;
}

template <typename T>
std::istream & operator>>(std::istream &s, Rational<T> &x)
{
char c;
s>>x.num>>c>>x.den;
return s;
}



template <typename T>
class Homographic;

template <typename T>
class BiHomographic;




template <typename T>
class ContFrac : public NRVec<T> {
private:
	int size() const; //prevent confusion with vector size
public:
	ContFrac():  NRVec<T>() {};
	explicit ContFrac(const std::list<T> &x) : NRVec<T>(x) {};
	template<int SIZE>  ContFrac(const T (&a)[SIZE]) : NRVec<T>(a) {};
	ContFrac(const NRVec<T> &v) : NRVec<T>(v) {}; //allow implicit conversion from NRVec
	ContFrac(const int n) : NRVec<T>(n+1) {};
	explicit ContFrac(double x, const int n, const T thres=0); //WARNING: it might yield a non-canonical form 
	//we could make a template for analogous conversion from an arbitrary-precision type
	ContFrac(T p, T q); //should yield a canonical form
	explicit ContFrac(const Rational<T> &r) : ContFrac(r.num,r.den) {};

	void canonicalize(); //requires a_1..a_n of the same sign (works on the terms, no overflow for fixed-size T)
	void convergent(T *p, T*q, const int trunc= -1) const;
	Rational<T> rational(const int trunc= -1) const {T p,q; convergent(&p,&q,trunc); return Rational<T>(p,q);};
	double value(const int trunc= -1) const; //we could make also a template usable with an arbitrary-precision type
	ContFrac reciprocal() const;
	ContFrac operator-() const; //unary minus
	int length() const {return NRVec<T>::size()-1;};
	void resize(const int n, const bool preserve=true) 
		{
		int nold=length();
		NRVec<T>::resize(n+1,preserve);
		if(preserve) for(int i=nold+1; i<=n;++i) (*this)[i]=0;
		} 

	//arithmetics with a single ContFrac operand
	ContFrac operator+(const Rational<T> &rhs) const {Homographic<T> h({{rhs.num,rhs.den},{rhs.den,0}}); return h.value(*this);};
	ContFrac operator-(const Rational<T> &rhs) const {Homographic<T> h({{-rhs.num,rhs.den},{rhs.den,0}}); return h.value(*this);};
	ContFrac operator*(const Rational<T> &rhs) const {Homographic<T> h({{0,rhs.num},{rhs.den,0}}); return h.value(*this);};
	ContFrac operator/(const Rational<T> &rhs) const {Homographic<T> h({{0,rhs.den},{rhs.num,0}}); return h.value(*this);};

	ContFrac & operator+=(const T  &rhs)  {this->copyonwrite(); (*this)[0]+=rhs; return *this;};
	ContFrac & operator-=(const T  &rhs)  {this->copyonwrite(); (*this)[0]-=rhs; return *this;};
	ContFrac operator+(const T  &rhs) const {ContFrac r(*this); r+=rhs; return r;};
	ContFrac operator-(const T  &rhs) const {ContFrac r(*this); r-=rhs; return r;};
	ContFrac operator*(const T &rhs)  const {Homographic<T> h({{0,rhs},{1,0}}); return h.value(*this);};
	ContFrac operator/(const T &rhs)  const {Homographic<T> h({{0,1},{rhs,0}}); return h.value(*this);};

	//arithmetics with two ContFrac operands
	ContFrac operator+(const ContFrac &rhs) const {BiHomographic<T> h({{{0,1},{1,0}},{{1,0},{0,0}}}); return h.value(*this,rhs);};
	ContFrac operator-(const ContFrac &rhs) const {BiHomographic<T> h({{{0,1},{-1,0}},{{1,0},{0,0}}}); return h.value(*this,rhs);};
	ContFrac operator*(const ContFrac &rhs) const {BiHomographic<T> h({{{0,0},{0,1}},{{1,0},{0,0}}}); return h.value(*this,rhs);};
	ContFrac operator/(const ContFrac &rhs) const {BiHomographic<T> h({{{0,1},{0,0}},{{0,0},{1,0}}}); return h.value(*this,rhs);};

	
	//relational operators, guaranteed only to work correctly for canonicalized CF!
	T compare(const ContFrac  &rhs) const;
	bool operator==(const ContFrac  &rhs) const {return compare(rhs)==0;};
	bool operator>(const ContFrac  &rhs) const {return compare(rhs)>0;};
	bool operator<(const ContFrac  &rhs) const {return rhs.operator>(*this);};
	bool operator!=(const ContFrac  &rhs) const {return ! this->operator==(rhs) ;}
	bool operator<=(const ContFrac  &rhs) const {return ! this->operator>(rhs) ;}
	bool operator>=(const ContFrac  &rhs) const {return ! this->operator<(rhs) ;}

	//iterator
        class iterator {
        private:
                T *p;
        public:
                iterator() {};
                ~iterator() {};
		iterator(T *v): p(v) {};
                bool operator==(const iterator rhs) const {return p==rhs.p;}
                bool operator!=(const iterator rhs) const {return p!=rhs.p;}
                iterator operator++() {return ++p;}
                iterator operator++(int) {return p++;}
                T& operator*() const {return *p;}
                T *operator->() const {return p;}
        };
        iterator begin() const {return NRVec<T>::v;}
        iterator end() const {return NRVec<T>::v+NRVec<T>::nn;}
        iterator beyondend() const {return NRVec<T>::v+NRVec<T>::nn+1;}

};

//for Gosper's arithmetic

template <typename T>
class Homographic {
public:
T v[2][2]; //{{a,b},{c,d}} for (a+b.x)/(c+d.x) i.e. [denominator][power_x]

	Homographic(){};
	explicit Homographic(const T (&a)[2][2]) {memcpy(v,a,2*2*sizeof(T));};
	ContFrac<T> value(const ContFrac<T>&z) const;

	Homographic input(const T &x, const bool inf) const; 
	Homographic output(const T &x) const;
	bool outputready(T &x, bool first) const;
	bool terminate() const;
	
};


template <typename T>
class BiHomographic {
public:
T v[2][2][2];  //{{{a,b},{c,d}},{{e,f},{g,h}}} i.e.[denominator][power_y][power_x]
	
	BiHomographic(){};
	explicit BiHomographic(const T (&a)[2][2][2]) {memcpy(v,a,2*2*2*sizeof(T));};
	ContFrac<T> value(const ContFrac<T>&x, const ContFrac<T>&y) const;

	BiHomographic inputx(const T &x, const bool inf) const;
	BiHomographic inputy(const T &y, const bool inf) const;
        BiHomographic output(const T &z) const;
	int inputselect() const;
        bool outputready(T &x,bool first) const;
        bool terminate() const;

};



}//namespace
#endif
