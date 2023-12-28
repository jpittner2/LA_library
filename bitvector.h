/*
    LA: linear algebra C++ interface library
    Copyright (C) 2008-2023 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>

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

#ifndef _BITVECTOR_H_
#define _BITVECTOR_H_

#include "vec.h"
#include "numbers.h"
#include <stdint.h>


namespace LA {
//compressed storage of large bit vectors
//let's now use 64-bit blocks exclusively  for simplicity

typedef uint64_t bitvector_block; 

#define blockbits (8*sizeof(bitvector_block))

inline unsigned int bitvector_rounded(unsigned int n)
{
return ((n+blockbits-1)/blockbits)*blockbits; 
}

class bitvector : public NRVec<bitvector_block>
	{
private:
	unsigned int modulo;
public:
	bitvector() : NRVec<bitvector_block>() {};
	explicit bitvector (const unsigned int n):NRVec<bitvector_block>((n+blockbits-1)/blockbits) {modulo=n%blockbits;};
	bitvector (const bitvector_block a, const unsigned int n):NRVec<bitvector_block>(a,(n+blockbits-1)/blockbits) {modulo=n%blockbits;};
	bitvector(const bitvector &rhs) : NRVec<bitvector_block>(rhs) {modulo=rhs.modulo;};
	//operator= seems to be correctly synthetized by the compiler
	//override dereferencing to address single bits, is however possible
	//only in the const context (otherwise we would have to define a type which, when assigned to, changes a single bit - possible but probably inefficient)
	void resize(const unsigned int n, bool preserve=false); //preserve data or clear
	unsigned int size() const {return (nn*blockbits)-blockbits+(modulo?modulo:blockbits);};
	//arguments must be unsigned to keep the resulting assembly code simple and efficient
        const bool operator[](const unsigned int i) const {return (v[i/blockbits] >>(i%blockbits))&1ULL;};
	const bool get(const unsigned int i) const {return (*this)[i];};
	bitvector_block getblock(const unsigned int i) const {return v[i];}; //integer interpretation
	void setblock(const unsigned int i, const bitvector_block b) {v[i]=b;};
	int getblocksize() const {return 8*sizeof(bitvector_block);};
	void set(const unsigned int i) {v[i/blockbits] |= (1UL<<(i%blockbits));};
	void reset(const unsigned int i) {v[i/blockbits] &= ~(1UL<<(i%blockbits));};
	const bool assign(const unsigned int i, const bool r) {if(r) set(i); else reset(i); return r;};
	void clear() {copyonwrite(true); memset(v,0,nn*sizeof(bitvector_block));};
	void fill() {memset(v,0xff,nn*sizeof(bitvector_block));};
	bool iszero() const {for(int i=0; i<nn; ++i) if(v[i]) return false; return true;};
	bool is_zero() const {return iszero();};
	bool is_one() const {if(v[0]!=1) return false; for(int i=1; i<nn; ++i) if(v[i]) return false; return true;};
	void randomize();
	bool operator!=(const bitvector &rhs) const;
	bool operator==(const bitvector &rhs) const {return !(*this != rhs);};
	bool operator>(const bitvector &rhs) const;
	bool operator<(const bitvector &rhs) const;
	bool operator>=(const bitvector &rhs) const {return !(*this < rhs);};
	bool operator<=(const bitvector &rhs) const {return !(*this > rhs);};
	bitvector operator~() const;
	bitvector& operator&=(const bitvector &rhs);
	bitvector& operator|=(const bitvector &rhs);
	bitvector& operator^=(const bitvector &rhs);
	bitvector& operator+=(const bitvector &rhs) {return (*this)^=rhs;}; //addition modulo 2
	bitvector& operator-=(const bitvector &rhs) {return (*this)^=rhs;}; //subtraction modulo 2
	bitvector operator&(const bitvector &rhs) const {return bitvector(*this) &= rhs;};
	bitvector operator|(const bitvector &rhs) const {return bitvector(*this) |= rhs;};
	bitvector operator^(const bitvector &rhs) const {return bitvector(*this) ^= rhs;};
	bitvector operator+(const bitvector &rhs) const {return *this ^ rhs;}; //addition modulo 2
	bitvector operator-(const bitvector &rhs) const {return *this ^ rhs;}; //subtraction modulo 2
	bitvector operator*(const bitvector &rhs) const; //multiplication of polynomials over GF(2) NOTE: naive algorithm, does not employ CLMUL nor fft-like approach, only for short vectors!!!
	bitvector division(const bitvector &rhs,  bitvector &remainder) const;
	bitvector operator/(const bitvector &rhs) const {bitvector rem(rhs.size()); return division(rhs,rem);};
	bitvector operator%(const bitvector &rhs) const {bitvector rem(rhs.size()); division(rhs,rem); return rem;};
	bitvector gcd(const bitvector &rhs) const;
	bitvector lcm(const bitvector &rhs) const {return (*this)*rhs/this->gcd(rhs);};
        unsigned int bitdiff(const bitvector &y) const; //number of differing bits
	unsigned int population(const unsigned int before=0) const; //number of 1's
	unsigned int nlz() const; //number of leading zeroes
	unsigned int degree() const {if(iszero()) return 0; else return size()-nlz()-1;}; //interprested as a polynomial over GF(2)
	unsigned int ntz() const;  //number of trailing zeroes
	//extended, truncated const i.e. not on *this but return new entity, take care of modulo's bits
	//logical shifts
	bitvector& operator>>=(unsigned int i);
	bitvector& leftshift(unsigned int i, bool autoresize=false);
	bitvector& operator<<=(unsigned int i) {return leftshift(i,true);};
	bitvector operator>>(unsigned int i) const {bitvector r(*this); return r>>=i;};
	bitvector operator<<(unsigned int i) const {bitvector r(*this); return r<<=i;};
	//logical rotations not implemented yet
	//unformatted file IO
	void read(int fd, bool dimensions=1, bool transp=0);
	void write(int fd, bool dimensions=1, bool transp=0);
	};

//expand to separate bytes or ints
template <typename T>
void bitvector_expand(const bitvector &v, NRVec<T> &r)
{
int n=v.size();
r.resize(n);
r.clear();
for(int i=0; i<n; ++i) if(v[i]) r[i]=1;
}

//mantissa of a floating number between 0 and 1
template <typename T>
bitvector mantissa(T x, int nbits, int shift=0)
{
while(shift >0) {x+=x; --shift;}
while(shift <0) {x*=.5; ++shift;}
if(x<0||x>=1) laerror("number not normalized in bitvector mantissa");
bitvector b(nbits);
b.clear();
T y= x+x;
for(int i=0; i<nbits-1; ++i)
	{
	int n= (int) y;
	if(n&1) b.set(i+1);
	y += y;
	}
return b;
}

template <typename T>
void bitvector_decimal(T &x, const bitvector &b, int shift=0) 
{
x=0;
for(int i=b.size()-1; i>=0; --i) if(b[i]) x += 1./(1ULL<<i);
while(shift >0) {x+=x; --shift;}
while(shift <0) {x*=.5; ++shift;}
}


template <typename T>
void bitvector_compress(bitvector &r, const NRVec<T> &v)
{
int n=v.size();
r.resize(n);
r.clear();
for(int i=0; i<n; ++i) if(v[i]) r.set(i);
}


extern std::ostream & operator<<(std::ostream &s, const bitvector &x);
extern std::istream & operator>>(std::istream  &s, bitvector &x);

class bitvector_from1 : public bitvector
	{
public:
	bitvector_from1() : bitvector() {};
	bitvector_from1(const bitvector &rhs) :bitvector(rhs) {};
	explicit bitvector_from1(const unsigned int n) : bitvector(n) {};
	const bool operator[](const unsigned int i) {return bitvector::operator[](i-1);};
	void set(const unsigned int i) {bitvector::set(i-1);};
	void reset(const unsigned int i) {bitvector::reset(i-1);};
	const bool get(const unsigned int i) {return bitvector::get(i-1);};
	const bool assign(const unsigned int i, const bool r) {return bitvector::assign(i-1,r);};
	unsigned int population(const unsigned int before=0) const {return bitvector::population(before?before-1:0);};
	};

}//namespace
#endif
