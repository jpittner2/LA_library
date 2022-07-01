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

#ifndef _BITVECTOR_H_
#define _BITVECTOR_H_

#include "vec.h"

namespace LA {

//compressed storage of large bit vectors
//any reasonable compiler changes the dividions and modulos to shift/and instructions

typedef unsigned long bitvector_block; //should be automatically portable and efficiently use wordlength of each machine (32 vs 64)

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
	void resize(const unsigned int n) {NRVec<bitvector_block>::resize((n+blockbits-1)/blockbits); modulo=n%blockbits;};
	unsigned int size() const {return (nn*blockbits)-blockbits+(modulo?modulo:blockbits);};
	//arguments must be unsigned to keep the resulting assembly code simple and efficient
        const bool operator[](const unsigned int i) const {return (v[i/blockbits] >>(i%blockbits))&1UL;};
	bitvector_block getblock(const unsigned int i) const {return v[i];}; //integer interpretation
	void setblock(const unsigned int i, const bitvector_block b) {v[i]=b;};
	int getblocksize() const {return 8*sizeof(bitvector_block);};
	void set(const unsigned int i) {v[i/blockbits] |= (1UL<<(i%blockbits));};
	void reset(const unsigned int i) {v[i/blockbits] &= ~(1UL<<(i%blockbits));};
	const bool get(const unsigned int i) {return (v[i/blockbits] >>(i%blockbits))&1UL;};
	const bool assign(const unsigned int i, const bool r) {if(r) set(i); else reset(i); return r;};
	void clear() {copyonwrite(true); memset(v,0,nn*sizeof(bitvector_block));};
	void fill() {memset(v,0xff,nn*sizeof(bitvector_block));};
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
	bitvector operator&(const bitvector &rhs) const {return bitvector(*this) &= rhs;};
	bitvector operator|(const bitvector &rhs) const {return bitvector(*this) |= rhs;};
	bitvector operator^(const bitvector &rhs) const {return bitvector(*this) ^= rhs;};
        unsigned int operator%(const bitvector &y) const; //number of differing bits
	unsigned int population(const unsigned int before=0) const; //number of 1's
	//extended, truncated const i.e. not on *this but return new entity, take care of modulo's bits
	//logical shifts  <<= >>= << >> not implemented yet
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
bitvector mantissa(const T &x, int nbits)
{
if(x<0||x>=1) laerror("number not normalized in bitvector mantissa");
bitvector b(nbits);
b.clear();
T y= x+x;
for(int i=0; i<nbits; ++i)
	{
	int n= (int) y;
	if(n&1) b.set(i);
	y += y;
	}
return b;
}

template <typename T>
void bitvector_decimal(T &x, const bitvector &b)
{
x=0;
for(int i=0; i<b.size(); ++i) if(b[i]) x += 1./(1ULL<<i);
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
