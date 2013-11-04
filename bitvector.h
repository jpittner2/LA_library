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
        const bool operator[](const unsigned int i) const {return (v[i/blockbits] >>(i%blockbits))&1;};
	void set(const unsigned int i) {v[i/blockbits] |= (1<<(i%blockbits));};
	void reset(const unsigned int i) {v[i/blockbits] &= ~(1<<(i%blockbits));};
	const bool get(const unsigned int i) {return (v[i/blockbits] >>(i%blockbits))&1;};
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
	};

extern std::ostream & operator<<(std::ostream &s, const bitvector &x);
extern std::istream & operator>>(std::istream  &s, bitvector &x);

}//namespace
#endif
