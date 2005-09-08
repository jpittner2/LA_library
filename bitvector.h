#ifndef _BITVECTOR_H_
#define _BITVECTOR_H_

#include "vec.h"

//compressed storage of large bit vectors
class bitvector : public NRVec<unsigned char>
	{
private:
	int modulo;
public:
	bitvector() : NRVec<unsigned char>() {};
	explicit bitvector (const int n):NRVec<unsigned char>((n+7)>>3) {modulo=n&7;};
	bitvector (const int a, const int n):NRVec<unsigned char>(a,(n+7)>>3) {modulo=n&7;};
	bitvector(const bitvector &rhs) : NRVec<unsigned char>(rhs) {};
	//overrige dereferencing to address single bits, is however possible
	//only in the const context (otherwise we would have to define a type which, when assigned to, changes a single bit - possible but probably inefficient)
	void resize(const int n) {NRVec<unsigned char>::resize((n+7)>>3); modulo=n&7;};
	const int size() const {return (nn<<3)-8+modulo?modulo:8;};
        const bool operator[](const int i) const {return (v[i>>3] >>(i&7))&1;};
	void set(const int i) {v[i>>3] |= (1<<(i&7));};
	void reset(const int i) {v[i>>3] &= ~(1<<(i&7));};
	const bool get(const int i) {return (v[i>>3] >>(i&7))&1;};
	const bool assign(const int i, const bool r) {if(r) set(i); else reset(i); return r;};
	void clear() {memset(v,0,nn);};
	void fill() {memset(v,0xff,nn);};
	};

#endif
