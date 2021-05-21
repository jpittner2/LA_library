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


#ifndef _PERMUTATION_H
#define _PERMUTATION_H

#include "la_traits.h"
#include "vec.h"

typedef   unsigned long long PERM_RANK_TYPE;

//permutations are always numbered from 1; offset is employed when applied to vectors and matrices

namespace LA {

//forward declaration
template <typename T> class NRVec_from1;
template <typename T> class CyclePerm;
template <typename T> class Partition;

template <typename T>
class NRPerm : public NRVec_from1<T> {
public:
	//basic constructors
	NRPerm(): NRVec_from1<T>() {};
	NRPerm(const int n) : NRVec_from1<T>(n) {};
	NRPerm(const NRVec_from1<T> &rhs): NRVec_from1<T>(rhs) {};
        NRPerm(const T *a, const int n): NRVec_from1<T>(a, n) {};
	explicit NRPerm(const CyclePerm<T> &rhs, const int n=0); 

	//specific operations
	void identity();
	bool is_valid() const; //is it really a permutation
	bool is_identity() const;
	NRPerm inverse() const;
	NRPerm operator*(const NRPerm q) const; //q is rhs and applied first, this applied second
	NRPerm conjugate_by(const NRPerm q) const; //q^-1 p q
	int parity() const;
	void randomize(void); //uniformly random by Fisher-Yates shuffle
	bool next(); //generate next permutation in lex order
	PERM_RANK_TYPE generate_all(void (*callback)(const NRPerm<T>&), int parity_select=0); //Algorithm from Knuth's vol.4, efficient but not in lex order!
	PERM_RANK_TYPE generate_all2(void (*callback)(const NRPerm<T>&)); //recursive method, also not lexicographic
	PERM_RANK_TYPE generate_all_lex(void (*callback)(const NRPerm<T>&)); //generate in lex order using next()
	PERM_RANK_TYPE rank() const; //counted from 0 to n!-1
	NRVec_from1<T> inversions(const int type, PERM_RANK_TYPE *prank=NULL) const; //inversion tables
	explicit NRPerm(const int type, const NRVec_from1<T> &inversions); //compute permutation from inversions
	explicit NRPerm(const int n, const PERM_RANK_TYPE rank); //compute permutation from its rank
};

extern PERM_RANK_TYPE factorial(const int n);

//permutations represented in the cycle format
template <typename T>
class CyclePerm : public NRVec_from1<NRVec_from1<T> > {
public:
	CyclePerm() :  NRVec_from1<NRVec_from1<T> >() {};
	explicit CyclePerm(const NRPerm<T> &rhs);

	bool is_valid() const; //is it really a permutation
	bool is_identity() const; //no cycles of length > 1
	void identity() {this->resize(0);};
	CyclePerm inverse() const; //reverse all cycles
	int parity() const; //negative if having odd number of even-length cycles
	T max() const {T m=0; for(int i=1; i<=this->size(); ++i) {T mm= (*this)[i].max(); if(mm>m) m=mm;} return m;}
	Partition<T> cycles(const T n) const;
	void readfrom(const std::string &line);
	CyclePerm operator*(const CyclePerm q) const; //q is rhs and applied first, this applied second
};


template <typename T>
std::istream & operator>>(std::istream &s, CyclePerm<T> &x);

template <typename T>
std::ostream & operator<<(std::ostream &s, const CyclePerm<T> &x);



//partitions stored as #of 1s, #of 2s, etc.
template <typename T>
class Partition : public NRVec_from1<T> {
public:
	Partition(): NRVec_from1<T>() {};
	Partition(const int n) : NRVec_from1<T>(n) {};
	T sum() const {T s=0; for(T i=1; i<=this->size(); ++i) s += i*(*this)[i]; return s;}
	T nparts() const {T s=0; for(T i=1; i<=this->size(); ++i) s += (*this)[i]; return s;}
	T nclasses() const {T s=0; for(T i=1; i<=this->size(); ++i) if((*this)[i]) ++s; return s;}
	bool is_valid() const {return this->size() == this->sum();}

//@@@generate all partitions, 
//@@@enumerator of partitions of n to r parts and total
//@@@adjoint partition, 
//@@@output formatted as in the group character table
//@@@Sn character table computation
};


}//namespace
#endif
