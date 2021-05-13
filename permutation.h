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
#include "mat.h"

//permutations are always numbered from 1; offset is employed when applied to vectors and matrices

namespace LA {

//forward declaration
template <typename T> class NRVec_from1;

template <typename T>
class NRPerm : public NRVec_from1<T> {
public:
	//basic constructors
	NRPerm(): NRVec_from1<T>() {};
	NRPerm(const int n) : NRVec_from1<T>(n) {};
	NRPerm(const NRVec_from1<T> &rhs): NRVec_from1<T>(rhs) {};
	NRPerm(const T &a, const int n): NRVec_from1<T>(a, n) {};
        NRPerm(const T *a, const int n): NRVec_from1<T>(a, n) {};

	//specific operations
	void identity();
	bool is_valid() const; //is it really a permutation
	bool is_identity() const;
	NRPerm inverse() const;
	NRPerm operator*(const NRPerm q) const; //q is rhs and applied first, this applied second
	NRPerm conjugate_by(const NRPerm q) const; //q^-1 p q
	int parity() const;


	//TODO:
	//@@@permutation matrix
	//@@@permgener
	//@@@next permutation
	//@@@lex rank
	//@@@inversion tables
	//@@@conversion to cycle structure and back
};




}//namespace
#endif
