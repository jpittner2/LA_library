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


template <typename T>
class ContFrac : public NRVec<T> {
public:
	ContFrac():  NRVec<T>() {};
	template<int SIZE>  ContFrac(const T (&a)[SIZE]) : NRVec<T>(a) {};
	ContFrac(const NRVec<T> &v) : NRVec<T>(v) {}; //allow implicit conversion from NRVec
	ContFrac(const int n) : NRVec<T>(n+1) {};

	int length() const {return NRVec<T>::size()-1;};
	void resize(const int n, const bool preserve=true) {NRVec<T>::resize(n+1,preserve);}
	
};



}//namespace
#endif
