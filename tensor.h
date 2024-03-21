/*
    LA: linear algebra C++ interface library
    Copyright (C) 2024 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>

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


//a simple tensor class with arbitrary summetry of index subgroups
//stored in an efficient way
//presently only a rudimentary implementation
//presently limited to 2G data size due to NRVec - maybe use a typedef LA_index 
//to uint64_t in the future in vector and matrix classes


#ifndef _TENSOR_H
#define _TENSOR_H

#include <stdint.h>
#include "vec.h"
#include "miscfunc.h"


namespace LA {

typedef int LA_index;
typedef int LA_largeindex;

typedef class indexgroup {
int number; //number of indices
int symmetry; //-1 0 or 1
LA_index offset; //indices start at
LA_index size;  //indices span this range
} INDEXGROUP;

template<typename T>
class Tensor {
	NRVec<indexgroup> shape;
	NRVec<T> data;	

public:
	Tensor() {};
	Tensor(const NRVec<indexgroup> &s) : shape(s), data((int)size()) {data.clear();};
	int rank() const; //is computed from shape
	LA_largeindex size() const; //expensive, is computed from shape
	void copyonwrite() {shape.copyonwrite(); data.copyonwrite();};
	//@@@operator() lhs and rhs both via vararg a via superindex of flat and nested types, get/put to file, stream i/o
};

template<typename T>
int Tensor<T>:: rank() const
{
int r=0;
for(int i=0; i<shape.size(); ++i)
        {
	if(shape[i].number==0) laerror("empty index group");
	r+=shape[i].number;
	}
return r;
}

template<typename T>
LA_largeindex Tensor<T>::size() const
{
LA_largeindex s=1;
for(int i=0; i<shape.size(); ++i)
	{
	if(shape[i].number==0) laerror("empty index group");
	if(shape[i].size==0) return 0;
	switch(shape[i].symmetry) 
		{
		case 0:
			s *= longpow(shape[i].size,shape[i].number);
			break;
		case 1:
			s *= binom(shape[i].size+shape[i].number-1,shape[i].number);
			break;
		case -1:
			s *= binom(shape[i].size,shape[i].number);
			break;
		default: 
			laerror("illegal index group symmetry");
			break;
		}
	}
return s;
}



}//namespace
#endif
