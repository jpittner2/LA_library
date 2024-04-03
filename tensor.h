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


//a simple tensor class with arbitrary symmetry of index subgroups
//stored in an efficient way
//presently only a rudimentary implementation
//presently limited to 2G data size due to NRVec - maybe use a typedef LA_index 
//to uint64_t in the future in vector and matrix classes


#ifndef _TENSOR_H
#define _TENSOR_H

#include <stdint.h>
#include <cstdarg>
#include "vec.h"
#include "miscfunc.h"


namespace LA {


template<typename T>
class Signedpointer
{
T *ptr;
int sgn;
public:
	Signedpointer(T *p, int s) : ptr(p),sgn(s) {};
	T& operator=(const T rhs) {if(sgn==0) return *ptr; if(sgn>0) *ptr=rhs; else *ptr = -rhs; return *ptr;}
//@@@@@@operations on singedpointer as LHS of the non-const tensor.operator() expressions
};


typedef int LA_index;
typedef int LA_largeindex;

typedef class indexgroup {
public:
int number; //number of indices
int symmetry; //-1 0 or 1
LA_index offset; //indices start at
LA_index range;  //indices span this range
} INDEXGROUP;

template<>
class LA_traits<indexgroup> {
	public:
	static bool is_plaindata() {return true;};
	static void copyonwrite(indexgroup& x) {};
	typedef INDEXGROUP normtype;
};


typedef NRVec<LA_index> FLATINDEX; //all indices but in a single vector
typedef NRVec<NRVec<LA_index> > SUPERINDEX; //all indices in the INDEXGROUP structure


template<typename T>
class Tensor {
//essential data
	NRVec<indexgroup> shape;
	NRVec<T> data;	

//redundant data to facilitate efficient indexing
	NRVec<LA_largeindex> cumsizes; //cumulative sizes of symmetry index groups

private:
	LA_largeindex index(int *sign, const SUPERINDEX &I) const; //map the tensor indices to the position in data
	LA_largeindex index(int *sign, const  FLATINDEX &I) const; //map the tensor indices to the position in data
	LA_largeindex vindex(int *sign, int i1, va_list args) const; //map list of indices  to the position in data @@@must call va_end

public:
	//constructors
	Tensor() {};
	Tensor(const NRVec<indexgroup> &s) : shape(s), data((int)getsize()) {data.clear();}; //general tensor
	Tensor(const indexgroup &g) {shape.resize(1); shape[0]=g; data.resize(getsize()); data.clear();}; //tensor with a single index group

	int getrank() const; //is computed from shape
	LA_largeindex getsize(); //set redundant data and return total size
	LA_largeindex size() const {return data.size();};
	void copyonwrite() {shape.copyonwrite(); data.copyonwrite();};
        inline Signedpointer<T> operator[](const SUPERINDEX &I) {int sign; LA_largeindex i=index(&sign,I); return Signedpointer<T>(&data[i],sign);};
	inline T operator()(const SUPERINDEX &I) {int sign; LA_largeindex i=index(&sign,I); if(sign==0) return 0; return sign>0 ?data[i] : -data[i];};
        inline Signedpointer<T> operator[](const FLATINDEX &I) {int sign; LA_largeindex i=index(&sign,I); return Signedpointer<T>(&data[i],sign);};
        inline T operator()(const FLATINDEX &I) {int sign; LA_largeindex i=index(&sign,I); if(sign==0) return 0; return sign>0 ?data[i] : -data[i];};
        //inline Signedpointer<T> operator[](int i1...) {va_list args; int sign; LA_largeindex i; va_start(args,i1); i= vindex(&sign, i1,args);  return Signedpointer<T>(&data[i],sign); };
	//cannot have operator[] with variable number of argmuments
        inline T operator()(int i1...) {va_list args; ; int sign; LA_largeindex i; va_start(args,i1);  i= vindex(&sign, i1,args); if(sign==0) return 0; return sign>0 ?data[i] : -data[i];};
	//@@@do a 'set' operatiaon with va_arg instead

	//NOTE: for sign==0 data[i]  can be undefined pointer, avoid dereferencing it

	//@@@TODO - unwinding to full size in a specified index
	//@@@TODO - contractions - basic and efficient
	//@@@TODO get/put to file, stream i/o
};




template<typename T>
int Tensor<T>:: getrank() const
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
LA_largeindex Tensor<T>::getsize()
{
cumsizes.resize(shape.size());
LA_largeindex s=1;
for(int i=0; i<shape.size(); ++i)
	{
	if(shape[i].number==0) laerror("empty index group");
	if(shape[i].range==0) return 0;
	cumsizes[i]=s;
	switch(shape[i].symmetry) 
		{
		case 0:
			s *= longpow(shape[i].range,shape[i].number);
			break;
		case 1:
			s *= simplicial(shape[i].number,shape[i].range);
			break;
		case -1:
			s *= simplicial(shape[i].number,shape[i].range-shape[i].number+1);
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
