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

#include <iostream>
#include "tensor.h"
#include "laerror.h"
#include "qsort.h"
#include "miscfunc.h"
#include <complex>


namespace LA {

LA_largeindex subindex(int *sign, const INDEXGROUP &g, const NRVec<LA_index> &I) //index of one subgroup 
{
#ifdef DEBUG
if(I.size()<=0) laerror("empty index group in subindex");
if(g.number!=I.size()) laerror("mismatch in the number of indices in a group");
for(int i=0; i<I.size(); ++i) if(I[i]<g.offset || I[i] >= g.offset+g.range) laerror("index out of range in tensor subindex");
#endif

switch(I.size()) //a few special cases for efficiency
	{
	case 0:
		*sign=0;
		return 0;
		break;
	case 1:
		*sign=1;
		return I[0]-g.offset;
		break;
	case 2:
		{
		*sign=1;
		if(g.symmetry==0) return (I[1]-g.offset)*g.range+I[0]-g.offset;
		LA_index i0,i1;
		if(I[0]>I[1]) {i1=I[0]; i0=I[1]; if(g.symmetry<0) *sign = -1;} else {i1=I[1]; i0=I[0];}
		i0 -= g.offset;
		i1 -= g.offset;
		if(g.symmetry<0)
			{
			if(i0==i1) {*sign=0; return -1;}
			return i1*(i1-1)/2+i0;
			}
		else
			{
			return i1*(i1+1)/2+i0;
			}
		}
		break;

	default: //general case
		{
		*sign=1;

                if(g.symmetry==0) //rectangular case
			{
			LA_largeindex r=0;
			for(int i=I.size()-1; i>=0; --i)
				{
				r*= g.range;
				r+= I[i]-g.offset;
				}
			return r;
			}
		}

		//compressed storage case
		NRVec<LA_index> II(I);
		II.copyonwrite();
		if(g.offset!=0) II -= g.offset;
		int parity=netsort(II.size(),&II[0]);
		if(g.symmetry<0 && (parity&1)) *sign= -1;
		if(g.symmetry<0) //antisymmetric
			{
			for(int i=0; i<I.size()-1; ++i) 
				if(II[i]==II[i+1])
					{*sign=0; return -1;} //identical indices of antisymmetric tensor

			LA_largeindex r=0;
                        for(int i=0; i<II.size(); ++i) r += simplicial(i+1,II[i]-i);
                        return r;
			}
		else	//symmetric
			{
			LA_largeindex r=0;
			for(int i=0; i<II.size(); ++i) r += simplicial(i+1,II[i]);
			return r;
			}
		break;
	}
laerror("this error should not happen");
return -1;
}




template<typename T>
LA_largeindex Tensor<T>::index(int *sign, const SUPERINDEX &I) const
{
//check index structure and ranges
#ifdef DEBUG
if(I.size()!=shape.size()) laerror("mismatch in the number of tensor index groups");
for(int i=0; i<I.size(); ++i)
	{
	if(shape[i].number!=I[i].size()) {std::cerr<<"error in index group no. "<<i<<std::endl; laerror("mismatch in the size of tensor index group");}
	for(int j=0; j<shape[i].number; ++j)
		{
		if(I[i][j] <shape[i].offset || I[i][j] >= shape[i].offset+shape[i].range) 
			{
			std::cerr<<"error in index group no. "<<i<<" index no. "<<j<<std::endl;
			laerror("tensor index out of range");
			}
		}
	}
#endif

LA_largeindex r=0;
*sign=1;
for(int g=0; g<shape.size(); ++g) //loop over index groups
	{
	int gsign;
	LA_largeindex groupindex =  subindex(&gsign,shape[g],I[g]);
	//std::cout <<"INDEX TEST group "<<g<<" cumsizes "<< cumsizes[g]<<" groupindex "<<groupindex<<std::endl;
	*sign *= gsign;
	if(groupindex == -1) return -1;
	r += groupindex * cumsizes[g];
	}
return r;
}


template<typename T>
LA_largeindex Tensor<T>::index(int *sign, const FLATINDEX &I) const
{
#ifdef DEBUG
if(rank()!=I.size()) laerror("tensor rank mismatch in index");
#endif

LA_largeindex r=0;
*sign=1;
int gstart=0;
for(int g=0; g<shape.size(); ++g) //loop over index groups
        {
        int gsign;
	int gend= gstart+shape[g].number-1;
	NRVec<LA_index> subI = I.subvector(gstart,gend); 
	gstart=gend+1;
        LA_largeindex groupindex =  subindex(&gsign,shape[g],subI);
        //std::cout <<"FLATINDEX TEST group "<<g<<" cumsizes "<< cumsizes[g]<<" groupindex "<<groupindex<<std::endl;
        *sign *= gsign;
        if(groupindex == -1) return -1;
        r += groupindex * cumsizes[g];
        }
return r;
}


template<typename T>
LA_largeindex Tensor<T>::vindex(int *sign,  LA_index i1, va_list args) const
{
NRVec<LA_index> I(rank());
I[0]=i1;
for(int i=1; i<rank(); ++i)
	{
	I[i] = va_arg(args,LA_index);
	}
va_end(args);

return index(sign,I);
}


//binary I/O

template<typename T>
void Tensor<T>::put(int fd) const
{
shape.put(fd,true);
cumsizes.put(fd,true);
data.put(fd,true);
}

template<typename T>
void Tensor<T>::get(int fd)
{
shape.get(fd,true);
myrank=calcrank(); //is not stored but recomputed
cumsizes.get(fd,true);
data.get(fd,true);
}




template class Tensor<double>;
template class Tensor<std::complex<double> >;


}//namespace
