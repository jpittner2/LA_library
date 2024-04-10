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

template<typename T>
int Tensor<T>:: calcrank()  
{
int r=0;
for(int i=0; i<shape.size(); ++i)
        {
        if(shape[i].number==0) laerror("empty index group");
        r+=shape[i].number;
        }
myrank=r;
return r;
}



template<typename T>
LA_largeindex Tensor<T>::calcsize()
{
groupsizes.resize(shape.size()); 
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
                        s *= groupsizes[i] = longpow(shape[i].range,shape[i].number);
                        break;
                case 1:
                        s *= groupsizes[i] = simplicial(shape[i].number,shape[i].range);
                        break;
                case -1:
                        s *= groupsizes[i] = simplicial(shape[i].number,shape[i].range-shape[i].number+1);
                        break;
                default: 
                        laerror("illegal index group symmetry");
                        break;
                }
        }
return s;
}




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


//inverse map of group superindex to canonically ordered index list
NRVec<LA_index>  inverse_subindex(const INDEXGROUP &g, LA_largeindex s)
{
NRVec<LA_index> I(g.number);
if(g.number==1) {I[0]=s+g.offset; return I;}
switch(g.symmetry)
	{
	case 0:
		for(int i=0; i<g.number; ++i)
			{
			I[i] = s%g.range;
			s /= g.range;
			}
		break;
	case 1:
		for(int i=g.number; i>0; --i)
			{
			I[i-1] = inverse_simplicial(i,s);
			s -= simplicial(i,I[i-1]);
			}
		break;
	case -1:
		for(int i=g.number-1; i>=0; --i)
                        {
                        I[i] = i + inverse_simplicial(i+1,s);
                        s -= simplicial(i+1,I[i]-i);
                        }
		break;
	default: laerror("illegal index symmetry");
	}

if(g.offset!=0) I += g.offset;
return I;
}


template<typename T>
SUPERINDEX Tensor<T>::inverse_index(LA_largeindex s) const
{
SUPERINDEX I(shape.size());
for(int g=shape.size()-1; g>=0; --g)
	{
	LA_largeindex groupindex;
	if(g>0) 
		{
		groupindex = s/cumsizes[g];
		s %= cumsizes[g];
		}
	else groupindex=s;
	I[g] = inverse_subindex(shape[g],groupindex);
	}
return I;
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
groupsizes.put(fd,true);
cumsizes.put(fd,true);
data.put(fd,true);
}

template<typename T>
void Tensor<T>::get(int fd)
{
shape.get(fd,true);
myrank=calcrank(); //is not stored but recomputed
groupsizes.put(fd,true);
cumsizes.get(fd,true);
data.get(fd,true);
}


template<typename T>
Tensor<T>::Tensor(const NRVec<T> &x)
: data(x)
{
myrank=1;
shape.resize(1);
shape[0].number=1;
shape[0].symmetry=0;
#ifndef LA_TENSOR_ZERO_OFFSET
shape[0].offset=0;
#endif
shape[0].range=x.size();
calcsize();
}

template<typename T>
Tensor<T>::Tensor(const NRMat<T> &x)
: data(&x(0,0),x.nrows()*x.ncols())
{
myrank=2;
if(x.nrows()==x.ncols())
	{
	shape.resize(1);
	shape[0].number=2;
	shape[0].symmetry=0;
#ifndef LA_TENSOR_ZERO_OFFSET
	shape[0].offset=0;
#endif
	shape[0].range=x.nrows();
	}
else
	{
	shape.resize(2);
	shape[0].number=1; shape[1].number=1;
	shape[0].symmetry=0; shape[1].symmetry=0;
#ifndef LA_TENSOR_ZERO_OFFSET
        shape[0].offset=0; shape[1].offset=0;
#endif
	shape[0].range=x.ncols();
	shape[1].range=x.nrows();
	}
calcsize();
}


template<typename T>
Tensor<T>::Tensor(const NRSMat<T> &x)
: data(NRVec<T>(x))
{
myrank=2;
shape.resize(1);
shape[0].number=2;
shape[0].symmetry=1;
#ifndef LA_TENSOR_ZERO_OFFSET
        shape[0].offset=0;
#endif
shape[0].range=x.nrows();
calcsize();
}


template<typename T>
void loopingroups(Tensor<T> &t, int ngroup, int igroup, T **p, SUPERINDEX &I, void (*callback)(const SUPERINDEX &, T *))
{
LA_index istart,iend;
switch(t.shape[ngroup].symmetry)
	{
	case 0:
		istart= t.shape[ngroup].offset;
		iend=   t.shape[ngroup].offset+t.shape[ngroup].range-1;
		break;
	case 1:
		istart= t.shape[ngroup].offset;
		if(igroup==t.shape[ngroup].number-1) iend=   t.shape[ngroup].offset+t.shape[ngroup].range-1;
		else iend = I[ngroup][igroup+1];
		break;
	case -1:
		istart= t.shape[ngroup].offset + igroup;
		if(igroup==t.shape[ngroup].number-1) iend=   t.shape[ngroup].offset+t.shape[ngroup].range-1;
                else iend = I[ngroup][igroup+1]-1;
		break;
	}

for(LA_index i = istart; i<=iend; ++i)
	{
	I[ngroup][igroup]=i;
	if(ngroup==0 && igroup==0) 
		{
		int sign;
		//std::cout <<"TEST  "<<t.index(&sign,I)<<" ";
		(*callback)(I,(*p)++);
		}
	else 
		{
		int newigroup= igroup-1;
		int newngroup=ngroup;
		if(newigroup<0)
			{
			--newngroup;
			newigroup=t.shape[newngroup].number-1;
			}
		loopingroups(t,newngroup,newigroup,p,I,callback);
		}
	}
}


template<typename T>
void Tensor<T>::loopover(void (*callback)(const SUPERINDEX &, T *))
{
SUPERINDEX I(shape.size());
for(int i=0; i<I.size(); ++i) {I[i].resize(shape[i].number); I[i] = shape[i].offset;}
T *pp=&data[0];
loopingroups(*this,shape.size()-1,shape[shape.size()-1].number-1,&pp,I,callback);
}


static std::ostream *sout;
template<typename T>
static void outputcallback(const SUPERINDEX &I, T *v)
{
//print indices flat
for(int i=0; i<I.size(); ++i)
	for(int j=0; j<I[i].size(); ++j) *sout << I[i][j]<<" ";
*sout<<" "<< *v<<std::endl;
}


std::ostream & operator<<(std::ostream &s, const INDEXGROUP &x)
{
s<<x.number <<" "<<x.symmetry<<" ";
#ifndef LA_TENSOR_ZERO_OFFSET
s<<x.offset<<" ";
#endif
s<< x.range<<std::endl;
return s;
}

std::istream & operator>>(std::istream &s, INDEXGROUP &x)
{
s>>x.number>>x.symmetry;
#ifndef LA_TENSOR_ZERO_OFFSET
s>>x.offset;
#endif
s>>x.range;
return s;
}



template<typename T>
std::ostream & operator<<(std::ostream &s, const Tensor<T> &x)
{
s<<x.shape;
sout= &s;
const_cast<Tensor<T> *>(&x)->loopover(&outputcallback<T>);
return s;
}

template <typename T>
std::istream & operator>>(std::istream &s, Tensor<T> &x)
{
s>>x.shape;
x.data.resize(x.calcsize()); x.calcrank();
FLATINDEX I(x.rank());
for(LA_largeindex i=0; i<x.data.size(); ++i)
	{
	for(int j=0; j<I.size(); ++j) s>>I[j];
	T val; s>>val;
	x.lhs(I) = val;
	}
return s;
}



template class Tensor<double>;
template class Tensor<std::complex<double> >;
template std::ostream & operator<<(std::ostream &s, const Tensor<double> &x);
template std::ostream & operator<<(std::ostream &s, const Tensor<std::complex<double> > &x);
template std::istream & operator>>(std::istream &s, Tensor<double> &x);
template std::istream & operator>>(std::istream &s, Tensor<std::complex<double> > &x);


}//namespace
