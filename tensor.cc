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
#include "bitvector.h"


namespace LA {

template<typename T>
int Tensor<T>:: calcrank()
{
int r=0;
for(int i=0; i<shape.size(); ++i)
        {
	const indexgroup *sh = &(* const_cast<const NRVec<indexgroup> *>(&shape))[i];
        if(sh->number<=0) laerror("empty index group"); //scalar must have shape.size()==0, not empty index group
        r+=sh->number;
        }
myrank=r;
return r;
}



template<typename T>
LA_largeindex Tensor<T>::calcsize()
{
if(shape.size()==0) return 1; //scalar
groupsizes.resize(shape.size()); 
cumsizes.resize(shape.size());
LA_largeindex s=1;
for(int i=0; i<shape.size(); ++i)
        {
	const indexgroup *sh = &(* const_cast<const NRVec<indexgroup> *>(&shape))[i];
        if(sh->number==0) laerror("empty index group");
        if(sh->range==0) return 0;
        cumsizes[i]=s;
        switch(sh->symmetry)
                {
                case 0:
                        s *= groupsizes[i] = longpow(sh->range,sh->number);
                        break;
                case 1:
                        s *= groupsizes[i] = simplicial(sh->number,sh->range);
                        break;
                case -1:
                        s *= groupsizes[i] = simplicial(sh->number,sh->range-sh->number+1);
                        break;
                default: 
                        laerror("illegal index group symmetry");
                        break;
                }
        }
if(s==0) laerror("empty tensor - perhaps antisymmetric in dim<rank");
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
void Tensor<T>::put(int fd, bool with_names) const
{
shape.put(fd,true);
groupsizes.put(fd,true);
cumsizes.put(fd,true);
data.put(fd,true);
if(with_names && names.size()!=0) names.put(fd,true);
}

template<typename T>
void Tensor<T>::get(int fd, bool with_names)
{
shape.get(fd,true);
myrank=calcrank(); //is not stored but recomputed
groupsizes.put(fd,true);
cumsizes.get(fd,true);
data.get(fd,true);
if(with_names) names.get(fd,true);
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
#ifdef LA_TENSOR_INDEXPOSITION
shape[0].upperindex=false;
#endif
shape[0].range=x.size();
calcsize();
}

template<typename T>
Tensor<T>::Tensor(const NRMat<T> &x, bool flat)
: data(&x(0,0),x.nrows()*x.ncols())
{
myrank=2;
if(x.nrows()==x.ncols() && !flat)
	{
	shape.resize(1);
	shape[0].number=2;
	shape[0].symmetry=0;
#ifndef LA_TENSOR_ZERO_OFFSET
	shape[0].offset=0;
#endif
#ifdef LA_TENSOR_INDEXPOSITION
shape[0].upperindex=false;
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
#ifdef LA_TENSOR_INDEXPOSITION
	shape[0].upperindex=false;
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
#ifdef LA_TENSOR_INDEXPOSITION
shape[0].upperindex=false;
#endif
shape[0].range=x.nrows();
calcsize();
}



template<typename T>
void loopingroups(Tensor<T> &t, int ngroup, int igroup, T **p, SUPERINDEX &I, void (*callback)(const SUPERINDEX &, T *))
{
LA_index istart,iend;
const indexgroup *sh = &(* const_cast<const NRVec<indexgroup> *>(&t.shape))[ngroup];
switch(sh->symmetry)
	{
	case 0:
		istart= sh->offset;
		iend=   sh->offset+sh->range-1;
		break;
	case 1:
		istart= sh->offset;
		if(igroup==sh->number-1) iend=   sh->offset+sh->range-1;
		else iend = I[ngroup][igroup+1];
		break;
	case -1:
		istart= sh->offset + igroup;
		if(igroup==sh->number-1) iend=   sh->offset+sh->range-1;
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
			const indexgroup *sh2 = &(* const_cast<const NRVec<indexgroup> *>(&t.shape))[newngroup];
			newigroup=sh2->number-1;
			}
		loopingroups(t,newngroup,newigroup,p,I,callback);
		}
	}
}


template<typename T>
void Tensor<T>::loopover(void (*callback)(const SUPERINDEX &, T *))
{
SUPERINDEX I(shape.size());
for(int i=0; i<I.size(); ++i)
	{
	const indexgroup *sh = &(* const_cast<const NRVec<indexgroup> *>(&shape))[i];
	I[i].resize(sh->number); 
	I[i] = sh->offset;
	}
T *pp=&data[0];
int ss=shape.size()-1;
const indexgroup *sh = &(* const_cast<const NRVec<indexgroup> *>(&shape))[ss];
loopingroups(*this,ss,sh->number-1,&pp,I,callback);
}


template<typename T>
void constloopingroups(const Tensor<T> &t, int ngroup, int igroup, const T **p, SUPERINDEX &I, void (*callback)(const SUPERINDEX &, const T *))
{
LA_index istart,iend;
const indexgroup *sh = &t.shape[ngroup];
switch(sh->symmetry)
	{
	case 0:
		istart= sh->offset;
		iend=   sh->offset+sh->range-1;
		break;
	case 1:
		istart= sh->offset;
		if(igroup==sh->number-1) iend=   sh->offset+sh->range-1;
		else iend = I[ngroup][igroup+1];
		break;
	case -1:
		istart= sh->offset + igroup;
		if(igroup==sh->number-1) iend=   sh->offset+sh->range-1;
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
			const indexgroup *sh2 = &(* const_cast<const NRVec<indexgroup> *>(&t.shape))[newngroup];
			newigroup=sh2->number-1;
			}
		constloopingroups(t,newngroup,newigroup,p,I,callback);
		}
	}
}


template<typename T>
void Tensor<T>::constloopover(void (*callback)(const SUPERINDEX &, const T *)) const
{
SUPERINDEX I(shape.size());
for(int i=0; i<I.size(); ++i)
	{
	const indexgroup *sh = &shape[i];
	I[i].resize(sh->number); 
	I[i] = sh->offset;
	}
const T *pp=&data[0];
int ss=shape.size()-1;
const indexgroup *sh = &shape[ss];
constloopingroups(*this,ss,sh->number-1,&pp,I,callback);
}


static std::ostream *sout;
template<typename T>
static void outputcallback(const SUPERINDEX &I, const T *v)
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
#ifdef LA_TENSOR_INDEXPOSITION
s<<x.upperindex<<" ";
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
#ifdef LA_TENSOR_INDEXPOSITION
s>> x.upperindex;
#endif
s>>x.range;
return s;
}


std::ostream & operator<<(std::ostream &s, const INDEXNAME &n)
{
s<<n.name<<" ";
return s;
}

std::istream & operator>>(std::istream &s, INDEXNAME &n)
{
s>>n.name;
return s;
}


template<typename T>
std::ostream & operator<<(std::ostream &s, const Tensor<T> &x)
{
s<<x.shape;
s<<x.names;
if(x.rank()==0) {s<<x.data[0]; return s;}
sout= &s;
x.constloopover(&outputcallback<T>);
return s;
}

template <typename T>
std::istream & operator>>(std::istream &s, Tensor<T> &x)
{
s>>x.shape;
s>>x.names;
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



template<typename T>
void loopovergroups(Tensor<T> &t, int ngroup, T **p, GROUPINDEX &I, void (*callback)(const GROUPINDEX &, T *))
{
for(LA_largeindex i = 0; i<t.groupsizes[ngroup]; ++i)
	{
	I[ngroup]=i;
	if(ngroup==0) (*callback)(I,(*p)++);
	else loopovergroups(t,ngroup-1,p,I,callback);
	}
}

template<typename T>
void Tensor<T>::grouploopover(void (*callback)(const GROUPINDEX &, T *))
{
GROUPINDEX I(shape.size());
T *pp= &data[0]; 
loopovergroups(*this,shape.size()-1,&pp,I,callback);
}




template<typename T>
void constloopovergroups(const Tensor<T> &t, int ngroup, const T **p, GROUPINDEX &I, void (*callback)(const GROUPINDEX &, const T *))
{
for(LA_largeindex i = 0; i<t.groupsizes[ngroup]; ++i)
	{
	I[ngroup]=i;
	if(ngroup==0) (*callback)(I,(*p)++);
	else constloopovergroups(t,ngroup-1,p,I,callback);
	}
}

template<typename T>
void Tensor<T>::constgrouploopover(void (*callback)(const GROUPINDEX &, const T *)) const
{
GROUPINDEX I(shape.size());
const T *pp= &data[0]; 
constloopovergroups(*this,shape.size()-1,&pp,I,callback);
}


const NRPerm<int> *help_p;
template<typename T> 
Tensor<T> *help_t;
template<typename T>
const Tensor<T> *help_tt;

template<typename T>
static void permutecallback(const GROUPINDEX &I, const T *v)
{
LA_largeindex target=0;
for(int i=0; i< help_t<T>->shape.size(); ++i)
	{
	target += I[(*help_p)[i+1]-1] * help_t<T>->cumsizes[i];
	}
help_t<T>->data[target] = *v;
}


//permutation of individual indices from permutation of index groups
NRPerm<int> group2flat_perm(const NRVec<indexgroup> &shape, const  NRPerm<int> &p)
{
int rank=0;
for(int i=0; i<shape.size(); ++i) rank+=shape[i].number;
NRPerm<int> q(rank);
int ii=1;
for(int i=0; i<shape.size(); ++i) 
	{
	int selgroup=p[i+1]-1;
	for(int j=0; j<shape[selgroup].number; ++j)
		{
		q[ii++] = 1+flatposition(selgroup,j,shape);
		}
	}
return q;
}


template<typename T>
Tensor<T> Tensor<T>::permute_index_groups(const NRPerm<int> &p) const
{
//std::cout <<"permute_index_groups permutation = "<<p<<std::endl;
NRVec<indexgroup> newshape=shape.permuted(p,true);
//std::cout <<"permute_index_groups newshape = "<<newshape<<std::endl;
Tensor<T> r(newshape);
if(is_named()) 
	{
	NRPerm<int> q=group2flat_perm(shape,p);
	r.names = names.permuted(q,true);
	}

//prepare statics for the callback
help_p = &p;
help_t<T> = &r;

//now rearrange the data
const_cast<Tensor<T> *>(this)->constgrouploopover(permutecallback<T>);
return r;
}


FLATINDEX superindex2flat(const SUPERINDEX &I)
{
int rank=0;
for(int i=0; i<I.size(); ++i) rank += I[i].size();
FLATINDEX J(rank);
int ii=0;
for(int i=0; i<I.size(); ++i)
	{
	for(int j=0; j<I[i].size(); ++j) J[ii++] = I[i][j];
	}
return J;
}

int flatposition(const INDEX &i, const NRVec<indexgroup> &shape) 
{return flatposition(i.group,i.index,shape);}

int flatposition(int group, int index, const NRVec<indexgroup> &shape)
{
int ii=0;
for(int g=0; g<group; ++g) ii+= shape[g].number;
ii += index;
return ii;
}

INDEX indexposition(int flatindex, const NRVec<indexgroup> &shape)
{
INDEX I={0,0};
if(flatindex<0) laerror("illegal index in indexposition");
for(int g=0; g<shape.size(); ++g)
	{
	I.group=g;
	if(flatindex<shape[g].number) {I.index=flatindex; return I;}
	flatindex-=shape[g].number;
	}
laerror("flatindex out of range");
return I;
}


template<typename T>
static void unwind_callback(const SUPERINDEX &I, T *v)
{
FLATINDEX J = superindex2flat(I);
FLATINDEX JP = J.permuted(*help_p,false);
//std::cout <<"TEST unwind_callback: from "<<JP<<" TO "<<J<<std::endl;
*v = (*help_tt<T>)(JP); //rhs operator() generates the redundant elements for the unwinded lhs tensor
}



template<typename T>
Tensor<T> Tensor<T>::unwind_index_group(int group) const
{
if(group==0) return *this; //is already the least significant group
if(group<0||group>=shape.size()) laerror("wrong group number in unwind_index_group");

NRPerm<int> p(shape.size());
p[1]= 1+group;
int ii=1;
if(ii==1+group) ii++; //skip this
for(int i=2; i<=shape.size(); ++i) 
	{
	p[i]=ii++;
	if(ii==1+group) ii++; //skip this
	}
if(!p.is_valid()) laerror("internal error in unwind_index");
return permute_index_groups(p);
}



template<typename T>
Tensor<T> Tensor<T>::unwind_index(int group, int index) const
{
if(group<0||group>=shape.size()) laerror("wrong group number in unwind_index");
if(index<0||index>=shape[group].number)  laerror("wrong index number in unwind_index");

if(group==0 && index==0 && shape[0].number == 1) return *this;
if(group==0 && index==0 && shape[0].symmetry==0) //formal split of 1 index without data rearrangement
	{
	Tensor<T> r(*this);
	r.split_index_group1(0);
	return r;
	}
if(shape[group].number==1) return unwind_index_group(group); //single index in the group

//general case - recalculate the shape and allocate the new tensor
NRVec<indexgroup> newshape(shape.size()+1);
newshape[0].number=1;
newshape[0].symmetry=0;
newshape[0].range=shape[group].range;
#ifndef LA_TENSOR_ZERO_OFFSET
newshape[0].offset = shape[group].offset;
#endif
#ifdef LA_TENSOR_INDEXPOSITION
newshape[0].upperindex =  shape[group].upperindex;
#endif
int flatindex=0; //(group,index) in flat form
for(int i=0; i<shape.size(); ++i)
	{
	newshape[i+1] = shape[i];
	if(i==group)
		{
		--newshape[i+1].number;
		flatindex += index;
		}
	else flatindex += shape[i].number;
	}


//std::cout <<"unwind new shape = "<<newshape<<std::endl;

Tensor<T> r(newshape);
if(r.rank()!=rank()) laerror("internal error 2 in unwind_index");

//compute the corresponding permutation of FLATINDEX for use in the callback
NRPerm<int> indexperm(rank());
indexperm[1]=flatindex+1;
int ii=1;
if(ii==flatindex+1) ii++;
for(int i=2; i<=rank(); ++i) 
	{
	indexperm[i] = ii++;
	if(ii==flatindex+1) ii++; //skip this
	}
if(!indexperm.is_valid()) 
	{
	//std::cout << "indexperm = "<<indexperm<<std::endl;
	laerror("internal error 3 in unwind_index");
	}

//std::cout <<"unwind permutation = "<<indexperm<<std::endl;

if(is_named())
	{
	r.names = names.permuted(indexperm,true);
	//std::cout <<"unwind new names = "<<r.names<<std::endl;
	}

//loop recursively and do the unwinding
help_tt<T> = this;
help_p = &indexperm;
r.loopover(unwind_callback);
return r;
}


template<typename T>
static void flatten_callback(const SUPERINDEX &I, T *v)
{
FLATINDEX J = superindex2flat(I);
//std::cout <<"TEST flatten_callback: from "<<JP<<" TO "<<J<<std::endl;
*v = (*help_tt<T>)(J); //rhs operator() generates the redundant elements for the unwinded lhs tensor
}       
//


template<typename T>
Tensor<T> Tensor<T>::flatten(int group) const
{
if(group>=shape.size()) laerror("too high group number in flatten");
if(is_flat())
	{
	if(has_symmetry()) //get rid of formal symemtry
		{
		Tensor<T> r(*this);
		r.shape.copyonwrite();
                for(int g=0; g<r.shape.size(); ++g) r.shape[g].symmetry=0;
                return r;
		}
	else
		return *this;
	}
if(group>=0) //single group
	{
	if(shape[group].number==1) 
		{
		if(shape[group].symmetry==0) return *this;
		else
			{
			Tensor<T> r(*this);
			r.shape[group].symmetry=0;
			return r;
			}
		}
	if(shape[group].symmetry==0)
		{
		Tensor<T> r(*this);
		r.split_index_group(group);
		return r;
		}
	}
if(group<0 && !is_compressed())
	{
	Tensor<T> r(*this);
        for(int g=0; g<shape.size(); ++g) 
		{
		if(shape[g].number>1) r.split_index_group(g);
		}
	for(int g=0; g<r.shape.size(); ++g) r.shape[g].symmetry=0;
        return r;
	}

//general case 
int newsize;
if(group<0) newsize=rank();
else newsize=shape.size()+shape[group].number-1;

//build new shape
NRVec<indexgroup> newshape(newsize);
int gg=0;
for(int g=0; g<shape.size(); ++g)
	{
	if((group<0 ||g==group) && shape[g].number>1) //flatten this group
		{
		for(int i=0; i<shape[g].number; ++i)
			{
			newshape[gg].symmetry=0;
			newshape[gg].number=1;
			newshape[gg].range=shape[g].range;
#ifndef LA_TENSOR_ZERO_OFFSET
			newshape[gg].offset = shape[g].offset;
#endif 
#ifdef LA_TENSOR_INDEXPOSITION
			newshape[gg].upperindex = shape[g].upperindex;
#endif
			gg++;
			}
		}
	else //preserve this group
		{
		newshape[gg++] = shape[g];
		}
	}

//std::cout <<"Flatten new shape = "<<newshape<<std::endl;


//decompress the tensor data
Tensor<T> r(newshape);
r.names=names;
help_tt<T> = this;
r.loopover(flatten_callback);
return r;
}




template<typename T>
Tensor<T> Tensor<T>::unwind_indices(const INDEXLIST &il) const
{
if(il.size()==0) return *this;
if(il.size()==1) return unwind_index(il[0]);

for(int i=0; i<il.size(); ++i)
	{
	if(il[i].group<0||il[i].group>=shape.size()) laerror("wrong group number in unwind_indices");
	if(il[i].index<0||il[i].index>=shape[il[i].group].number)  laerror("wrong index number in unwind_indices");
	}

//all indices are solo in their groups - permute groups
bool sologroups=true;
int nonsolo=0;
for(int i=0; i<il.size(); ++i)
	if(shape[il[i].group].number!=1) {sologroups=false; ++nonsolo;}
if(sologroups)
	{
	NRPerm<int> p(shape.size());
	bitvector waslisted(shape.size());
	waslisted.clear();
	for(int i=0; i<il.size(); ++i) 
		{
		p[1+i] = 1+il[i].group;
		waslisted.set(il[i].group);
		}
	int ii=il.size();
	for(int i=0; i<shape.size(); ++i)
		{
		if(!waslisted[i])
			{
			waslisted.set(i);
			p[1+ii] = 1+i;
			ii++;
			}
		}

	if(!p.is_valid()) laerror("internal error in unwind_indices");
	if(p.is_identity()) return *this;
	else return permute_index_groups(p);
	}


//general case - recalculate the shape and allocate the new tensor
NRVec<indexgroup> oldshape(shape); 
oldshape.copyonwrite();
NRVec<indexgroup> newshape(shape.size()+nonsolo);

//first the unwound indices as solo groups
for(int i=0; i<il.size(); ++i)
	{
	newshape[i].number=1;
	newshape[i].symmetry=0;
	newshape[i].range=shape[il[i].group].range;
#ifndef LA_TENSOR_ZERO_OFFSET
	newshape[i].offset = shape[il[i].group].offset;
#endif
#ifdef LA_TENSOR_INDEXPOSITION
	newshape[i].upperindex =  shape[il[i].group].upperindex;
#endif
	oldshape[il[i].group].number --;
	}

//then the remaining groups with one index removed, if nonempty
int ii=il.size();
int emptied_groups=0;
for(int i=0; i<oldshape.size(); ++i)
	if(oldshape[i].number>0)
		{
		newshape[ii++] = oldshape[i];
		}
	else
		++emptied_groups;

if(emptied_groups) newshape.resize(newshape.size()-emptied_groups,true);

Tensor<T> r(newshape);
if(r.rank()!=rank()) laerror("internal error 2 in unwind_indces");

//compute the corresponding permutation of FLATINDEX for use in the callback
NRPerm<int> indexperm(rank());
bitvector waslisted(rank());
waslisted.clear();
//first unwound indices
ii=0;
for(int i=0; i<il.size(); ++i)
	{
	int pos= flatposition(il[i],shape);
	indexperm[1+ii] = 1+pos;
	waslisted.set(pos);
	++ii;
	}

//the remaining index groups
for(int g=0; g<shape.size(); ++g)
	for(int i=0; i<shape[g].number; ++i)
		{
		int pos= flatposition(g,i,shape);
		if(!waslisted[pos])
			{
			waslisted.set(pos);
			indexperm[1+ii] = 1+pos;
			++ii;	
			}
		}

if(!indexperm.is_valid()) 
	{
	std::cout << "indexperm = "<<indexperm<<std::endl;
	laerror("internal error 3 in unwind_indices");
	}

if(is_named())
        {
        r.names = names.permuted(indexperm,true);
        //std::cout <<"unwind new names = "<<r.names<<std::endl;
        }

//loop recursively and do the unwinding
help_tt<T> = this;
help_p = &indexperm;
r.loopover(unwind_callback);
return r;
}

template<typename T>
static void auxmatmult(int nn, int mm, int kk, T *r, const T *a, const T *b,  T alpha=1, T beta=0, bool conjugate=false) //R(nn,mm) = A(nn,kk) * B^T(mm,kk)
{
for(int i=0; i<nn; ++i) for(int j=0; j<mm; ++j)
	{
	if(beta==0)  r[i*mm+j]=0; else r[i*mm+j] *= beta;
	for(int k=0; k<kk; ++k) r[i*mm+j] += alpha * a[i*kk+k] * conjugate? LA_traits<T>::conjugate(b[j*kk+k]) : b[j*kk+k];
	}
}


template<>
void auxmatmult<double>(int nn, int mm, int kk, double *r, const double *a, const double *b, double alpha, double beta, bool conjugate)
{
cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nn, mm, kk, alpha, a, kk, b, kk, beta, r, mm);
}

template<>
void auxmatmult<std::complex<double> >(int nn, int mm, int kk, std::complex<double> *r, const std::complex<double> *a, const std::complex<double> *b,  std::complex<double> alpha,  std::complex<double> beta, bool conjugate)
{
cblas_zgemm(CblasRowMajor, CblasNoTrans, (conjugate?CblasConjTrans:CblasTrans), nn, mm, kk, &alpha, a, kk, b, kk, &beta, r, mm);
}



template<typename T>
void Tensor<T>::addgroupcontraction(const Tensor &rhs1, int group, const Tensor &rhs, int rhsgroup, T alpha, T beta, bool doresize, bool conjugate1, bool conjugate)
{
if(group<0||group>=rhs1.shape.size()) laerror("wrong group number in contraction");
if(rhsgroup<0||rhsgroup>=rhs.shape.size()) laerror("wrong rhsgroup number in contraction");
if(rhs1.shape[group].offset != rhs.shape[rhsgroup].offset) laerror("incompatible index offset in contraction");
if(rhs1.shape[group].range != rhs.shape[rhsgroup].range) laerror("incompatible index range in contraction");
if(rhs1.shape[group].symmetry != rhs.shape[rhsgroup].symmetry) laerror("incompatible index symmetry in addgroupcontraction");
if(rhs1.shape[group].symmetry == 1) laerror("addgroupcontraction not implemented for symmetric index groups");
#ifdef LA_TENSOR_INDEXPOSITION
if(rhs1.shape[group].upperindex ^ rhs.shape[rhsgroup].upperindex == false) laerror("can contact only upper with lower index");
#endif

const Tensor<T> u = conjugate1? (rhs1.unwind_index_group(group)).conjugate() : rhs1.unwind_index_group(group);
const Tensor<T> rhsu = rhs.unwind_index_group(rhsgroup);

NRVec<indexgroup> newshape(u.shape.size()+rhsu.shape.size()-2);
int ii=0;
for(int i=1; i<rhsu.shape.size(); ++i) newshape[ii++] = rhsu.shape[i];
for(int i=1; i<u.shape.size(); ++i) newshape[ii++] = u.shape[i]; //this tensor will have more significant indices than the rhs one

NRVec<INDEXNAME> newnames;
if(u.is_named() && rhsu.is_named())
	{
	for(int i=0; i<u.shape[0].number; ++i) 
		{
		if(u.names[i]!=rhsu.names[i]) laerror("contraction index name mismatch in addgroupcontraction");
		}
	newnames.resize(u.names.size()+rhsu.names.size()-2*u.shape[0].number);
	int ii=0;
	for(int i=rhsu.shape[0].number; i<rhsu.names.size(); ++i)  newnames[ii++] = rhsu.names[i];
	for(int i=u.shape[0].number; i<u.names.size(); ++i)  newnames[ii++] = u.names[i];
	}

if(doresize)
	{
	if(beta!= (T)0) laerror("resize in addgroupcontraction requires beta=0");
	resize(newshape);
	names=newnames;
	}
else
	{
	if(shape!=newshape) laerror("tensor shape mismatch in addgroupcontraction");
	if(is_named() && names!=newnames) laerror("remaining index names mismatch in addgroupcontraction"); 
	}

int nn,mm,kk;
kk=u.groupsizes[0];
if(kk!=rhsu.groupsizes[0]) laerror("internal error in addgroupcontraction");
T factor=alpha;
if(u.shape[0].symmetry== -1) factor=alpha*(T)factorial(u.shape[0].number);
if(u.shape[0].symmetry== 1) laerror("addgroupcontraction not implemented for symmetric index groups");
nn=1; for(int i=1; i<u.shape.size(); ++i) nn*= u.groupsizes[i];
mm=1; for(int i=1; i<rhsu.shape.size(); ++i) mm*= rhsu.groupsizes[i];
data.copyonwrite();
auxmatmult<T>(nn,mm,kk,&data[0],&u.data[0], &rhsu.data[0],factor,beta,conjugate);
}




//Conntraction could be implemented without the temporary storage for unwinding, but then we would need
//double recursion over indices of both tensors. Hopefully using the matrix multiplication here
//makes it also more efficient, even for (anti)symmetric indices
//The index unwinding is unfortunately a big burden, and in principle could be eliminated in case of non-symmetric indices
//
template<typename T>
void Tensor<T>::addcontraction(const Tensor &rhs1, int group, int index, const Tensor &rhs, int rhsgroup, int rhsindex, T alpha, T beta, bool doresize, bool conjugate1, bool conjugate)
{
if(group<0||group>=rhs1.shape.size()) laerror("wrong group number in contraction");
if(rhsgroup<0||rhsgroup>=rhs.shape.size()) laerror("wrong rhsgroup number in contraction");
if(index<0||index>=rhs1.shape[group].number)  laerror("wrong index number in conntraction");
if(rhsindex<0||rhsindex>=rhs.shape[rhsgroup].number)  laerror("wrong index number in conntraction");
if(rhs1.shape[group].offset != rhs.shape[rhsgroup].offset) laerror("incompatible index offset in contraction");
if(rhs1.shape[group].range != rhs.shape[rhsgroup].range) laerror("incompatible index range in contraction");
#ifdef LA_TENSOR_INDEXPOSITION
if(rhs1.shape[group].upperindex ^ rhs.shape[rhsgroup].upperindex == false) laerror("can contact only upper with lower index");
#endif

const Tensor<T> u = conjugate1? (rhs1.unwind_index(group,index)).conjugate() : rhs1.unwind_index(group,index);
const Tensor<T> rhsu = rhs.unwind_index(rhsgroup,rhsindex);


NRVec<indexgroup> newshape(u.shape.size()+rhsu.shape.size()-2);
int ii=0;
for(int i=1; i<rhsu.shape.size(); ++i) newshape[ii++] = rhsu.shape[i];
for(int i=1; i<u.shape.size(); ++i) newshape[ii++] = u.shape[i]; //this tensor will have more significant indices than the rhs one

NRVec<INDEXNAME> newnames;
if(u.is_named() && rhsu.is_named())
	{
	if(u.names[0]!=rhsu.names[0]) laerror("contraction index name mismatch in addcontraction");
	newnames.resize(u.names.size()+rhsu.names.size()-2);
	int ii=0;
	for(int i=1; i<rhsu.names.size(); ++i)  newnames[ii++] = rhsu.names[i];
	for(int i=1; i<u.names.size(); ++i)  newnames[ii++] = u.names[i];
	}

if(doresize)
	{
	if(beta!= (T)0) laerror("resize in addcontraction requires beta=0");
	resize(newshape);
	names=newnames;
	}
else
	{
	if(shape!=newshape) laerror("tensor shape mismatch in addcontraction");
	if(is_named() && names!=newnames) laerror("remaining index names mismatch in addcontraction"); 
	}

int nn,mm,kk;
kk=u.groupsizes[0];
if(kk!=rhsu.groupsizes[0]) laerror("internal error in addcontraction");
nn=1; for(int i=1; i<u.shape.size(); ++i) nn*= u.groupsizes[i];
mm=1; for(int i=1; i<rhsu.shape.size(); ++i) mm*= rhsu.groupsizes[i];
data.copyonwrite();
auxmatmult<T>(nn,mm,kk,&data[0],&u.data[0], &rhsu.data[0],alpha,beta,conjugate);
}



template<typename T>
void Tensor<T>::addcontractions(const Tensor &rhs1, const INDEXLIST &il1, const Tensor &rhs2, const INDEXLIST &il2,  T alpha, T beta, bool doresize, bool conjugate1, bool conjugate2)
{
if(il1.size()==0) laerror("empty contraction - outer product not implemented");
if(il1.size()!=il2.size()) laerror("mismatch in index lists in addcontractions");
for(int i=0; i<il1.size(); ++i)
	{
	if(il1[i].group<0||il1[i].group>=rhs1.shape.size()) laerror("wrong group1 number in contractions");
	if(il2[i].group<0||il2[i].group>=rhs2.shape.size()) laerror("wrong group2 number in contractions");
	if(il1[i].index<0||il1[i].index>=rhs1.shape[il1[i].group].number)  laerror("wrong index1 number in conntractions");
	if(il2[i].index<0||il2[i].index>=rhs2.shape[il2[i].group].number)  laerror("wrong index2 number in conntractions");
	if(rhs1.shape[il1[i].group].offset != rhs2.shape[il2[i].group].offset) laerror("incompatible index offset in contractions");
	if(rhs1.shape[il1[i].group].range  != rhs2.shape[il2[i].group].range) laerror("incompatible index range in contractions");
#ifdef LA_TENSOR_INDEXPOSITION
	if(rhs1.shape[il1[i].group].upperindex ^ rhs2.shape[il2[i].group].upperindex == false) laerror("can contact only upper with lower index");
#endif
	}

const Tensor<T> u = conjugate1? (rhs1.unwind_indices(il1)).conjugateme() : rhs1.unwind_indices(il1);
const Tensor<T> rhsu = rhs2.unwind_indices(il2);


NRVec<indexgroup> newshape(u.shape.size()+rhsu.shape.size()-2*il1.size());
int ii=0;
for(int i=il1.size(); i<rhsu.shape.size(); ++i) newshape[ii++] = rhsu.shape[i];
for(int i=il1.size(); i<u.shape.size(); ++i) newshape[ii++] = u.shape[i]; //this tensor will have more significant indices than the rhs one

NRVec<INDEXNAME> newnames;
if(u.is_named() && rhsu.is_named())
        {
        for(int i=0; i<il1.size(); ++i) if(u.names[i]!=rhsu.names[i]) laerror("contraction index name mismatch in addcontractions");
        newnames.resize(u.names.size()+rhsu.names.size()-2*il1.size());
        int ii=0;
        for(int i=il1.size(); i<rhsu.names.size(); ++i)  newnames[ii++] = rhsu.names[i];
        for(int i=il1.size(); i<u.names.size(); ++i)  newnames[ii++] = u.names[i];
        }


if(doresize)
	{
	if(beta!= (T)0) laerror("resize in addcontractions requires beta=0");
	resize(newshape);
	names=newnames;
	}
else
	{
	if(shape!=newshape) laerror("tensor shape mismatch in addcontraction");
	if(is_named() && names!=newnames) laerror("remaining index names mismatch in addcontractions");
	}

int nn,mm,kk;
kk=1;
int kk2=1;
for(int i=0; i<il1.size(); ++i)
	{
	kk *= u.groupsizes[i];
	kk2 *= rhsu.groupsizes[i];
	}
if(kk!=kk2) laerror("internal error in addcontractions");

nn=1; for(int i=il1.size(); i<u.shape.size(); ++i) nn*= u.groupsizes[i];
mm=1; for(int i=il1.size(); i<rhsu.shape.size(); ++i) mm*= rhsu.groupsizes[i];
data.copyonwrite();
auxmatmult<T>(nn,mm,kk,&data[0],&u.data[0], &rhsu.data[0],alpha,beta,conjugate2);
}




template<typename T>
static const PermutationAlgebra<int,T> *help_pa;

static bool help_inverse;

template<typename T>
static T help_alpha;

template<typename T>
static void permutationalgebra_callback(const SUPERINDEX &I, T *v)
{
FLATINDEX J = superindex2flat(I);
for(int p=0; p<help_pa<T>->size(); ++p)
	{
	FLATINDEX Jp = J.permuted((*help_pa<T>)[p].perm,help_inverse);
	*v += help_alpha<T> * (*help_pa<T>)[p].weight * (*help_t<T>)(Jp);
	}
}


static int help_tn;
template<typename T>
static Tensor<T> *help_tv;

template<typename T>
static void permutationalgebra_callback2(const SUPERINDEX &I, T *v)
{
FLATINDEX J = superindex2flat(I);
for(int p=0; p<help_pa<T>->size(); ++p)
        {
        FLATINDEX Jp = J.permuted((*help_pa<T>)[p].perm,help_inverse);
	T product;
	//build product from the vector of tensors using the Jp index
	int rankshift=0;
	for(int i=0; i<help_tn; ++i)
		{
		int rank = help_tv<T>[i].rank();
		FLATINDEX indi = Jp.subvector(rankshift,rankshift+rank-1);
		if(i==0) product= help_tv<T>[i](indi);
		else product *= help_tv<T>[i](indi);
		rankshift +=  rank;
		}
        *v += help_alpha<T> * (*help_pa<T>)[p].weight * product;
        }
}


template<typename T>
void Tensor<T>::apply_permutation_algebra(const Tensor<T> &rhs, const PermutationAlgebra<int,T> &pa, bool inverse, T alpha, T beta)
{
if(beta!=(T)0) {if(beta!=(T)1) *this *= beta;} else clear();
if(alpha==(T)0) return;
copyonwrite();

if(rank()!=rhs.rank()) laerror("rank mismatch in apply_permutation_algebra");

help_t<T> = const_cast<Tensor<T> *>(&rhs);
help_pa<T> = &pa;
help_inverse = inverse;
help_alpha<T> = alpha;
loopover(permutationalgebra_callback);
}


template<typename T>
void Tensor<T>::apply_permutation_algebra(const NRVec<Tensor<T> > &rhsvec, const PermutationAlgebra<int,T> &pa, bool inverse, T alpha, T beta)
{
if(beta!=(T)0) {if(beta!=(T)1) *this *= beta;} else clear();
if(alpha==(T)0) return;
copyonwrite();

int totrank=0;
for(int i=0; i<rhsvec.size(); ++i) totrank+=rhsvec[i].rank();
if(totrank!=rank()) laerror("rank mismatch in apply_permutation_algebra");

help_tv<T> = const_cast<Tensor<T> *>(&rhsvec[0]);
help_tn = rhsvec.size();
help_pa<T> = &pa;
help_inverse = inverse;
help_alpha<T> = alpha;
loopover(permutationalgebra_callback2);
}





template<typename T>
void Tensor<T>::split_index_group(int group)
{
const indexgroup *sh = &(* const_cast<const NRVec<indexgroup> *>(&shape))[0];
if(group<0||group >= shape.size()) laerror("illegal index group number");
if(sh[group].number==1) return; //nothing to split
if(sh[group].symmetry!=0) laerror("only non-symmetric index group can be splitted, use flatten instead");

NRVec<indexgroup> newshape(shape.size()+sh[group].number-1);
int gg=0;
for(int g=0; g<shape.size(); ++g)
	{
	if(g==group)
		{
		for(int i=0; i<sh[group].number; ++i)
			{
			newshape[gg] = sh[group]; 
			newshape[gg].number = 1;
			gg++;
			}
		}
	else
		newshape[gg++] = sh[g];
	}

shape=newshape;
LA_largeindex newsize = calcsize(); //recalculate auxiliary arrays
if(data.size()!=newsize) laerror("internal error in split_index_group");
}


template<typename T>
void Tensor<T>::split_index_group1(int group)
{
if(group<0||group >= shape.size()) laerror("illegal index group number");
if(shape[group].number==1) return; //nothing to split
if(shape[group].symmetry!=0) laerror("only non-symmetric index group can be splitted, use flatten instead");

NRVec<indexgroup> newshape(shape.size()+1);
int gg=0;
for(int g=0; g<shape.size(); ++g)
	{
	if(g==group)
		{
		newshape[gg] = shape[group]; 
		newshape[gg].number = 1;
		gg++;
		newshape[gg] = shape[group]; 
                newshape[gg].number -= 1;
                gg++;
		}
	else
		newshape[gg++] = shape[g];
	}

shape=newshape;
LA_largeindex newsize = calcsize(); //recalculate auxiliary arrays
if(data.size()!=newsize) laerror("internal error in split_index_group");
}


template<typename T>
void Tensor<T>:: merge_adjacent_index_groups(int groupfrom, int groupto)
{
if(groupfrom<0||groupfrom>= shape.size()) laerror("illegal index group number");
if(groupto<0||groupto>= shape.size()) laerror("illegal index group number");
if(groupfrom==groupto) return;
if(groupfrom>groupto) {int t=groupfrom; groupfrom=groupto; groupto=t;}

int newnumber=0;
for(int g=groupfrom; g<=groupto; ++g)
	{
	if(shape[g].symmetry!=0) laerror("only non-symmetric index groups can be merged");
	if(shape[g].offset!=shape[groupfrom].offset) laerror("incompatible offset in merge_adjacent_index_groups");
	if(shape[g].range!=shape[groupfrom].range) laerror("incompatible range in merge_adjacent_index_groups");
	newnumber += shape[g].number;
	}

NRVec<indexgroup> newshape(shape.size()-(groupto-groupfrom+1)+1);
for(int g=0; g<=groupfrom; ++g) newshape[g]=shape[g];
newshape[groupfrom].number=newnumber;
for(int g=groupfrom+1; g<newshape.size(); ++g) newshape[g]=shape[g+groupto-groupfrom];

shape=newshape;
LA_largeindex newsize = calcsize(); //recalculate auxiliary arrays
if(data.size()!=newsize) laerror("internal error in merge_adjacent_index_groups");
}


template<typename T>
Tensor<T> Tensor<T>::merge_index_groups(const NRVec<int> &groups) const
{
if(groups.size()<=1) return *this;
NRPerm<int> p(shape.size());
int gg=0;
bitvector selected(shape.size());
selected.clear();
for(int g=0; g<groups.size(); ++g) 
	{
	if(groups[g]<0||groups[g]>=shape.size()) laerror("illegal group number in merge_index_groups");
	if(selected[g]) laerror("repeated group number in merge_index_groups");
	selected.set(g);
	p[gg++] = 1+groups[g];
	}
for(int g=0; g<shape.size(); ++g)
	if(!selected[g]) 
		p[gg++] = 1+g;
if(gg!=shape.size() || !p.is_valid()) laerror("internal error in merge_index_groups");

Tensor<T> r =  permute_index_groups(p); //takes care of names permutation too
r.merge_adjacent_index_groups(0,groups.size()-1); //flat names invariant
return r;
}



template<typename T>
T Tensor<T>::dot(const Tensor &rhs) const
{       
if(shape!=rhs.shape) laerror("incompatible tensor shapes in dot");
if(is_named() && rhs.is_named() && names!=rhs.names) laerror("incompatible tensor index names in dot");
T factor=1;     
for(int i=0; i<shape.size(); ++i)
        {       
        if(shape[i].symmetry==1) laerror("unsupported index group symmetry in dot");
        if(shape[i].symmetry== -1) factor *= (T)factorial(shape[i].number);
        }
return factor * data.dot(rhs.data);
}       



//NOTE: Tucker of rank=2 is inherently inefficient - result is a diagonal tensor stored in full and 2 calls to SVD
//we could avoid the second SVD, but the wasteful storage and reconstruction would remain
//
template<typename T> 
NRVec<NRMat<T> > Tensor<T>::Tucker(typename LA_traits<T>::normtype thr, bool inverseorder)
{
int r=rank();
NRVec<NRMat<T> > ret(r);
if(r<1) laerror("illegal rank in Tucker");
copyonwrite();

if(r==1) //create an analogous output for the trivial case 
	{
	typename LA_traits<T>::normtype N=data.norm();
	data*= (1/N);
	ret[0]=NRMat<T>(data,data.size(),1);
	shape[0].range=1;
	data.resize(calcsize()); 
	calcrank();
	data[0]=(T)N;
	return ret;
	}

//loop over all indices; relies on the fact that unwinding does not change order of remaining indices
for(int i=0; i<r; ++i)
	{
	INDEX I=indexposition(i,shape);
	NRMat<T> um;
	NRVec<indexgroup> ushape;
	{
	Tensor<T> uu=unwind_index(I);
	ushape=uu.shape; //ushape.copyonwrite(); should not be needed
	um=uu.matrix();
	}
	int mini=um.nrows(); if(um.ncols()<mini) mini=um.ncols(); //compact SVD, expect descendingly sorted values
	NRMat<T> u(um.nrows(),mini),vt(mini,um.ncols());
	NRVec<typename LA_traits<T>::normtype> w(mini);
	//std::cout << "decomposing "<<um<<std::endl;
	singular_decomposition(um,&u,w,&vt,0);
	//std::cout << "resulting U "<<u<<std::endl;
	//std::cout << "resulting W "<<w<<std::endl;
	//std::cout << "resulting VT "<<vt<<std::endl;
	int umnr=um.nrows();
	int umnc=um.ncols();
	um.resize(0,0); //deallocate
	int preserve=mini;
	for(int k=0; k<mini; ++k) if(w[k]<thr) {preserve=k; break;}
	if(preserve==0) laerror("singular tensor in Tucker decomposition");
	NRMat<T> umnew;
	//std::cout <<"TEST "<<i<<" mini preserve "<<mini<<" "<<preserve<<std::endl;
	if(preserve<mini)
		{
		vt=vt.submatrix(0,preserve-1,0,umnc-1);
		w=w.subvector(0,preserve-1);
		umnew=u.submatrix(0,umnr-1,0,preserve-1);
		}
	else umnew=u;
	ret[(inverseorder? r-i-1 : i)]=vt.transpose(true);
	umnew.diagmultr(w);
	//rebuild tensor of the preserved shape from matrix
	ushape[0].range=preserve;
	{
	NRVec<T> newdata(umnew);
	umnew.resize(0,0);//deallocate
	*this = Tensor(ushape,newdata);
	}
	}
if(!is_flat()) laerror("this should not happen");
if(!inverseorder)
	{
	NRPerm<int> p(r);
	for(int i=1; i<=r; ++i) p[r-i+1]=i;
	*this =  permute_index_groups(p);
	}
return ret;
}



template<typename T>
Tensor<T> Tensor<T>::inverseTucker(const NRVec<NRMat<T> > &x, bool inverseorder) const
{
if(rank()!=x.size()) laerror("input of inverseTucker does not match rank");
Tensor<T> tmp(*this);
Tensor<T> r;
if(!is_flat()) laerror("inverseTucker only for flat tensors as produced by Tucker");
for(int i=0; i<rank(); ++i)
	{
	Tensor<T> mat(x[i],true);
	r= tmp.contraction(i,0,mat,0,0,(T)1,false,false);
	if(i<rank()-1) 
		{
		tmp=r;
		r.deallocate();
		}
	}
if(!inverseorder)
        {
        NRPerm<int> p(r.rank());
        for(int i=1; i<=r.rank(); ++i) p[r.rank()-i+1]=i;
        return r.permute_index_groups(p);
        }
else
	return r;
}


template<typename T> 
int Tensor<T>::findflatindex(const INDEXNAME nam) const
{
if(!is_named()) laerror("tensor indices were not named");
for(int i=0; i<names.size(); ++i)
	{
	if(nam==names[i]) return i;
	}
return -1;
}

template<typename T>
INDEX Tensor<T>::findindex(const INDEXNAME nam) const
{
int n=findflatindex(nam);
if(n<0) laerror("index with this name was not found");
return indexposition(n,shape);
}

template<typename T>
NRVec<INDEX> Tensor<T>::findindexlist(const NRVec<INDEXNAME> &names) const
{
int n=names.size();
NRVec<INDEX> ind(n);
for(int i=0; i<n; ++i) ind[i] = findindex(names[i]);
return ind;
}



template class Tensor<double>;
template class Tensor<std::complex<double> >;
template std::ostream & operator<<(std::ostream &s, const Tensor<double> &x);
template std::ostream & operator<<(std::ostream &s, const Tensor<std::complex<double> > &x);
template std::istream & operator>>(std::istream &s, Tensor<double> &x);
template std::istream & operator>>(std::istream &s, Tensor<std::complex<double> > &x);


}//namespace
