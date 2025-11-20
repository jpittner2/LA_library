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
#include <string.h>
#include <bits/stdc++.h>
#include <time.h>


namespace LA {

template<typename T>
int Tensor<T>:: calcrank()
{
int r=0;
for(int i=0; i<shape.size(); ++i)
        {
	const INDEXGROUP *sh = &(* const_cast<const NRVec<INDEXGROUP> *>(&shape))[i];
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
	const INDEXGROUP *sh = &(* const_cast<const NRVec<INDEXGROUP> *>(&shape))[i];
        if(sh->number==0) laerror("empty index group");
        if(sh->range==0) return 0;
        cumsizes[i]=s;
        switch(sh->symmetry)
                {
                case 0:
                        s *= groupsizes[i] = longpow(sh->range,sh->number);
                        break;
                case 2:
                case -2:
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
for(int i=0; i<I.size(); ++i) if(I[i]<g.offset || I[i] >= g.offset+g.range) 
	{
	std::cout<<"TENSOR INDEX PROBLEM in group " <<g<<" with index "<<I<<std::endl;
	laerror("index out of range in tensor subindex");
	}
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
		if(g.symmetry==0) {*sign=1; return (I[1]-g.offset)*g.range+I[0]-g.offset;};
		LA_index i0,i1;
		if(I[0]>I[1]) {i1=I[0]; i0=I[1]; *sign=g.symmetry;} else {i1=I[1]; i0=I[0]; *sign=1;}
		i0 -= g.offset;
		i1 -= g.offset;
		if(g.symmetry == -1) //antisymmetric
			{
			if(i0==i1) {*sign=0; return -1;}
			return i1*(i1-1)/2+i0;
			}
		else	//symmetric, hermitian, antihermitian
			{
			return i1*(i1+1)/2+i0;
			}
		}
		break;

	default: //general case
		{
                if(g.symmetry==0) //rectangular case
			{
			*sign=1;
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
		int nswaps=netsort(II.size(),&II[0]);
		*sign= (nswaps&1) ? g.symmetry : 1;
		if(g.symmetry == -1) //antisymmetric - do not store zero diagonal
			{
			for(int i=0; i<I.size()-1; ++i) 
				if(II[i]==II[i+1])
					{*sign=0; return -1;} //identical indices of antisymmetric tensor

			LA_largeindex r=0;
                        for(int i=0; i<II.size(); ++i) r += simplicial(i+1,II[i]-i);
                        return r;
			}
		else	//symmetric, hermitian, antihermitian
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
	case 2:
	case -2:
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

//group-like multiplication table to combine symmetry adjustments due to several index groups
static const int signmultab[5][5] = {
{1,2,0,-2,-1},
{2,1,0,-1,-2},
{0,0,0,0,0},
{-2,-1,0,1,2},
{-1,-2,0,2,1}
};


static inline  int signmult(int s1, int s2)
{
return signmultab[s1+2][s2+2];
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
			std::cout<<"TENSOR INDEX PROBLEM group no. "<<i<<" index no. "<<j<<" should be between "<<shape[i].offset<<" and "<<shape[i].offset+shape[i].range-1<<std::endl;
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
	if(LA_traits<T>::is_complex()) *sign = signmult(*sign,gsign); else *sign *= gsign;
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
	if(LA_traits<T>::is_complex()) *sign = signmult(*sign,gsign); else *sign *= gsign;
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
const INDEXGROUP *sh = &(* const_cast<const NRVec<INDEXGROUP> *>(&t.shape))[ngroup];
switch(sh->symmetry)
	{
	case 0:
		istart= sh->offset;
		iend=   sh->offset+sh->range-1;
		break;
	case 2:
	case -2:
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
			const INDEXGROUP *sh2 = &(* const_cast<const NRVec<INDEXGROUP> *>(&t.shape))[newngroup];
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
	const INDEXGROUP *sh = &(* const_cast<const NRVec<INDEXGROUP> *>(&shape))[i];
	I[i].resize(sh->number); 
	I[i] = sh->offset;
	}
T *pp=&data[0];
int ss=shape.size()-1;
const INDEXGROUP *sh = &(* const_cast<const NRVec<INDEXGROUP> *>(&shape))[ss];
loopingroups(*this,ss,sh->number-1,&pp,I,callback);
}


template<typename T>
void constloopingroups(const Tensor<T> &t, int ngroup, int igroup, const T **p, SUPERINDEX &I, void (*callback)(const SUPERINDEX &, const T *))
{
LA_index istart,iend;
const INDEXGROUP *sh = &t.shape[ngroup];
switch(sh->symmetry)
	{
	case 0:
		istart= sh->offset;
		iend=   sh->offset+sh->range-1;
		break;
	case 2:
	case -2:
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
			const INDEXGROUP *sh2 = &(* const_cast<const NRVec<INDEXGROUP> *>(&t.shape))[newngroup];
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
	const INDEXGROUP *sh = &shape[i];
	I[i].resize(sh->number); 
	I[i] = sh->offset;
	}
const T *pp=&data[0];
int ss=shape.size()-1;
const INDEXGROUP *sh = &shape[ss];
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
if(x.rank()==0) {s>>x.data[0]; return s;}
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
NRPerm<int> group2flat_perm(const NRVec<INDEXGROUP> &shape, const  NRPerm<int> &p)
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
NRVec<INDEXGROUP> newshape=shape.permuted(p,true);
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


template<typename T>
Tensor<T> Tensor<T>::permute_index_groups(const NRVec<INDEXNAME> &names) const
{
NRPerm<int> p(shape.size());
if(names.size()!=shape.size()) laerror("names list does not match shape in permute_index_groups");
for(int i=0; i<names.size(); ++i)
	{
	INDEX ii=findindex(names[i]);
	if(ii.index>0) laerror("indices inside groups cannot be permuted in permute_index_groups, define permutation by first group's index name or flatten the tensor first");
	p[1+i] = 1+ii.group;
	}
if(!p.is_valid()) laerror("illegal permutation from names in permute_index_groups, perhaps repeated or not unique names?");
return permute_index_groups(p);
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

int flatposition(const INDEX &i, const NRVec<INDEXGROUP> &shape) 
{return flatposition(i.group,i.index,shape);}

int flatposition(int group, int index, const NRVec<INDEXGROUP> &shape)
{
int ii=0;
for(int g=0; g<group; ++g) ii+= shape[g].number;
ii += index;
return ii;
}

INDEX indexposition(int flatindex, const NRVec<INDEXGROUP> &shape)
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
NRVec<INDEXGROUP> newshape(shape.size()+1);
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
//std::cout <<"TEST flatten_callback: "<<J<<std::endl;
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
		r.canonicalize_shape();
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
NRVec<INDEXGROUP> newshape(newsize);
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
	for(int j=0; j<i; ++j) if(il[i]==il[j]) laerror("repeated index in the list");
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
NRVec<INDEXGROUP> oldshape(shape); 
oldshape.copyonwrite();
NRVec<INDEXGROUP> newshape(shape.size()+nonsolo);

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
clock_t t0=clock();
cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nn, mm, kk, alpha, a, kk, b, kk, beta, r, mm);
#ifdef TIMING
std::cout<<"cblas_dgemm CPU time " << (clock()-t0)/(double)CLOCKS_PER_SEC<<std::endl;
#endif
}

template<>
void auxmatmult<std::complex<double> >(int nn, int mm, int kk, std::complex<double> *r, const std::complex<double> *a, const std::complex<double> *b,  std::complex<double> alpha,  std::complex<double> beta, bool conjugate)
{
clock_t t0=clock();
cblas_zgemm(CblasRowMajor, CblasNoTrans, (conjugate?CblasConjTrans:CblasTrans), nn, mm, kk, &alpha, a, kk, b, kk, &beta, r, mm);
#ifdef TIMING
std::cout<<"cblas_zgemm CPU time " << (clock()-t0)/(double)CLOCKS_PER_SEC<<std::endl;
#endif
}



template<typename T>
void Tensor<T>::addgroupcontraction(const Tensor &rhs1, int group, const Tensor &rhs, int rhsgroup, T alpha, T beta, bool doresize, bool conjugate1, bool conjugate)
{
if(group<0||group>=rhs1.shape.size()) laerror("wrong group number in contraction");
if(rhsgroup<0||rhsgroup>=rhs.shape.size()) laerror("wrong rhsgroup number in contraction");
if(rhs1.shape[group].offset != rhs.shape[rhsgroup].offset) laerror("incompatible index offset in contraction");
if(rhs1.shape[group].range != rhs.shape[rhsgroup].range) laerror("incompatible index range in contraction");
if(rhs1.shape[group].symmetry != rhs.shape[rhsgroup].symmetry) laerror("incompatible index symmetry in addgroupcontraction");
if(rhs1.shape[group].symmetry !=0 && rhs1.shape[group].symmetry != -1) laerror("addgroupcontraction only implemented for nonsymmetric and antisymmetric index groups");
#ifdef LA_TENSOR_INDEXPOSITION
if(rhs1.shape[group].upperindex ^ rhs.shape[rhsgroup].upperindex == false) laerror("can contact only upper with lower index");
#endif

const Tensor<T> u = conjugate1? (rhs1.unwind_index_group(group)).conjugate() : rhs1.unwind_index_group(group);
const Tensor<T> rhsu = rhs.unwind_index_group(rhsgroup);

NRVec<INDEXGROUP> newshape(u.shape.size()+rhsu.shape.size()-2);
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
if(u.shape[0].symmetry== 2) laerror("addgroupcontraction not implemented for hermitean index groups");
if(u.shape[0].symmetry== -2) laerror("addgroupcontraction not implemented for antihermitean index groups");
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
if(rhs1.shape[group].upperindex ^ rhs.shape[rhsgroup].upperindex == false) laerror("can contract only upper with lower index");
#endif

const Tensor<T> u = conjugate1? (rhs1.unwind_index(group,index)).conjugate() : rhs1.unwind_index(group,index);
const Tensor<T> rhsu = rhs.unwind_index(rhsgroup,rhsindex);


NRVec<INDEXGROUP> newshape(u.shape.size()+rhsu.shape.size()-2);
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
bool Tensor<T>::is_uniquely_named() const
{
if(!is_named()) return false;
bool r=true;
for(int i=1; i<names.size();++i) for(int j=0; j<i; ++j) if(names[i]==names[j]) r=false;
return r;
}


template<typename T>
void Tensor<T>::addcontractions(const Tensor &rhs1, const INDEXLIST &il1, const Tensor &rhs2, const INDEXLIST &il2,  T alpha, T beta, bool doresize, bool conjugate1, bool conjugate2)
{
if(il1.size()!=il2.size()) laerror("mismatch in index lists in addcontractions");
if(il1.size()==0) //add just outer product
	{
	if(beta!=(T)0) *this *= beta; else this->clear();
	*this += (conjugate1?rhs1.conjugate():rhs1) * (conjugate2?rhs2.conjugate():rhs2) * alpha;
	return;
	} 

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
	for(int j=0; j<i; ++j) if(il1[i]==il1[j]||il2[i]==il2[j]) laerror("repeated index in the list");
	}

clock_t t0=clock();
const Tensor<T> u = conjugate1? (rhs1.unwind_indices(il1)).conjugateme() : rhs1.unwind_indices(il1);
const Tensor<T> rhsu = rhs2.unwind_indices(il2);
#ifdef TIMING
std::cout <<"unwinding CPU time = "<<(clock()-t0)/(double)CLOCKS_PER_SEC<<std::endl;
#endif


NRVec<INDEXGROUP> newshape(u.shape.size()+rhsu.shape.size()-2*il1.size());
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


int help_tdim;
const SUPERINDEX *help_dummyindex;

template<typename T>
static void innercontraction_callback(const SUPERINDEX &I, T *r)
{       
//compute address like in operarator() but we need true address
//this could be done more efficiently if needed, avoiding explicit SUPERINDEX J
SUPERINDEX J = help_dummyindex->concat(I);
int sign; 
LA_largeindex ii=help_tt<T>->index(&sign,J); 
if(sign<0) laerror("internal error in innercontraction");
const T *matrix00 = &help_tt<T>->data[ii];
*r = 0;
for(int i=0; i<help_tdim; ++i) *r += matrix00[i*help_tdim+i]; //trace
}                       
                


template<typename T>
Tensor<T> Tensor<T>::innercontraction(const INDEXLIST &il1, const INDEXLIST &il2) const
{
if(il1.size()!=il2.size()) laerror("mismatch in index lists in innercontraction");
if(il1.size()==0) return *this;
for(int i=0; i<il1.size(); ++i)
        {
        if(il1[i].group<0||il1[i].group>=shape.size()) laerror("wrong group1 number in innercontraction");
        if(il2[i].group<0||il2[i].group>=shape.size()) laerror("wrong group2 number in innercontraction");
        if(il1[i].index<0||il1[i].index>=shape[il1[i].group].number)  laerror("wrong index1 number in innerconntraction");
        if(il2[i].index<0||il2[i].index>=shape[il2[i].group].number)  laerror("wrong index2 number in innerconntraction");
        if(shape[il1[i].group].offset != shape[il2[i].group].offset) laerror("incompatible index offset in innercontraction");
        if(shape[il1[i].group].range  != shape[il2[i].group].range) laerror("incompatible index range in innercontraction");
#ifdef LA_TENSOR_INDEXPOSITION
        if(shape[il1[i].group].upperindex ^ shape[il2[i].group].upperindex == false) laerror("can contact only upper with lower index");
#endif
        for(int j=0; j<i; ++j) if(il1[i]==il1[j]||il2[i]==il2[j]) laerror("repeated index in the innercontraction list");
	 for(int j=0; j<il1.size(); ++j) if(il1[i]==il2[j])  laerror("repeated index between the innercontraction lists");
        }

INDEXLIST il=il1.concat(il2);
const Tensor u = unwind_indices(il);
int tracedim = u.cumsizes[il1.size()];
//std::cout <<"trace dim = "<<tracedim<<" unwound shape = "<<u.shape<<std::endl;

if(il.size()>(u.shape).size()) laerror("larger innercontraction index list than tensor rank");
if(il.size()==(u.shape).size()) //result is scalar
	{
	T s=0;
	for(int i=0; i<tracedim; ++i) s += u.data[i*tracedim+i]; //trace
	return Tensor<T>(s);
	}

//result is tensor
NRVec<INDEXGROUP> newshape = (u.shape).subvector(il.size(),(u.shape).size()-1);
//std::cout <<"new shape = "<<newshape<<std::endl;

Tensor r(newshape);
if(u.is_named()) r.names=(u.names).subvector(il.size(),(u.shape).size()-1);
//loop over result's elements and do matrix trace
help_tt<T> = &u;
help_tdim = tracedim;
SUPERINDEX dummyindex(il.size());
for(int i=0; i<il.size(); ++i)
	{
	dummyindex[i].resize(1);
	dummyindex[i][0]=u.shape[i].offset; 
	}
help_dummyindex = &dummyindex;
r.loopover(innercontraction_callback);
return r;
}



template<typename T>
static const PermutationAlgebra<int,T> *help_pa;

static bool help_inverse;
static bool help_conjugate;

template<typename T>
static T help_alpha;

template<typename T>
static void permutationalgebra_callback(const SUPERINDEX &I, T *v)
{
FLATINDEX J = superindex2flat(I);
for(int p=0; p<help_pa<T>->size(); ++p)
	{
	FLATINDEX Jp = J.permuted((*help_pa<T>)[p].perm,help_inverse);
	bool conj=false;
	if(help_conjugate) conj= ((*help_pa<T>)[p].perm).parity()<0;
	T tmp= (*help_tt<T>)(Jp);
	*v += help_alpha<T> * (*help_pa<T>)[p].weight * (conj ? LA_traits<T>::conjugate(tmp) : tmp);
	}
}


static int help_tn;
template<typename T>
static const Tensor<T> *help_tv;

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
void Tensor<T>::apply_permutation_algebra(const Tensor<T> &rhs, const PermutationAlgebra<int,T> &pa, bool inverse, T alpha, T beta, bool conjugate)
{
if(beta!=(T)0) {if(beta!=(T)1) *this *= beta;} else clear();
if(alpha==(T)0) return;
copyonwrite();

if(rank()!=rhs.rank()) laerror("rank mismatch in apply_permutation_algebra");

if(rhs.is_named())
	{
	NRVec<INDEXNAME> namperm = rhs.names.permuted(pa[0].perm,!inverse);
	if(is_named())
		{
		if(names!=namperm) 
			{
			std::cout <<"LHS names = "<<names<<std::endl;
			std::cout <<"permuted RHS names = "<<namperm<<std::endl;
			laerror("inconsistent index names in apply_permutation_algebra");
			}
		}
	else	
		names=namperm;
	}

help_tt<T> = &rhs;
help_pa<T> = &pa;
help_inverse = inverse;
help_conjugate= LA_traits<T>::is_complex() ? conjugate : false;
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

bool allnamed=true;
for(int i=0; i<rhsvec.size(); ++i) if(!rhsvec[i].is_named()) allnamed=false;
if(allnamed)
	{
	NRVec<INDEXNAME> allrhsnames=rhsvec[0].names;
	for(int i=1; i<rhsvec.size(); ++i) allrhsnames.concatme(rhsvec[i].names);
	NRVec<INDEXNAME> namperm = allrhsnames.permuted(pa[0].perm,!inverse);
        if(is_named())
                {
                if(names!=namperm) laerror("inconsistent index names in apply_permutation_algebra");
                }
        else
                names=namperm;
	}

help_tv<T> = &rhsvec[0];
help_tn = rhsvec.size();
help_pa<T> = &pa;
help_inverse = inverse;
help_alpha<T> = alpha;
loopover(permutationalgebra_callback2);
}





template<typename T>
void Tensor<T>::split_index_group(int group)
{
const INDEXGROUP *sh = &(* const_cast<const NRVec<INDEXGROUP> *>(&shape))[0];
if(group<0||group >= shape.size()) laerror("illegal index group number");
if(sh[group].number==1) return; //nothing to split
if(sh[group].symmetry!=0) laerror("only non-symmetric index group can be splitted, use flatten instead");

NRVec<INDEXGROUP> newshape(shape.size()+sh[group].number-1);
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
const INDEXGROUP *sh = &(* const_cast<const NRVec<INDEXGROUP> *>(&shape))[0];
if(group<0||group >= shape.size()) laerror("illegal index group number");
if(sh[group].number==1) return; //nothing to split
if(sh[group].symmetry!=0) laerror("only non-symmetric index group can be splitted, use flatten instead");

NRVec<INDEXGROUP> newshape(shape.size()+1);
int gg=0;
for(int g=0; g<shape.size(); ++g)
	{
	if(g==group)
		{
		newshape[gg] = sh[group]; 
		newshape[gg].number = 1;
		gg++;
		newshape[gg] = sh[group]; 
                newshape[gg].number -= 1;
                gg++;
		}
	else
		newshape[gg++] = sh[g];
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

NRVec<INDEXGROUP> newshape(shape.size()-(groupto-groupfrom+1)+1);
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
        if(shape[i].symmetry==1||shape[i].symmetry==2||shape[i].symmetry== -2) laerror("unsupported index group symmetry in dot");
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
//for inverseorder=false, to avoid inverting order by permute_index_groups we repeatedly unwind the LAST index, and all indices rotate at this position with the first one in the last iteration due to the unwind!
NRVec<INDEXNAME> names_saved = names;
for(int i=0; i<r; ++i)
	{
	INDEX I= inverseorder? indexposition(i,shape) : indexposition(r-1,shape);
	NRMat<T> um;
	NRVec<INDEXGROUP> ushape;
	{
	Tensor<T> uu=unwind_index(I);
	ushape=uu.shape; 
	um=uu.matrix(1);
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
	ret[r-i-1]=vt.transpose(true);
	umnew.diagmultr(w);
	//rebuild tensor of the preserved shape from matrix
	ushape.copyonwrite();
	ushape[0].range=preserve;
	{
	NRVec<T> newdata(umnew);
	umnew.resize(0,0);//deallocate
	*this = Tensor(ushape,newdata);
	names=names_saved;
	}
	}
if(!is_flat()) laerror("this should not happen");
if(inverseorder)
	{
	NRPerm<int> p(rank()); p.identity();
	names=names_saved.permuted(p.reverse());
	}
else 
	names=names_saved;
return ret;
}


template<typename T>
Tensor<T> Tensor<T>::inverseTucker(const NRVec<NRMat<T> > &x, bool inverseorder) const
{
if(rank()!=x.size()) laerror("input of inverseTucker does not match rank");
Tensor<T> tmp(*this);
Tensor<T> r;
if(!is_flat()) laerror("inverseTucker only for flat tensors as produced by Tucker");
if(inverseorder)
{
for(int i=0; i<rank(); ++i)
	{
	Tensor<T> mat(x[i],true); //flat tensor from a matrix
	r= tmp.contraction(i,0,mat,0,0,(T)1,false,false);
	if(i<rank()-1) 
		{
		tmp=r;
		r.deallocate();
		}
	}
}
else //not inverseroder
{
for(int i=rank()-1; i>=0; --i)
        {
        Tensor<T> mat(x[i],true); //flat tensor from a matrix
        r= tmp.contraction(rank()-1,0,mat,0,0,(T)1,false,false); //the current index will be the last after previous contractions
        if(i>0)
                {
                tmp=r;
                r.deallocate();
                }
        }
}
if(inverseorder)
        {
        NRPerm<int> p(rank()); p.identity();
        r.names=names.permuted(p.reverse());
        }
else
        r.names=names;
return r;
}


template<typename T>
Tensor<T> Tensor<T>::linear_transform(const NRVec<NRMat<T> > &x) const
{
if(x.size()!=shape.size()) laerror("wrong number of transformation matrices in linear_transform");
for(int i=0; i<shape.size(); ++i) if(x[i].ncols()!=shape[i].range) laerror("mismatch of transformation matrix size in linear_transform");

Tensor<T> tmp(*this);
Tensor<T> r;

//do the groups in reverse order to preserve index ordering
for(int g=shape.size()-1; g>=0; --g)
	{
	Tensor<T> mat(x[g],true); //flat tensor from a matrix
	for(int i=shape[g].number-1; i>=0; --i) //indices in the group in reverse order
		{
		r= tmp.contraction(indexposition(rank()-1,tmp.shape),mat,INDEX(0,0),(T)1,false,false); //always the last index
		if(i>0)
                	{
                	tmp=r;
                	r.deallocate();
			}
		}
	//the group's indices are now individual ones leftmost in tensor r, restore group size and symmetry
	if(shape[g].number>1)
		{
		INDEXLIST il(shape[g].number);
		for(int i=0; i<shape[g].number; ++i)
			{
			il[i].group=i;
			il[i].index=0;
			}
		r = r.merge_indices(il,shape[g].symmetry);
		}
	if(g>0) 
		{
		tmp=r;
                r.deallocate();
		}
	}

r.names=names;
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


template<typename T>
Tensor<T> Tensor<T>::merge_indices(const INDEXLIST &il, int sym) const
{
if(il.size()==0) laerror("empty index list for merge_indices");
if(il.size()==1) return unwind_index(il[0]); //result should be index group of size 1

bool samegroup=true;
bool isordered=true;
for(int i=0; i<il.size(); ++i)
	{
	if(il[i].group<0||il[i].group>=shape.size()) laerror("wrong group number in merge_indices");
	if(il[i].index<0||il[i].index>=shape[il[i].group].number)  laerror("wrong index number in merge_indices");
	for(int j=0; j<i; ++j) if(il[i]==il[j]) laerror("repeated index in the list");
#ifdef LA_TENSOR_INDEXPOSITION
	if(shape[il[0].group].upperindex != shape[il[i].group].upperindex == false) laerror("can merge only within lower or upper separately");
#endif
	if(shape[il[0].group].range != shape[il[i].group].range) 
		{
		std::cout << "indices "<<il[0]<<" and "<<il[i]<< " have ranges "<<shape[il[0].group].range<< " and "<< shape[il[i].group].range <<" respectively\n";
		laerror("incompatible range in merge_indices");
		}
	if(shape[il[0].group].offset != shape[il[i].group].offset) laerror("incompatible offset in merge_indices");
	if(il[0].group != il[i].group) samegroup=false;
	if(il[i].index!=i) isordered=false;
	}

if(samegroup && isordered && il.size()==shape[il[0].group].number) return unwind_index_group(il[0].group);


//calculate new shape and flat index permutation
NRVec<INDEXGROUP> workshape(shape); 
workshape.copyonwrite();
NRPerm<int> basicperm(rank());

bitvector was_in_list(rank());
was_in_list.clear();
for(int i=0; i<il.size(); ++i)
	{
	int fp=flatposition(il[i],shape);
	was_in_list.set(fp);
	basicperm[i+1] = 1+fp;
	if( --workshape[il[i].group].number <0) laerror("inconsistent index list with index group size");
	}
int newshapesize=1; //newly created group
for(int i=0; i<workshape.size(); ++i) if(workshape[i].number>0) ++newshapesize; //this group survived index removal

NRVec<INDEXGROUP> newshape(newshapesize);
newshape[0].number=il.size();
newshape[0].symmetry=sym;
newshape[0].offset=shape[il[0].group].offset;
newshape[0].range=shape[il[0].group].range;
#ifdef LA_TENSOR_INDEXPOSITION
newshape[0].upperindex=shape[il[0].group].upperindex;
#endif 
int ii=1;
for(int i=0; i<workshape.size(); ++i) 
	if(workshape[i].number>0) 
		newshape[ii++] = workshape[i];
int jj=1+il.size();
for(int i=0; i<rank(); ++i) 
	if(!was_in_list[i])
		basicperm[jj++] = 1+i;
if(!basicperm.is_valid()) laerror("internal error in merge_indices");

//std::cout <<"newshape = "<<newshape<<std::endl;
//std::cout <<"basicperm = "<<basicperm<<std::endl;


//prepare permutation algebra
PermutationAlgebra<int,T> pa;
bool doconjugate=false;
if(sym==0)
	{
	pa.resize(1);
	pa[0].weight=1;
	pa[0].perm=basicperm;
	}
else
	{
	doconjugate = sym>=2||sym<= -2;
	PermutationAlgebra<int,int> sa = sym>0 ? symmetrizer<int>(il.size()) : antisymmetrizer<int>(il.size());
	//std::cout <<"SA = "<<sa<<std::endl;
	pa.resize(sa.size());
	for(int i=0; i<sa.size(); ++i)
		{
		pa[i].weight = (T) sa[i].weight;
		pa[i].perm.resize(rank());
		for(int j=1; j<=il.size(); ++j) pa[i].perm[j] = basicperm[sa[i].perm[j]];
		for(int j=il.size()+1; j<=rank(); ++j)   pa[i].perm[j] = basicperm[j];
		}
	}

//std::cout <<"Use PA = "<<pa<<std::endl;

Tensor<T> r(newshape);
r.apply_permutation_algebra(*this,pa,false,(T)1/(T)pa.size(),0,doconjugate);
if(is_named()) r.names=names.permuted(basicperm,true);	
return r;
}



template<typename T>
void Tensor<T>::canonicalize_shape()
{
if(shape.size()==0) return;
const INDEXGROUP *sh = &(* const_cast<const NRVec<INDEXGROUP> *>(&shape))[0];
for(int i=0; i<shape.size(); ++i)
	{
	if(sh[i].number==1 && sh[i].symmetry!=0) {shape.copyonwrite(); shape[i].symmetry=0;}
	int maxlegal = LA_traits<T>::is_complex() ? 2 : 1;
	if(sh[i].symmetry> maxlegal || sh[i].symmetry< -maxlegal) laerror("illegal index group symmetry specified");
	}
}


std::ostream & operator<<(std::ostream &s, const INDEX &x)
{
s<<x.group<<" "<<x.index;
return s;
}

std::istream & operator>>(std::istream &s, INDEX &x)
{ 
s>>x.group>>x.index;
return s;
}

static NRPerm<int> find_indexperm(const NRVec<INDEXNAME> &what, const NRVec<INDEXNAME> &where)
{
if(what.size()!=where.size()) laerror("sizes of index name vectors do not match in find_indexperm");
NRPerm<int> basicperm(what.size());
basicperm.clear();
for(int i=0; i<what.size(); ++i) 
	{
	for(int j=0; j<what.size(); ++j)
             if(what[i] == where[j])
		{
		basicperm[i+1]=j+1;
		break;
		}
	}
if(!basicperm.is_valid()) laerror("indices mismatch between lhs and rhs in add_permuted_contractions");
return basicperm;
}


static void parse_antisymmetrizer(const char *txt0, NRVec<NRVec_from1<int> > &groups0, NRVec<INDEXNAME> &names0)
{
int igroup= -1;
int iname= -1;
int iclass=0;

char *txt=strdup(txt0);
char *p=txt;
std::list<std::list<int> > groups;
std::list<INDEXNAME> names;
std::list<int> currentgroup;

foundspace:
	while(isspace(*p)) ++p;
	if(*p==0) goto finished;
	if(!isalpha(*p)) laerror("letter expected in parse_antisymmetrizer");
	++iname;
	++igroup;
	++iclass;
foundname:
	{
	INDEXNAME n;
	int i=0;
	while(isalnum(*p)) 
		{
		n.name[i]=*p;
		++i;
		if(i>N_INDEXNAME) laerror("too long index name in parse_antisymmetrizer");
		++p;
		}
	if(i>=N_INDEXNAME) laerror("too long index name in parse_antisymmetrizer");
	n.name[i]=0;

	//store
	names.push_back(n);	
	currentgroup.push_back(iclass);
	
	//now decide based on next character
	if(*p==',')
		{
		++iname;
		++p;
		if(!isalpha(*p)) laerror("letter expected in parse_antisymmetrizer");
		goto foundname;
		}
	if(*p=='/')
		{
		++iname;
		++p;
		++iclass;
		if(!isalpha(*p)) laerror("letter expected in parse_antisymmetrizer");
                goto foundname;
		}
	if(isspace(*p))
		{
		groups.push_back(currentgroup);
		currentgroup.clear();
		++p;
		goto foundspace;
		}
	if(*p==0) 
		{
		groups.push_back(currentgroup);
		goto finished;
		}
	laerror("unexpected character in parse_antisymmetrizer");
	}
	
finished:

names0= NRVec<INDEXNAME> (names);
//std::cout <<"names parsed "<<names0;

groups0.resize(groups.size());
int i=0;
for(typename std::list<std::list<int> >::const_iterator ii=groups.begin(); ii!=groups.end(); ++ii)
	{
	groups0[i++] = NRVec_from1<int>(*ii);
	}
//std::cout<<"groups parsed "<<groups0;

free(txt);
}


template<typename T>
void Tensor<T>::add_permuted_contractions(const char *antisymmetrizer, const Tensor &rhs1, const Tensor &rhs2, T alpha, T beta, bool conjugate1, bool conjugate2)
{
if(!rhs1.is_uniquely_named()||!rhs2.is_uniquely_named()|| !is_uniquely_named()) laerror("tensors must have unique named indices in add_permuted_contractions");

//find contraction indices
int nc=0;
for(int i=0; i<rhs1.names.size(); ++i) for(int j=0; j<rhs1.names.size(); ++j) 
	if(rhs1.names[i]==rhs2.names[j])
		{
		//std::cout << "found contraction "<<nc<<" th. index = "<<rhs1.names[i]<<std::endl;
#ifdef LA_TENSOR_INDEXPOSITION
		int group1=indexposition(i,rhs1.shape);
		int group2=indexposition(j,rhs2.shape);
		if(rhs1.shape[group1].upperindex ^ rhs2.shape[group2].upperindex == false) laerror("can contract only upper with lower index");
#endif
		++nc;
		}

INDEXLIST il1(nc),il2(nc);
bitvector is_c_index1(rhs1.names.size());
bitvector is_c_index2(rhs2.names.size());
int ii=0;
for(int i=0; i<rhs1.names.size(); ++i) for(int j=0; j<rhs2.names.size(); ++j)
        if(rhs1.names[i]==rhs2.names[j])
		{
		is_c_index1.set(i);
		is_c_index2.set(j);
		il1[ii] = indexposition(i,rhs1.shape);
		il2[ii] = indexposition(j,rhs2.shape);
		++ii;
		}

//std::cout<<"contraction list1 = "<<il1<<std::endl;
//std::cout<<"contraction list2 = "<<il2<<std::endl;

//make a dry-run to get only the list of names, concatenate names of both tensors unless they are the contraction ones
std::list<INDEXNAME> listnames;
for(int j=0; j<rhs2.names.size(); ++j) if(!is_c_index2[j]) listnames.push_back(rhs2.names[j]);
for(int j=0; j<rhs1.names.size(); ++j) if(!is_c_index1[j]) listnames.push_back(rhs1.names[j]);
NRVec<INDEXNAME> tmpnames(listnames);

//generate the antisymmetrizer, adding also indices not involved as a constant subpermutation
//
//first parse the antisymmetrizer
NRVec<INDEXNAME> antinames;
NRVec<NRVec_from1<int> > antigroups;
parse_antisymmetrizer(antisymmetrizer,antigroups,antinames);

//check the names make sense and fill in the possibly missing ones as separate group
if(antinames.size()>tmpnames.size()) laerror("too many indices in the antisymmetrizer");
bitvector isexplicit(tmpnames.size());
isexplicit.clear();
for(int i=0; i<antinames.size(); ++i)
	{
	for(int j=0; j<i; ++j) if(antinames[i]==antinames[j]) laerror("repeated names in the antisymmetrizer");
	for(int j=0; j<tmpnames.size(); ++j) 
            if(antinames[i]==tmpnames[j]) 
		{
		isexplicit.set(j);
		goto namefound;
		}
	laerror("index name from the antisymmetrized not found in tensor");
	namefound: ;
	}
if(isexplicit.population()!=antinames.size()) laerror("internal error in add_permuted_contractions");

//fill in additional names
if(antinames.size()<tmpnames.size())
	{
	int lastgroup=antigroups.size()-1;
	int lastclass;
	if(antigroups.size()==0) lastclass=0;
	else lastclass=antigroups[antigroups.size()-1][antigroups[antigroups.size()-1].size()];
	int lastname=antinames.size()-1;
	antigroups.resize(antigroups.size()+tmpnames.size()-antinames.size(),true);
	antinames.resize(tmpnames.size(),true);
	for(int j=0; j<names.size(); ++j)
		if(!isexplicit[j])
			{
			++lastclass;
			++lastname;
			++lastgroup;
			antigroups[lastgroup].resize(1);
			antigroups[lastgroup][1]=lastclass;
			antinames[lastname] = tmpnames[j];
			}
	}
//std::cout <<"final antisymmmetrizer names and groups"<<antinames<<antigroups;
//std::cout <<"LHS names = "<<names<<std::endl; 
//std::cout <<"TMP names = "<<tmpnames<<std::endl;

//prepare the antisymmetrizer
PermutationAlgebra<int,int> pa = general_antisymmetrizer(antigroups,-2,true);
//std::cout <<"initial antisymmetrizer = "<<pa;

//find permutation between antisym and TMP index order 
NRPerm<int> antiperm=find_indexperm(antinames,tmpnames);
//std::cout<<"permutation from rhs to antisymmetrizer = "<<antiperm<<std::endl;

//conjugate the PA by antiperm or its inverse
pa= pa.conjugated_by(antiperm,true);

//find permutation which will bring indices of tmp to the order as in *this: this->names[i] == tmpnames[p[i]]
NRPerm<int> basicperm=find_indexperm(names,tmpnames);
PermutationAlgebra<int,T> pb = basicperm.inverse()*pa;
//std::cout <<"permutation from rhs to lhs = "<<basicperm<<std::endl;

//save some work if the PA is trivial - contract only
if(pb.is_identity())
	{
	addcontractions(rhs1,il1,rhs2,il2,alpha,beta,false,conjugate1,conjugate2);
	return;
	}

//create an intermediate contracted tensor and the permute it
clock_t t0=clock();
Tensor<T> tmp=rhs1.contractions(il1,rhs2,il2,alpha,conjugate1,conjugate2);
#ifdef TIMING
std::cout<<"contractions CPU time = "<<(1.*clock()-t0)/CLOCKS_PER_SEC<<std::endl;
#endif

if(rank()!=tmp.rank()) laerror("rank mismatch in add_permuted_contractions");
if(tmp.names!=tmpnames) laerror("internal error in add_permuted_contractions");
clock_t t1=clock();
apply_permutation_algebra(tmp,pb,true,(T)1,beta);
#ifdef TIMING
std::cout<<"apply_permutation_algebra CPU time = "<<(1.*clock()-t1)/CLOCKS_PER_SEC<<std::endl;
#endif
}
        

template<typename T>
void Tensor<T>::permute_inside_group(int g, const NRPerm<int> &p, bool inverse)
{
if(g<0||g>=shape.size()) laerror("group out of range");
if(shape[g].symmetry==0) laerror("permutation possible only inside symmetric index groups");
if(shape[g].number!=p.size()) laerror("permutation size mismatch to index number");
if(!p.is_valid()) laerror("invalid permutation in permute_inside_group");
int par=p.parity();
if(par<0)
	{
	switch(shape[g].symmetry)
		{
		case 1:
			break;
		case -1:
			data.negateme();
			break;
		case 2:
			data.conjugateme();
			break;
		case -2:
			data.negateconjugateme();
			break;
		}
	}
if(is_named())
	{
	int ii=0;
	for(int i=0; i<g; ++i) ii+= shape[i].number;
	NRVec<INDEXNAME> tmp=(names.subvector(ii,ii+p.size()-1)).permuted(p,inverse);
	for(int i=0; i<p.size(); ++i) names[ii+i]=tmp[i];
	}
}


static void checkindex(const NRVec<INDEXGROUP> &shape, const  SUPERINDEX &I, bool &zeroreal, bool &zeroimag)
{
#ifdef DEBUG
if(shape.size()!=I.size()) laerror("inconsistent shape and index in checkindex");
#endif
zeroreal=zeroimag=false;
for(int g=0; g<I.size(); ++g)
	{
#ifdef DEBUG
	if(I[g].size()!=shape[g].number) laerror("inconsistent2 shape and index in checkindex");
#endif
	if(shape[g].symmetry == 2 || shape[g].symmetry == -2)
		{
		bool sameindex=false;
		for(int i=1; i<shape[g].number; ++i) for(int j=0; j<i; ++j)
			if(I[g][i]==I[g][j]) 
				{
				sameindex=true;
				goto checkfailed; //for this group, but do the remaining ones for further restrictions
				}
		
		checkfailed:
		if(sameindex)
			{
			if(shape[g].symmetry == 2) zeroimag=true;
			if(shape[g].symmetry == -2) zeroreal=true;
			}
		}
	}
}


template<typename T>
static void hermiticity_callback(const SUPERINDEX &I, T *v)
{
bool zeroimag;
bool zeroreal;
checkindex(help_t<T>->shape,I,zeroreal,zeroimag);
if(zeroreal) LA_traits<T>::setrealpart(*v,0);
if(zeroimag) LA_traits<T>::setimagpart(*v,0);
}


template<typename T>
static void hermiticity_callback2(const SUPERINDEX &I, const T *v)
{
bool zeroimag;
bool zeroreal;
checkindex(help_tt<T>->shape,I,zeroreal,zeroimag);
if(zeroreal && LA_traits<T>::realpart(*v)!=(T)0 || zeroimag  &&  LA_traits<T>::imagpart(*v)!=(T)0) throw true;
}



template<typename T>
void Tensor<T>::enforce_hermiticity()
{
if(!has_hermiticity()) return;
help_t<T> = this;
loopover(hermiticity_callback);
}


template<typename T>
bool Tensor<T>::fulfills_hermiticity() const
{
if(!has_hermiticity()) return true;
help_tt<T> = this;
try	{
	constloopover(hermiticity_callback2);
	}
catch(bool failed) {return false;}
return true;
}



template class Tensor<double>;
template class Tensor<std::complex<double> >;
template std::ostream & operator<<(std::ostream &s, const Tensor<double> &x);
template std::ostream & operator<<(std::ostream &s, const Tensor<std::complex<double> > &x);
template std::istream & operator>>(std::istream &s, Tensor<double> &x);
template std::istream & operator>>(std::istream &s, Tensor<std::complex<double> > &x);


}//namespace
