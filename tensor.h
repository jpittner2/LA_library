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
//each index group has a specific symmetry (nosym,sym,antisym)
//additional symmetry between index groups (like in 2-electron integrals) is not supported directly, you would need to nest the class to Tensor<Tensor<T> >
//presently only a rudimentary implementation
//presently limited to 2G data size due to NRVec - maybe use a typedef LA_index 
//to uint64_t in the future in vector and matrix classes


#ifndef _TENSOR_H
#define _TENSOR_H

#include <stdint.h>
#include <cstdarg>
#include "vec.h"
#include "mat.h"
#include "smat.h"
#include "miscfunc.h"

//@@@permutation of individual indices??? how to treat the symmetry groups
//@@@todo - index names and contraction by named index list
//@@@contraction inside one tensor

namespace LA {


template<typename T>
class Signedpointer
{
T *ptr;
int sgn;
public:
	Signedpointer(T *p, int s) : ptr(p),sgn(s) {};
	//dereferencing *ptr  should intentionally segfault for sgn==0
	T& operator=(const T rhs) {if(sgn>0) *ptr=rhs; else *ptr = -rhs; return *ptr;}
	T& operator*=(const T rhs) {*ptr *= rhs; return *ptr;}
	T& operator/=(const T rhs) {*ptr /= rhs; return *ptr;}
	T& operator+=(const T rhs) {if(sgn>0) *ptr += rhs; else *ptr -= rhs; return *ptr;}
	T& operator-=(const T rhs) {if(sgn>0) *ptr -= rhs; else *ptr += rhs; return *ptr;}
};


typedef int LA_index;
typedef int LA_largeindex;

typedef class indexgroup {
public:
int number; //number of indices
int symmetry; //-1 0 or 1, later 2 for hermitian and -2 for antihermitian? - would need change in operator() and Signedpointer
#ifdef LA_TENSOR_ZERO_OFFSET
static const LA_index offset = 0; //compiler can optimize away some computations 
#else
LA_index offset; //indices start at a general offset
#endif
LA_index range;  //indices span this range

	bool operator==(const indexgroup &rhs) const {return number==rhs.number && symmetry==rhs.symmetry && offset==rhs.offset && range==rhs.range;};
	inline bool operator!=(const indexgroup &rhs) const {return !((*this)==rhs);};
} INDEXGROUP;


std::ostream & operator<<(std::ostream &s, const INDEXGROUP &x);
std::istream & operator>>(std::istream &s, INDEXGROUP &x);



template<>
class LA_traits<indexgroup> {
	public:
	static bool is_plaindata() {return true;};
	static void copyonwrite(indexgroup& x) {};
	typedef INDEXGROUP normtype;
	static inline int gencmp(const indexgroup *a, const indexgroup *b, int n) {return memcmp(a,b,n*sizeof(indexgroup));};
        static inline void put(int fd, const indexgroup &x, bool dimensions=1) {if(sizeof(indexgroup)!=write(fd,&x,sizeof(indexgroup))) laerror("write error 1 in indexgroup put"); }
        static inline void multiput(int nn, int fd, const indexgroup *x, bool dimensions=1) {if(nn*sizeof(indexgroup)!=write(fd,x,nn*sizeof(indexgroup))) laerror("write error 1 in indexgroup multiiput"); }
        static inline void get(int fd, indexgroup &x, bool dimensions=1) {if(sizeof(indexgroup)!=read(fd,&x,sizeof(indexgroup))) laerror("read error 1 in indexgroup get");}
        static inline void multiget(int nn, int fd, indexgroup *x, bool dimensions=1) {if(nn*sizeof(indexgroup)!=read(fd,x,nn*sizeof(indexgroup))) laerror("read error 1 in indexgroup get");}

};


typedef NRVec<LA_index> FLATINDEX; //all indices but in a single vector
typedef NRVec<NRVec<LA_index> > SUPERINDEX; //all indices in the INDEXGROUP structure
typedef NRVec<LA_largeindex> GROUPINDEX; //set of indices in the symmetry groups
struct  INDEX
{
int group;
int index;
};
typedef NRVec<INDEX> INDEXLIST; //collection of several indices

int flatposition(const INDEX &i, const NRVec<indexgroup> &shape); //position of that index in FLATINDEX
int flatposition(const INDEX &i, const NRVec<indexgroup> &shape);

FLATINDEX superindex2flat(const SUPERINDEX &I);

template<typename T>
class Tensor {
public:
	NRVec<indexgroup> shape;
	NRVec<T> data;	
	int myrank;
	NRVec<LA_largeindex> groupsizes; //group sizes of symmetry index groups (a function of shape but precomputed for efficiency)
	NRVec<LA_largeindex> cumsizes; //cumulative sizes of symmetry index groups (a function of shape but precomputed for efficiency); always cumsizes[0]=1, index group 0 is the innermost-loop one

public:
	LA_largeindex index(int *sign, const SUPERINDEX &I) const; //map the tensor indices to the position in data
	LA_largeindex index(int *sign, const  FLATINDEX &I) const; //map the tensor indices to the position in data
	LA_largeindex vindex(int *sign, LA_index i1, va_list args) const; //map list of indices  to the position in data
	SUPERINDEX inverse_index(LA_largeindex s) const; //inefficient, but possible if needed

	//constructors
	Tensor() : myrank(0) {};
	Tensor(const NRVec<indexgroup> &s) : shape(s) { data.resize(calcsize()); calcrank();}; //general tensor
	Tensor(const indexgroup &g) {shape.resize(1); shape[0]=g; data.resize(calcsize()); calcrank();}; //tensor with a single index group
	Tensor(const Tensor &rhs): myrank(rhs.myrank), shape(rhs.shape), groupsizes(rhs.groupsizes), cumsizes(rhs.cumsizes), data(rhs.data) {};
	Tensor(int xrank, const NRVec<indexgroup> &xshape, const NRVec<LA_largeindex> &xgroupsizes,  const  NRVec<LA_largeindex> xcumsizes, const  NRVec<T> &xdata) :  myrank(xrank), shape(xshape), groupsizes(xgroupsizes), cumsizes(xcumsizes), data(xdata) {};
	explicit Tensor(const NRVec<T> &x);
	explicit Tensor(const NRMat<T> &x);
	explicit Tensor(const NRSMat<T> &x);

	void clear() {data.clear();};
	int rank() const {return myrank;};
	int calcrank(); //is computed from shape
	LA_largeindex calcsize(); //set redundant data and return total size
	LA_largeindex size() const {return data.size();};
	void copyonwrite() {shape.copyonwrite(); groupsizes.copyonwrite(); cumsizes.copyonwrite(); data.copyonwrite();};
	void resize(const NRVec<indexgroup> &s) {shape=s; data.resize(calcsize()); calcrank();};
        inline Signedpointer<T> lhs(const SUPERINDEX &I) {int sign; LA_largeindex i=index(&sign,I); return Signedpointer<T>(&data[i],sign);};
	inline T operator()(const SUPERINDEX &I) const {int sign; LA_largeindex i=index(&sign,I); if(sign==0) return 0; return sign>0 ?data[i] : -data[i];};
        inline Signedpointer<T> lhs(const FLATINDEX &I) {int sign; LA_largeindex i=index(&sign,I); return Signedpointer<T>(&data[i],sign);};
        inline T operator()(const FLATINDEX &I) const {int sign; LA_largeindex i=index(&sign,I); if(sign==0) return 0; return sign>0 ?data[i] : -data[i];};
        inline Signedpointer<T> lhs(LA_index i1...) {va_list args; int sign; LA_largeindex i; va_start(args,i1); i= vindex(&sign, i1,args);  return Signedpointer<T>(&data[i],sign); };
        inline T operator()(LA_index i1...) const {va_list args; ; int sign; LA_largeindex i; va_start(args,i1);  i= vindex(&sign, i1,args); if(sign==0) return 0; return sign>0 ?data[i] : -data[i];};
	
	inline Tensor& operator=(const Tensor &rhs) {myrank=rhs.myrank; shape=rhs.shape; groupsizes=rhs.groupsizes; cumsizes=rhs.cumsizes; data=rhs.data; return *this;};

	inline Tensor& operator*=(const T &a) {data*=a; return *this;};
	inline Tensor operator*(const T &a) const {Tensor r(*this); r *=a; return r;};
	inline Tensor& operator/=(const T &a) {data/=a; return *this;};
	inline Tensor operator/(const T &a) const {Tensor r(*this); r /=a; return r;};

	Tensor operator*(const Tensor &rhs) const; //outer product

	Tensor& conjugateme() {data.conjugateme(); return *this;};
        inline Tensor conjugate() const {Tensor r(*this); r.conjugateme(); return r;};



	inline Tensor& operator+=(const Tensor &rhs) 
		{
#ifdef DEBUG
		if(shape!=rhs.shape) laerror("incompatible tensors for operation"); 
#endif
		data+=rhs.data; 
		return *this;
		}

        inline Tensor& operator-=(const Tensor &rhs) 
                {
#ifdef DEBUG
                if(shape!=rhs.shape) laerror("incompatible tensors for operation"); 
#endif
                data-=rhs.data; 
                return *this;
                }
	inline Tensor operator+(const Tensor &rhs) const {Tensor r(*this); r+=rhs; return r;};
	inline Tensor operator-(const Tensor &rhs) const {Tensor r(*this); r-=rhs; return r;};

	Tensor operator-() const {return Tensor(myrank,shape,groupsizes,cumsizes,-data);}; //unary-

	void put(int fd) const;
	void get(int fd);

	inline void randomize(const typename LA_traits<T>::normtype &x) {data.randomize(x);};

	void loopover(void (*callback)(const SUPERINDEX &, T *)); //loop over all elements
	void grouploopover(void (*callback)(const GROUPINDEX &, T *)); //loop over all elements disregarding the internal structure of index groups

	Tensor permute_index_groups(const NRPerm<int> &p) const; //rearrange the tensor storage permuting index groups as a whole
	Tensor unwind_index(int group, int index) const; //separate an index from a group and expand it to full range as the least significant one
	Tensor unwind_indices(const INDEXLIST &il) const; //the same for a list of indices
	void addcontraction(const Tensor &rhs1, int group, int index, const Tensor &rhs2, int rhsgroup, int rhsindex, T alpha=1, T beta=1, bool doresize=false, bool conjugate1=false, bool conjugate=false); //rhs1 will have more significant non-contracted indices in the result than rhs2
	inline Tensor contraction(int group, int index, const Tensor &rhs, int rhsgroup, int rhsindex, T alpha=1, bool conjugate1=false, bool conjugate=false) const {Tensor<T> r; r.addcontraction(*this,group,index,rhs,rhsgroup,rhsindex,alpha,0,true, conjugate1, conjugate); return r; };

	void addcontractions(const Tensor &rhs1, const INDEXLIST &il1, const Tensor &rhs2, const INDEXLIST &il2,  T alpha=1, T beta=1, bool doresize=false, bool conjugate1=false, bool conjugate2=false);
	inline Tensor contractions( const INDEXLIST &il1, const Tensor &rhs2, const INDEXLIST &il2,  T alpha=1, bool conjugate1=false, bool conjugate2=false) const {Tensor<T> r; r.addcontractions(*this,il1,rhs2,il2,alpha,0,true,conjugate1, conjugate2); return r; };

	void apply_permutation_algebra(const  Tensor &rhs, const PermutationAlgebra<int,T> &pa, bool inverse=false, T alpha=1, T beta=0); //general (not optimally efficient) symmetrizers, antisymmetrizers etc. acting on the flattened index list:
//								 this *=beta; for I over this: this(I) += alpha * sum_P c_P rhs(P(I))
//								 PermutationAlgebra can represent e.g. general_antisymmetrizer in Kucharski-Bartlett notation
	void split_index_group(int group); //formal split of a non-symmetric index group WITHOUT the need for data reorganization
	void merge_adjacent_index_groups(int groupfrom, int groupto); //formal merge of non-symmetric index groups  WITHOUT the need for data reorganization
	Tensor merge_index_groups(const NRVec<int> &groups) const;

	//TODO perhaps implement application of a permutation algebra to a product of several tensors
};



template <typename T>
std::ostream & operator<<(std::ostream &s, const Tensor<T> &x);

template <typename T>
std::istream & operator>>(std::istream &s, Tensor<T> &x);





}//namespace
#endif
