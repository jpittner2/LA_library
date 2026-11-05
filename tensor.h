/*
    LA: linear algebra C++ interface library
    Copyright (C) 2024-2025 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>

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
//indices can optionally have names and be handled by name
//each index group has a specific symmetry (antihermitean= -2, antisym= -1, nosymmetry= 0, symmetric= 1,hermitean=2)
//NOTE: diagonal elements of antihermitean and hermitean matrices are stored including the zero imag/real part and the zeroness is NOT checked and similarly for higher rank tensors
//additional symmetry between index groups (like in 2-electron integrals) is not supported directly, you would need to nest the class to Tensor<Tensor<T> >
//leftmost index is least significant (changing fastest) in the storage order
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
#include "fourindex.h"
#include "miscfunc.h"

//TODO:
//?maybe optional negative range for beta spin handling in some cases of fourindex-tensor conversions combining AA and AB etc. cases - if needed


//do not distinguish covariant/contravariant indices
#undef LA_TENSOR_INDEXPOSITION

namespace LA {

template<typename T>
static inline T signeddata(const int sgn, const T &data)
{
if(LA_traits<T>::is_complex()) //condition known at compile time
                                {
                                switch(sgn)
                                        {
                                        case 2:
                                                return LA_traits<T>::conjugate(data);
                                                break;
                                        case 1: 
                                                return data;
                                                break;
                                        case -1:
                                                return -data;
                                                break;
                                        case -2:
                                                return -LA_traits<T>::conjugate(data);
                                                break;
                                        case 0:
						return 0;
						break;
                                        }
					return 0;
                                }
                        else // for real
                                {
                                if(sgn>0) return data;
                                if(sgn<0) return -data;
				return 0;
                                }

}


static bool didwarn=false;
static inline bool ptr_ok(void *ptr)
{
#ifdef DEBUG
if(ptr==NULL && !didwarn) 
	{
	std::cout<<"Warning: write to zero element of antisym tensor ignored\n";
	std::cerr<<"Warning: write to zero element of antisym tensor ignored\n";
	didwarn=true;
	}
#endif
return (bool)ptr;
}

template<typename T>
class Signedpointer
{
T *ptr;
int sgn;
public:
	Signedpointer(T *p, int s) : ptr(p),sgn(s) {};
	//dereferencing *ptr  should be ignored for sgn==0
	const T operator=(const T rhs) {if(ptr_ok(ptr)) *ptr = signeddata(sgn,rhs); return rhs;}
	//the element L is related to the stored value S by L=signeddata(sgn,S): S, -S, conj(S) (sgn 2) or -conj(S) (sgn -2)
	//for sgn +-1: L*a = +-S*a, so S*=a; for sgn +-2: L*a = +-conj(S)*a = +-conj(S*conj(a)), so S*=conj(a)
	//(signeddata(sgn,rhs) cannot be used here, for negative sgn it would also negate the factor)
	void operator*=(const T rhs) {if(ptr_ok(ptr)) *ptr *= (sgn==2||sgn== -2)? LA_traits<T>::conjugate(rhs) : rhs;}
	void operator/=(const T rhs) {if(ptr_ok(ptr)) *ptr /= (sgn==2||sgn== -2)? LA_traits<T>::conjugate(rhs) : rhs;}
	void operator+=(T rhs) {if(ptr_ok(ptr)) *ptr += signeddata(sgn,rhs);}
	void operator-=(T rhs) {if(ptr_ok(ptr)) *ptr -= signeddata(sgn,rhs);}
};


typedef int LA_index;
typedef int LA_largeindex;

//indexname must not be an array due to its use as a return value in NRVec functions
#define N_INDEXNAME 8
struct INDEXNAME {
	char name[N_INDEXNAME];
	INDEXNAME() {};
	INDEXNAME(const char *n) {strncpy(name,n,N_INDEXNAME);};
	INDEXNAME & operator=(const INDEXNAME &rhs) {strncpy(name,rhs.name,N_INDEXNAME); return *this;};
	bool operator==(const INDEXNAME &rhs) const {return 0==strncmp(name,rhs.name,N_INDEXNAME);};
	bool operator!=(const INDEXNAME &rhs) const {return !(rhs == *this);};
};
template<>
class LA_traits<INDEXNAME> {
        public:
        static bool is_plaindata() {return true;};
        static void copyonwrite(INDEXNAME& x) {};
        typedef INDEXNAME normtype;
        static inline int gencmp(const INDEXNAME *a, const INDEXNAME *b, int n) {return memcmp(a,b,n*sizeof(INDEXNAME));};
        static inline void put(int fd, const INDEXNAME &x, bool dimensions=1) {if(sizeof(INDEXNAME)!=write(fd,&x,sizeof(INDEXNAME))) laerror("write error 1 in INDEXNAME put"); }
        static inline void multiput(int nn, int fd, const INDEXNAME *x, bool dimensions=1) {if(nn*sizeof(INDEXNAME)!=write(fd,x,nn*sizeof(INDEXNAME))) laerror("write error 1 in INDEXNAME multiiput"); }
        static inline void get(int fd, INDEXNAME &x, bool dimensions=1) {if(sizeof(INDEXNAME)!=read(fd,&x,sizeof(INDEXNAME))) laerror("read error 1 in INDEXNAME get");}
        static inline void multiget(int nn, int fd, INDEXNAME *x, bool dimensions=1) {if(nn*sizeof(INDEXNAME)!=read(fd,x,nn*sizeof(INDEXNAME))) laerror("read error 1 in INDEXNAME get");}
	static inline void clear(INDEXNAME *dest, size_t n) {memset(dest,0,n*sizeof(INDEXNAME));}
};


typedef class INDEXGROUP {
public:
int number; //number of indices
int symmetry; //-1 0 or 1, later 2 for hermitian and -2 for antihermitian
#ifdef LA_TENSOR_ZERO_OFFSET
static const LA_index offset = 0; //compiler can optimize away some computations 
#else
LA_index offset; //indices start at a general offset
#endif
LA_index range;  //indices span this range
#ifdef LA_TENSOR_INDEXPOSITION
bool upperindex;
#endif

	bool operator==(const INDEXGROUP &rhs) const {return number==rhs.number && symmetry==rhs.symmetry && offset==rhs.offset && range==rhs.range
#ifdef LA_TENSOR_INDEXPOSITION
	&& upperindex == rhs.upperindex
#endif
	;};
	inline bool operator!=(const INDEXGROUP &rhs) const {return !((*this)==rhs);};
} INDEXGROUP;


std::ostream & operator<<(std::ostream &s, const INDEXGROUP &x);
std::istream & operator>>(std::istream &s, INDEXGROUP &x);

std::ostream & operator<<(std::ostream &s, const INDEXNAME &x);
std::istream & operator>>(std::istream &s, INDEXNAME &x);




template<>
class LA_traits<INDEXGROUP> {
	public:
	static bool is_plaindata() {return true;};
	static void copyonwrite(INDEXGROUP& x) {};
	typedef INDEXGROUP normtype;
	static inline int gencmp(const INDEXGROUP *a, const INDEXGROUP *b, int n) {return memcmp(a,b,n*sizeof(INDEXGROUP));};
        static inline void put(int fd, const INDEXGROUP &x, bool dimensions=1) {if(sizeof(INDEXGROUP)!=write(fd,&x,sizeof(INDEXGROUP))) laerror("write error 1 in INDEXGROUP put"); }
        static inline void multiput(int nn, int fd, const INDEXGROUP *x, bool dimensions=1) {if(nn*sizeof(INDEXGROUP)!=write(fd,x,nn*sizeof(INDEXGROUP))) laerror("write error 1 in INDEXGROUP multiiput"); }
        static inline void get(int fd, INDEXGROUP &x, bool dimensions=1) {if(sizeof(INDEXGROUP)!=read(fd,&x,sizeof(INDEXGROUP))) laerror("read error 1 in INDEXGROUP get");}
        static inline void multiget(int nn, int fd, INDEXGROUP *x, bool dimensions=1) {if(nn*sizeof(INDEXGROUP)!=read(fd,x,nn*sizeof(INDEXGROUP))) laerror("read error 1 in INDEXGROUP get");}
	static inline void clear(INDEXGROUP *dest, size_t n) {memset(dest,0,n*sizeof(INDEXGROUP));}
};


typedef NRVec<LA_index> FLATINDEX; //all indices but in a single vector
typedef NRVec<FLATINDEX> SUPERINDEX; //all indices in the INDEXGROUP structure
typedef NRVec<LA_largeindex> GROUPINDEX; //set of indices in the symmetry groups
typedef NRMat<LA_index> INDEXMATRIX; //list of FLATINDEXes (rows of the matrix) of all tensor elements - convenient to be able to run over the whole tensor in a for loop rather than via recursive loopovers with a callback
struct  INDEX
{
int group;
int index;
bool operator==(const INDEX &rhs) const {return group==rhs.group && index==rhs.index;};
	INDEX() {};
	INDEX(int g, int i): group(g), index(i) {};
};
typedef NRVec<INDEX> INDEXLIST; //collection of several indices

std::ostream & operator<<(std::ostream &s, const INDEX &x);
std::istream & operator>>(std::istream &s, INDEX &x);


template<>
class LA_traits<INDEX> {
        public:
        static bool is_plaindata() {return true;};
        static void copyonwrite(INDEX& x) {};
        typedef INDEX normtype;
        static inline int gencmp(const INDEX *a, const INDEX *b, int n) {return memcmp(a,b,n*sizeof(INDEX));};
        static inline void put(int fd, const INDEX &x, bool dimensions=1) {if(sizeof(INDEX)!=write(fd,&x,sizeof(INDEX))) laerror("write error 1 in INDEX put"); }
        static inline void multiput(int nn, int fd, const INDEX *x, bool dimensions=1) {if(nn*sizeof(INDEX)!=write(fd,x,nn*sizeof(INDEX))) laerror("write error 1 in INDEX multiiput"); }
        static inline void get(int fd, INDEX &x, bool dimensions=1) {if(sizeof(INDEX)!=read(fd,&x,sizeof(INDEX))) laerror("read error 1 in INDEX get");}
        static inline void multiget(int nn, int fd, INDEX *x, bool dimensions=1) {if(nn*sizeof(INDEX)!=read(fd,x,nn*sizeof(INDEX))) laerror("read error 1 in INDEX get");}
        static inline void clear(INDEX *dest, size_t n) {memset(dest,0,n*sizeof(INDEX));}
};







int flatposition(int group, int index, const NRVec<INDEXGROUP> &shape);
int flatposition(const INDEX &i, const NRVec<INDEXGROUP> &shape); //position of that index in FLATINDEX
INDEX indexposition(int flatindex, const NRVec<INDEXGROUP> &shape); //inverse to flatposition

LA_largeindex subindex(int *sign, const INDEXGROUP &g, const NRVec<LA_index> &I); //index of one subgroup
NRVec<LA_index>  inverse_subindex(const INDEXGROUP &g, LA_largeindex s);


//useful for negative offsets and 0 index excluded
bool zero_in_index(const FLATINDEX &);
bool zero_in_index(const SUPERINDEX &);
bool zero_in_index(const INDEXMATRIX &, const LA_largeindex row);

bool shares_index(const FLATINDEX &I, const FLATINDEX &J);

FLATINDEX superindex2flat(const SUPERINDEX &I);

template<typename T>
class Tensor {
public:
	NRVec<INDEXGROUP> shape;
	NRVec<T> data;	
	int myrank;
	NRVec<LA_largeindex> groupsizes; //group sizes of symmetry index groups (a function of shape but precomputed for efficiency)
	NRVec<LA_largeindex> cumsizes; //cumulative sizes of symmetry index groups (a function of shape but precomputed for efficiency); always cumsizes[0]=1, index group 0 is the innermost-loop one
	NRVec<INDEXNAME> names;

	//indexing
	LA_largeindex index(int *sign, const SUPERINDEX &I) const; //map the tensor indices to the position in data
	LA_largeindex index(int *sign, const  FLATINDEX &I) const; //map the tensor indices to the position in data
	LA_largeindex vindex(int *sign, LA_index i1, va_list args) const; //map list of indices  to the position in data
	SUPERINDEX inverse_index(LA_largeindex s) const; //inefficient, but possible if needed
	int myflatposition(int group, int index) const {return flatposition(group,index,shape);};
	int myflatposition(const INDEX &i) const {return flatposition(i,shape);};
	INDEX myindexposition(int flatindex) const {return indexposition(flatindex,shape);};	

	//constructors
	Tensor() : myrank(-1) {};
	explicit Tensor(const T &x) : myrank(0), data(1) {data[0]=x;}; //scalar
	Tensor(const NRVec<INDEXGROUP> &s) : shape(s) { data.resize(calcsize()); calcrank(); canonicalize_shape();}; //general tensor
	Tensor(const NRVec<INDEXGROUP> &s, const NRVec<INDEXNAME> &newnames) : shape(s), names(newnames) { data.resize(calcsize());  calcrank(); canonicalize_shape(); if(names.size()!=myrank && names.size()!=0) laerror("bad number of index names");}; //general tensor
	Tensor(const NRVec<INDEXGROUP> &s, const NRVec<T> &mydata) : shape(s) { LA_largeindex dsize=calcsize(); calcrank(); canonicalize_shape(); if(mydata.size()!=dsize) laerror("inconsistent data size with shape"); data=mydata;} 
	Tensor(const NRVec<INDEXGROUP> &s, const NRVec<T> &mydata, const NRVec<INDEXNAME> &newnames) : shape(s), names(newnames) { LA_largeindex dsize=calcsize(); calcrank(); canonicalize_shape(); if(mydata.size()!=dsize) laerror("inconsistent data size with shape"); data=mydata;  if(names.size()!=myrank && names.size()!=0) laerror("bad number of index names");} 
	Tensor(const INDEXGROUP &g) {shape.resize(1); shape[0]=g; data.resize(calcsize()); calcrank(); canonicalize_shape();}; //tensor with a single index group
	Tensor(const INDEXGROUP &g, const NRVec<INDEXNAME> &newnames) : names(newnames) {shape.resize(1); shape[0]=g; data.resize(calcsize()); calcrank(); canonicalize_shape(); if(names.size()!=myrank && names.size()!=0) laerror("bad number of index names");}; //tensor with a single index group
	Tensor(const Tensor &rhs): myrank(rhs.myrank), shape(rhs.shape), groupsizes(rhs.groupsizes), cumsizes(rhs.cumsizes), data(rhs.data), names(rhs.names) {};
	Tensor(int xrank, const NRVec<INDEXGROUP> &xshape, const NRVec<LA_largeindex> &xgroupsizes,  const  NRVec<LA_largeindex> xcumsizes, const  NRVec<T> &xdata) :  myrank(xrank), shape(xshape), groupsizes(xgroupsizes), cumsizes(xcumsizes), data(xdata) {};
	Tensor(int xrank, const NRVec<INDEXGROUP> &xshape, const NRVec<LA_largeindex> &xgroupsizes,  const  NRVec<LA_largeindex> xcumsizes, const  NRVec<T> &xdata, const NRVec<INDEXNAME>  &xnames) :  myrank(xrank), shape(xshape), groupsizes(xgroupsizes), cumsizes(xcumsizes), data(xdata), names(xnames) {};

	//subtensor - todo: more general versions
	Tensor subtensor1(int i) const; //for one value of rightmost index

	//numpy interface
	void numpy_write(const char *name, const char *descr="<f8") const;
	void numpy_read(const char *name);	

	//conversions from/to matrix and vector
	explicit Tensor(const NRVec<T> &x);
	explicit Tensor(const NRMat<T> &x, bool flat=false);
	explicit Tensor(const NRSMat<T> &x);
	NRMat<T> matrix(int n=1) const //reinterpret as matrix with column index being the tensor's n leftmost index groups (typically the unwound single index)
		{
		if(n<0||n>shape.size()) laerror("wrong number n in tensor:: matrix()");
		LA_largeindex ncol = (n==shape.size()) ? data.size() : cumsizes[n];
		return NRMat<T>(data,data.size()/ncol,ncol,0);
		};

	bool is_named() const {if(names.size()==0) return false; if(names.size()!=myrank) laerror("bad number of index names"); return true;};
	bool is_uniquely_named() const; //no repeated names
	bool is_flat() const {for(int i=0; i<shape.size(); ++i) if(shape[i].number>1) return false; return true;};
	bool is_compressed() const {for(int i=0; i<shape.size(); ++i) if(shape[i].number>1&&shape[i].symmetry!=0) return true; return false;};
	bool has_symmetry() const {for(int i=0; i<shape.size(); ++i) if(shape[i].symmetry!=0) return true; return false;};
	bool has_hermiticity() const {if(!LA_traits<T>::is_complex()) return false; for(int i=0; i<shape.size(); ++i) if(shape[i].symmetry < -1 || shape[i].symmetry > 1) return true; return false;};
	void clear() {data.clear();};
	void defaultnames(const char *basename="i") {names.resize(rank()); for(int i=0; i<rank(); ++i) sprintf(names[i].name,"%s%03d",basename,i);}
	int rank() const {return myrank;};
	int calcrank(); //is computed from shape
	void canonicalize_shape(); 
	LA_largeindex calcsize(); //set redundant data and return total size
	LA_largeindex size() const {return data.size();};
	void copyonwrite() {shape.copyonwrite(); groupsizes.copyonwrite(); cumsizes.copyonwrite(); data.copyonwrite(); names.copyonwrite();};
	void resize(const NRVec<INDEXGROUP> &s) {shape=s; data.resize(calcsize()); calcrank(); names.clear();};
	void deallocate() {data.resize(0); shape.resize(0); groupsizes.resize(0); cumsizes.resize(0); names.resize(0);};

        inline Signedpointer<T> lhs(const SUPERINDEX &I) {int sign; LA_largeindex i=index(&sign,I);
				if(sign==0)  return Signedpointer<T>(NULL,sign);
				else return Signedpointer<T>(&data[i],sign);};
	inline T operator()(const SUPERINDEX &I) const {int sign; LA_largeindex i=index(&sign,I); if(sign==0) return 0; else return signeddata(sign,data[i]);};
        inline Signedpointer<T> lhs(const FLATINDEX &I) {int sign; LA_largeindex i=index(&sign,I); 
                                if(sign==0) return Signedpointer<T>(NULL,sign);
				else return Signedpointer<T>(&data[i],sign);};
        inline T operator()(const FLATINDEX &I) const {int sign; LA_largeindex i=index(&sign,I); if(sign==0) return 0; else return signeddata(sign,data[i]);};
        inline Signedpointer<T> lhs(LA_index i1...) {va_list args; int sign; LA_largeindex i; va_start(args,i1); i= vindex(&sign, i1,args);  
                                if(sign==0) return Signedpointer<T>(NULL,sign);
				else return Signedpointer<T>(&data[i],sign); };
        inline T operator()(LA_index i1...) const {va_list args; ; int sign; LA_largeindex i; va_start(args,i1);  i= vindex(&sign, i1,args);  if(sign==0) return 0; else return signeddata(sign,data[i]);};
	
	inline Tensor& operator=(const Tensor &rhs) {myrank=rhs.myrank; shape=rhs.shape; groupsizes=rhs.groupsizes; cumsizes=rhs.cumsizes; data=rhs.data; names=rhs.names; return *this;};

	inline Tensor& operator*=(const T &a) {data*=a; return *this;};
	inline Tensor operator*(const T &a) const {Tensor r(*this); r *=a; return r;};
	inline Tensor& operator/=(const T &a) {data/=a; return *this;};
	inline Tensor operator/(const T &a) const {Tensor r(*this); r /=a; return r;};

	typename LA_traits<T>::normtype norm() const {return data.norm();};

	inline Tensor operator*(const Tensor &rhs) const {return Tensor(rhs.shape.concat(shape),data.otimes2vec(rhs.data),rhs.names.concat(names));} //outer product,  rhs indices will be the less significant 
	T dot(const Tensor &rhs) const; //scalar product (full contraction), in complex case is automatically conjugated, not for symmetric group indices

	Tensor& conjugateme() {data.conjugateme(); return *this;};
        inline Tensor conjugate() const {Tensor r(*this); r.conjugateme(); return r;};

	//find index by name
	int findflatindex(const INDEXNAME nam) const;
	INDEX findindex(const INDEXNAME nam) const;
	NRVec<INDEX> findindexlist(const NRVec<INDEXNAME> &names) const;
	void renameindex(const  INDEXNAME namfrom, const INDEXNAME nameto) {int i=findflatindex(namfrom); names.copyonwrite(); names[i]=nameto;};

	Tensor& operator+=(const Tensor &rhs) 
		{
		if(shape!=rhs.shape) laerror("incompatible tensors for operation"); 
		if(is_named() && rhs.is_named() && names!=rhs.names) laerror("incompatible names for operation");
		data+=rhs.data; 
		return *this;
		}

        Tensor& operator-=(const Tensor &rhs) 
                {
                if(shape!=rhs.shape) laerror("incompatible tensors for operation"); 
		if(is_named() && rhs.is_named() && names!=rhs.names) laerror("incompatible names for operation");
                data-=rhs.data; 
                return *this;
                }

	Tensor& axpy(const T alpha, const Tensor &rhs) 
		{
		if(shape!=rhs.shape) laerror("incompatible tensors for operation");
                if(is_named() && rhs.is_named() && names!=rhs.names) laerror("incompatible names for operation");
		data.axpy(alpha,rhs.data);
                return *this;
		}
	inline Tensor operator+(const Tensor &rhs) const {Tensor r(*this); r+=rhs; return r;};
	inline Tensor operator-(const Tensor &rhs) const {Tensor r(*this); r-=rhs; return r;};

	Tensor operator-() const {return Tensor(myrank,shape,groupsizes,cumsizes,-data);}; //unary-

	void put(int fd, bool with_names=false) const;
	void get(int fd, bool with_names=false);

	void enforce_hermiticity(); //zero out real/imag parts for repeated indices appropriately
	bool fulfills_hermiticity() const; //check it is so
	inline void randomize(const typename LA_traits<T>::normtype &x) {data.randomize(x); enforce_hermiticity();};


	void loopover(void (*callback)(const SUPERINDEX &, T *), bool skipzeros=false); //loop over all elements, optionally skip zero indices (i.e. run over ...-2,-1,1,2...) which is useful for special applications
	void constloopover(void (*callback)(const SUPERINDEX &, const T *), bool skipzeros=false) const; //loop over all elements
	void grouploopover(void (*callback)(const GROUPINDEX &, T *)); //loop over all elements disregarding the internal structure of index groups
	void constgrouploopover(void (*callback)(const GROUPINDEX &, const T *)) const; //loop over all elements disregarding the internal structure of index groups

	INDEXMATRIX indexmatrix() const; //get indexmatrix - rows store FLATINDEXes matching data[]

	Tensor permute_index_groups(const NRPerm<int> &p) const; //rearrange the tensor storage permuting index groups as a whole: result_i = source_p_i
	Tensor permute_index_groups(const NRVec<INDEXNAME> &names) const; //permute to requested order of group's first indices (or permute individual indices of a flat tensor)

	Tensor unwind_index_group(int group) const; //make the index group leftmost (least significant)
	Tensor unwind_index(int group, int index) const; //separate an index from a group and expand it to full range as the least significant one (the leftmost one)
	Tensor unwind_index(const INDEX &I) const {return unwind_index(I.group,I.index);};
	Tensor unwind_index(const INDEXNAME &N) const {return unwind_index(findindex(N));};

	Tensor unwind_indices(const INDEXLIST &il) const; //the same for a list of indices
	Tensor unwind_indices(const NRVec<INDEXNAME> &names) const {return unwind_indices(findindexlist(names));};

	void addcontraction(const Tensor &rhs1, int group, int index, const Tensor &rhs2, int rhsgroup, int rhsindex, T alpha=1, T beta=1, bool doresize=false, bool conjugate1=false, bool conjugate=false); //rhs1 will have more significant non-contracted indices in the result than rhs2
	inline void addcontraction(const Tensor &rhs1, const INDEX &I1, const Tensor &rhs2, const INDEX &I2, T alpha=1, T beta=1, bool doresize=false, bool conjugate1=false, bool conjugate=false) {addcontraction(rhs1, I1.group, I1.index, rhs2, I2.group, I2.index, alpha, beta, doresize, conjugate1, conjugate);};
	inline void addcontraction(const Tensor &rhs1, const Tensor &rhs2, const INDEXNAME &iname, T alpha=1, T beta=1, bool doresize=false, bool conjugate1=false, bool conjugate=false) {addcontraction(rhs1, rhs1.findindex(iname), rhs2, rhs2.findindex(iname), alpha, beta, doresize, conjugate1, conjugate);};
	inline Tensor contraction(int group, int index, const Tensor &rhs, int rhsgroup, int rhsindex, T alpha=1, bool conjugate1=false, bool conjugate=false) const {Tensor<T> r; r.addcontraction(*this,group,index,rhs,rhsgroup,rhsindex,alpha,0,true, conjugate1, conjugate); return r; };
	inline Tensor contraction(const INDEX &I, const Tensor &rhs, const INDEX &RHSI, T alpha=1, bool conjugate1=false, bool conjugate=false) const {return  contraction(I.group,I.index, rhs, RHSI.group, RHSI.index,alpha, conjugate1, conjugate);};
	inline Tensor contraction(const Tensor &rhs, const INDEXNAME &iname,  T alpha=1, bool conjugate1=false, bool conjugate=false) const {return  contraction(findindex(iname),rhs,rhs.findindex(iname),alpha, conjugate1, conjugate);};

	//NOTE: (add)contractions() performs all listed contractions in a single matrix multiplication step
	//it might be more efficient to perform the contractions subsequently one at a time
	//this routine does not care about this, it is the responsibility of the user to generate optimal contraction sequence
	void addcontractions(const Tensor &rhs1, const INDEXLIST &il1, const Tensor &rhs2, const INDEXLIST &il2,  T alpha=1, T beta=1, bool doresize=false, bool conjugate1=false, bool conjugate2=false);
	inline void addcontractions(const Tensor &rhs1, const Tensor &rhs2, const NRVec<INDEXNAME> &names,  T alpha=1, T beta=1, bool doresize=false, bool conjugate1=false, bool conjugate2=false) {addcontractions(rhs1, rhs1.findindexlist(names), rhs2, rhs2.findindexlist(names), alpha, beta, doresize, conjugate1,conjugate2);};
	inline Tensor contractions( const INDEXLIST &il1, const Tensor &rhs2, const INDEXLIST &il2,  T alpha=1, bool conjugate1=false, bool conjugate2=false) const {Tensor<T> r; r.addcontractions(*this,il1,rhs2,il2,alpha,0,true,conjugate1, conjugate2); return r; };
	inline Tensor contractions(const Tensor &rhs2, const NRVec<INDEXNAME> &names,  T alpha=1, bool conjugate1=false, bool conjugate2=false) const {return contractions(findindexlist(names),rhs2,rhs2.findindexlist(names),alpha,conjugate1,conjugate2); };

	void addgroupcontraction(const Tensor &rhs1, int group, const Tensor &rhs2, int rhsgroup, T alpha=1, T beta=1, bool doresize=false, bool conjugate1=false, bool conjugate=false); //over all indices in a group of same symmetry; rhs1 will have more significant non-contracted indices in the result than rhs2
	inline Tensor groupcontraction(int group, const Tensor &rhs, int rhsgroup, T alpha=1, bool conjugate1=false, bool conjugate=false) const {Tensor<T> r; r.addgroupcontraction(*this,group,rhs,rhsgroup,alpha,0,true, conjugate1, conjugate); return r; };

	void add_permuted_contractions(const char *antisymmetrizer, const Tensor &rhs1, const Tensor &rhs2, T alpha=1, T beta=1, bool conjugate1=false, bool conjugate2=false); //must be used on named tensors, finds repeated indices to contract and antisymmetrizes and permutes remaining indices to match *this


	Tensor innercontraction(const INDEXLIST &il1, const INDEXLIST &il2) const; //contraction(s) inside this tensor
	Tensor innercontraction(const INDEX &i1, const INDEX &i2) const {INDEXLIST il1(1); il1[0]=i1; INDEXLIST il2(1); il2[0]=i2; return innercontraction(il1,il2);};
	Tensor innercontraction(const NRVec<INDEXNAME> &nl1, const NRVec<INDEXNAME> &nl2) const {return innercontraction(findindexlist(nl1),findindexlist(nl2));};
	Tensor innercontraction(const INDEXNAME &n1, const INDEXNAME &n2) const {return innercontraction(findindex(n1),findindex(n2));};

	void apply_permutation_algebra(const  Tensor &rhs, const PermutationAlgebra<int,T> &pa, bool inverse=false, T alpha=1, T beta=0, bool conjugate_by_parity=false); //general (not optimally efficient) symmetrizers, antisymmetrizers etc. acting on the flattened index list:
	void apply_permutation_algebra(const  NRVec<Tensor> &rhsvec, const PermutationAlgebra<int,T> &pa, bool inverse=false, T alpha=1, T beta=0); //avoids explicit outer product but not vectorized, rather inefficient
//								 this *=beta; for I over this: this(I) += alpha * sum_P c_P rhs(P(I))
//								 PermutationAlgebra can represent e.g. general_antisymmetrizer in Kucharski-Bartlett notation, or Grassmann products building RDM from cumulants
//								 Note that *this tensor can be e.g. antisymmetric while rhs is not and is being antisymmetrized by the PermutationAlgebra
//								 The efficiency is not optimal, even when avoiding the outer product, the calculation is done indexing element by element
//								 More efficient would be applying permutation algebra symbolically and efficiently computing term by term

	void split_index_group(int group); //formal in-place split of a non-symmetric index group WITHOUT the need for data reorganization or names rearrangement
	void split_index_group1(int group); //formal in-place split of the leftmost index in a non-symmetric index group WITHOUT the need for data reorganization or names rearrangement
	void merge_adjacent_index_groups(int groupfrom, int groupto); //formal merge of non-symmetric index groups  WITHOUT the need for data reorganization or names rearrangement
	void permute_inside_group(int group, const NRPerm<int> &p, bool inverse=false); //permute indices inside a symmetric index group only

	Tensor merge_index_groups(const NRVec<int> &groups) const; 
	Tensor flatten(int group= -1) const; //split and uncompress a given group or all of them, leaving flat index order the same
	Tensor merge_indices(const INDEXLIST &il, int symmetry=0) const;	//opposite to flatten (merging with optional symmetrization/antisymmetrization and compression)
	Tensor merge_indices(const NRVec<INDEXNAME> &nl, int symmetry=0) const {return merge_indices(findindexlist(nl),symmetry);};

	NRVec<NRMat<T> > Tucker(typename LA_traits<T>::normtype thr=LA_precision<typename LA_traits<T>::normtype>::svd_threshold(), bool inverseorder=false); //HOSVD-Tucker decomposition, return core tensor in *this, flattened 
	Tensor inverseTucker(const NRVec<NRMat<T> > &x, bool inverseorder=false) const; //rebuild the original tensor from Tucker
	Tensor linear_transform(const NRVec<NRMat<T> > &x) const; //linear transform by a different matrix per each index group, preserving group symmetries
	Tensor linear_transform(const int g, const Tensor<T> &x) const; //linear transform a single group, preserve its position and symmetry, x must be rank-2 flat tensor (this is useful for raising/lowering indices)
	Tensor linear_transform(const int g, const NRMat<T> &x) const {Tensor<T> mat(x,true); return linear_transform(g,mat);}; //linear transform a single group, preserve its position and symmetry
};


//due to template nesting and specialization limitation, this cannot be class member - the S parameter has to be the outer one
//note also that the class Tnesor does not support symmetry between groups of indices (only symmetry inside each group), the full fourindex symmetry cannot thus be reflected in some cases
//
template<fourindexsymtype S, typename T, typename I> 
Tensor<T> fourindex2tensor(const fourindex_dense<S,T,I> &f);
template<fourindexsymtype S, typename T, typename I>          
void tensor2fourindex(const Tensor<T> &t, fourindex_dense<S,T,I> &f);

//conversions from/to fourindex specialized by symmetry type
//NOTE also different index order due to tensor's leftmost index being the least significant
//
//operator() : (i,j,k,l) in fourindex -> (k,l,i,j) in tensor
template<typename T, typename I>
Tensor<T>  fourindex2tensor(const fourindex_dense<nosymmetry,T,I> &f)
{
NRVec<INDEXGROUP> shape(4);
int n=f.nbas();
for(int i=0; i<4; ++i)
	{
	shape[i].number=1;
	shape[i].symmetry=0;
	shape[i].offset=1;
	shape[i].range=n;
	}
NRVec<T> data(f);
return Tensor<T>(shape,data);
}

template<typename T, typename I>
void tensor2fourindex(const Tensor<T> &t, fourindex_dense<nosymmetry,T,I> &f)
{
if(t.rank()!=4) laerror("wrong rank in tensor2fourindex");
int range=t.shape[0].range;
int offset=t.shape[0].offset;
for(int i=0; i<t.shape.size(); ++i)
	{
	if(range!=t.shape[i].range) laerror("range mismatch in tensor2fourindex");
	if(offset!=t.shape[i].offset) laerror("offset mismatch in tensor2fourindex");
	if(0!=t.shape[i].symmetry) laerror("symmetry mismatch in tensor2fourindex");
	}
f=fourindex_dense<nosymmetry,T,I>(range,NRMat<T>(t.data,range*range,range*range));
}


//operator() : (i,j,k,l) in fourindex -> (k,l,i,j)=(l,k,j,i) in tensor
template<typename T, typename I>
Tensor<T>  fourindex2tensor(const fourindex_dense<twoelectronrealmullikanAB,T,I> &f)
{
NRVec<INDEXGROUP> shape(2);
int n=f.nbas();
for(int i=0; i<2; ++i)
	{
	shape[i].number=2;
	shape[i].symmetry=1;
	shape[i].offset=1;
	shape[i].range=n;
	}
NRVec<T> data(f);
return Tensor<T>(shape,data);
}

template<typename T, typename I>
void tensor2fourindex(const Tensor<T> &t, fourindex_dense<twoelectronrealmullikanAB,T,I> &f)
{
if(t.rank()!=4) laerror("wrong rank in tensor2fourindex");
if(t.shape.size()!=2) laerror("wrong symmetry groups in tensor2fourindex");
int range=t.shape[0].range;
int offset=t.shape[0].offset;
for(int i=0; i<t.shape.size(); ++i)
	{
	if(range!=t.shape[i].range) laerror("range mismatch in tensor2fourindex");
	if(offset!=t.shape[i].offset) laerror("offset mismatch in tensor2fourindex");
	if(1!=t.shape[i].symmetry) laerror("symmetry mismatch in tensor2fourindex");
	}
f=fourindex_dense<twoelectronrealmullikanAB,T,I>(NRMat<T>(t.data,range*(range+1)/2,range*(range+1)/2));
}


//operator() : (i,j,k,l) in fourindex -> (i,j,k,l)=(l,k,j,i) in tensor
template<typename T, typename I>
Tensor<T>  fourindex2tensor(const fourindex_dense<twoelectronrealmullikan,T,I> &f)
{
NRVec<INDEXGROUP> shape(2);
int n=f.nbas();
for(int i=0; i<2; ++i)
	{
	shape[i].number=2;
	shape[i].symmetry=1;
	shape[i].offset=1;
	shape[i].range=n;
	}
NRMat<T> mat(f);
NRVec<T> data(mat); //expand symmetric to full matrix for index group symmetry
return Tensor<T>(shape,data);
}

template<typename T, typename I>
void tensor2fourindex(const Tensor<T> &t, fourindex_dense<twoelectronrealmullikan,T,I> &f)
{
if(t.rank()!=4) laerror("wrong rank in tensor2fourindex");
if(t.shape.size()!=2) laerror("wrong symmetry groups in tensor2fourindex");
int range=t.shape[0].range;
int offset=t.shape[0].offset;
for(int i=0; i<t.shape.size(); ++i)
	{
	if(range!=t.shape[i].range) laerror("range mismatch in tensor2fourindex");
	if(offset!=t.shape[i].offset) laerror("offset mismatch in tensor2fourindex");
	if(1!=t.shape[i].symmetry) laerror("symmetry mismatch in tensor2fourindex");
	}
NRMat<T> mat(t.data,range*(range+1)/2,range*(range+1)/2);
f=fourindex_dense<twoelectronrealmullikan,T,I>(NRSMat<T>(mat)); //symmetrize mat 
}


//operator() : (i,j,k,l) in fourindex -> (k,l,i,j) in tensor
template<typename T, typename I>
Tensor<T>  fourindex2tensor(const fourindex_dense<T2IjAb_aces,T,I> &f)
{
NRVec<INDEXGROUP> shape(4);
for(int i=0; i<4; ++i)
	{
	shape[i].number=1;
	shape[i].symmetry=0;
	shape[i].offset=1;
	}
shape[2].range=f.noca;
shape[3].range=f.nocb;
shape[0].range=f.nvra;
shape[1].range=f.nvrb;
NRVec<T> data(f);
return Tensor<T>(shape,data);
}

template<typename T, typename I>
void tensor2fourindex(const Tensor<T> &t, fourindex_dense<T2IjAb_aces,T,I> &f)
{
if(t.rank()!=4 ||t.shape.size()!=4) laerror("wrong rank/shape in tensor2fourindex");
int noca = t.shape[2].range;
int nocb = t.shape[3].range;
int nvra = t.shape[0].range;
int nvrb = t.shape[1].range;

int offset=t.shape[0].offset;
for(int i=0; i<t.shape.size(); ++i)
	{
	if(offset!=t.shape[i].offset) laerror("offset mismatch in tensor2fourindex");
	if(0!=t.shape[i].symmetry) laerror("symmetry mismatch in tensor2fourindex");
	}
NRMat<T> mat(t.data,noca*nocb,nvra*nvrb);
f=fourindex_dense<T2IjAb_aces,T,I>(noca,nocb,nvra,nvrb,mat);
}


//operator() : (i,j,k,l) in fourindex -> (k,l,i,j) in tensor
template<typename T, typename I>
Tensor<T>  fourindex2tensor(const fourindex_dense<T2ijab_aces,T,I> &f)
{
NRVec<INDEXGROUP> shape(2);
for(int i=0; i<2; ++i)
	{
	shape[i].number=2;
	shape[i].symmetry= -1;
	shape[i].offset=1;
	}
shape[0].range=f.nvrt;
shape[1].range=f.nocc;
NRVec<T> data(f);
return Tensor<T>(shape,data);
}

template<typename T, typename I>
void tensor2fourindex(const Tensor<T> &t, fourindex_dense<T2ijab_aces,T,I> &f)
{
if(t.rank()!=4) laerror("wrong rank in tensor2fourindex");
if(t.shape.size()!=2) laerror("wrong symmetry groups in tensor2fourindex");
int nvrt=t.shape[0].range;
int nocc=t.shape[1].range;
int offset=t.shape[0].offset;
for(int i=0; i<t.shape.size(); ++i)
	{
	if(offset!=t.shape[i].offset) laerror("offset mismatch in tensor2fourindex");
	if(-1!=t.shape[i].symmetry) laerror("symmetry mismatch in tensor2fourindex");
	}
NRMat<T> mat(t.data,nocc*(nocc-1)/2,nvrt*(nvrt-1)/2);
f=fourindex_dense<T2ijab_aces,T,I>(nocc,nvrt,mat);
}


//operator() : (i,j,k,l) in fourindex -> (k,l,i,j)=(i,j,k,l) in tensor
template<typename T, typename I>
Tensor<T>  fourindex2tensor(const fourindex_dense<antisymtwoelectronrealdiracAB,T,I> &f)
{
NRVec<INDEXGROUP> shape(4);
int n=f.nbas();
for(int i=0; i<4; ++i)
	{
	shape[i].number=1;
	shape[i].symmetry=0;
	shape[i].offset=1;
	shape[i].range=n;
	}
NRMat<T> data(f); //expand to full matrix
return Tensor<T>(shape,NRVec<T>(data));
}

template<typename T, typename I>
void tensor2fourindex(const Tensor<T> &t, fourindex_dense<antisymtwoelectronrealdiracAB,T,I> &f)
{
if(t.rank()!=4) laerror("wrong rank in tensor2fourindex");
if(t.shape.size()!=4) laerror("wrong symmetry groups in tensor2fourindex");
int range=t.shape[0].range;
int offset=t.shape[0].offset;
for(int i=0; i<t.shape.size(); ++i)
	{
	if(range!=t.shape[i].range) laerror("range mismatch in tensor2fourindex");
	if(offset!=t.shape[i].offset) laerror("offset mismatch in tensor2fourindex");
	if(0!=t.shape[i].symmetry) laerror("symmetry mismatch in tensor2fourindex");
	}
NRMat<T> mat(t.data,range*range,range*range);
f=fourindex_dense<antisymtwoelectronrealdiracAB,T,I>(range,NRSMat<T>(mat));
}


//operator() : (i,j,k,l) in fourindex -> (k,l,i,j)=(i,j,k,l) in tensor
template<typename T, typename I>
Tensor<T>  fourindex2tensor(const fourindex_dense<antisymtwoelectronrealdirac,T,I> &f)
{
NRVec<INDEXGROUP> shape(2);
int n=f.nbas();
for(int i=0; i<2; ++i)
	{
	shape[i].number=2;
	shape[i].symmetry= -1;
	shape[i].offset=1;
	shape[i].range=n;
	}
NRMat<T> mat(f);
NRVec<T> data(mat); //expand symmetric to full matrix for index group symmetry
return Tensor<T>(shape,data);
}

template<typename T, typename I>
void tensor2fourindex(const Tensor<T> &t, fourindex_dense<antisymtwoelectronrealdirac,T,I> &f)
{
if(t.rank()!=4) laerror("wrong rank in tensor2fourindex");
if(t.shape.size()!=2) laerror("wrong symmetry groups in tensor2fourindex");
int range=t.shape[0].range;
int offset=t.shape[0].offset;
for(int i=0; i<t.shape.size(); ++i)
	{
	if(range!=t.shape[i].range) laerror("range mismatch in tensor2fourindex");
	if(offset!=t.shape[i].offset) laerror("offset mismatch in tensor2fourindex");
	if(-1!=t.shape[i].symmetry) laerror("symmetry mismatch in tensor2fourindex");
	}
NRMat<T> mat(t.data,range*(range-1)/2,range*(range-1)/2);
f=fourindex_dense<antisymtwoelectronrealdirac,T,I>(range,NRSMat<T>(mat)); //symmetrize mat 
}



template <typename T>
std::ostream & operator<<(std::ostream &s, const Tensor<T> &x);

template <typename T>
std::istream & operator>>(std::istream &s, Tensor<T> &x);





}//namespace
#endif
