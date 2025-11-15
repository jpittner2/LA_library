/* vim: set ts=8 sw=8 sts=8 noexpandtab cindent: */
/*******************************************************************************
    LA: linear algebra C++ interface library
    Copyright (C) 2008 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>
		  complex versions written by Roman Curik <roman.curik@jh-inst.cas.cz>

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
*******************************************************************************/
#ifndef _LA_VEC_H_
#define _LA_VEC_H_

#include "la_traits.h"
#include "vecmat3.h"
#include <list>

using namespace LA_Vecmat3;

namespace LA {

/***************************************************************************//**
 * forward declarations
 ******************************************************************************/
template <typename T> void lawritemat(FILE *file, const T *a, int r, int c,
		const char *form0, int nodim, int modulo, int issym);

template <typename T> class NRPerm;
template <typename T> class CyclePerm;

/***************************************************************************//**
 * auxiliary macro to avoid compilation errors for some types
 ******************************************************************************/
template <typename T>
inline  typename LA_traits<T>::normtype  MYABS(const T &x) {return std::abs(x);}

template <> inline unsigned char MYABS(const  unsigned char &x) {return x;}
template <> inline unsigned short MYABS(const  unsigned short &x) {return x;}
template <> inline unsigned int MYABS(const  unsigned int &x) {return x;}
template <> inline unsigned long MYABS(const  unsigned long &x) {return x;}
template <> inline unsigned long long MYABS(const  unsigned long long &x) {return x;}



/***************************************************************************//**
 * static constants used in several cblas-routines
 ******************************************************************************/
const static std::complex<double> CONE = 1.0, CMONE = -1.0, CZERO = 0.0;
#ifdef CUDALA
const static cuDoubleComplex CUONE = {1.,0.}, CUMONE = {-1.,0.}, CUZERO = {0.,0.};
#endif

/***************************************************************************//**
 * macros to construct binary operators +,-,*, from +=, -=, *=
 * for 3 cases: X + a, a + X, X + Y
 ******************************************************************************/
#define NRVECMAT_OPER(E,X) \
template<class T> \
	inline const NR##E<T> NR##E<T>::operator X(const T &a) const \
{ return NR##E(*this) X##= a; } \
	\
	template<class T> \
	inline const NR##E<T> operator X(const T &a, const NR##E<T> &rhs) \
{ return NR##E<T>(rhs) X##= a; }

#define NRVECMAT_OPER2(E,X) \
template<class T> \
inline const NR##E<T> NR##E<T>::operator X(const NR##E<T> &a) const \
{ return NR##E(*this) X##= a; }


/***************************************************************************//**
 * \brief NRVec<T> class template implementing the vector interface
 * @see NRMat<T>, NRSMat<T>
 ******************************************************************************/
template <typename T>
class NRVec {
protected:
	int nn;//!< size of the vector
	T *v;//!< pointer to the underlying data structure
	int *count;//!< pointer to the reference-counter
#ifdef CUDALA
	GPUID location;//!< determines the memory address space of this object (CPU/GPU)
#endif
public:
	friend class NRSMat<T>;
	friend class NRMat<T>;
	template <typename U> friend NRVec<std::complex<U> > complexify(const NRVec<U>&);

	typedef T ROWTYPE;

	//! standard destructor 
	~NRVec();

	/***************************************************************************//**
	 * inlined constructor creating zero vector of general type <code>T</code>
	 ******************************************************************************/
	inline NRVec(): nn(0), v(0), count(0) {
		#ifdef CUDALA
		location = DEFAULT_LOC;
		#endif
        };

	/***************************************************************************//**
	 * Explicit inlined constructor creating vector of given size and location.
	 * Because of performance reasons, no incialization is done.
	 * @param[in] n vector size (count of elements)
	 * @param[in] loc location of the underlying data (CPU/GPU)
	 ******************************************************************************/
	explicit inline NRVec(const int n, const GPUID loc = undefined): nn(n), count(new int(1)) {
		#ifdef CUDALA
			location = (loc == undefined)?DEFAULT_LOC:loc;
			if(location == cpu){
		#endif
				v = new T[n];
		#ifdef CUDALA
			}else{
				v = (T*) gpualloc(n*sizeof(T));
			}
		#endif
	};
	
	//! inlined constructor creating vector of given size filled with prescribed value 
	//inline NRVec(const T &a, const int n);
	inline NRVec(const T &a, const int n, const GPUID loc = undefined);

	//! inlined constructor creating vector from an array
	template<int SIZE> inline NRVec(const T (&a)[SIZE]);
	
	//! inlined constructor creating vector of given size filled with data located at given memory location 
        inline NRVec(const T *a, const int n);
	inline NRVec(const Vec3<T> &rhs) : NRVec(&rhs[0],3) {};
	
	//! inlined constructor creating vector of given size filled with data located at given memory location 
	inline NRVec(T *a, const int n, bool skeleton);

	//! inlined copy constructor 
	inline NRVec(const NRVec &rhs);

	//! constructor from std::list
	explicit NRVec(const std::list<T> l);

	//! complexifying constructor 
	NRVec(const typename LA_traits_complex<T>::NRVec_Noncomplex_type &rhs, bool imagpart=false);//construct complex from real

	//! explicit inlined constructor converting symmetric matrix into a vector 
	inline explicit NRVec(const NRSMat<T> & S);
	
	/***************************************************************************//**

	******************************************************************************/
	#ifdef MATPTR
		explicit NRVec(const NRMat<T> &rhs): NRVec(&rhs[0][0], rhs.nrows()*rhs.ncols()) {};
	#else
		explicit NRVec(const NRMat<T> &rhs);
	#endif

	/***************************************************************************//**
	 * routines for CUDA related stuff
	 * \li <code>getlocation()</code> gets the protected data member location
	 * \li <code>moveto(const GPUID)</code> moves underlying data between CPU/GPU memory
	******************************************************************************/
	#ifdef CUDALA
	        inline GPUID getlocation() const { return location; }
	        void moveto(const GPUID dest);
	#else
	        inline GPUID getlocation() const { return cpu; }
	        void moveto(const GPUID dest) {};
	#endif

	//! create separate copy of the data corresponding to this vector 
	void copyonwrite(bool detachonly=false, bool deep=true);

	//! purge this vector 
	void clear() { copyonwrite(LA_traits<T>::is_plaindata()); LA_traits<T>::clear(v, nn); };

	//! assignment operator assigns given vector
	NRVec& operator=(const NRVec &rhs);

	//! assigment operator assigns given scalar to each element of this vector 
	NRVec& operator=(const T &a);
	
	//! fills in this vector with pseudo-random numbers generated using uniform distribution  
        void randomize(const typename LA_traits<T>::normtype &x);

	//! perform deep-copy of given vector 
	NRVec& operator|=(const NRVec &rhs);

        //! extract specified subvector
        const NRVec subvector(const int from, const int to) const;
	const NRVec subvector(const NRVec<int> &selection) const;

        //! store given vector at given position into the current vector
        void storesubvector(const int from, const NRVec &rhs);
	void storesubvector(const NRVec<int> &selection, const NRVec &rhs);

	//! relational operators 
	const bool operator!=(const NRVec &rhs) const 
		{
		if(nn!=rhs.nn) return 1; 
		if(LA_traits<T>::is_plaindata()) return LA_traits<T>::gencmp(v,rhs.v,nn);
		else
			{
			for(int i=0; i<nn; ++i) if(v[i]!=rhs.v[i]) return 1;
			return 0;
			}
		}
	const bool operator==(const NRVec &rhs) const {return !(*this != rhs);};
	const bool operator>(const NRVec &rhs) const;
	const bool operator<(const NRVec &rhs) const;
	const bool operator>=(const NRVec &rhs) const {return !(*this < rhs);};
	const bool operator<=(const NRVec &rhs) const {return !(*this > rhs);};

	//! unary minus 
	const NRVec operator-() const;
	
	//! bunch of vector-vector arithmetic operators defined element-wise 
	inline NRVec& operator+=(const NRVec &rhs);
	inline NRVec& operator-=(const NRVec &rhs);
	inline NRVec& operator*=(const NRVec &rhs);
	inline NRVec& operator/=(const NRVec &rhs);

	inline const NRVec operator+(const NRVec &rhs) const;
	inline const NRVec operator-(const NRVec &rhs) const;

	//! bunch of scalar-vector arithmetic operators defined element-wise 
	inline NRVec& operator+=(const T &a);
	inline NRVec& operator-=(const T &a);
	inline NRVec& operator*=(const T &a);
	inline NRVec& operator/=(const T &a);

	inline const NRVec operator+(const T &a) const;
	inline const NRVec operator-(const T &a) const;
	inline const NRVec operator*(const T &a) const;
	inline const NRVec operator/(const T &a) const;
	
	//!concatenate vectors
	NRVec concat(const NRVec &rhs) const 
		{
		if(nn==0) return rhs;
		if(rhs.nn==0) return *this;
		NRVec r(nn+rhs.nn);
		for(int i=0; i<nn; ++i) r[i] = (*this)[i];
		for(int i=0; i<rhs.nn; ++i) r[nn+i] = rhs[i];
		return r;
		}

        NRVec concatscaled(const NRVec &rhs,  const T alpha=1) const
                {
                if(nn==0) return rhs;
                if(rhs.nn==0) return *this;
                NRVec r(nn+rhs.nn);
                for(int i=0; i<nn; ++i) r[i] = (*this)[i];
                for(int i=0; i<rhs.nn; ++i) r[nn+i] = rhs[i]*alpha;
                return r;
                }


	//!concatenate vector into *this 
	void concatme(const NRVec &rhs) 
		{
		NOT_GPU(*this);
		if(rhs.nn==0) return;
		int nnold=nn;
		resize(nn+rhs.nn,true);
		for(int i=0; i<rhs.nn; ++i) v[nnold+i] = rhs[i];
		}
	void concatmescaled(const NRVec &rhs, const T alpha=1)
		{
                NOT_GPU(*this);
                if(rhs.nn==0) return;
                int nnold=nn;
                resize(nn+rhs.nn,true);
                for(int i=0; i<rhs.nn; ++i) v[nnold+i] = rhs[i]*alpha;
                }

	void append(const T &a) //not efficient if done repeatedly
		{
		NOT_GPU(*this);
		int nnold=nn;
                resize(nn+1,true);
                v[nnold] = a;
                }

	void prepend(const T &a) //not efficient if done repeatedly
                {
		NOT_GPU(*this); 
                int nnold=nn;
                resize(nn+1,true,true);
                v[0] = a;
                }

	//! complex conjugate 
        NRVec& conjugateme();
        inline NRVec conjugate() const {NRVec r(*this); r.conjugateme(); return r;};

	//! determine the actual value of the reference counter 
	inline int getcount() const {return count?*count:0;}

	//! compute the Euclidean inner product (with conjugation in complex case) 
	inline const T operator*(const NRVec &rhs) const;
	inline const T dot(const NRVec &rhs) const {return *this * rhs;};

	//! compute the Euclidean inner product (with conjugation in complex case) with a stride-vector
	inline const T dot(const T *a, const int stride = 1) const;

	void gemv(const T beta, const NRMat<T> &a, const char trans, const T alpha, const NRVec &x);
	void gemv(const T beta, const NRSMat<T> &a, const char trans /**< just for compatibility reasons */, const T alpha, const NRVec &x);
	void gemv(const T beta, const SparseMat<T> &a, const char trans, const T alpha, const NRVec &x,const bool treat_as_symmetric = false);

	void gemv(	const typename LA_traits_complex<T>::Component_type beta, 
			const typename LA_traits_complex<T>::NRMat_Noncomplex_type &a, 
			const char trans,
			const typename LA_traits_complex<T>::Component_type alpha,
			const NRVec &x);

	void gemv(	const typename LA_traits_complex<T>::Component_type beta,
			const typename LA_traits_complex<T>::NRSMat_Noncomplex_type &a,
			const char trans,
			const typename LA_traits_complex<T>::Component_type alpha, const NRVec &x);

	//! multiply given matrix with this vector from left 
	const NRVec operator*(const NRMat<T> &mat) const {
		SAME_LOC(*this, mat);

		NRVec<T> result(mat.ncols(), mat.getlocation());
		result.gemv((T)0, mat, 't', (T)1, *this);
		return result;
	};

	//! multiply given symmetric matrix in packed form with this vector from left 
	const NRVec operator*(const NRSMat<T> &mat) const {
		SAME_LOC(*this, mat);

		NRVec<T> result(mat.ncols(), mat.getlocation());
		result.gemv((T)0, mat, 't', (T)1, *this);
		return result;
	};

	//! multiply given sparse matrix with this vector from left 
	const NRVec operator*(const SparseMat<T> &mat) const {
		NOT_GPU(*this);

		NRVec<T> result(mat.ncols());
		result.gemv((T)0, mat, 't', (T)1, *this);
		return result;
	};
	
	//! compute the outer product of two vectors 
	const NRMat<T> otimes(const NRVec<T> &rhs, const bool conjugate = false, const T &scale = 1) const;

	//! opeartor for outer product computation
	inline const NRMat<T> operator|(const NRVec<T> &rhs) const { return otimes(rhs,true); };

        //! compute the outer product of two vectors, result interpreted as a vector
        const NRVec otimes2vec(const NRVec<T> &rhs, const bool conjugate = false, const T &scale = 1) const;

	//! compute the sum of the vector elements 
	const T sum() const;

	//! compute the sum of squares the vector elements
        inline const T sumsqr() const {
                T sum(v[0]);
                for(register int i=1; i<nn; i++){ sum += v[i]*v[i];}
                return sum;
        };


	//! compute the product of the vector elements
        inline const T prod() const {
                T prod(v[0]);
                for(register int i=1; i<nn; i++){ prod *= v[i];}
                return prod;
        };

	//! permute vector elements
	const NRVec permuted(const NRPerm<int> &p, const bool inverse=false) const;
	void permuteme(const CyclePerm<int> &p); //in place
	void permuteme(const NRPerm<int> &p, const bool inverse=false)
			{
			NRVec<T> tmp=permuted(p,inverse); 
			copyonwrite();
			for(int i=0; i<size(); ++i) v[i] = tmp.v[i];
			}



	//! compute the sum of the absolute values of the elements of this vector 
	inline const typename LA_traits<T>::normtype asum() const;

	//! indexing operator - index running from zero
	inline T & operator[](const int i);
	inline const T & operator[](const int i) const;

	//! dummy routine 
	inline void setcoldim(int i) {};

	//! get the pointer to the underlying data structure 
	inline operator T*(); 

	//! get the constant pointer to the underlying data structure 
	inline operator const T*() const;

	//! add up a scalar multiple of a given vector
	void axpy(const T alpha, const NRVec &x);
	
	//! add up a scalar multiple of a given vector with given stride
	void axpy(const T alpha, const T *x, const int stride=1);

	//! determine the number of elements
	inline int size() const;

	//! resize the current vector, optionally preserving data
	void resize(const int n, const bool preserve=false, const bool preserve_at_end=false);

	//!deallocate the current vector
	void dealloc(void) {resize(0);}

	//! determine the norm of this vector 
	inline const typename LA_traits<T>::normtype norm(int length= -1, int offset=0, int stride=1) const;

	//! normalize this vector and optionally save the norm 
	NRVec& normalize(typename LA_traits<T>::normtype* norm = 0);

	//! get normalized copy of this vector 
	inline const NRVec unitvector() const;

	//! find an element by value with threshold, first from left 
	const int find(const T &val) const;
	const int findthr(const T &val, const  typename LA_traits<T>::normtype &thr=0) const;

	//! determine the maximal element (in the absolute value) of this vector 
	inline const T amax() const;
	//! determine the minimal element (in the absolute value) of this vector 
	inline const T amin() const;

        //! determine the maximal element of this vector
        const T max() const;
        //! determine the minimal element of this vector
        const T min() const;


	//! routine for formatted output 
	void fprintf(FILE *f, const char *format, const int modulo) const;
	//! routine for unformatted output 
        void put(int fd, bool dimensions=1, bool transp=0) const;
	
	//! routine for formatted input 
	void fscanf(FILE *f, const char *format);
	//! routine for unformatted input 
        void get(int fd, bool dimensions=1, bool transp=0);

	//! constructor creating vector from sparse matrix
	explicit NRVec(const SparseMat<T> &rhs);
	
	//! routine for compatibility with sparse types
	inline void simplify() {}; 

	//! determine whether the i<sup>th</sup> element is bigger than the j<sup>th</sup> element
	bool bigger(int i, int j) const {
		NOT_GPU(*this);
		return LA_traits<T>::bigger(v[i], v[j]);
	};

	//! determine whether the i<sup>th</sup> element is bigger than the j<sup>th</sup> element
	bool smaller(int i, int j) const {
		NOT_GPU(*this);
		return LA_traits<T>::smaller(v[i], v[j]);
	};

	//! swap the i<sup>th</sup> and j<sup>th</sup> element
	void swap(int i, int j) {
		const T tmp(v[i]);
		v[i] = v[j];
		v[j] = tmp;
	};

	//! sort by default in ascending order and return the parity of corresponding permutation resulting to this order
	int sort(int direction = 0, int from = 0, int to = -1, int *perm = NULL, bool stable=false);
	int sort(int direction, NRPerm<int> &perm, bool stable=false);
	void printsorted(std::ostream &s, int direction=1, bool absvalue=false, bool stable=false, typename LA_traits<T>::normtype thr=0, int topmax=0, const NRMat<int> *indexmap=NULL) const;

	//! apply given function to each element
	NRVec& call_on_me(T (*_F)(const T &) ){
		NOT_GPU(*this);

		copyonwrite();
		for(int i=0; i<nn; ++i) v[i] = _F(v[i]);
		return *this;
	};

	void swap(NRVec &rhs) //more efficient swap than via tmp and constructors and operator=
		{
		int tmpnn=nn; nn=rhs.nn; rhs.nn=tmpnn;
		T *tmpv=v; v=rhs.v; rhs.v=tmpv;
		int *tmpcount=count; count=rhs.count; rhs.count=tmpcount;
#ifdef CUDALA
		GPUID tmplocation=location; location=rhs.location; rhs.location=tmplocation;
#endif

		}

	NRVec<typename LA_traits<T>::normtype> diffabs(const NRVec &rhs) const; //difference of absolute values
	NRVec<typename LA_traits<T>::normtype> abs() const; //element-wise absolute values
};




/***************************************************************************//**
 * implements \c NRVec<T> functionality with indexing from 1
 * all possible constructors have to be given explicitly, other stuff is inherited
 * with exception of the operator() which differs
 ******************************************************************************/
template<typename T>
class NRVec_from1 : public NRVec<T> {
public:
	NRVec_from1(): NRVec<T>() {};
	template<int SIZE>  NRVec_from1(const T (&a)[SIZE]) : NRVec<T>(a) {};
	explicit NRVec_from1(const int n): NRVec<T>(n) {};
	NRVec_from1(const NRVec<T> &rhs): NRVec<T>(rhs) {};//!< be able to convert the parent class transparently to this
	NRVec_from1(const T &a, const int n): NRVec<T>(a, n) {};
	NRVec_from1(const T *a, const int n): NRVec<T>(a, n) {};
	explicit NRVec_from1(const std::list<T> l) : NRVec<T>(l) {};
	inline const T& operator[] (const int i) const;
        inline T& operator[] (const int i);
};


}//namespace

//due to mutual includes this has to be after full class declaration
#include "mat.h"
#include "smat.h"
#include "sparsemat.h"
#include "sparsesmat.h"
#include "qsort.h"




//needs NRVec_from1
#include "permutation.h"
namespace LA {


template<typename T>
int NRVec<T>::sort(int direction, int from, int to, int *perm, bool stable) {
	NOT_GPU(*this);

	copyonwrite();
	if(to == -1) to = nn - 1;
	if(stable)
		{
		if(direction) return memqsortstable<1, NRVec<T>, int, int>(*this, perm, from, to);
                else return memqsortstable<0, NRVec<T>, int, int>(*this, perm, from, to);
		}
	else
		{	
		if(direction) return memqsort<1, NRVec<T>, int, int>(*this, perm, from, to);
		else return memqsort<0, NRVec<T>, int, int>(*this, perm, from, to);
		}
}

template<typename T>
int NRVec<T>::sort(int direction, NRPerm<int> &perm, bool stable)
{
if(nn!=perm.size()) laerror("incompatible vector and permutation");
perm.identity();
int r=sort(direction,0,nn-1,&perm[1],stable);
return r;
}


template<typename T>
void NRVec<T>::printsorted(std::ostream &s, int direction, bool absvalue, bool stable, typename LA_traits<T>::normtype thr, int topmax,  const NRMat<int> *indexmap) const
{
NRPerm<int> perm(nn); perm.identity();
NRVec_from1<T> tmp;
NRVec_from1<typename LA_traits<T>::normtype> tmp2;
if(absvalue) 
	{
	tmp2.resize(nn);
	for(int i=1; i<=nn; ++i) tmp2[i] = MYABS((*this)[i-1]);
	tmp2.sort(direction,perm,stable);
	}
else
	{
	tmp= *this;
	tmp.sort(direction,perm,stable);
	}
int ii=1;
for(int i=1; i<=nn; ++i)
	{
	if(topmax && ii>topmax) break;
	if(thr!=0 && (absvalue?tmp2[i]:MYABS(tmp[i]))<thr) continue;
	if(indexmap) s<<(*indexmap)[perm[i]-1][0]<<" "<<(*indexmap)[perm[i]-1][1]; else s<<perm[i]-1;
	s<<' ' << (*this)[perm[i]-1]<<std::endl;
	++ii;
	}
}


/***************************************************************************//**
 * indexing operator giving the element at given position with range checking in
 * the DEBUG mode
 * @param[in] i position of the required vector element (starting from 0)
 * @return reference to the requested element
 ******************************************************************************/
template <typename T>
inline T& NRVec_from1<T>::operator[](const int i) {
#ifdef DEBUG
        if(_LA_count_check && *NRVec<T>::count != 1) laerror("possible use of NRVec[] with count>1 as l-value");
        if(i < 1 || i > NRVec<T>::nn) laerror("out of range");
        if(!NRVec<T>::v) laerror("unallocated NRVec");
#endif
        NOT_GPU(*this);

        return NRVec<T>::v[i-1];
}

/***************************************************************************//**
 * indexing operator giving the element at given position with range checking in
 * the DEBUG mode
 * @param[in] i position of the required vector element (starting from 0)
 * @return constant reference to the requested element
 ******************************************************************************/
template <typename T>
inline const T& NRVec_from1<T>::operator[](const int i) const {
#ifdef DEBUG
        if(i < 1 || i > NRVec<T>::nn) laerror("out of range");
        if(!NRVec<T>::v) laerror("unallocated NRVec");
#endif
        NOT_GPU(*this);

        return NRVec<T>::v[i-1];
}


/***************************************************************************//**
 * output operator
 * @param[in,out] s output stream
 * @param[in] x vector of general type intended for output
 * @return modified stream
 ******************************************************************************/
template <typename T>
std::ostream & operator<<(std::ostream &s, const NRVec<T> &x) {
#ifdef CUDALA
	if(x.getlocation() == cpu){
#endif
		const int n = x.size();
		s << n << std::endl;
		for(register int i = 0; i<n; i++){
			s << (typename LA_traits_io<T>::IOtype)x[i] << (i == n-1 ? '\n' : ' ');
		}
		return s;
#ifdef CUDALA
	}else{
		NRVec<T> tmp(x);
		tmp.moveto(cpu);
		return s << tmp;
	}
#endif
}

/***************************************************************************//**
 * input operator
 * @param[in,out] s input stream
 * @param[in] x vector of general type intended for input
 * @return modified stream
 ******************************************************************************/
template <typename T>
std::istream & operator>>(std::istream &s, NRVec<T> &x) {
#ifdef CUDALA
	if(x.getlocation() == cpu){
#endif
		int i,n;
		s >> n;
		x.resize(n);
		typename LA_traits_io<T>::IOtype tmp;
		for(i=0; i<n; i++){
			s >> tmp;
			x[i] = tmp;
		}
		return s;
#ifdef CUDALA
	}else{
		NRVec<T> tmp;
		tmp.moveto(cpu);
		s >> tmp;
		tmp.moveto(x.getlocation());
		x = tmp;
		return s;
	}
#endif
}


/***************************************************************************//**
 * inline constructor creating vector of given size filled with prescribed value
 * @param[in] a value to be assigned to all vector elements
 * @param[in] n required vector size
 ******************************************************************************/

/* replaced by the one with optional GPUID
template <typename T>
inline NRVec<T>::NRVec(const T& a, const int n): nn(n), count(new int) {
	*count = 1;
#ifdef CUDALA
	location = DEFAULT_LOC;
	if(location == cpu){
#endif
		v = new T[n];
                if(!LA_traits<T>::is_plaindata() || a != (T)0){
                        for(register int i=0; i<n; i++) v[i] = a;
                }else{
                        memset(v, 0, nn*sizeof(T));
               }
#ifdef CUDALA
	}else{
		v = (T*) gpualloc(n*sizeof(T));
		if(!LA_traits<T>::is_plaindata()) laerror("only implemented for plain data");
		smart_gpu_set(n, a, v);
	}
#endif
}
*/

template <typename T>
inline NRVec<T>::NRVec(const T& a, const int n,  const GPUID loc): nn(n), count(new int) {
	*count = 1;
#ifdef CUDALA
	location = (loc==undefined?DEFAULT_LOC:loc);
	if(location == cpu){
#endif
		v = new T[n];
                if(!LA_traits<T>::is_plaindata() || a != (T)0){
                        for(register int i=0; i<n; i++) v[i] = a;
                }else{
                        memset(v, 0, nn*sizeof(T));
               }
#ifdef CUDALA
	}else{
		if(sizeof(T)%sizeof(float) != 0)laerror("memory alignment error");
		v = (T*) gpualloc(n*sizeof(T));
		if(!LA_traits<T>::is_plaindata()) laerror("only implemented for plain data");
		smart_gpu_set(n, a, v);
	}
#endif
}

/***************************************************************************//**
 * inline constructor creating vector from an array
 ******************************************************************************/


template <typename T> template<int SIZE>
inline NRVec<T>::NRVec(const T (&a)[SIZE]) : count(new int) {
	nn = SIZE;
        *count = 1;
#ifdef CUDALA
        location = DEFAULT_LOC;
        if(location == cpu){
#endif
                v = new T[nn];
                if(LA_traits<T>::is_plaindata()) memcpy(v, a, nn*sizeof(T));
		else for( int i=0; i<nn; i++) v[i] = a[i];
#ifdef CUDALA
        }else{
                v = (T*) gpualloc(nn*sizeof(T));
		if(!LA_traits<T>::is_plaindata()) laerror("only implemented for plain data");
		cublasSetVector(nn, sizeof(T), a, 1, v, 1);
                TEST_CUBLAS("cublasSetVector");
        }
#endif
}





/***************************************************************************//**
 * inline constructor creating vector of given size filled with given data
 * @param[in] a pointer to the data
 * @param[in] n required vector size
 ******************************************************************************/
template <typename T>
inline NRVec<T>::NRVec(const T *a, const int n): nn(n), count(new int) {
#ifdef CUDALA
	location = DEFAULT_LOC;
	if(location == cpu) {
#endif
		v = new T[n];
		*count = 1;
		if(LA_traits<T>::is_plaindata()) memcpy(v, a, n*sizeof(T));
		else for( int i=0; i<n; i++) v[i] = a[i];
#ifdef CUDALA
	}else{
		v = (T*) gpualloc(n*sizeof(T));
		if(!LA_traits<T>::is_plaindata()) laerror("only implemented for plain data");
		cublasSetVector(n, sizeof(T), a, 1, v, 1);
		TEST_CUBLAS("cublasSetVector");
	}
#endif

}

/***************************************************************************//**
 * inline constructor creating vector of given size filled with given data
 * @param[in] a pointer to the data
 * @param[in] n required vector size
 * @param[in] skeleton if equal to true, only the internal data pointer is modified
 * and reference counter is set to two, i.e. no data deallocation occurs in destructor
 ******************************************************************************/
template <typename T>
inline NRVec<T>::NRVec(T *a, const int n, bool skeleton) : nn(n), count(new int) {
	if(!skeleton){
#ifdef CUDALA
		location = DEFAULT_LOC;
		if(location == cpu){
#endif
			v = new T[n];
			*count = 1;
			if(LA_traits<T>::is_plaindata()) memcpy(v, a, n*sizeof(T));
			else for( int i=0; i<n; i++) v[i] = a[i];
#ifdef CUDALA
		}else{
			v= (T*) gpualloc(n*sizeof(T));
			if(!LA_traits<T>::is_plaindata()) laerror("only implemented for plain data");
			cublasSetVector(n, sizeof(T), a, 1, v, 1);
			TEST_CUBLAS("cublasSetVector");
		}
#endif
	}else{
#ifdef CUDALA
		if(location != cpu) laerror("NRVec() with skeleton option cannot be on GPU");
#endif
		*count = 2;
		v = a;
	}
}

/***************************************************************************//**
 * inline copy constructor
 * @param[in] rhs reference vector being copied
 ******************************************************************************/
template <typename T>
inline NRVec<T>::NRVec(const NRVec<T> &rhs) {
#ifdef CUDALA
	location = rhs.location;
#endif
	v = rhs.v;
	nn = rhs.nn;
	count = rhs.count;
	if(count) (*count)++;
}

/***************************************************************************//**
 * inline constructor interpreting symmetric matrix of order \f$n\f$ stored in packed form
 * as a linear vector consisting of \f$n(n+1)/2\f$ elements
 * @param[in] rhs symmetric matrix of type <code>NRSMat<T></code>
 * @see NRSMat<T>
 ******************************************************************************/
template <typename T>
inline NRVec<T>::NRVec(const NRSMat<T> &rhs) {
#ifdef CUDALA
        location = rhs.location;
#endif
	nn = rhs.nn;
	//! using macro NN2 defined in smat.h 
	nn = NN2;
	v = rhs.v;
	count = rhs.count;
	(*count)++;
}

/***************************************************************************//**
 * adds given scalar value of type T to all vector elements
 * @param[in] a scalar value being added
 * @return reference to the modified vector
 ******************************************************************************/
template <typename T>
inline NRVec<T> & NRVec<T>::operator+=(const T &a) {
	NOT_GPU(*this);

        copyonwrite();

	if(a != (T)0){ for(register int i=0; i<nn; ++i) v[i] += a; }
        return *this;
}


/***************************************************************************//**
 * subtracts given scalar value of type T from all vector elements
 * @param[in] a scalar value being subtracted
 * @return reference to the modified vector
 ******************************************************************************/
template <typename T>
inline NRVec<T>& NRVec<T>::operator-=(const T &a) {
	NOT_GPU(*this);

	copyonwrite();

	if(a != (T)0){ for(register int i=0; i<nn; ++i) v[i] -= a; }
	return *this;
}


/***************************************************************************//**
 * adds a vector \f$\vec{y}\f$ of general type <code>T</code> to this vector \f$\vec{x}\f$
 * \f[\vec{x}\leftarrow\vec{x}+\vec{y}\f]
 * @param[in] rhs vector \f$\vec{y}\f$ of type <code>T</code>
 * @return reference to the modified vector
 ******************************************************************************/
template <typename T>
inline NRVec<T>& NRVec<T>::operator+=(const NRVec<T> &rhs) {
#ifdef DEBUG
	if (nn != rhs.nn) laerror("incompatible dimensions");
#endif
	NOT_GPU(*this);
	NOT_GPU(rhs);

	copyonwrite();

	for(register int i=0; i<nn; ++i) v[i] += rhs.v[i];
	return *this;
}

/***************************************************************************//**
 * multiplies this vector \f$\vec{y}\f$ componentwise by general vector \f$\vec{x}\f$
 * \f[\vec{x}_i = \vec{x}_i\times\vec{y}_i\f]
 * @param[in] rhs general vector \f$\vec{y}\f$
 * @return reference to the modified vector
 ******************************************************************************/
template <typename T>
inline NRVec<T>& NRVec<T>::operator*=(const NRVec<T>& rhs) {
#ifdef DEBUG
	if (nn != rhs.nn) laerror("incompatible dimensions");
#endif
	NOT_GPU(*this);
	NOT_GPU(rhs);
        
	copyonwrite();

	for(register int i=0; i<nn; ++i) v[i] *= rhs.v[i];
        return *this;
}

/***************************************************************************//**
 * divides this vector \f$\vec{y}\f$ componentwise by general vector \f$\vec{x}\f$
 * \f[\vec{x}_i = \vec{x}_i\slash\vec{y}_i\f]
 * @param[in] rhs general vector \f$\vec{y}\f$
 * @return reference to the modified vector
 ******************************************************************************/
template <typename T>
inline NRVec<T> & NRVec<T>::operator/=(const NRVec<T> &rhs) {
#ifdef DEBUG
	if (nn != rhs.nn) laerror("incompatible dimensions");
#endif
	NOT_GPU(*this);
	NOT_GPU(rhs);

        copyonwrite();

        for(register int i=0; i<nn; ++i) v[i] /= rhs.v[i];
        return *this;
}


/***************************************************************************//**
 * subtracts given vector \f$\vec{y}\f$ from this vector \f$\vec{x}\f$
 * \f[\vec{x}_i = \vec{x}_i-\vec{y}_i\f]
 * @param[in] rhs vector \f$\vec{y}\f$
 * @return reference to the modified vector
 ******************************************************************************/
template <typename T>
inline NRVec<T> & NRVec<T>::operator-=(const NRVec<T> &rhs) {
#ifdef DEBUG
	if (nn != rhs.nn) laerror("incompatible dimensions");
#endif
	NOT_GPU(*this);
	NOT_GPU(rhs);

	copyonwrite();

	for(register int i=0; i<nn; ++i) v[i] -= rhs.v[i];
        return *this;
}

/***************************************************************************//**
 * difference of elements of two vectors in absolute values
 * \f[\vec{z}_i = \vec{x}_i-\vec{y}_i\f]
 * @param[in] rhs vector \f$\vec{y}\f$
 * @return reference to the modified vector
 ******************************************************************************/
template <typename T>
NRVec<typename LA_traits<T>::normtype>  NRVec<T>::diffabs(const NRVec<T> &rhs) const {
#ifdef DEBUG
	if (nn != rhs.nn) laerror("incompatible dimensions");
#endif
	NOT_GPU(*this);
	NOT_GPU(rhs);

NRVec<typename LA_traits<T>::normtype> r(nn);
	for(int i=0; i<nn; ++i) r[i] = MYABS(v[i]) - MYABS(rhs.v[i]);
        return r;
}

template <typename T>
NRVec<typename LA_traits<T>::normtype>  NRVec<T>::abs() const {
        NOT_GPU(*this);

NRVec<typename LA_traits<T>::normtype> r(nn);
        for(int i=0; i<nn; ++i) r[i] = MYABS(v[i]);
        return r;
}


/***************************************************************************//**
 * multiply this general vector \f$\vec{x}\f$ by scalar value \f$\lambda\f$
 * \f[\vec{x}_i \leftarrow \lambda\vec{x}_i\f]
 * @param[in] a scalar value \f$\lambda\f$
 * @return reference to the modified vector
 ******************************************************************************/
template <typename T>
inline NRVec<T> & NRVec<T>::operator*=(const T &a) {
	NOT_GPU(*this);
	copyonwrite();

	for(register int i=0; i<nn; ++i) v[i] *= a;
	return *this;
}

template <typename T>
inline NRVec<T> & NRVec<T>::operator/=(const T &a) {
        NOT_GPU(*this);
        copyonwrite();

        for(register int i=0; i<nn; ++i) v[i] /= a;
        return *this;
}


/***************************************************************************//**
 * compute scalar product \f$d\f$ of this vector \f$\vec{x}\f$ of general type <code>T</code> 
 * with given vector \f$\vec{y}\f$ of type <code>T</code> and order \f$N\f$
 * \f[d = \sum_{i=1}^N\vec{x}_i\cdot\vec{y}_i\f]
 * @param[in] rhs general vector \f$\vec{y}\f$
 * @return reference to the modified vector
 ******************************************************************************/
template<typename T>
inline const T NRVec<T>::operator*(const NRVec<T> &rhs) const {
#ifdef DEBUG
	if (nn != rhs.nn) laerror("incompatible dimensions");
#endif
	NOT_GPU(*this);
	NOT_GPU(rhs);

	T dot(0);
	for(register int i=0; i<nn; ++i) dot += v[i]*rhs.v[i];
	return dot;
}

/***************************************************************************//**
 * indexing operator giving the element at given position with range checking in
 * the DEBUG mode
 * @param[in] i position of the required vector element (starting from 0) 
 * @return reference to the requested element
 ******************************************************************************/
template <typename T>
inline T& NRVec<T>::operator[](const int i) {
#ifdef DEBUG
	if(_LA_count_check && *count != 1) laerror("possible use of NRVec[] with count>1 as l-value");
	if(i < 0 || i >= nn) laerror("out of range");
	if(!v) laerror("unallocated NRVec");
#endif
	NOT_GPU(*this);

	return v[i];
}

/***************************************************************************//**
 * indexing operator giving the element at given position with range checking in
 * the DEBUG mode
 * @param[in] i position of the required vector element (starting from 0) 
 * @return constant reference to the requested element
 ******************************************************************************/
template <typename T>
inline const T& NRVec<T>::operator[](const int i) const {
#ifdef DEBUG
	if(i < 0 || i >= nn) laerror("out of range");
	if(!v) laerror("unallocated NRVec");
#endif
	NOT_GPU(*this);

	return v[i];
}

/***************************************************************************//**
 * determine the number of elements of this vector
 * @return length of this vector
 ******************************************************************************/
template <typename T>
inline int NRVec<T>::size() const {
	return nn;
}

/***************************************************************************//**
 * get the pointer to the underlying data of this vector
 * @return pointer to the first vector element
 ******************************************************************************/
template <typename T>
inline NRVec<T>::operator T*() {
#ifdef DEBUG
	if(!v) laerror("unallocated NRVec");
#endif
	return v;
}

/***************************************************************************//**
 * get the constant pointer to the underlying data of this vector
 * @return constant pointer to the first vector element
 ******************************************************************************/
template <typename T>
inline NRVec<T>::operator const T*() const {
#ifdef DEBUG
	if(!v) laerror("unallocated NRVec");
#endif
	return v;
}


/***************************************************************************//**
 * create normalized copy of this vector
 * @return copy of this vector after normalization
 * @see NRVec<T>::normalize()
 ******************************************************************************/
template <typename T>
inline const NRVec<T> NRVec<T>::unitvector() const {
	return NRVec<T>(*this).normalize();
}

/***************************************************************************//**
 * generate operators involving vector and scalar			     
 ******************************************************************************/
NRVECMAT_OPER(Vec,+)
NRVECMAT_OPER(Vec,-)
NRVECMAT_OPER(Vec,*)
NRVECMAT_OPER(Vec,/)

/***************************************************************************//**
 * generate operators involving vector and vector
 ******************************************************************************/
NRVECMAT_OPER2(Vec,+)
NRVECMAT_OPER2(Vec,-)

/***************************************************************************//**
 * destructor for general vector decreases the reference count and performs
 * deallocation if neccessary
 ******************************************************************************/
template <typename T>
NRVec<T>::~NRVec() {
	if(!count) return;
	if(--(*count) <= 0) {
		if(v){
#ifdef CUDALA
			if(location == cpu){
#endif
				delete[] v;
#ifdef CUDALA
			}else{ gpufree(v); }
#endif
		}
                delete count;
        }
}

/***************************************************************************//**
 * make own copy of the underlying data connected with this vector
 ******************************************************************************/
template <typename T>
void NRVec<T>::copyonwrite(bool detachonly,bool deep) {
	if(!count)
		{
		if(nn) laerror("nonempty vector without reference count encountered");
		if(_LA_warn_empty_copyonwrite) std::cout <<"Warning: copyonwrite of empty vector\n";
		return;
		}
	if(*count == 1 && !LA_traits<T>::is_plaindata() && !detachonly && deep) //type-nested copyonwrite
		{
#ifdef CUDALA
                if(location != cpu) laerror("nested types not supported on gpu memory");
#endif
		for(int i=0; i<nn; ++i) LA_traits<T>::copyonwrite(v[i]);
		}
	if(*count > 1) {
		(*count)--;
		 count = new int;
		*count = 1;
		T *newv;
#ifdef CUDALA
		if(location == cpu){
#endif
			newv = new T[nn];
			if(!detachonly) 
				{
				if(LA_traits<T>::is_plaindata()) memcpy(newv, v, nn*sizeof(T));
				else
					{
					for(int i=0; i<nn; ++i) {newv[i]=v[i]; LA_traits<T>::copyonwrite(newv[i]);}
					}
				}
#ifdef CUDALA
		}else{
			if(!LA_traits<T>::is_plaindata()) laerror("nested types not supported on gpu memory");
			newv = (T *) gpualloc(nn*sizeof(T));
			if(sizeof(T)%sizeof(float) != 0) laerror("memory alignment problem in NRVec<T>::copyonwrite()");
			if(!detachonly) cublasScopy(nn*sizeof(T)/sizeof(float), (const float *) v, 1, (float *)newv, 1);
			TEST_CUBLAS("cublasScopy");//"NRVec<T>::copyonwrite()"
		}
#endif
		v = newv;
	}
}


/***************************************************************************//**
 * assigns general vector \f$\vec{y}\f$ to this vector \f$\vec{x}\f$
 * \li checks for self-assignment
 * \li decreases the reference count and performs deallocation if neccesary
 * \li links the internal data structures with corresponding properties of vector \f$\vec{y}\f$
 * \li updates the reference count properly
 ******************************************************************************/
template <typename T>
NRVec<T> & NRVec<T>::operator=(const NRVec<T> &rhs) {
	//check for self-assignment
	if(this != &rhs){
		if(count){
			if(--(*count) == 0){
#ifdef CUDALA
				if(location == cpu){
#endif
					delete[] v;
#ifdef CUDALA
				}else{
					gpufree(v);
				}
#endif
				delete count;
			}
		}
		v = rhs.v;
		nn = rhs.nn;
		count = rhs.count;
#ifdef CUDALA
		location = rhs.location;
#endif
		if(count){ (*count)++; }
	}
	return *this;
}


/***************************************************************************//**
 * resizes this vector
 * @param[in] n requested size
 ******************************************************************************/
template <typename T>
void NRVec<T>::resize(const int n, const bool preserve, const bool preserve_at_end) 
{
#ifdef DEBUG
	if(n < 0) laerror("illegal dimension in NRVec::resize");
#endif
T *vold=0;
int nnold=0;
bool preserved=false;
bool do_delete=false;

	if(count) //we are allocated
		{
		if(n == 0) //just deallocate
			{
			if(--(*count) <= 0)
				{
				if(v)
					{
#ifdef CUDALA
					if(location == cpu){
#endif
						delete[] (v);
#ifdef CUDALA
					}else{
						gpufree(v);
					}
#endif
					}
				delete count;
				}
			count = 0;
			nn = 0;
			v = 0;
			return;
			}
		if(*count > 1) //detach from shared data
			{
			(*count)--;
			count = 0;
			vold=v;
			v = 0;
			nnold=nn;
			nn = 0;
			preserved=true;
			}
		}
	if(!count) //we were not allocated or we just detached
		{
		count = new int;
		*count = 1;
		nn = n;
#ifdef CUDALA
		if(location == cpu)
#endif
			v = new T[nn];
#ifdef CUDALA
		else
			v = (T*) gpualloc(nn*sizeof(T));
#endif
		if(preserved && preserve) goto do_preserve;
		return;
		}
// *count == 1 in this branch
if (n == nn)  return; //nothing to do
nnold=nn;
nn = n;
#ifdef CUDALA
		if(location == cpu)
			{
#endif
			if(preserve) {vold=v; preserved= do_delete=true;} else delete[] v;
			v = new T[nn];
#ifdef CUDALA
			}
		else
			{
			 if(preserve) {vold=v; d preserved= o_delete=true;} else gpufree(v);
			v = (T*) gpualloc(nn*sizeof(T));
			}
#endif

if(!preserve) return;

//copy data from old location and zero excess allocated memory
do_preserve:
if(!preserve || !preserved) laerror("assertion failed in NRVec::resize");
// omit this check since we would need to have traits for presently unknown user defined classes
// if(!LA_traits<T>::is_plaindata()) laerror("do not know how to preserve non-plain data");

int nnmin=nnold;
if(nn<nnmin) nnmin=nn;

#ifdef CUDALA
if(location == cpu)
        {
#endif       
	if(preserve_at_end)
		{
		for(int i=0; i<nnmin; ++i) v[nn-i-1]=vold[nnold-i-1]; //preserve even non-plain data classes
                if(nn>nnold) memset(v,0,(nn-nnold)*sizeof(T)); //just zero the new memory
		}
	else
		{
		for(int i=0; i<nnmin; ++i) v[i]=vold[i]; //preserve even non-plain data classes
		if(nn>nnold) memset(v+nnold,0,(nn-nnold)*sizeof(T)); //just zero the new memory
		}
	if(do_delete) delete[] vold;
#ifdef CUDALA
        }
else
        {
	//!!!works only with plain data
	if(preserve_at_end)
                {
		cublasSetVector(nnmin, sizeof(T), vold+nnold-nnmin, 1, v+nn-nnmin, 1);
                TEST_CUBLAS("cublasSetVector");
                T a(0);
                if(nn>nnold) smart_gpu_set(nn-nnold, a, v);
		}
	else
		{
		cublasSetVector(nnmin, sizeof(T), vold, 1, v, 1);
		TEST_CUBLAS("cublasSetVector"); 
		T a(0);
		if(nn>nnold) smart_gpu_set(nn-nnold, a, v+nnold);
		}
	if(do_delete) gpufree(vold);
        }
#endif

return;
}


/***************************************************************************//**
 * perfrom deep copy
 * @param[in] rhs vector being copied
 * @see NRVec<T>::copyonwrite()
 ******************************************************************************/
template <typename T>
NRVec<T> & NRVec<T>::operator|=(const NRVec<T> &rhs) {
#ifdef DEBUG
	if(!rhs.v) laerror("unallocated vector");
#endif
        if(this == &rhs) return *this;
        *this = rhs;
        this->copyonwrite();
        return *this;
}


/***************************************************************************//**
 * complexify given vector of general type <code>T</code>, i.e. convert its
 * elements to type <code>complex<T></code>
 * @param[in] rhs vector being complexified
 * @see NRVec<T>::copyonwrite()
 ******************************************************************************/
template<typename T>
NRVec<std::complex<T> > complexify(const NRVec<T> &rhs) {
	NOT_GPU(rhs);

	NRVec<std::complex<T> > r(rhs.size(), rhs.getlocation());
	for(register int i=0; i<rhs.size(); ++i)  r[i] = rhs[i];
	return r;
}
template<> NRVec<std::complex<double> > complexify<double>(const NRVec<double> &rhs);

/***************************************************************************//**
 * routine for moving vector data between CPU and GPU memory
 * @param[in] dest required location
 * @see NRVec<T>::location, NRVec<T>::getlocation()
 ******************************************************************************/
#ifdef CUDALA
template<typename T>
void NRVec<T>::moveto(const GPUID dest) {
	if(location == dest) return;

	CPU_GPU(location, dest);
	location = dest;

	if(v && !count) laerror("internal");
	if (!count) return;

	if(v && *count == 0) laerror("internal");
	if(!v) return;

	T *vold = v;

	if(dest == cpu){ // moving from GPU to CPU
		v = new T[nn];
		gpuget(nn,sizeof(T),vold,v);
		if(*count == 1) gpufree(vold);
		else {--(*count); count = new int(1);}

	}else{ // moving from CPU to GPU
		v = (T *) gpualloc(nn*sizeof(T));
		gpuput(nn,sizeof(T),vold,v);
		if(*count == 1) delete[] vold;
		else {--(*count); count = new int(1);}
	}
}
#endif



/***************************************************************************//**
 * adds a real scalar value \f$\alpha\f$ to all elements of this real vector \f$\vec{x}\f$
 * \f[\vec{x}_i\leftarrow\vec{x}_i+\alpha\f]
 * @param[in] a real scalar value \f$\alpha\f$ being added
 * @return reference to the modified vector
 ******************************************************************************/
template<>
inline NRVec<double>& NRVec<double>::operator+=(const double &a) {
	copyonwrite();
#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_daxpy(nn, 1.0, &a, 0, v, 1);
#ifdef CUDALA
	}else{
		double *d = gpuputdouble(a);
		cublasDaxpy(nn, 1.0, d, 0, v, 1);
		TEST_CUBLAS("cublasDaxpy");
		gpufree(d);
	}
#endif
	return *this;
}

/***************************************************************************//**
 * adds a complex scalar value \f$\alpha\f$ to all elements of this complex vector \f$\vec{x}\f$
 * \f[\vec{x}_i\leftarrow\vec{x}_i+\alpha\f]
 * @param[in] a complex scalar value \f$\alpha\f$ being added
 * @return reference to the modified vector
 ******************************************************************************/
template<>
inline NRVec<std::complex<double> >& NRVec<std::complex<double> >::operator+=(const std::complex<double> &a) {
	copyonwrite();
#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_zaxpy(nn, &CONE, &a, 0, v, 1);
#ifdef CUDALA
	}else{
		std::complex<double> *d = gpuputcomplex(a);
                cublasZaxpy(nn, CUONE, (cuDoubleComplex *)d, 0, (cuDoubleComplex *)v, 1);
		TEST_CUBLAS("cublasZaxpy");
                gpufree(d);
        }
#endif
	return *this;
}

/***************************************************************************//**
 * subtracts a real scalar value \f$\alpha\f$ from all elements of this real vector \f$\vec{x}\f$
 * \f[\vec{x}_i\leftarrow\vec{x}_i-\alpha\f]
 * @param[in] a real scalar value \f$\alpha\f$ being subtracted
 * @return reference to the modified vector
 ******************************************************************************/
template<>
inline NRVec<double>& NRVec<double>::operator-=(const double &a) {
	copyonwrite();
#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_daxpy(nn, -1.0, &a, 0, v, 1);
#ifdef CUDALA
	}else{
		double *d = gpuputdouble(a);
		cublasDaxpy(nn, -1.0, d, 0, v, 1);
		TEST_CUBLAS("cublasDaxpy");
		gpufree(d);
	}
#endif
	return *this;
}

/***************************************************************************//**
 * subtracts a complex scalar value \f$\alpha\f$ from all elements of this complex vector \f$\vec{x}\f$
 * \f[\vec{x}_i\leftarrow\vec{x}_i-\alpha\f]
 * @param[in] a complex scalar value \f$\alpha\f$ being subtracted
 * @return reference to the modified vector
 ******************************************************************************/
template<>
inline NRVec<std::complex<double> >& NRVec<std::complex<double> >::operator-=(const std::complex<double> &a) {
	copyonwrite();
#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_zaxpy(nn, &CMONE, &a, 0, v, 1);
#ifdef CUDALA
	}else{
		std::complex<double> *d = gpuputcomplex(a);
		cublasZaxpy(nn, CUMONE, (cuDoubleComplex *)d, 0, (cuDoubleComplex *)v, 1);
		TEST_CUBLAS("cublasZaxpy");
		gpufree(d);
	}
#endif
	return *this;
}

/***************************************************************************//**
 * adds a real vector \f$\vec{y}\f$ to this real vector \f$\vec{x}\f$
 * \f[\vec{x}\leftarrow\vec{x}+\vec{y}\f]
 * @param[in] rhs real vector \f$\vec{y}\f$
 * @return reference to the modified vector
 ******************************************************************************/
template<>
inline NRVec<double>& NRVec<double>::operator+=(const NRVec<double> &rhs) {
#ifdef DEBUG
        if (nn != rhs.nn) laerror("incompatible dimensions");
#endif
	SAME_LOC(*this, rhs);
        copyonwrite();
#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_daxpy(nn, 1.0, rhs.v, 1, v, 1);
#ifdef CUDALA
	}else{
		cublasDaxpy(nn, 1.0, rhs.v, 1, v, 1);
		TEST_CUBLAS("cubasDaxpy");
	}
#endif
        return *this;
}

/***************************************************************************//**
 * adds a complex vector \f$\vec{y}\f$ to this complex vector \f$\vec{x}\f$
 * \f[\vec{x}\leftarrow\vec{x}+\vec{y}\f]
 * @param[in] rhs complex vector \f$\vec{y}\f$
 * @return reference to the modified vector
 ******************************************************************************/
template<>
inline NRVec<std::complex<double> >& NRVec<std::complex<double> >::operator+=(const NRVec<std::complex<double> > &rhs) {
#ifdef DEBUG
        if (nn != rhs.nn) laerror("incompatible dimensions");
#endif
	SAME_LOC(*this, rhs);
        copyonwrite();
#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_zaxpy(nn, &CONE, rhs.v, 1, v, 1);
#ifdef CUDALA
	}else{
		cublasZaxpy(nn, CUONE, (cuDoubleComplex*)rhs.v, 1, (cuDoubleComplex*)v, 1);
		TEST_CUBLAS("cublasZaxpy");
	}
#endif
	return *this;
}

/***************************************************************************//**
 * subtracts a real vector \f$\vec{y}\f$ from this real vector \f$\vec{x}\f$
 * \f[\vec{x}\leftarrow\vec{x}-\vec{y}\f]
 * @param[in] rhs real vector \f$\vec{y}\f$
 * @return reference to the modified vector
 ******************************************************************************/
template<>
inline NRVec<double> & NRVec<double>::operator-=(const NRVec<double> &rhs) {
#ifdef DEBUG
        if (nn != rhs.nn) laerror("incompatible dimensions");
#endif
	SAME_LOC(*this,rhs);
	copyonwrite();
#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_daxpy(nn, -1.0, rhs.v, 1, v, 1);
#ifdef CUDALA
	}else{
		cublasDaxpy(nn, -1.0, rhs.v, 1, v, 1);
		TEST_CUBLAS("cubasDaxpy");
	}
#endif
        return *this;
}

/***************************************************************************//**
 * subtracts a complex vector \f$\vec{y}\f$ from this complex vector \f$\vec{x}\f$
 * \f[\vec{x}\leftarrow\vec{x}-\vec{y}\f]
 * @param[in] rhs double-precision complex vector \f$\vec{y}\f$
 * @return reference to the modified vector
 ******************************************************************************/
template<>
inline NRVec<std::complex<double> >& NRVec<std::complex<double> >::operator-=(const NRVec<std::complex<double> > &rhs) {
#ifdef DEBUG
        if (nn != rhs.nn) laerror("incompatible dimensions");
#endif
	SAME_LOC(*this, rhs);
	copyonwrite();
#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_zaxpy(nn, &CMONE, rhs.v, 1, v, 1);
#ifdef CUDALA
	}else{
		cublasZaxpy(nn, CUMONE, (cuDoubleComplex*)rhs.v, 1, (cuDoubleComplex*)v, 1);
		TEST_CUBLAS("cublasZaxpy");
	}
#endif
        return *this;
}

/***************************************************************************//**
 * multiplies this real vector \f$\vec{x}\f$ by a real scalar value \f$\alpha\f$
 * \f[\vec{x}_i\leftarrow\alpha\vec{x}_i\f]
 * @param[in] a real scalar value \f$\alpha\f$
 * @return reference to the modified vector
 ******************************************************************************/
template<>
inline NRVec<double>& NRVec<double>::operator*=(const double &a) {
        copyonwrite();
#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_dscal(nn, a, v, 1);
#ifdef CUDALA
	}else{
		cublasDscal(nn, a, v, 1);
		TEST_CUBLAS("cublasDscal");
	}
#endif
        return *this;
}

template<>
inline NRVec<double>& NRVec<double>::operator/=(const double &a) {return *this *= (1./a);}

/***************************************************************************//**
 * multiplies this complex vector \f$\vec{x}\f$ by a complex scalar value \f$\alpha\f$
 * \f[\vec{x}_i\leftarrow\alpha\vec{x}_i\f]
 * @param[in] a complex scalar value \f$\alpha\f$
 * @return reference to the modified vector
 ******************************************************************************/
template<>
inline NRVec<std::complex<double> >& NRVec<std::complex<double> >::operator*=(const std::complex<double> &a) {
        copyonwrite();
#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_zscal(nn, &a, v, 1);
#ifdef CUDALA
	}else{
		const cuDoubleComplex alpha = make_cuDoubleComplex(a.real(), a.imag());
		cublasZscal(nn, alpha, (cuDoubleComplex*)v, 1);
		TEST_CUBLAS("cublasZscal");
	}
#endif
        return *this;
}

template<>
inline NRVec<std::complex<double> >& NRVec<std::complex<double> >::operator/=(const std::complex<double> &a) {return *this *= (1./a);}

/***************************************************************************//**
 * computes the inner product of this real vector \f$\vec{x}\f$ with given real vector \f$\vec{y]\f$
 * @param[in] rhs real vector \f$\vec{y}\f$
 * @return \f$\sum_{i=1}^N\vec{x}_i\cdot\vec{y}_i\f$
 ******************************************************************************/
template<>
inline const double NRVec<double>::operator*(const NRVec<double> &rhs) const {
	double ret(0.0);
#ifdef DEBUG
        if(nn != rhs.nn) laerror("incompatible dimensions");
#endif
	SAME_LOC(*this, rhs);
#ifdef CUDALA
	if(location == cpu){
#endif
		ret = cblas_ddot(nn, v, 1, rhs.v, 1);
#ifdef CUDALA
	}else{
		ret = cublasDdot(nn, v, 1, rhs.v, 1);
		TEST_CUBLAS("cublasDdot");
	}
#endif
	return ret;
}

/***************************************************************************//**
 * computes the inner product of this complex vector \f$\vec{x}\f$ with given complex vector \f$\vec{y}\f$
 * taking conjugation of vector \f$\vec{x}\f$ into account
 * @param[in] rhs complex vector \f$\vec{y}\f$
 * @return \f$\sum_{i=1}^N\overbar{\vec{x}_i}\cdot\vec{y}_i\f$
 ******************************************************************************/
template<>
inline const std::complex<double> NRVec<std::complex<double> >::operator*(const NRVec< std::complex<double> > &rhs) const {
#ifdef DEBUG
        if(nn != rhs.nn) laerror("incompatible dimensions");
#endif
        std::complex<double> dot;
	SAME_LOC(*this, rhs);
#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_zdotc_sub(nn, v, 1, rhs.v, 1, &dot);
#ifdef CUDALA
	}else{
		const cuDoubleComplex val = cublasZdotc(nn, (cuDoubleComplex*)v, 1, (cuDoubleComplex*)rhs.v, 1);
		TEST_CUBLAS("cublasZdotc");
		dot = std::complex<double>(cuCreal(val), cuCimag(val)); 
	}
#endif
        return dot;
}

/***************************************************************************//**
 * computes the inner product of this real vector \f$\vec{x}\f$ with given real data
 * @param[in] y pointer to the double-precision real array (sufficient length assumed)
 * @param[in] stride specifies the stride regarding the data pointe to by <tt>y</tt>
 * @return \f$\sum_{i=1}^N\vec{x}_{i}\cdot y_{\mathrm{stride}\cdot(i-1) + 1}\f$
 ******************************************************************************/
template<>
inline const double NRVec<double>::dot(const double *y, const int stride) const {
	NOT_GPU(*this);
	return cblas_ddot(nn, y, stride, v, 1);
}

/***************************************************************************//**
 * computes the inner product of this complex vector \f$\vec{x}\f$ with given complex data
 * @param[in] y pointer to the double-precision complex array (sufficient length assumed)
 * @param[in] stride specifies the stride regarding the data pointe to by <tt>y</tt>
 * @return \f$\sum_{i=1}^N\vec{x}_{i}\cdot \overbar{y_{\mathrm{stride}\cdot(i-1) + 1}}\f$
 ******************************************************************************/
template<>
inline const std::complex<double> NRVec<std::complex<double> >::dot(const std::complex<double> *y, const int stride) const {
        std::complex<double> dot;
	NOT_GPU(*this);
        cblas_zdotc_sub(nn, y, stride, v, 1, &dot);
        return dot;
}

/***************************************************************************//**
 * computes the sum of the absolute values of the elements of this real vector \f$\vec{x}\f$
 * @return \f$\sum_{i=1}^N\left|\vec{x}_i\right|\f$
 ******************************************************************************/
template<>
inline const double NRVec<double>::asum() const {
	double ret(0.0);
#ifdef CUDALA
	if(location == cpu){
#endif
		ret = cblas_dasum(nn, v, 1);
#ifdef CUDALA
	}else{
		ret = cublasDasum(nn, v, 1);
		TEST_CUBLAS("cublasDasum");
	}
#endif
	return ret;
}


/***************************************************************************//**
 * for this complex vector \f$\vec{x}\f$ compute the expression
 * \f[\sum_{i=1}^N\left|\Re{}\vec{x}_i\right| + \left|\Im{}\vec{x}_i\right|\f]
 * @return the value of this sum
 ******************************************************************************/
template<>
inline const double NRVec<std::complex<double> >::asum() const {
	double ret(0.0);
#ifdef CUDALA
	if(location == cpu){
#endif
		ret = cblas_dzasum(nn, v, 1);
#ifdef CUDALA
	}else{
		ret = cublasDzasum(nn, (cuDoubleComplex*)v, 1);
		TEST_CUBLAS("cublasDzasum");
	}
#endif
	return ret;
}

/***************************************************************************//**
 * for this real vector \f$\vec{x}\f$ (of \f$N\f$ elements) determine the Frobenius norm
 * @return \f$\sum_{i=1}^N\left|\vec{x}_i\right|^2\f$
 ******************************************************************************/
template<>
inline const double  NRVec<double>::norm(int length, int offset, int stride) const {
	double ret(0.);
#ifdef CUDALA
	if(location == cpu){
#endif
		ret = cblas_dnrm2((length>=0?length:nn), v+offset, stride);
#ifdef CUDALA
	}else{
		ret = cublasDnrm2((length>=0?length:nn), v+offset, stride);
		TEST_CUBLAS("cublasDnrm2");
	}
#endif
	return ret;
}

/***************************************************************************//**
 * for this complex vector \f$\vec{x}\f$ (of \f$N\f$ elements) determine the Frobenius norm
 * @return \f$\sum_{i=1}^N\left|\vec{x}_i\right|^2\f$
 ******************************************************************************/
template<>
inline const double NRVec< std::complex<double> >::norm(int length, int offset, int stride) const {
	double ret(0.);
#ifdef CUDALA
	if(location == cpu){
#endif
		ret = cblas_dznrm2((length>=0?length:nn), v+offset, stride);
#ifdef CUDALA
	}else{
		ret = cublasDznrm2((length>=0?length:nn), ((cuDoubleComplex*)v)+offset, stride);
		TEST_CUBLAS("cublasDzrm2");
	}
#endif
	return ret;
}

/***************************************************************************//**
 * for this real vector \f$\vec{x}\f$ determine the element with largest absolute value
 * @return \f$\vec{x}_i\f$ where \f$\left|\vec{x]_i\right|=\mathrm{max}_{j}\left|\vec{x}_{j}\right|\f$
 ******************************************************************************/
template<>
inline const double NRVec<double>::amax() const {
	double ret(0.0);
#ifdef CUDALA
	if(location == cpu){
#endif
		ret = v[cblas_idamax(nn, v, 1)];
#ifdef CUDALA
	}else{
		const int pozice = cublasIdamax(nn, v, 1) ;
		TEST_CUBLAS("cublasIdamax");

		gpuget(1, sizeof(double), v + pozice, &ret);
	}
#endif
	return ret;
}

/***************************************************************************//**
 * for this real vector \f$\vec{x}\f$ determine the element with smallest absolute value
 * @return \f$\vec{x}_i\f$ where \f$\left|\vec{x]_i\right|=\mathrm{min}_{j}\left|\vec{x}_{j}\right|\f$
 ******************************************************************************/
template<>
inline const double NRVec<double>::amin() const {
	double ret(std::numeric_limits<double>::max());
#ifdef CUDALA
	if(location == cpu){
#endif
		//BLAS routine idamin seems no to be supported
		double val(0.0);
		int index(-1);
		for(register int i = 0; i < nn; i++){
			val = std::abs(v[i]);
			if(val < ret){ index = i; ret = val; }
		}
		ret = v[index];
#ifdef CUDALA
	}else{
		const int pozice = cublasIdamin(nn, v, 1);
		TEST_CUBLAS("cublasIdamin");
		gpuget(1, sizeof(double), v + pozice, &ret);
	}
#endif
	return ret;
}

/***************************************************************************//**
 * for a given complex vector \f$\vec{v}\f$, determine the smallest index of the maximum 
 * magnitude element, i.e. maximal element in the 1-norm
 * @return \f$\vec{v}_{j}\f$ which maximizes \f$\left\{\left|\Re{}\vec{v}_{i}\right|+\left|\Im{}\vec{v}_{i}\right|\right}\f$
 ******************************************************************************/
template<>
inline const std::complex<double> NRVec<std::complex<double> >::amax() const {
	std::complex<double> ret(0., 0.);
#ifdef CUDALA
	if(location == cpu){
#endif
		ret = v[cblas_izamax(nn, v, 1) ];
#ifdef CUDALA
	}else{
		const int pozice = cublasIzamax(nn, (cuDoubleComplex*)v, 1);
		TEST_CUBLAS("cublasIzamax");
		gpuget(1, sizeof(std::complex<double>), v + pozice, &ret);
	}
#endif
	return ret;
}

/***************************************************************************//**
 * for a given complex vector \f$\vec{v}\f$, determine the smallest index of the minimum 
 * magnitude element, i.e. minimal element in the 1-norm
 * @return \f$\vec{v}_{j}\f$ which minimizes \f$\left\{\left|\Re{}\vec{v}_{i}\right|+\left|\Im{}\vec{v}_{i}\right|\right}\f$
 ******************************************************************************/
template<>
inline const std::complex<double> NRVec<std::complex<double> >::amin() const {
	std::complex<double> ret(0., 0.);
#ifdef CUDALA
	if(location == cpu){
#endif
		// izamin seems not to be supported
		int index(0);
		double val(0.0), min_val(std::numeric_limits<double>::max());
		std::complex<double> z_val(0.0, 0.0);

		for(register int i=0; i < nn; i++){
			z_val = v[i];
			val = std::abs(z_val.real()) + std::abs(z_val.imag());
			if(val < min_val){ index = i; min_val = val; }	
		}
		ret = v[index];
#ifdef CUDALA
	}else{
		const int pozice = cublasIzamin(nn, (cuDoubleComplex*)v, 1);
		TEST_CUBLAS("cublasIzamin");
		gpuget(1, sizeof(std::complex<double>), v + pozice, &ret);
	}
#endif
	return ret;
}


/***************************************************************************//**
 * routine for raw output
 * @param[in] fd file descriptor for output
 * @param[in] dim number of elements intended for output
 * @param[in] transp reserved
 * @see NRMat<T>::put()
 ******************************************************************************/
template <typename T>
void NRVec<T>::put(int fd, bool dim, bool transp) const {
#ifdef CUDALA
	if(location != cpu){
		NRVec<T> tmp = *this;
		tmp.moveto(cpu);
		tmp.put(fd,dim,transp);
		return;
	}
#endif
	errno = 0;
	int pad(1); //align at least 8-byte
	if(dim){
		if(sizeof(int) != write(fd,&nn,sizeof(int))) laerror("write failed");
		if(sizeof(int) != write(fd,&pad,sizeof(int))) laerror("write failed");
	}
	LA_traits<T>::multiput(nn,fd,v,dim);
}

/***************************************************************************//**
 * routine for raw input 
 * @param[in] fd file descriptor for input
 * @param[in] dim number of elements intended for input, for dim=0 perform copyonwrite
 * @param[in] transp reserved
 * @see NRMat<T>::get(), copyonwrite()
 ******************************************************************************/
template <typename T>
void NRVec<T>::get(int fd, bool dim, bool transp) {
#ifdef CUDALA
	if(location != cpu){
		NRVec<T> tmp;
		tmp.moveto(cpu);
		tmp.get(fd,dim,transp);
		tmp.moveto(location);
		*this = tmp;
		return;
	}
#endif
	int nn0[2]; //align at least 8-byte
	errno = 0;
	if(dim){
		if(2*sizeof(int) != read(fd,&nn0,2*sizeof(int))) laerror("read failed");
		resize(nn0[0]);
	}else{
		copyonwrite();
	}
	LA_traits<T>::multiget(nn,fd,v,dim);
}


//constructor from a list
template<typename T>
NRVec<T>::NRVec(const std::list<T> l) : NRVec<T>(l.size())
{
int ii=0;
for(typename std::list<T>::const_iterator i=l.begin(); i!=l.end(); ++i) (*this)[ii++] = *i;
}


//general simplification template for a NRVec of a class consisting from a coefficient and an element
//the class T must have traits for sorting, normtype, elementtype, coefficienttype, coefficient, abscoefficient, and operator== which ignores the coefficient and uses just the element
//it is not a member function to avoid the need of extra traits when this is not needed

template<typename T>
void NRVec_simplify(NRVec<T> &x, const typename LA_traits<T>::normtype thr=0, bool alwayskeepfirst=false)
{
if(x.size()==0) return;

x.copyonwrite();

//first sort to identify identical terms
x.sort();

//the following operations conserve the sorting, so no need to reset the issorted flag

int newsize=1;
//add factors of identical elements
for(int next=1; next<x.size(); ++next)
	{
	if(x[next] == x[newsize-1])
		{
		LA_traits<T>::coefficient(x[newsize-1]) += LA_traits<T>::coefficient(x[next]);
		}
	else
		{
		if(next!=newsize) x[newsize] = x[next];
		++newsize;
		}
	}

//now skip terms with zero coefficients
int newsize2=0;
for(int next=0; next<newsize; ++next)
	{
	if(next==0 && alwayskeepfirst || !(LA_traits<T>::coefficient(x[next])==(typename LA_traits<T>::elementtype)0 || LA_traits<T>::abscoefficient(x[next]) <thr))
		{
		if(next!=newsize2) x[newsize2] = x[next];
		++newsize2;
		}
	}

x.resize(newsize2,true);
}

//convert whole vector between types
template <typename T, typename S>
void NRVec_convert(NRVec<T> &a, const NRVec<S> &b)
{
a.resize(b.size());
for(int i=0; i<b.size(); ++i)
                a[i] = (T) b[i];
}

//these must be here rather than in vec.cc to be used on unknown new types
/***************************************************************************//**
 * extract block subvector
 * @param[in] from starting position
 * @param[in] to final position
 * @return extracted block subvector
 ******************************************************************************/
template <typename T>
const NRVec<T> NRVec<T>::subvector(const int from, const int to) const 
{
#ifdef DEBUG
	if(from<0 || from>=nn|| to<0 || to>=nn || from>to){
		laerror("invalid subvector specification");
	}
#endif
//trivial case of whole vector for efficiency
	if(from==0 && to == nn-1) return *this;

	const int n = to - from + 1;
	NRVec<T> r(n, getlocation());
	if(!LA_traits<T>::is_plaindata()) laerror("only implemented for plain data");

#ifdef CUDALA
	if(location == cpu){
#endif
		memcpy(r.v, v+from, n*sizeof(T));
#ifdef CUDALA
	}else{
		if(sizeof(T)%sizeof(float) != 0) laerror("cpu memcpy alignment problem");
			cublasScopy(n*sizeof(T)/sizeof(float), (const float *)(v+from), 1, (float*)r.v, 1);
			TEST_CUBLAS("cublasScopy");
	}
#endif	
	return r;
}


template <typename T>
const NRVec<T> NRVec<T>::subvector(const NRVec<int> &selection) const
{
        NOT_GPU(*this);
        const int n = selection.size();
        NRVec<T> r(n);

        for(int i=0; i<n; ++i)
                {
                int ii=selection[i];
                if(ii<0||ii>=nn) laerror("bad row index in subvector");
                r[i] = (*this)[ii];
                }
return r;

}

/***************************************************************************//**
 * places given vector as subvector at given position
 * @param[in] from coordinate 
 * @param[in] rhs input vector
 ******************************************************************************/
template <typename T>
void NRVec<T>::storesubvector(const int from, const NRVec &rhs) 
{
	const int to = from + rhs.size() - 1;
#ifdef DEBUG
	if(from<0 || from>=nn || to>=nn) laerror("bad indices in storesubvector");
#endif
	SAME_LOC(*this, rhs);
	if(!LA_traits<T>::is_plaindata()) laerror("only implemented for plain data");

	#ifdef CUDALA
		if(location == cpu){
	#endif
			memcpy(v+from, rhs.v, rhs.size()*sizeof(T));

	#ifdef CUDALA
		}else{
			if(sizeof(T)%sizeof(float) != 0) laerror("cpu memcpy alignment problem");
			cublasScopy(rhs.size()*sizeof(T)/sizeof(float), (const float *) (rhs.v), 1, (float *)(v + from), 1);
		}
	#endif
}


template <typename T>
void NRVec<T>::storesubvector(const NRVec<int> &selection, const NRVec &rhs)
{
        NOT_GPU(*this);
        const int n = selection.size();
	if(n!=rhs.size()) laerror("size mismatch in storesubvector");

        for(int i=0; i<n; ++i)
                {
                int ii=selection[i];
                if(ii<0||ii>=nn) laerror("bad  index in storesubvector");
                (*this)[ii] = rhs[i];
                }
}



}//namespace

#endif /* _LA_VEC_H_ */
