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

namespace LA {

/***************************************************************************//**
 * forward declarations
 ******************************************************************************/
template <typename T> void lawritemat(FILE *file, const T *a, int r, int c,
		const char *form0, int nodim, int modulo, int issym);

/***************************************************************************//**
 * static constants used in several cblas-routines
 ******************************************************************************/
const static complex<double> CONE = 1.0, CMONE = -1.0, CZERO = 0.0;
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
	template <typename U> friend NRVec<complex<U> > complexify(const NRVec<U>&);

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
	inline NRVec(const T &a, const int n);
	
	//! inlined constructor creating vector of given size filled with data located at given memory location 
        inline NRVec(const T *a, const int n);
	
	//! inlined constructor creating vector of given size filled with data located at given memory location 
	inline NRVec(T *a, const int n, bool skeleton);

	//! inlined copy constructor 
	inline NRVec(const NRVec &rhs);

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
	void copyonwrite(bool detachonly=false);

	//! purge this vector 
	void clear() { copyonwrite(true); LA_traits<T>::clear(v, nn); };

	//! assignment operator assigns given vector
	NRVec& operator=(const NRVec &rhs);

	//! assigment operator assigns given scalar to each element of this vector 
	NRVec& operator=(const T &a);
	
	//! fills in this vector with pseudo-random numbers generated using uniform distribution  
        void randomize(const typename LA_traits<T>::normtype &x);

	//! perform deep-copy of given vector 
	NRVec& operator|=(const NRVec &rhs);

	//! relational operators 
	const bool operator!=(const NRVec &rhs) const {if(nn!=rhs.nn) return 1; return LA_traits<T>::gencmp(v,rhs.v,nn);}
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

	inline const NRVec operator+(const T &a) const;
	inline const NRVec operator-(const T &a) const;
	inline const NRVec operator*(const T &a) const;


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

	//! compute the sum of the vector elements 
	inline const T sum() const {
		T sum(0);
		for(register int i=0; i<nn; i++){ sum += v[i];}
		return sum;
	};

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

	//! resize the current vector
	void resize(const int n);

	//!deallocate the current vector
	void dealloc(void) {resize(0);}

	//! determine the norm of this vector 
	inline const typename LA_traits<T>::normtype norm() const;

	//! normalize this vector and optionally save the norm 
	NRVec& normalize(typename LA_traits<T>::normtype* norm = 0);

	//! get normalized copy of this vector 
	inline const NRVec unitvector() const;

	//! determine the maximal element (in the absolute value) of this vector 
	inline const T amax() const;
	//! determine the minimal element (in the absolute value) of this vector 
	inline const T amin() const;

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
	int sort(int direction = 0, int from = 0, int to = -1, int *perm = NULL);

	//! apply given function to each element
	NRVec& call_on_me(T (*_F)(const T &) ){
		NOT_GPU(*this);

		copyonwrite();
		for(int i=0; i<nn; ++i) v[i] = _F(v[i]);
		return *this;
	};
};

}//namespace

//due to mutual includes this has to be after full class declaration
#include "mat.h"
#include "smat.h"
#include "sparsemat.h"
#include "sparsesmat.h"


namespace LA {

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
template <typename T>
inline NRVec<T>::NRVec(const T& a, const int n): nn(n), count(new int) {
	*count = 1;
#ifdef CUDALA
	location = DEFAULT_LOC;
	if(location == cpu){
#endif
		v = new T[n];
		if(a != (T)0){
			for(register int i=0; i<n; i++) v[i] = a;
		}else{
			memset(v, 0, nn*sizeof(T));
		}
#ifdef CUDALA
	}else{
		v = (T*) gpualloc(n*sizeof(T));
		smart_gpu_set(n, a, v);
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
		memcpy(v, a, n*sizeof(T));
#ifdef CUDALA
	}else{
		v = (T*) gpualloc(n*sizeof(T));
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
			memcpy(v, a, n*sizeof(T));
#ifdef CUDALA
		}else{
			v= (T*) gpualloc(n*sizeof(T));
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
void NRVec<T>::copyonwrite(bool detachonly) {
	if(!count) laerror("copyonwrite of an undefined vector");
	if(*count > 1) {
		(*count)--;
		 count = new int;
		*count = 1;
		T *newv;
#ifdef CUDALA
		if(location == cpu){
#endif
			newv = new T[nn];
			if(!detachonly) memcpy(newv, v, nn*sizeof(T));
#ifdef CUDALA
		}else{
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
void NRVec<T>::resize(const int n) {
#ifdef DEBUG
	if(n < 0) laerror("illegal dimension");
#endif
	if(count){
		if(n == 0){
			if(--(*count) <= 0){
				if(v){
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
		if(*count > 1) {
			(*count)--;
			count = 0;
			v = 0;
			nn = 0;
		}
	}
	if(!count){
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
		return;
	}
	// *count = 1 in this branch
	if (n != nn) {
		nn = n;
#ifdef CUDALA
		if(location == cpu){
#endif
		
			delete[] v;
			v = new T[nn];
#ifdef CUDALA
		}else{
		
			gpufree(v);
			v = (T*) gpualloc(nn*sizeof(T));
		}
#endif
	}
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
NRVec<complex<T> > complexify(const NRVec<T> &rhs) {
	NOT_GPU(rhs);

	NRVec<complex<T> > r(rhs.size(), rhs.getlocation());
	for(register int i=0; i<rhs.size(); ++i)  r[i] = rhs[i];
	return r;
}
template<> NRVec<complex<double> > complexify<double>(const NRVec<double> &rhs);

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
inline NRVec<complex<double> >& NRVec<complex<double> >::operator+=(const complex<double> &a) {
	copyonwrite();
#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_zaxpy(nn, &CONE, &a, 0, v, 1);
#ifdef CUDALA
	}else{
		complex<double> *d = gpuputcomplex(a);
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
inline NRVec<complex<double> >& NRVec<complex<double> >::operator-=(const complex<double> &a) {
	copyonwrite();
#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_zaxpy(nn, &CMONE, &a, 0, v, 1);
#ifdef CUDALA
	}else{
		complex<double> *d = gpuputcomplex(a);
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
inline NRVec<complex<double> >& NRVec<complex<double> >::operator+=(const NRVec<complex<double> > &rhs) {
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
inline NRVec<complex<double> >& NRVec<complex<double> >::operator-=(const NRVec<complex<double> > &rhs) {
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

/***************************************************************************//**
 * multiplies this complex vector \f$\vec{x}\f$ by a complex scalar value \f$\alpha\f$
 * \f[\vec{x}_i\leftarrow\alpha\vec{x}_i\f]
 * @param[in] a complex scalar value \f$\alpha\f$
 * @return reference to the modified vector
 ******************************************************************************/
template<>
inline NRVec<complex<double> >& NRVec<complex<double> >::operator*=(const complex<double> &a) {
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
inline const complex<double> NRVec<complex<double> >::operator*(const NRVec< complex<double> > &rhs) const {
#ifdef DEBUG
        if(nn != rhs.nn) laerror("incompatible dimensions");
#endif
        complex<double> dot;
	SAME_LOC(*this, rhs);
#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_zdotc_sub(nn, v, 1, rhs.v, 1, &dot);
#ifdef CUDALA
	}else{
		const cuDoubleComplex val = cublasZdotc(nn, (cuDoubleComplex*)v, 1, (cuDoubleComplex*)rhs.v, 1);
		TEST_CUBLAS("cublasZdotc");
		dot = complex<double>(cuCreal(val), cuCimag(val)); 
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
inline const complex<double> NRVec<complex<double> >::dot(const complex<double> *y, const int stride) const {
        complex<double> dot;
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
inline const double NRVec<complex<double> >::asum() const {
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
inline const double  NRVec<double>::norm() const {
	double ret(0.);
#ifdef CUDALA
	if(location == cpu){
#endif
		ret = cblas_dnrm2(nn, v, 1);
#ifdef CUDALA
	}else{
		ret = cublasDnrm2(nn, v, 1);
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
inline const double NRVec< complex<double> >::norm() const {
	double ret(0.);
#ifdef CUDALA
	if(location == cpu){
#endif
		ret = cblas_dznrm2(nn, v, 1);
#ifdef CUDALA
	}else{
		ret = cublasDznrm2(nn, (cuDoubleComplex*)v, 1);
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
		ret = v[cblas_idamax(nn, v, 1) - 1];
#ifdef CUDALA
	}else{
		const int pozice = cublasIdamax(nn, v, 1) - 1;
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
		const int pozice = cublasIdamin(nn, v, 1) - 1;
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
inline const complex<double> NRVec<complex<double> >::amax() const {
	complex<double> ret(0., 0.);
#ifdef CUDALA
	if(location == cpu){
#endif
		ret = v[cblas_izamax(nn, v, 1) - 1];
#ifdef CUDALA
	}else{
		const int pozice = cublasIzamax(nn, (cuDoubleComplex*)v, 1) - 1;
		TEST_CUBLAS("cublasIzamax");
		gpuget(1, sizeof(complex<double>), v + pozice, &ret);
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
inline const complex<double> NRVec<complex<double> >::amin() const {
	complex<double> ret(0., 0.);
#ifdef CUDALA
	if(location == cpu){
#endif
		// izamin seems not to be supported
		int index(0);
		double val(0.0), min_val(std::numeric_limits<double>::max());
		complex<double> z_val(0.0, 0.0);

		for(register int i=0; i < nn; i++){
			z_val = v[i];
			val = std::abs(z_val.real()) + std::abs(z_val.imag());
			if(val < min_val){ index = i; min_val = val; }	
		}
		ret = v[index];
#ifdef CUDALA
	}else{
		const int pozice = cublasIzamin(nn, (cuDoubleComplex*)v, 1) - 1;
		TEST_CUBLAS("cublasIzamin");
		gpuget(1, sizeof(complex<double>), v + pozice, &ret);
	}
#endif
	return ret;
}
}//namespace

#endif /* _LA_VEC_H_ */
