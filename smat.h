//------------------------------------------------------------------------------
/* vim: set ts=8 sw=8 sts=8 noexpandtab cindent: */
//------------------------------------------------------------------------------
/*
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
*/
#ifndef _LA_SMAT_H_
#define _LA_SMAT_H_
#include "la_traits.h"

namespace LA {
#define NN2 ((size_t)nn*(nn+1)/2)


/***************************************************************************//**
 * This class implements a general symmetric or hermitian matrix the elements
 * of which are stored in packed form. Particularly the lower triangular part
 * of a symmetric or hermitian matrix of order \f$N\f$ is interpreted as a
 * vector of length \f$N(N+1)/2\f$ in row-major storage scheme.
 ******************************************************************************/
template <class T>
class NRSMat{
protected:
	int nn;//!< number of rows/columns of this symmetric matrix
	T *v;//!< internal pointer to the underlying data structure
	int *count;//!< pointer to the reference counter
#ifdef CUDALA
       GPUID location;//!< specification of memory (GPU/CPU) location where this objects resides
#endif
public:
	friend class NRVec<T>;
	friend class NRMat<T>;
	
	~NRSMat();
	
	//! default constructor of null-matrix
	inline NRSMat() : nn(0),v(0),count(0) {
	#ifdef CUDALA
		location = DEFAULT_LOC;
	#endif
	};

	//! default constructor
	inline explicit NRSMat(const int n, const GPUID loc = undefined);
	
	//! constructor initializing the matrix being created by given scalar value
	inline NRSMat(const T &a, const int n);

	//! constructor initializing the matrix being created by data located at given memory position
	inline NRSMat(const T *a, const int n);

	//! copy constructor
	inline NRSMat(const NRSMat &rhs);

	//! constructor converting real matrix to its complex counterpart
	NRSMat(const typename LA_traits_complex<T>::NRSMat_Noncomplex_type &rhs, bool imagpart = false);

	//! constructor creating symmetric part of a general matrix
	explicit NRSMat(const NRMat<T> &rhs);

	//! construct symmetric matrix by filling the lower triangle with data stored in a vector
	explicit NRSMat(const NRVec<T> &rhs, const int n);

	//! assignment operator performing shallow copy
	NRSMat & operator=(const NRSMat &rhs);

	//! assignment operator performing deep copy
	NRSMat & operator|=(const NRSMat &rhs);

	//! fill the matrix with pseudorandom numbers (uniform distribution)
        void randomize(const typename LA_traits<T>::normtype &x);

	//! assign scalar value to diagonal elements
	NRSMat & operator=(const T &a);


	inline int getcount() const {return count?*count:0;}

	#ifdef CUDALA
		inline GPUID getlocation() const {return location;}
	        void moveto(const GPUID dest);
	#else
		inline GPUID getlocation() const {return cpu;}
	        void moveto(const GPUID dest) {};
	#endif

	//! relational operator for testing nonequality
        const bool operator!=(const NRSMat &rhs) const {if(nn!=rhs.nn) return 1; return LA_traits<T>::gencmp(v,rhs.v,NN2);};
	//! relational operator for testing equality
        const bool operator==(const NRSMat &rhs) const {return !(*this != rhs);};

	inline NRSMat & operator*=(const T &a);
	inline NRSMat & operator+=(const T &a);
	inline NRSMat & operator-=(const T &a);
	inline NRSMat & operator+=(const NRSMat &rhs);
	inline NRSMat & operator-=(const NRSMat &rhs);
	const NRSMat operator-() const;

	inline const NRSMat operator*(const T &a) const;
	inline const NRSMat operator+(const T &a) const;
	inline const NRSMat operator-(const T &a) const;
	inline const NRSMat operator+(const NRSMat &rhs) const;
	inline const NRSMat operator-(const NRSMat &rhs) const;

	inline const NRMat<T> operator+(const NRMat<T> &rhs) const;
	inline const NRMat<T> operator-(const NRMat<T> &rhs) const;
	const NRMat<T> operator*(const NRSMat &rhs) const;
	const NRMat<T> operator*(const NRMat<T> &rhs) const;

	const T dot(const NRSMat &rhs) const;
	const T dot(const NRVec<T> &rhs) const;

	const NRVec<T> operator*(const NRVec<T> &rhs) const {NRVec<T> result(nn,rhs.getlocation()); result.gemv((T)0,*this,'n',(T)1,rhs); return result;};
	const NRVec<complex<T> > operator*(const NRVec<complex<T> > &rhs) const {NRVec<complex<T> > result(nn,rhs.getlocation()); result.gemv((T)0,*this,'n',(T)1,rhs); return result;};

	const T* diagonalof(NRVec<T> &, const bool divide = 0, bool cache = false) const;

	void gemv(const T beta, NRVec<T> &r, const char trans, const T alpha, const NRVec<T> &x) const {r.gemv(beta,*this,trans,alpha,x);};
	void gemv(const T beta, NRVec<complex<T> > &r, const char trans, const T alpha, const NRVec<complex<T> > &x) const {r.gemv(beta,*this,trans,alpha,x);};

	inline const T& operator[](const size_t ij) const;
	inline T& operator[](const size_t ij);

	inline const T& operator()(const int i, const int j) const;
	inline T& operator()(const int i, const int j);

	inline int nrows() const;
	inline int ncols() const;
	inline size_t size() const;

	inline bool transp(const int i, const int j) const {return i>j;} //this can be used for compact storage of matrices, which are actually not symmetric, but one triangle of them is redundant
	const typename LA_traits<T>::normtype norm(const T scalar = (T)0) const;
	void axpy(const T alpha, const NRSMat &x); // this+= a*x

	inline const T amax() const;
	inline const T amin() const;

	const T trace() const;
	void get(int fd, bool dimensions = 1, bool transp = 0);
        void put(int fd, bool dimensions = 1, bool transp = 0) const;

	void copyonwrite(bool detachonly=false);

	void clear() {copyonwrite(true); LA_traits<T>::clear(v,NN2);}; //zero out
	void resize(const int n);
	void dealloc(void) {resize(0);}

	inline operator T*();
	inline operator const T*() const;
	void fprintf(FILE *f, const char *format, const int modulo) const; 
	void fscanf(FILE *f, const char *format); 
//members concerning sparse matrix
	explicit NRSMat(const SparseMat<T> &rhs);               // dense from sparse
	explicit NRSMat(const SparseSMat<T> &rhs);               // dense from sparse
	inline void simplify() {}; //just for compatibility with sparse ones
	bool issymmetric() const {return 1;}
};

}//namespace
//due to mutual includes this has to be after full class declaration
#include "vec.h"
#include "mat.h"

namespace LA {

/***************************************************************************//**
 * constructor of a symmetric matrix stored in packed form
 * @param[in]   n number of rows of the matrix being created
 * @param[in] loc location for storing the matrix
 * @see count, v, location
 ******************************************************************************/
template <typename T>
inline NRSMat<T>::NRSMat(const int n, const GPUID loc): nn(n), count(new int(1)) {
#ifdef CUDALA
        location = (loc == undefined?DEFAULT_LOC:loc);
        if(location == cpu){
#endif
		v = new T[NN2];
#ifdef CUDALA
	}else{
		v = (T*) gpualloc(NN2*sizeof(T));
	}
#endif
}

/***************************************************************************//**
 * constructor of a symmetric matrix stored in packed form (default location in used)
 * @param[in]   a set all matrix elements equal to this value
 * @param[in]   n number of rows of the matrix being created
 * @see count, v, location, NRSMat<T>::NRSMat(const int, const GPUID)
 ******************************************************************************/
template <typename T>
inline NRSMat<T>::NRSMat(const T& a, const int n) : nn(n), count(new int(1)) {
#ifdef CUDALA
        location = DEFAULT_LOC;
	if(location == cpu){
#endif
		v = new T[NN2];
		if(a != (T)0) for(register size_t i = 0; i<NN2; i++) v[i] = a;
		else memset(v, 0, NN2*sizeof(T));

#ifdef CUDALA
	}else{
		v = (T*) gpualloc(NN2*sizeof(T));
		cublasSetVector(NN2, sizeof(T), &a, 0, v, 1);
	}
#endif
}

/***************************************************************************//**
 * constructor of a symmetric matrix stored in packed form (default location in used)
 * @param[in]   a pointer to data of type T used for matrix inicialization
 * @param[in]   n number of rows of the matrix being created
 * @see count, v, location, NRSMat<T>::NRSMat(const int, const GPUID), NRSMat<T>::NRSMat(const T&, const int)
 ******************************************************************************/
template <typename T>
inline NRSMat<T>::NRSMat(const T *a, const int n) : nn(n), count(new int(1)) {
#ifdef CUDALA
        location = DEFAULT_LOC;
        if(location == cpu){
#endif
		memcpy(v, a, NN2*sizeof(T));
#ifdef CUDALA
	}else{
		v = (T*) gpualloc(NN2*sizeof(T));
		cublasSetVector(NN2, sizeof(T), a, 1, v, 1);
	}
#endif

}

/***************************************************************************//**
 * copy constructor implementing shallow copy
 * @param[in] rhs reference matrix being copied
 * @see count, v, location
 ******************************************************************************/
template <typename T>
inline NRSMat<T>::NRSMat(const NRSMat<T> &rhs) {
#ifdef CUDALA
	location = rhs.location;
#endif
	v = rhs.v;
	nn = rhs.nn;
	count = rhs.count;
	if(count) (*count)++;
}

/***************************************************************************//**
 * constructor interpreting a vector of \f$n(n+1)/2\f$ elements as a symmetric
 * matrix stored in packed form having \f$n\f$ rows
 * @param[in] rhs reference matrix being copied
 * @param[in] n count of rows of the matrix being created
 ******************************************************************************/
template <typename T>
NRSMat<T>::NRSMat(const NRVec<T> &rhs, const int n) {
#ifdef CUDALA
	location = rhs.location;
#endif
	nn = n;
#ifdef DEBUG
	if(NN2 != rhs.size()){ laerror("incompatible dimensions in NRSMat<T>::NRSMat(const NRVec<T>&, const int)"); }
#endif
	count = rhs.count;
	v = rhs.v;
	(*count)++;
}


/***************************************************************************//**
 * multiply this real symmetric matrix with real scalar value
 * @param[in] a real multiplicative factor
 * @return reference to the modified matrix
 ******************************************************************************/
template<>
inline NRSMat<double> & NRSMat<double>::operator*=(const double &a) {
	copyonwrite();
#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_dscal(NN2, a, v, 1);
#ifdef CUDALA
	}else{
		cublasDscal(NN2, a, v, 1);
		TEST_CUBLAS("cublasDscal");//"NRSMat<double>& NRSMat<double>::operator*=(const double &)"
	}
#endif
	return *this;
}

/***************************************************************************//**
 * multiply this complex symmetric matrix with complex scalar value
 * @param[in] a complex multiplicative factor
 * @return reference to the modified matrix
 ******************************************************************************/
template<>
inline NRSMat<complex<double> > &
NRSMat<complex<double> >::operator*=(const complex<double> &a) {
	copyonwrite();
#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_zscal(NN2, &a, v, 1);
#ifdef CUDALA
	}else{
		const cuDoubleComplex _a = make_cuDoubleComplex(a.real(), a.imag());
		cublasZscal(NN2, _a, (cuDoubleComplex*)v, 1);
		TEST_CUBLAS("cublasZscal");//"NRSMat<complex<double> >& NRSMat<complex<double> >::operator*=(const complex<double> &)"
	}
#endif
	return *this;
}


/***************************************************************************//**
 * multiply this symmetric matrix of general type <code>T</code> stored in packed form
 * with scalar value of type <code>T</code>
 * @param[in] a multiplicative factor of type <code>T</code>
 * @return reference to the modified matrix
 ******************************************************************************/
template <typename T>
inline NRSMat<T> & NRSMat<T>::operator*=(const T &a) {
	NOT_GPU(*this);

	copyonwrite();
	for(register size_t i = 0; i<NN2; ++i) v[i] *= a;
        return *this;
}


/***************************************************************************//**
 * add a scalar value \f$\alpha\f$ of general type <code>T</code> to the diagonal elements of this symmetric matrix of type <code>T</code>
 * @param[in] a scalar value \f$\alpha\f$
 * @return reference to the modified matrix
 ******************************************************************************/
template <typename T>
inline NRSMat<T> & NRSMat<T>::operator+=(const T &a) {
	NOT_GPU(*this);

	copyonwrite();
	for(register int i = 0; i<nn; i++) v[i*(size_t)(i+1)/2 + i] += a;
	return *this;
}

/***************************************************************************//**
 * subtract a scalar value \f$\alpha\f$ of general type <code>T</code> from the
 * diagonal elements of this symmetric matrix of type <code>T</code>
 * @param[in] a scalar value \f$\alpha\f$
 * @return reference to the modified matrix
 ******************************************************************************/
template <typename T>
inline NRSMat<T> & NRSMat<T>::operator-=(const T &a) {
	NOT_GPU(*this);

	copyonwrite();
	for(register int i = 0; i<nn; i++) v[i*(size_t)(i+1)/2+i] -= a;
	return *this;
}

/***************************************************************************//**
 * add up this real symmetric matrix with given symmetric matrix
 * @param[in] rhs real symmetric matrix to be added
 * @return reference to the modified matrix
 ******************************************************************************/
template<>
inline NRSMat<double>& NRSMat<double>::operator+=(const NRSMat<double> & rhs) {
#ifdef DEBUG
	if(nn != rhs.nn) laerror("incompatible dimensions in NRSMat<double>& NRSMat<double>::operator+=(const NRSMat<double> &)");
#endif
	SAME_LOC(*this, rhs);
	copyonwrite();

#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_daxpy(NN2, 1.0, rhs.v, 1, v, 1);
#ifdef CUDALA
	}else{
		cublasDaxpy(NN2, 1.0, rhs.v, 1, v, 1);
		TEST_CUBLAS("cublasDaxpy");//" NRSMat<double>& NRSMat<double>::operator+=(const NRSMat<double> &)"
	}
#endif
	return *this;
}

/***************************************************************************//**
 * add up this complex symmetric matrix with given symmetric matrix
 * @param[in] rhs complex symmetric matrix to be added
 * @return reference to the modified matrix
 ******************************************************************************/
template<>
inline NRSMat<complex<double> >& NRSMat<complex<double> >::operator+=(const NRSMat<complex<double> > & rhs) {
#ifdef DEBUG
	if(nn != rhs.nn) laerror("incompatible dimensions in NRSMat<complex<double> >& NRSMat<complex<double> >::operator+=(const NRSMat<complex<double> > &)");
#endif
	SAME_LOC(*this, rhs);
	copyonwrite();

#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_zaxpy(NN2, &CONE, rhs.v, 1, v, 1);
#ifdef CUDALA
	}else{
		cublasZaxpy(NN2, CUONE, (cuDoubleComplex*)(rhs.v), 1, (cuDoubleComplex*)v, 1);
		TEST_CUBLAS("cublasZaxpy");//"NRSMat<complex<double> >& NRSMat<complex<double> >::operator+=(const NRSMat<complex<double> > &)"
	}
#endif
	return *this;
}

/***************************************************************************//**
 * add up this symmetric matrix of general type <code>T</code> with given symmetric matrix
 * @param[in] rhs complex matrix of general type <code>T</code> to be added
 * @return reference to the modified matrix
 ******************************************************************************/
template <typename T>
inline NRSMat<T>& NRSMat<T>::operator+=(const NRSMat<T>& rhs) {
#ifdef DEBUG
        if(nn != rhs.nn) laerror("incompatible NRSMat<T>& NRSMat<T>::operator+=(const NRSMat<T> &)");
#endif
	NOT_GPU(*this);
	SAME_LOC(*this, rhs);

        copyonwrite();
	for(register size_t i = 0; i<NN2; ++i) v[i] += rhs.v[i];
        return *this;
}

/***************************************************************************//**
 * subtracts given real symmetric matrix from this real symmetric matrix
 * @param[in] rhs real symmetric matrix to be subtracted
 * @return reference to the modified matrix
 ******************************************************************************/
template<>
inline NRSMat<double>& NRSMat<double>::operator-=(const NRSMat<double>& rhs) {
#ifdef DEBUG
	if(nn != rhs.nn) laerror("incompatible dimensions in NRSMat<double>& NRSMat<double>::operator-=(const NRSMat<double> &)");
#endif
	SAME_LOC(*this, rhs);
	copyonwrite();

#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_daxpy(NN2, -1.0, rhs.v, 1, v, 1);
#ifdef CUDALA
	}else{
		cublasDaxpy(NN2, -1.0, rhs.v, 1, v, 1);
		TEST_CUBLAS("cublasDaxpy");//" NRSMat<double>& NRSMat<double>::operator-=(const NRSMat<double> &)"
	}
#endif
	return *this;
}

/***************************************************************************//**
 * subtracts given complex symmetric matrix from this complex symmetric matrix
 * @param[in] rhs complex symmetric matrix to be subtracted
 * @return reference to the modified matrix
 ******************************************************************************/
template<>
inline NRSMat<complex<double> >& NRSMat<complex<double> >::operator-=(const NRSMat<complex<double> >& rhs) {
#ifdef DEBUG
	if(nn != rhs.nn) laerror("incompatible dimensions in NRSMat<complex<double> >& NRSMat<complex<double> >::operator-=(const NRSMat<complex<double> > &)");
#endif
	SAME_LOC(*this, rhs);
	copyonwrite();

#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_zaxpy(NN2, &CMONE, rhs.v, 1, v, 1);
#ifdef CUDALA
	}else{
		cublasZaxpy(NN2, CUMONE, (cuDoubleComplex*)(rhs.v), 1, (cuDoubleComplex*)v, 1);
		TEST_CUBLAS("cublasZaxpy");//"NRSMat<complex<double> >& NRSMat<complex<double> >::operator-=(const NRSMat<complex<double> > &)"
	}
#endif
	return *this;
}

/***************************************************************************//**
 * subtracts given symmetric matrix of general type <code>T</code> from this symmetric matrix of type <code>T</code>
 * @param[in] rhs symmetric matrix of general type <code>T</code> to be subtracted
 * @return reference to the modified matrix
 ******************************************************************************/
template <typename T>
inline NRSMat<T>& NRSMat<T>::operator-=(const NRSMat<T>& rhs) {
#ifdef DEBUG
        if(nn != rhs.nn) laerror("incompatible NRSMat<T>& NRSMat<T>::operator-=(const NRSMat<T> &)");
#endif
	NOT_GPU(*this);
        copyonwrite();

        for(register size_t i = 0; i<NN2; ++i) v[i] -= rhs.v[i];
        return *this;
}


/***************************************************************************//**
 * add up given dense matrix of general type <code>T</code> with this symmetric matrix of type <code>T</code>
 * @param[in] rhs dense matrix of type <code>T</code> to be added
 * @return reference to the modified matrix
 ******************************************************************************/
template <typename T>
inline const NRMat<T> NRSMat<T>::operator+(const NRMat<T> &rhs) const {
	return NRMat<T>(rhs) += *this;
}

/***************************************************************************//**
 * subtracts given dense matrix of general type <code>T</code> from this symmetric matrix of type <code>T</code>
 * @param[in] rhs dense matrix of type <code>T</code> to be added
 * @return reference to the modified matrix
 ******************************************************************************/
template <typename T>
inline const NRMat<T> NRSMat<T>::operator-(const NRMat<T> &rhs) const {
	return NRMat<T>(-rhs) += *this;
}

/***************************************************************************//**
 * determine matrix element of this symmetric matrix of general type <code>T</code>
 * using cumulative index increasing in a row-major way and corresponding to the 
 * lower triangular part of the respective dense matrix
 * @param[in] ij index of the requested element
 * @return reference to the corresponding matrix element
 ******************************************************************************/
template <typename T>
inline T& NRSMat<T>::operator[](const size_t ij) {
#ifdef DEBUG
	if(_LA_count_check && *count != 1) laerror("T& NRSMat<T>::operator[] used for matrix with count>1");
	if(ij<0 || ij>=NN2) laerror("T& NRSMat<T>::operator[] out of range");
	if(!v) laerror("T& NRSMat<T>::operator[] used for unallocated NRSmat<T> object");
#endif
	NOT_GPU(*this);

	return v[ij];
}

/***************************************************************************//**
 * determine matrix element of this symmetric matrix of general type <code>T</code>
 * using cumulative index increasing in a row-major way and corresponding to the 
 * lower triangular part of the respective dense matrix, i.e. \f$A_{i,j}\f$ for
 * \f$N>i\geq{}j\geq0\f$ corresponds to cumulative index \f$i(i+1)/2+j\f$
 * @param[in] ij index of the requested element
 * @return constant reference to the corresponding matrix element
 ******************************************************************************/
template <typename T>
inline const T & NRSMat<T>::operator[](const size_t ij) const {
#ifdef DEBUG
	if(ij<0 || ij>=NN2) laerror("T& NRSMat<T>::operator[] out of range");
	if(!v) laerror("T& NRSMat<T>::operator[] used for unallocated NRSmat<T> object");
#endif
	NOT_GPU(*this);

	return v[ij];
}

/***************************************************************************//**
 * determine the cumulative index or matrix element in row \f$i\f$ and column \f$j\f$
 * where \f$0\leq{}i,j<N\f$
 * @param[in] i row index
 * @param[in] i column index
 * @return cumulative index
 ******************************************************************************/
template<typename T>
inline size_t SMat_index(T i, T j) {
	return (i>=j) ? i*(size_t)(i+1)/2+j : j*(size_t)(j+1)/2+i;
}

/***************************************************************************//**
 * determine the cumulative index or matrix element in row \f$i\f$ and column \f$j\f$
 * where \f$0\leq{}i,j<N\f$ for special case \f$i\geq{}j\f$
 * @param[in] i row index
 * @param[in] i column index
 * @return cumulative index
 ******************************************************************************/
template<typename T>
inline size_t SMat_index_igej(T i, T j) {
	return i*(size_t)(i+1)/2+j;
}

/***************************************************************************//**
 * determine the cumulative index or matrix element in row \f$i\f$ and column \f$j\f$
 * where \f$0\leq{}i,j<N\f$ for special case \f$i\leq{}j\f$
 * @param[in] i row index
 * @param[in] i column index
 * @return cumulative index
 ******************************************************************************/
template<typename T>
inline size_t SMat_index_ilej(T i, T j) {
	return j*(size_t)(j+1)/2+i;
}

/***************************************************************************//**
 * determine the cumulative index or matrix element in row \f$i\f$ and column \f$j\f$
 * where \f$1\leq{}i,j\leq{}N\f$
 * @param[in] i row index
 * @param[in] i column index
 * @return cumulative index
 ******************************************************************************/
template<typename T>
inline size_t SMat_index_1(T i, T j) {
	return (i>=j)? i*(size_t)(i-1)/2+j-1 : j*(size_t)(j-1)/2+i-1;
}

/***************************************************************************//**
 * determine the cumulative index or matrix element in row \f$i\f$ and column \f$j\f$
 * where \f$1\leq{}i,j\leq{}N\f$ for special case \f$i\geq{}j\f$
 * @param[in] i row index
 * @param[in] i column index
 * @return cumulative index
 ******************************************************************************/
template<typename T>
inline size_t SMat_index_1igej(T i, T j) {
	return i*(size_t)(i-1)/2+j-1;
}

/***************************************************************************//**
 * determine the cumulative index or matrix element in row \f$i\f$ and column \f$j\f$
 * where \f$1\leq{}i,j\leq{}N\f$ for special case \f$i\leq{}j\f$
 * @param[in] i row index
 * @param[in] i column index
 * @return cumulative index
 ******************************************************************************/
template<typename T>
inline size_t SMat_index_1ilej(T i, T j) {
	return j*(size_t)(j-1)/2+i-1;
}

//indexing for antisymmetric matrix (used by fourindex)

template<typename T>
inline size_t ASMat_index(T i, T j)
{
if(i == j) return -1;
return (i>j) ? i*(size_t)(i-1)/2+j : j*(size_t)(j-1)/2+i;
}

template<typename T>
inline size_t ASMat_index_1(T i, T j)
{
if(i == j) return -1;
return (i>j)? (i-2)*(i-1)/2+j-1 : (j-2)*(j-1)/2+i-1;
}

/***************************************************************************//**
 * determine matrix element of this symmetric matrix of general type <code>T</code>
 * @param[in] i row index running from 0
 * @param[in] j column index running from 0
 * @return reference to the corresponding matrix element
 * @see count, SMat_index, NRSMat<T>::operator[]
 ******************************************************************************/
template <typename T>
inline T & NRSMat<T>::operator()(const int i, const int j) {
#ifdef DEBUG
	if(_LA_count_check && *count != 1) laerror("T & NRSMat<T>::operator()(const int, const int) used for matrix with count > 1");
	if(i<0 || i>=nn || j<0 || j>=nn) laerror("T & NRSMat<T>::operator()(const int, const int) out of range");
	if(!v) laerror("T & NRSMat<T>::operator()(const int, const int) used for unallocated NRSmat<T> object");
#endif
	NOT_GPU(*this);

	return v[SMat_index(i,j)];
}

/***************************************************************************//**
 * determine matrix element of this symmetric matrix of general type <code>T</code>
 * @param[in] i row index running from 0
 * @param[in] j column index running from 0
 * @return constant reference to the corresponding matrix element
 * @see count, SMat_index, NRSMat<T>::operator[]
 ******************************************************************************/
template <typename T>
inline const T & NRSMat<T>::operator()(const int i, const int j) const {
#ifdef DEBUG
	if(i<0 || i>=nn || j<0 || j>=nn) laerror("T & NRSMat<T>::operator()(const int, const int) out of range");
	if(!v) laerror("T & NRSMat<T>::operator()(const int, const int) used for unallocated NRSmat<T> object");
#endif
	NOT_GPU(*this);

	return v[SMat_index(i,j)];
}

/***************************************************************************//**
 * @return number of rows of this symmetric matrix of generalt type <code>T</code>
 ******************************************************************************/
template <typename T>
inline int NRSMat<T>::nrows() const {
	return nn;
}

/***************************************************************************//**
 * @return number of columns of this symmetric matrix of generalt type <code>T</code>
 ******************************************************************************/
template <typename T>
inline int NRSMat<T>::ncols() const {
	return nn;
}

/***************************************************************************//**
 * @return number of elements of this symmetric matrix of generalt type <code>T</code>
 ******************************************************************************/
template <typename T>
inline size_t NRSMat<T>::size() const {
        return NN2;
}


/***************************************************************************//**
 * for this real symmetric matrix \f$A\f$, determine the 
 * first element with largest absolute value
 * @return \f$A_{l,m}\f$ which maximizes \f$\left|A_{i,j}\right|\f$
 ******************************************************************************/
template<>
inline const double NRSMat<double>::amax() const {
	double ret(0.0);
#ifdef CUDALA
	if(location == cpu){
#endif
		ret = v[cblas_idamax(NN2, v, 1) - 1];
#ifdef CUDALA
	}else{
		const int pozice = cublasIdamax(NN2, v, 1) - 1;
		TEST_CUBLAS("cublasIdamax");//"double NRSMat<double>::amax()"

		gpuget(1, sizeof(double), v + pozice, &ret);
	}
#endif
	return ret;
}

/***************************************************************************//**
 * for this real symmetric matrix \f$A\f$, determine the 
 * first element with smallest absolute value
 * @return \f$A_{l,m}\f$ which minimizes \f$\left|A_{i,j}\right|\f$
 ******************************************************************************/
template<>
inline const double NRSMat<double>::amin() const {
	double ret(0.0);
#ifdef CUDALA
	if(location == cpu){
#endif
		// idamin seems not to be supported
		double val(0.0);
		int index(-1);
		ret = std::numeric_limits<double>::max();
		for(register size_t i = 0; i < NN2; i++){
			val = std::abs(v[i]);
			if(val < ret){ index = i; ret = val; }
		}
		ret = v[index];
#ifdef CUDALA
	}else{
		const int pozice = cublasIdamin(nn, v, 1) - 1;
		TEST_CUBLAS("cublasIdamin");//"double NRSMat<double>::amin()"
		gpuget(1, sizeof(double), v + pozice, &ret);
	}
#endif
	return ret;
}

/***************************************************************************//**
 * for this complex symmetric matrix \f$A\f$, determine the 
 * first element with largest "absolute value"
 * @return \f$A_{l,m}\f$ which maximizes \f$\left|\Re{}A_{i,j}\right| + \left|\Im{}A_{i,j}\right|\f$
 ******************************************************************************/
template<>
inline const complex<double> NRSMat< complex<double> >::amax() const{
	complex<double> ret(0., 0.);
#ifdef CUDALA
	if(location == cpu){
#endif
		ret = v[cblas_izamax(NN2, v, 1) - 1];
#ifdef CUDALA
	}else{
		const int pozice = cublasIzamax(NN2, (cuDoubleComplex*)v, 1) - 1;
		TEST_CUBLAS("cublasIzamax");//"complex<double> NRSMat<complex<double> >::amax()"
		gpuget(1, sizeof(complex<double>), v + pozice, &ret);
	}
#endif
	return ret;
}

/***************************************************************************//**
 * for this complex symmetric matrix \f$A\f$, determine the 
 * first element with smallest "absolute value"
 * @return \f$A_{l,m}\f$ which minimizes \f$\left|\Re{}A_{i,j}\right| + \left|\Im{}A_{i,j}\right|\f$
 ******************************************************************************/
template<>
inline const complex<double> NRSMat<complex<double> >::amin() const{
	complex<double> ret(0., 0.);
#ifdef CUDALA
	if(location == cpu){
#endif
		// izamin seems not to be supported
		int index(0);
		double val(0.0), min_val(0.0);
		complex<double> z_val(0.0, 0.0);

		min_val = std::numeric_limits<double>::max();
		for(register size_t i = 0; i < NN2; i++){
			z_val = v[i];
			val = std::abs(z_val.real()) + std::abs(z_val.imag());
			if(val < min_val){ index = i; min_val = val; }	
		}
		ret = v[index];
#ifdef CUDALA
	}else{
		const int pozice = cublasIzamin(nn, (cuDoubleComplex*)v, 1) - 1;
		TEST_CUBLAS("cublasIzamin");//"complex<double> NRSMat<complex<double> >::amin()"
		gpuget(1, sizeof(complex<double>), v + pozice, &ret);
	}
#endif
	return ret;
}

/***************************************************************************//**
 * @return pointer of general type T to the underlying data structure
 ******************************************************************************/
template <typename T>
inline NRSMat<T>::operator T*() {
#ifdef DEBUG
	if(!v) laerror("unallocated NRSMat object in NRSMat<T>::operator T*()");
#endif
	return v;
}

/***************************************************************************//**
 * @return constant pointer of general type T to the underlying data structure
 ******************************************************************************/
template <typename T>
inline NRSMat<T>::operator const T*() const {
#ifdef DEBUG
	if(!v) laerror("unallocated NRSMat object in NRSMat<T>::operator const T*()");
#endif
	return v;
}

/***************************************************************************//**
 * destructor for general type <code>T</code>
 * @see NRSMat<T>::count, NRSMat<T>::v
 ******************************************************************************/
template <typename T>
NRSMat<T>::~NRSMat() {
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
 * assigment operator implementing deep copy of the reference NRSMat<T> object
 * @see NRSMat<T>::operator=, NRSMat<T>::copyonwrite()
 ******************************************************************************/
template <typename T>
NRSMat<T> & NRSMat<T>::operator|=(const NRSMat<T> &rhs) {
#ifdef DEBUG
	if(!rhs.v) laerror("unallocated NRSMat<T> object in NRSMat<T> & NRSMat<T>::operator|=(const NRSMat<T> &)");
#endif
	if(this == &rhs) return *this;
	*this = rhs;
	this->copyonwrite();
	return *this;
}


/***************************************************************************//**
 * assignment operator implementing shallow copy of reference NRSMat<T> object
 * @see NRSMat<T>::operator|=, NRSMat<T>::copyonwrite()
 ******************************************************************************/
template <typename T>
NRSMat<T> & NRSMat<T>::operator=(const NRSMat<T> & rhs) {
	if(this == &rhs) return *this;
	if(count)
		if(--(*count) == 0){
#ifdef CUDALA
			if(location == cpu){
#endif
				delete [] v;
#ifdef CUDALA
			}else{ gpufree(v); }
#endif
			delete count;
		}
	v = rhs.v;
	nn = rhs.nn;
	count = rhs.count;
#ifdef CUDALA
	location = rhs.location;
#endif
        if(count) (*count)++;
        return *this;
}

/***************************************************************************//**
 * detach this NRSMat<T> object and create own physical copy of the data
 * @see NRSMat<T>::operator|=, NRSMat<T>::copyonwrite()
 ******************************************************************************/
template <typename T>
void NRSMat<T>::copyonwrite(bool detachonly) {
	if(!count) laerror("calling NRSMat<T>::copyonwrite() for undefined NRSMat<T> object");
	if(*count > 1){
		(*count)--;
		count = new int;
		*count = 1;
		T *newv;
#ifdef CUDALA
		if(location == cpu) {
#endif
			newv = new T[NN2];
			if(!detachonly) memcpy(newv, v, NN2*sizeof(T));
#ifdef CUDALA
		}else{
			newv = (T *) gpualloc(NN2*sizeof(T));
			if(sizeof(T)%sizeof(float) != 0) laerror("memory alignment problem in NRSMat<T>::copyonwrite()");
			if(!detachonly) cublasScopy(NN2*sizeof(T)/sizeof(float), (const float *) v, 1, (float *)newv, 1);
			TEST_CUBLAS("cublasScopy");//"NRSMat<T>::copyonwrite()"
		}
#endif

                 v = newv;
        }
}

/***************************************************************************//**
 * resize this symmetric matrix of general type <code>T</code>
 * @param[in] n requested number of rows (columns)
 ******************************************************************************/
template <typename T>
void NRSMat<T>::resize(const int n) {
#ifdef DEBUG
	if(n < 0) laerror("illegal dimension in NRSMat<T>::resize(const int)");
#endif
	if(count){
		if(n == 0){
			if(--(*count) <= 0) {
				if(v) {
#ifdef CUDALA
					if(location == cpu){
#endif
						delete[] (v);
#ifdef CUDALA
					}else{ gpufree(v); }
#endif
				}
				delete count;
			}
			count = 0;
			nn = 0;
			v = 0;
			return;
		}
		if(*count > 1){
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
		if(location == cpu){
#endif
			v = new T[NN2];
#ifdef CUDALA
		}else{ v = (T*) gpualloc(NN2*sizeof(T)); }
#endif

		return;
	}
	if(n != nn){
		nn = n;
#ifdef CUDALA
		if(location == cpu){
#endif
			delete[] v;
			v = new T[NN2];
#ifdef CUDALA
		}else{

			gpufree(v);
			v = (T*) gpualloc(NN2*sizeof(T));
		}
#endif

	}
}

/***************************************************************************//**
 * perform memory transfers between CPU and GPU memory
 * @param[in] dest memory destination
 * @see NRSMat<T>::location, NRSMat<T>::getlocation()
 ******************************************************************************/
#ifdef CUDALA
template<typename T>
void NRSMat<T>::moveto(const GPUID dest) {
	if(location == dest) return;

	CPU_GPU(location,dest);
	location = dest;

	if(v && !count) laerror("internal inconsistency of reference counting 1");
	if(!count) return;

	if(v && *count == 0) laerror("internal inconsistency of reference counting 2");
	if(!v) return;

	T *vold = v;

	if(dest == cpu){ //moving from GPU to CPU
		v = new T[NN2];
		gpuget(NN2, sizeof(T), vold, v);
		if(*count == 1) gpufree(vold);
		else {--(*count); count = new int(1);}

	}else{    //moving from CPU to GPU

		v = (T *) gpualloc(NN2*sizeof(T));
		gpuput(NN2, sizeof(T), vold, v);
		if(*count == 1) delete[] vold;
		else {--(*count); count = new int(1);}
	}
}
#endif


/***************************************************************************//**
 * complexify a given matrix \f$A\f$ of general type <code>T</code>
 * @param[in] rhs matrix \f$A\f$ intended for this operation
 * @return matrix \f$B\f$ where \f$\Re B=A\f$ and \f$\Im B = 0\f$
 ******************************************************************************/
template<typename T>
NRSMat<complex<T> > complexify(const NRSMat<T> &rhs) {
	NOT_GPU(rhs);

	NRSMat<complex<T> > r(rhs.nrows());
	for(register int i = 0; i<rhs.nrows(); ++i) 
		for(register int j = 0; j<=i; ++j) r(i,j) = rhs(i,j);
	return r;
}


/***************************************************************************//**
 * complexify a given double-precision real matrix \f$A\f$
 * @param[in] rhs matrix \f$A\f$ intended for this operation
 * @return matrix \f$B\f$ where \f$\Re B=A\f$ and \f$\Im B = 0\f$
 ******************************************************************************/
/*
template<>
NRSMat<complex<double> > complexify(const NRSMat<double> &rhs) {
	NRSMat<complex<double> > r(rhs.nrows(), rhs.getlocation());
#ifdef CUDALA
	if(rhs.getlocation() == cpu){
#endif
		cblas_dcopy(rhs.size(), &(rhs[0]), 1, (double*)(&(r[0])), 2);
#ifdef CUDALA
	}else{
		cublasDcopy(rhs.size(), &(rhs[0]), 1, (double*)(&(r[0])), 2);
		TEST_CUBLAS("cublasDcopy");//"NRSMat<complex<double> > complexify(const NRSMat<double> &)"
	}
#endif
	return r;
}
*/
/***************************************************************************//**
 * output operator
 * @param[in,out] s output stream
 * @param[in] x NRSMat<T> matrix to be printed out
 * @return modified stream
 ******************************************************************************/
template <typename T>
std::ostream& operator<<(std::ostream &s, const NRSMat<T> &x) {
#ifdef CUDALA
	if(x.getlocation() == cpu){
#endif
		int i,j,n;
		n = x.nrows();
		s << n << ' ' << n << '\n';
		for(i = 0;i<n;i++){
			for(j = 0; j<n;j++) s << (typename LA_traits_io<T>::IOtype)x(i,j) << (j == n-1 ? '\n' : ' ');
		}
		return s;
#ifdef CUDALA
	}else{
		NRSMat<T> tmp = x;
		tmp.moveto(cpu);
		return s<<tmp;
	}
#endif
}


/***************************************************************************//**
 * input operator
 * @param[in,out] s input stream
 * @param[in] x NRSMat<T> matrix for storing the input
 * @return modified stream
 ******************************************************************************/
template <typename T>
std::istream& operator>>(std::istream  &s, NRSMat<T> &x) {
#ifdef CUDALA
	if(x.getlocation() == cpu){
#endif
		int i,j,n,m;
		s >> n >> m;
		if(n!=m) laerror("input symmetric matrix not square");
		x.resize(n);
		typename LA_traits_io<T>::IOtype tmp;
		for(i = 0;i<n;i++) for(j = 0; j<m;j++) {s>>tmp; x(i,j)=tmp;}
		return s;
#ifdef CUDALA
	}else{
		NRSMat<T> tmp;
		tmp.moveto(cpu);
		s >> tmp;
		tmp.moveto(x.getlocation());
		x = tmp;
		return s;
	}
#endif
}


/***************************************************************************//**
 * generate operators relating NRSMat<T> objects and scalars
 * corresponding macro is defined in vec.h
 ******************************************************************************/
NRVECMAT_OPER(SMat,+)
NRVECMAT_OPER(SMat,-)
NRVECMAT_OPER(SMat,*)

/***************************************************************************//**
 * generate operators relating in general two NRSMat<T> objects
 * corresponding macro is defined in vec.h
 ******************************************************************************/
NRVECMAT_OPER2(SMat,+)
NRVECMAT_OPER2(SMat,-)

/***************************************************************************//**
 * class implementing NRSMat<T> funcitonality with indices running from 1
 * allmost all function members are inherited, only constructors are given explicitly
 ******************************************************************************/
template<typename T>
class NRSMat_from1 : public NRSMat<T> {
public:
	NRSMat_from1(): NRSMat<T>() {};
	explicit NRSMat_from1(const int n): NRSMat<T>(n) {};
	NRSMat_from1(const NRSMat<T> &rhs): NRSMat<T>(rhs) {}; //be able to convert the parent class  transparently to this
	NRSMat_from1(const T &a, const int n): NRSMat<T>(a,n) {}; 
	NRSMat_from1(const T *a, const int n): NRSMat<T>(a,n) {}; 
	explicit NRSMat_from1(const NRMat<T> &rhs): NRSMat<T>(rhs) {};
	explicit NRSMat_from1(const NRVec<T> &rhs, const int n): NRSMat<T>(rhs,n) {};

	inline const T& operator() (const int i, const int j) const {
	#ifdef DEBUG
		if(i<=0||j<=0||i>NRSMat<T>::nn||j>NRSMat<T>::nn) laerror("index in const T& NRSMat<T>::operator() (const int, const int) out of range");
	#endif
		return NRSMat<T>::v[SMat_index_1(i,j)];
	}
	
	inline  T& operator() (const int i, const int j){
	#ifdef DEBUG
		if(i<=0||j<=0||i>NRSMat<T>::nn||j>NRSMat<T>::nn) laerror("index in T& NRSMat<T>::operator() (const int, const int) out of range");
	#endif
		return NRSMat<T>::v[SMat_index_1(i,j)];
	}
};

}//namespace
#endif /* _LA_SMAT_H_ */
