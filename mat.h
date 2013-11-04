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
#ifndef _LA_MAT_H_
#define _LA_MAT_H_
#include "la_traits.h"

namespace LA {

/***************************************************************************//**
 * \brief NRMat<T> class template implementing the matrix interface
 * @see NRVec<T>, NRSMat<T>
 ******************************************************************************/
template <typename T>
class NRMat{
protected:
	int nn;//!< number of rows
	int mm;//!< number of columns
#ifdef MATPTR
	T **v;//!< pointer to the array of pointers pointing to the beginings of individual rows
#else
	T *v;//!< pointer to the data stored continuously in emmory
#endif
	int *count;//!< reference counter
public:
#ifdef CUDALA
	GPUID location;
#endif
	friend class NRVec<T>;
	friend class NRSMat<T>;
	
	//! standard destructor
	~NRMat();

	/***************************************************************************//**
	 * \brief inlined constructor creating zero matrix of general type <code>T</code>
	 ******************************************************************************/
	inline NRMat() : nn(0), mm(0), v(0), count(0){
		#ifdef CUDALA
		location = DEFAULT_LOC;
		#endif
	};

	/***************************************************************************//**
	 * \brief Inlined constructor creating matrix of given size and location.
	 * Because of performance reasons, no incialization is done.
	 * @param[in] n vector size (count of elements)
	 * @param[in] loc location of the underlying data (CPU/GPU)
	 ******************************************************************************/
	inline NRMat(const int n, const int m, const GPUID loc = undefined);

	//! inlined constructor creating matrix of given size filled with prescribed value and stored at given location
	inline NRMat(const T &a, const int n, const int m, const GPUID loc);

	//! inlined constructor creating matrix of given size filled with prescribed value 
	inline NRMat(const T &a, const int n, const int m);

	//! inlined constructor creating matrix of given size filled with data located at given memory location 
	NRMat(const T *a, const int n, const int m);

	//! inlined copy-constructor
	inline NRMat(const NRMat &rhs);

	//! complexifying constructor 
	NRMat(const typename LA_traits_complex<T>::NRMat_Noncomplex_type &rhs, bool imagpart = false);
	//! explicit decomplexifying constructor
	explicit NRMat(const NRMat<complex<T> > &rhs, bool imagpart = false);

	//! explicit constructor converting symmetric matrix stored in packed form into a <code>NRMat<T></code> object
	explicit NRMat(const NRSMat<T> &rhs);

	//! explicit constructor converting vector into a <code>NRMat<T></code> object
#ifdef MATPTR
	explicit NRMat(const NRVec<T> &rhs, const int n, const int m, const int offset = 0):NRMat(&rhs[0][0] + offset , n, m){
		if (offset < 0 || (size_t)n*m + offset > rhs.nn) laerror("matrix dimensions and offset incompatible with vector length");
	};
#else
	explicit NRMat(const NRVec<T> &rhs, const int n, const int m, const int offset = 0);
#endif

#ifdef MATPTR
	const bool operator!=(const NRMat &rhs) const {if(nn!=rhs.nn || mm!=rhs.mm) return 1; return LA_traits<T>::gencmp(v[0],rhs.v[0],(size_t)nn*mm);} //memcmp for scalars else elementwise
#else
        const bool operator!=(const NRMat &rhs) const {if(nn!=rhs.nn || mm!=rhs.mm) return 1; return LA_traits<T>::gencmp(v,rhs.v,(size_t)nn*mm);} //memcmp for scalars else elementwise
#endif

	const bool operator==(const NRMat &rhs) const {return !(*this != rhs);};

	//! determine the count of references to this object
	inline int getcount() const {return count?*count:0;}

	//! ensure that the data of this matrix are referenced exactly once
	void copyonwrite(bool detachonly=false);

	/***************************************************************************//**
	 * routines for CUDA related stuff
	 * \li <code>getlocation()</code> gets the protected data member location
	 * \li <code>moveto(const GPUID)</code> moves underlying data between CPU/GPU memory
	******************************************************************************/
#ifdef CUDALA
	inline GPUID getlocation() const {return location;}
	void moveto(const GPUID dest);
#else
	inline GPUID getlocation() const {return cpu;}
	void moveto(const GPUID dest) {};
#endif

	//! fill the matrix with pseudorandom numbers (uniform distribution)
	void randomize(const typename  LA_traits<T>::normtype &x);

	//! assigment operator performing shallow copy
	NRMat & operator=(const NRMat &rhs);
	//! assigment operator performing deep copy
	NRMat & operator|=(const NRMat &rhs);

	//! assign scalar value to the diagonal elements
	NRMat & operator=(const T &a);
	//! add scalar value to the diagonal elements
	NRMat & operator+=(const T &a);
	//! subtract scalar value to the diagonal elements
	NRMat & operator-=(const T &a);

	//! multiply by a scalar value
	NRMat & operator*=(const T &a);

	//! add given matrix
	NRMat & operator+=(const NRMat &rhs);
	//! subtract given matrix
	NRMat & operator-=(const NRMat &rhs);
	//! Hadamard element-wise product
	NRMat & operator^=(const NRMat<T> &rhs);

	//! add symmetric matrix stored in packed form
	NRMat & operator+=(const NRSMat<T> &rhs);
	//! subtract symmetric matrix stored in packed form
	NRMat & operator-=(const NRSMat<T> &rhs);

	//! unary minus
	const NRMat operator-() const;

	//! add scalar value to all matrix elements and return the result by value
	inline const NRMat operator+(const T &a) const;
	//! subtract scalar value from all matrix elements and return the result by value
	inline const NRMat operator-(const T &a) const;
	//! multiply all matrix elements by a scalar value and return the result by value
	inline const NRMat operator*(const T &a) const;

	//! add given matrix and return the result by value
	inline const NRMat operator+(const NRMat &rhs) const;
	//! add given symmetric matrix stored in packed form and return the result by value
	inline const NRMat operator+(const NRSMat<T> &rhs) const;
	
	//! subtract given matrix and return the result by value
	inline const NRMat operator-(const NRMat &rhs) const;
	//! subtract given symmetric matrix stored in packed form and return the result by value
	inline const NRMat operator-(const NRSMat<T> &rhs) const;
	
	//! multiply by given matrix and return the result by value
	const NRMat operator*(const NRMat &rhs) const;
	//! multiply by given symmetric matrix stored in packed form and return the result by value
	const NRMat operator*(const NRSMat<T> &rhs) const; 

	//! direct sum of two matrices
	const NRMat operator&(const NRMat &rhs) const;
	//! direct product of two matrices
	const NRMat operator|(const NRMat<T> &rhs) const;

	//! multiply by a vector
	const NRVec<T> operator*(const NRVec<T> &rhs) const {
		NRVec<T> result(nn, rhs.getlocation());
		result.gemv((T)0, *this, 'n', (T)1, rhs);
		return result;
	};
	//! multiply this matrix of general type <code>T</code> by vector of type <code>complex<T></code>
	const NRVec<complex<T> > operator*(const NRVec<complex<T> > &rhs) const {
		NRVec<complex<T> > result(nn, rhs.getlocation());
		result.gemv((T)0, *this, 'n', (T)1, rhs);
		return result;
	};

	//! inner product of two matrices (taking conjugation into account in the complex case)
	const T dot(const NRMat &rhs) const;

	//! direct sum
	const NRMat oplus(const NRMat &rhs) const;
	//! direct product
	const NRMat otimes(const NRMat &rhs, bool reversecolumns = false) const;

	//! multiply by diagonal matrix from left
	void diagmultl(const NRVec<T> &rhs);
	//! multiply by diagonal matrix from right
	void diagmultr(const NRVec<T> &rhs);

	//! for this matrix \f$A\f$ compute \f$A^\mathrm{T}\cdot{}A\f$
	const NRSMat<T> transposedtimes() const;
	//! for this matrix \f$A\f$ compute \f$A\cdot{}A^\mathrm{T}\f$
	const NRSMat<T> timestransposed() const;

	//! sum the rows
	const NRVec<T> rsum() const;
	//! sum the columns
	const NRVec<T> csum() const;

	//! orthonormalize this matrix
	void orthonormalize(const bool rowcol, const NRSMat<T> *metric = NULL);

	//! get the i<sup>th</sup> row
	const NRVec<T> row(const int i, int l = -1) const;

	//! get the j<sup>th</sup> column
	const NRVec<T> column(const int j, int l = -1) const {
		NOT_GPU(*this);
		if(l < 0) l = nn;
		NRVec<T> r(l);
		for(register int i=0; i<l; ++i) r[i] = (*this)(i,j);
		return r;
	};

	//! extract the digonal elements of this matrix and store them into a vector
	const T* diagonalof(NRVec<T> &, const bool divide = 0, bool cache = false) const;
	//! set diagonal elements
	void diagonalset(const NRVec<T> &);

	//! perform the <b>gemv</b> operation with vector of type <code>T</code>
	void gemv(const T beta, NRVec<T> &r, const char trans, const T alpha, const NRVec<T> &x) const { r.gemv(beta, *this, trans, alpha, x); };
	//! perform the <b>gemv</b> operation with vector of type <code>complex<T></code>
	void gemv(const T beta, NRVec<complex<T> > &r, const char trans, const T alpha, const NRVec<complex<T> > &x) const { r.gemv(beta, *this, trans, alpha, x); };

	//! determine the pointer to the i<sup>th</sup> row
	inline T* operator[](const int i);
	//! determine the const pointer to the i<sup>th</sup> row
	inline const T* operator[](const int i) const;

	//! get the reference to the element with indices (i,j)
	inline T& operator()(const int i, const int j);
	//! get the const reference to the element with indices (i,j)
	inline const T& operator()(const int i, const int j) const;
	//! get the copy of the element with indices (i,j)
	inline const T get_ij(const int i, const int j) const;

	//! get the number of rows
	inline int nrows() const;
	//! get the number of columns
	inline int ncols() const;
	//! get the number of matrix elements
	inline size_t size() const;

	//! unformatted input
        void get(int fd, bool dimensions = 1, bool transposed = false);
	//! unformatted output
        void put(int fd, bool dimensions = 1, bool transposed = false) const;
	//! formatted output
	void fprintf(FILE *f, const char *format, const int modulo) const;
	//! formatted input
	void fscanf(FILE *f, const char *format);

	//! set all matrix elements equal to zero
        void clear(){
		if(nn&&mm){
			copyonwrite(true);
			LA_traits<T>::clear((*this)[0], (size_t)nn*mm);
		}
	};

	//! resize the matrix
	void resize(int n, int m);

	//! deallocate the matrix
	void dealloc(void) {resize(0,0);}

	//! get the pointer to the data
	inline operator T*();
	//! get the const pointer to the data
	inline operator const T*() const;

	//! in case of square matrix, transpose the leading minor of order <code>n</code>
	NRMat& transposeme(const int n = 0);
	//! conjugate a square matrix
	NRMat& conjugateme();

	//! transpose this matrix and return the result by value
	const NRMat transpose(bool conj = false) const;
	//! conjugate this matrix and return the result by value
	const NRMat conjugate() const;

	//! extract specified submatrix
        const NRMat submatrix(const int fromrow, const int torow, const int fromcol, const int tocol) const;

	//! store given matrix at given position into the current matrix
	void storesubmatrix(const int fromrow, const int fromcol, const NRMat &rhs);

	//! perform the \b gemm operation
	void gemm(const T &beta, const NRMat &a, const char transa, const NRMat &b, const char transb, const T &alpha);

	//! compute the norm of this matrix
	const typename LA_traits<T>::normtype norm(const T scalar = (T)0) const;

	//! add up a scalar multiple of given matrix to the current matrix
	void axpy(const T alpha, const NRMat &x);

	//! maximal element in the absolute value
	inline const T amax() const;
	//! minimal element in the absolute value
	inline const T amin() const;

	//! determine the sum of the diagonal elements
	const T trace() const;

	//! swap the order of the rows of the current matrix
	NRMat & swap_rows();
	//! swap the order of the columns of the current matrix
	NRMat & swap_cols();
	//! swap the order of the rows and columns of the current matrix
	NRMat & swap_rows_cols();

	//! multiply by sparse matrix
	SparseSMat<T> operator*(const SparseSMat<T> &rhs) const;

	//! explicit constructor converting sparse matrix into \c NRMat<T> object
	explicit NRMat(const SparseMat<T> &rhs);                // dense from sparse
	//! explicit constructor converting sparse symmetric matrix into \c NRMat<T> object
	explicit NRMat(const SparseSMat<T> &rhs);
	//! explicit constructor converting sparse CSR matrix into \c NRMat<T> object
        explicit NRMat(const CSRMat<T> &rhs);

	//! add up given sparse matrix
	NRMat & operator+=(const SparseMat<T> &rhs);
	//! subtract given sparse matrix
        NRMat & operator-=(const SparseMat<T> &rhs);

	//! perform the \b gemm operation
	void gemm(const T &beta, const SparseMat<T> &a, const char transa, const NRMat &b, const char transb, const T &alpha);

        inline void simplify() {};
	bool issymmetric() const { return 0; };

#ifndef NO_STRASSEN
	//! Strassen's multiplication (better than \f$\mathacal{O}(n^3)\f$, analogous syntax to \see NRMat<T>::gemm() )
	void strassen(const T beta, const NRMat &a, const char transa, const NRMat &b, const char transb, const T alpha);
	void s_cutoff(const int,const int,const int,const int) const;
#endif
};

}//namespace

//due to mutual includes this has to be after full class declaration
#include "vec.h"
#include "smat.h"
#include "sparsemat.h"
#include "sparsesmat.h"

namespace LA {

/***************************************************************************//**
 * matrix constructor
 * @param[in]   n number of rows of the matrix being created
 * @param[in]   m number of cols of the matrix being created
 * @param[in] loc location for storing the matrix
 * @see count, v, location
 ******************************************************************************/
template <typename T>
NRMat<T>::NRMat(const int n, const int m, const GPUID loc) : nn(n), mm(m), count(new int) {
	T* p;
	*count = 1;
	const size_t nm = (size_t)n*m;
#ifdef CUDALA
	location = (loc==undefined?DEFAULT_LOC:loc);
	if(location == cpu) {
#endif
	#ifdef MATPTR
		v = new T*[n];
		p = v[0] = new T[nm];
		for (int i=1; i<n; i++) v[i] = v[i-1] + m;
	#else
		p = v = new T[nm];
	#endif
#ifdef CUDALA
	}else{
		const T val = 0;
		v = (T*) gpualloc(nm*sizeof(T));
	}
#endif
}

/***************************************************************************//**
 * matrix constructor
 * @param[in] a value of type T intended for matrix inicialization
 * @param[in] n number of rows of the matrix being created
 * @param[in] m number of cols of the matrix being created
 * @see count, v
 ******************************************************************************/
template <typename T>
NRMat<T>::NRMat(const T &a, const int n, const int m, const GPUID loc) : nn(n), mm(m), count(new int) {
	const size_t nm = (size_t)n*m;
	T *p;
	*count = 1;

#ifdef CUDALA
	location = (loc==undefined?DEFAULT_LOC:loc);
	if(location==cpu){
#endif
	#ifdef MATPTR
		v = new T*[n];
		p = v[0] = new T[nm];
		for (register int i=1; i<n; i++) v[i] = v[i-1] + m;
	#else
		p = v = new T[nm];
	#endif
		if (a != (T)0){
			for (register int i=0; i<nm; i++) *p++ = a;
		}else{
			memset(p, 0, nm*sizeof(T));
		}
#ifdef CUDALA
	}else{
		if(sizeof(T)%sizeof(float) != 0)laerror("memory alignment error");

		v = (T*)gpualloc(nm*sizeof(T));
		smart_gpu_set(nm, a, v);
	}
#endif
}

/***************************************************************************//**
 * matrix constructor
 * @param[in] a value of type T intended for matrix inicialization
 * @param[in] n number of rows of the matrix being created
 * @param[in] m number of cols of the matrix being created
 * @see count, v
 ******************************************************************************/
template <typename T>
NRMat<T>::NRMat(const T &a, const int n, const int m) : nn(n), mm(m), count(new int) {
	const size_t nm = (size_t)n*m;
	T *p;
	*count = 1;

#ifdef CUDALA
	location = DEFAULT_LOC;
	if(location==cpu){
#endif
	#ifdef MATPTR
		v = new T*[n];
		p = v[0] = new T[nm];
		for (register int i=1; i<n; i++) v[i] = v[i-1] + m;
	#else
		p = v = new T[nm];
	#endif
	if (a != (T)0)
		for (register int i=0; i<nm; i++) *p++ = a;
	else
		memset(p, 0, nm*sizeof(T));
#ifdef CUDALA
	}else{
		v = (T*)gpualloc(nm*sizeof(T));
		smart_gpu_set(nm, a, v);
	}
#endif
}

/***************************************************************************//**
 * matrix constructor
 * @param[in] a pointer to values of type T intended for matrix inicialization
 * @param[in] n number of rows of the matrix being created
 * @param[in] m number of cols of the matrix being created
 * @see count, v
 ******************************************************************************/
template <typename T>
NRMat<T>::NRMat(const T *a, const int n, const int m) : nn(n), mm(m), count(new int) {
	const size_t nm = (size_t)n*m;
#ifdef CUDALA
	location = DEFAULT_LOC;
#endif

	*count = 1;
#ifdef CUDALA
	if(location==cpu){
#endif
	#ifdef MATPTR
		v = new T*[n];
		v[0] = new T[nm];
		for (register int i=1; i<n; i++) v[i] = v[i-1] + m;
		memcpy(v[0], a, nm*sizeof(T));
	#else
		v = new T[nm];
		memcpy(v, a, nm*sizeof(T));
	#endif
#ifdef CUDALA
	}else{
		v = (T*) gpualloc(nm*sizeof(T));
		cublasSetVector(nm, sizeof(T), a, 1, v, 1);
	}
#endif

}

/***************************************************************************//**
 * copy constructor implementing shallow copy
 * @param[in] rhs reference object to be copied
 * @see count, v
 ******************************************************************************/
template <typename T>
NRMat<T>::NRMat(const NRMat &rhs) {
#ifdef CUDALA
	location = rhs.location;
#endif
	nn = rhs.nn;
	mm = rhs.mm;
	count = rhs.count;
	v = rhs.v;
	if (count) ++(*count);
}

/***************************************************************************//**
 * create matrix from a \c NRSMat object
 * @param[in] rhs \c NRSMat input object to be converted
 * @see count, v, vec.h, NRSMat<T>
 ******************************************************************************/
template <typename T>
NRMat<T>::NRMat(const NRSMat<T> &rhs) {
	NOT_GPU(rhs); 

#ifdef CUDALA
	location = rhs.location;
#endif

	int i(0), j(0), k(0);
	nn = mm = rhs.nrows();
	 count = new int;
	*count = 1;
#ifdef MATPTR
	v = new T*[nn];
	v[0] = new T[(size_t)mm*nn];
	for (int i=1; i<nn; i++) v[i] = v[i-1] + mm;
#else
	v = new T[(size_t)mm*nn];
#endif

#ifdef MATPTR
	for (i=0; i<nn; i++){
		for (j=0; j<=i; j++){
			v[i][j] = v[j][i] = rhs[k++];
		}
	}
#else
	for (i=0; i<nn; i++){
		for (j=0; j<=i; j++){
			v[i*(size_t)nn + j] = v[j*(size_t)nn + i] = rhs[k++];
		}
	}
#endif
}
 
/***************************************************************************//**
 * create matrix from a vector (shallow copy)
 * @param[in] rhs NRVec vector containing the data
 * @param[in] n number of rows of the matrix being created
 * @param[in] m number of cols of the matrix being created
 * @see count, v, vec.h
 ******************************************************************************/
#ifndef MATPTR
template <typename T>
NRMat<T>::NRMat(const NRVec<T> &rhs, const int n, const int m, const int offset)
{
	if (offset < 0 || (size_t)n*m + offset > rhs.nn) laerror("matrix dimensions and offset incompatible with vector length");

#ifdef CUDALA
	location=rhs.location;
#endif

	nn = n;
	mm = m;
	count = rhs.count;
	v = rhs.v + offset;//!< make just shallow copy
	(*count)++;//!< therefore increase the reference counter
}
#endif

/***************************************************************************//**
 * \c NRMat + \c NRSmat via operator +=
 * @param[in] rhs NRSMat matrix to be subtracted from current matrix
 * @return result of the subtraction
 * @see NRMat<T>::operator+=(const NRSMat<T> &)
 ******************************************************************************/
template <typename T>
inline const NRMat<T> NRMat<T>::operator+(const NRSMat<T> &rhs) const {
	return NRMat<T>(*this) += rhs;
}

/***************************************************************************//**
 * \c NRMat - \c NRSmat via operator -=
 * @param[in] rhs NRSMat matrix to be subtracted from current matrix
 * @return result of the subtraction
 * @see NRMat<T>::operator-=(const NRSMat<T> &)
 ******************************************************************************/
template <typename T>
inline const NRMat<T> NRMat<T>::operator-(const NRSMat<T> &rhs) const {
	return NRMat<T>(*this) -= rhs;
}

/***************************************************************************//**
 * @param[in] i row number
 * @return pointer to the first element in the i-th row
 ******************************************************************************/
template <typename T>
inline T* NRMat<T>::operator[](const int i) {
#ifdef DEBUG
	if (_LA_count_check && *count != 1) laerror("matrix with *count>1 used as l-value");
	if (i < 0 || i >= nn) laerror("Mat [] out of range");
	if (!v) laerror("unallocated matrix");
#endif
	#ifdef MATPTR
		return v[i];
	#else
		return v + i*(size_t)mm;
	#endif
}

/***************************************************************************//**
 * @param[in] i row number
 * @return const pointer to the first element in the i-th row
 ******************************************************************************/
template <typename T>
inline const T* NRMat<T>::operator[](const int i) const {
#ifdef DEBUG
	if (i < 0 || i >= nn) laerror("index out of range");
	if (!v) laerror("unallocated matrix");
#endif
	NOT_GPU(*this);
	#ifdef MATPTR
		return v[i];
	#else
		return v + i*(size_t)mm;
	#endif
}

/***************************************************************************//**
 * for a given matrix \f$A\f$, determine the element with indices (i,j)
 * @param[in] i row number
 * @param[in] j col number
 * @return reference to \f$A_{i,j}\f$
 * @see NRMat<T>::count
 ******************************************************************************/
template <typename T>
inline T& NRMat<T>::operator()(const int i, const int j){
#ifdef DEBUG
	if (_LA_count_check && *count != 1) laerror("NRMat::operator(,) used as l-value for a matrix with count > 1");
	if (i < 0 || i >= nn && nn > 0 || j < 0 || j >= mm && mm > 0) laerror("index out of range");
	if (!v) laerror("unallocated matrix");
#endif
	NOT_GPU(*this);
	#ifdef MATPTR
		return v[i][j];
	#else
		return v[i*(size_t)mm + j];
	#endif
}

/***************************************************************************//**
 * for a given matrix \f$A\f$, determine the element with indices (i,j)
 * @param[in] i row number
 * @param[in] j col number
 * @return const reference to \f$A_{i,j}\f$
 ******************************************************************************/
template <typename T>
inline const T& NRMat<T>::operator()(const int i, const int j) const{
	T ret;
#ifdef DEBUG
	if (i<0 || i>=nn && nn>0 || j<0 || j>=mm && mm>0) laerror("index out of range");
	if (!v) laerror("unallocated matrix");
#endif
	NOT_GPU(*this);
	#ifdef MATPTR
		return v[i][j];
	#else
		return v[i*(size_t)mm + j];
	#endif
}

/***************************************************************************//**
 * for a given matrix \f$A\f$, determine the element with indices (i,j)
 * @param[in] i row number
 * @param[in] j col number
 * @return const reference to \f$A_{i,j}\f$
 ******************************************************************************/
template <typename T>
inline const T NRMat<T>::get_ij(const int i, const int j) const{
	T ret;
#ifdef DEBUG
	if (i<0 || i>=nn || j<0 || j>=mm) laerror("index out of range");
	if (!v) laerror("unallocated matrix");
#endif
#ifdef CUDALA
	if(location == cpu){
#endif
	#ifdef MATPTR
		return v[i][j];
	#else
		return v[i*(size_t)mm + j];
	#endif
#ifdef CUDALA
	}else{
		const size_t pozice = i*(size_t)mm + j;
		gpuget(1, sizeof(T), v + pozice, &ret);
		return ret;
	}
#endif
}

/***************************************************************************//**
 * @return number of rows
 ******************************************************************************/
template <typename T>
inline int NRMat<T>::nrows() const{
	return nn;
}

/***************************************************************************//**
 * @return number of columns
 ******************************************************************************/
template <typename T>
inline int NRMat<T>::ncols() const{
	return mm;
}

/***************************************************************************//**
 * @return number of elements
 ******************************************************************************/
template <typename T>
inline size_t NRMat<T>::size() const{
	return (size_t)nn*mm;
}

/***************************************************************************//**
 * @return pointer of general type T to the underlying data structure
 ******************************************************************************/
template <typename T>
inline NRMat<T>::operator T*(){
	#ifdef DEBUG
		if (!v) laerror("unallocated matrix");
	#endif
	#ifdef MATPTR
		return v[0];
	#else
		return v;
	#endif
}

/***************************************************************************//**
 * @return const pointer of general type T to the underlying data
 ******************************************************************************/
template <typename T>
inline NRMat<T>::operator const T*() const{
	#ifdef DEBUG
		if (!v) laerror("unallocated matrix");
	#endif
	#ifdef MATPTR
		return v[0];
	#else
		return v;
	#endif
}

/***************************************************************************//**
 * for this real matrix \f$A\f$, determine the first element
 * with largest absolute value
 * @return \f$A_{l,m}\f$ which maximizes \f$\left|A_{i,j}\right|\f$
 ******************************************************************************/
template<>
inline const double NRMat<double>::amax() const{
#ifdef CUDALA
	if(location == cpu){
#endif
	#ifdef MATPTR
		return v[0][cblas_idamax(nn*mm, v[0], 1) - 1];
	#else
		return v[cblas_idamax(nn*mm, v, 1) - 1];
	#endif
#ifdef CUDALA
	}else{
		double ret(0.0);
		const size_t pozice = cublasIdamax((size_t)nn*mm, v, 1) - 1;
		TEST_CUBLAS("cublasIdamax");
		gpuget(1, sizeof(double), v + pozice, &ret);
		return ret;
	}
#endif
}

/***************************************************************************//**
 * for this real matrix \f$A\f$, determine the first element
 * with smallest absolute value
 * @return \f$A_{l,m}\f$ which minimizes \f$\left|A_{i,j}\right|\f$
 ******************************************************************************/
template<>
inline const double NRMat<double>::amin() const{
	double ret(0.0);
#ifdef CUDALA
	if(location == cpu){
#endif
		// idamin seems not to be supported
		const size_t nm = (size_t)nn*mm;
		double val(0.0);
		int index(-1);
		ret = std::numeric_limits<double>::max();
		for(register int i=0; i < nm; i++){
		#ifdef MATPTR			
			val = std::abs(v[0][i]);
		#else
			val = std::abs(v[i]);
		#endif
			if(val < ret){ index = i; ret = val; }
		}
		#ifdef MATPTR
			ret = v[0][index];
		#else
			ret = v[index];
		#endif	
#ifdef CUDALA
	}else{
		const size_t pozice = cublasIdamin((size_t)nn*mm, v, 1) - 1;
		TEST_CUBLAS("cublasIdamin");
		gpuget(1, sizeof(double), v + pozice, &ret);
	}
#endif
	return ret;
}

/***************************************************************************//**
 * for this complex matrix \f$A\f$, determine the smallest index of the maximum 
 * magnitude element, i.e. maximal element in the 1-norm
 * @return \f$A_{l,m}\f$ which maximizes \f$\left\{\left|\Re{}A_{i,j}\right|+\left|\Im{}A_{i,j}\right|\right}\f$
 ******************************************************************************/
template<>
inline const complex<double> NRMat<complex<double> >::amax() const{
#ifdef CUDALA
	if(location == cpu){
#endif
	#ifdef MATPTR
		return v[0][cblas_izamax(nn*mm, v[0], 1) - 1];
	#else
		return v[cblas_izamax(nn*mm, v, 1) - 1];
	#endif
#ifdef CUDALA
	}else{
		complex<double> ret(0.0, 0.0);
		const size_t pozice = cublasIzamax((size_t)nn*mm, (cuDoubleComplex*)v, 1) - 1;
		TEST_CUBLAS("cublasIzamax");
		gpuget(1, sizeof(complex<double>), v + pozice, &ret);
		return ret;
	}
#endif

}

/***************************************************************************//**
 * for this complex matrix \f$A\f$, determine the smallest index of the minimum 
 * magnitude element, i.e. minimal element in the 1-norm
 * @return \f$A_{l,m}\f$ which minimizes \f$\left\{\left|\Re{}A_{i,j}\right|+\left|\Im{}A_{i,j}\right|\right}\f$
 ******************************************************************************/
template<>
inline const complex<double> NRMat<complex<double> >::amin() const{
	complex<double> ret(0.0, 0.0);
#ifdef CUDALA
	if(location == cpu){
#endif
		// idamin seems not to be supported
		const size_t nm = (size_t)nn*mm;
		int index(-1);
		double val(0.0), min_val(0.0);
		complex<double> z_val(0.0, 0.0);

		min_val = std::numeric_limits<double>::max();
		for(register int i=0; i < nm; i++){
		#ifdef MATPTR			
			z_val = v[0][i];
		#else
			z_val = v[i];
		#endif
			val = std::abs(z_val.real()) + std::abs(z_val.imag());
			if(val < min_val){ index = i; min_val = val; }
		}
		#ifdef MATPTR
			ret = v[0][index];
		#else
			ret = v[index];
		#endif	
#ifdef CUDALA
	}else{
		const size_t pozice = cublasIzamin((size_t)nn*mm, (cuDoubleComplex*)v, 1) - 1;
		TEST_CUBLAS("cublasIzamin");
		gpuget(1, sizeof(complex<double>), v + pozice, &ret);
	}
#endif
	return ret;
}

/***************************************************************************//**
 * destructor for general type
 * @see NRMat<T>::count 
 ******************************************************************************/
template <typename T>
NRMat<T>::~NRMat() {
	if(!count) return;
	if(--(*count) <= 0) {
		if (v){
#ifdef CUDALA
			if(location == cpu){
#endif
			#ifdef MATPTR
				delete[] (v[0]);
			#endif
				delete[] v;
#ifdef CUDALA
			}else{
				gpufree(v);
			}
#endif
                }
                delete count;
        }
}

/***************************************************************************//**
 * assigment operator for general type between NRMat and NRMat
 * @see count 
 * @return reference to the newly assigned matrix 
 ******************************************************************************/
template <typename T>
NRMat<T> & NRMat<T>::operator=(const NRMat<T> &rhs) {
	if (this != &rhs){
		if (count){
			if (--(*count) ==0 ){
#ifdef CUDALA
				if(location == cpu){
#endif
				#ifdef MATPTR
					delete[] (v[0]);
				#endif
					delete[] v;
#ifdef CUDALA
				}else{ gpufree(v); }
#endif
				delete count;
                	}
		}
                v = rhs.v;
#ifdef CUDALA
                location = rhs.location;
#endif
                nn = rhs.nn;
                mm = rhs.mm;
                count = rhs.count;
                if(count) (*count)++;
	}
        return *this;
}


/***************************************************************************//**
 * perform an explicit deep copy of \c NRMat object
 * @see count 
 * @return reference to the newly copied matrix 
 ******************************************************************************/
template <typename T>
NRMat<T> & NRMat<T>::operator|=(const NRMat<T> &rhs) {
        if(this == &rhs) return *this; // test to avoid self-assignment
	*this = rhs;
	this->copyonwrite();
        return *this;
}

/***************************************************************************//**
 * create own deep copy
 * @see NRMat<T>::count, NRMat<T>::operator|=() 
 ******************************************************************************/
template <typename T>
void NRMat<T>::copyonwrite(bool detachonly) {
	if(!count) laerror("attempt to call copyonwrite() for a matrix with count == 0");
	if(*count > 1){
		(*count)--;
		count = new int;
		*count = 1;
#ifdef CUDALA
		if(location == cpu){ //matrix is in CPU memory
#endif
		#ifdef MATPTR
			T **newv = new T*[nn];
			newv[0] = new T[(size_t)mm*nn];
			if(!detachonly) memcpy(newv[0], v[0], (size_t)mm*nn*sizeof(T));
			v = newv;
			for(register int i=1; i<nn; i++) v[i] = v[i-1] + mm;
		#else
			T *newv = new T[(size_t)mm*nn];
			if(!detachonly) memcpy(newv, v, (size_t)mm*nn*sizeof(T));
			v = newv;
		#endif
#ifdef CUDALA
		}else{ //matrix is in GPU memory
			T *newv = (T *) gpualloc((size_t)mm*nn*sizeof(T));
			if(sizeof(T)%sizeof(float) != 0) laerror("cpu memcpy alignment problem");
			if(!detachonly) cublasScopy(nn*mm*sizeof(T)/sizeof(float), (const float *) v, 1, (float *)newv, 1);
			TEST_CUBLAS("cublasScopy");
			v = newv;
		}
#endif
        }
}

/***************************************************************************//**
 * resize given matrix
 * @param[in] n number of rows
 * @param[in] m number of cols
 * @see count, NRMat<T>::copyonwrite(), NRMat<T>::operator|=() 
 * @return reference to the newly copied matrix 
 ******************************************************************************/
template <typename T>
void NRMat<T>::resize(int n, int m) {
#ifdef DEBUG
	if (n<0 || m<0) laerror("illegal dimensions");
#endif

	//allow trivial dimensions
	if(n == 0 || m == 0) m = n =0;

	if(count){
		if(n==0 && m==0){
			if(--(*count) <= 0){
#ifdef CUDALA
				if(location==cpu){
#endif
				#ifdef MATPTR
					if(v) delete[] (v[0]);
				#endif
					if(v) delete[] v;
#ifdef CUDALA
				}
				else { gpufree(v); }
#endif
				delete count;
			}
			count = 0;
			nn = mm = 0;
			v = 0;
			return;
		}
		/*
		   if we have more than one reference to this matrix, set count to NULL
		   in order to reach the if-branch below where new memory resources are allocated
		*/
		if(*count > 1){
			(*count)--;
			count = 0;
			nn = mm = 0;
			v = 0;
		}
	}

	if(!count){
		 count = new int;
		*count = 1;
		nn = n;
                mm = m;
#ifdef CUDALA
		if(location==cpu){
#endif
		#ifdef MATPTR
			v = new T*[nn];
			v[0] = new T[(size_t)m*n];
			for (register int i=1; i< n; i++) v[i] = v[i-1] + m;
		#else
			v = new T[(size_t)m*n];
		#endif
#ifdef CUDALA
		}else{
			v = (T *) gpualloc((size_t)n*m*sizeof(T));
		}
#endif
                return;
        }

	// at this point *count = 1, check if resize is necessary
	if (n != nn || m != mm) {
		nn = n;
		mm = m;
#ifdef CUDALA
		if(location==cpu){
#endif
		#ifdef MATPTR
			delete[] (v[0]);
		#endif
			delete[] v;
		#ifdef MATPTR
			v = new T*[nn];
			v[0] = new T[(size_t)m*n];
			for (int i=1; i< n; i++) v[i] = v[i-1] + m;
		#else
			v = new T[(size_t)m*n];
		#endif
#ifdef CUDALA
		}else{
			gpufree(v);
			v=(T *) gpualloc((size_t)n*m*sizeof(T));
		}
#endif
	}
}


/***************************************************************************//**
 * complexify a given matrix \f$A\f$
 * @param[in] rhs matrix \f$A\f$ intended for this operation
 * @return matrix \f$B\f$ where \f$\Re B=A\f$ and \f$\Im B = 0\f$
 ******************************************************************************/
template<typename T>
NRMat<complex<T> > complexify(const NRMat<T> &rhs) {
	NOT_GPU(rhs);

	NRMat<complex<T> > r(rhs.nrows(), rhs.ncols(), rhs.getlocation());
	for(register int i=0; i<rhs.nrows(); ++i){ 
		for(register int j=0; j<rhs.ncols(); ++j) r(i,j) = rhs(i,j);
	}
	return r;
}

/***************************************************************************//**
 * output operator
 * @param[in,out] s output stream
 * @param[in] x NRMat matrix to be prited out
 * @return modified stream
 ******************************************************************************/
template <typename T>
std::ostream& operator<<(std::ostream &s, const NRMat<T> &x) {
#ifdef CUDALA
	if(x.getlocation() == cpu){
#endif
		int i(0),j(0);
		int n(x.nrows()), m(x.ncols());
		s << n << ' ' << m << '\n';
		for(i=0; i<n; i++){
			for(j=0; j<m; j++){
				// endl cannot be used in the conditional expression, since it is an overloaded function
				s << (typename LA_traits_io<T>::IOtype) x[i][j] << (j==m-1 ? '\n' : ' ');
			}
		}
		return s;
#ifdef CUDALA
	}else{
		NRMat<T> tmp = x;
		tmp.moveto(cpu);
		return s << tmp;
	}
#endif
}

/***************************************************************************//**
 * input operator
 * @param[in,out] s input stream
 * @param[in] x NRMat matrix for storing the input 
 * @return modified stream
 ******************************************************************************/
template <typename T>
std::istream& operator>>(std::istream  &s, NRMat<T> &x)
{
#ifdef CUDALA
	if(x.getlocation() == cpu){
#endif
		int i(0), j(0), n(0), m(0);
		s >> n >> m;
		x.resize(n, m);
		typename LA_traits_io<T>::IOtype tmp;
		for(i=0;i<n;i++){
			for(j=0; j<m;j++){
				s >> tmp;
				x[i][j] = tmp;
			}
		}
		return s;
#ifdef CUDALA
	}else{
		NRMat<T> tmp;
		tmp.moveto(cpu);
		s >> tmp;
		tmp.moveto(x.getlocation());
		x = tmp;
		return s;
	}
#endif
}

/***************************************************************************//**
 * implements \c NRMat<T> functionality with indexing from 1
 * all possible constructors have to be given explicitly, other stuff is inherited
 * with exception of the operator() which differs
 ******************************************************************************/
template<typename T>
class NRMat_from1 : public NRMat<T> {
public:
	NRMat_from1(): NRMat<T>() {};
	explicit NRMat_from1(const int n): NRMat<T>(n) {};
	NRMat_from1(const NRMat<T> &rhs): NRMat<T>(rhs) {};//!< be able to convert the parent class transparently to this
	NRMat_from1(const int n, const int m): NRMat<T>(n, m) {};
	NRMat_from1(const T &a, const int n, const int m): NRMat<T>(a, n, m) {};
	NRMat_from1(const T *a, const int n, const int m): NRMat<T>(a, n, m) {};

	inline const T& operator() (const int i, const int j) const {
	#ifdef DEBUG
		if (i<1 || i>NRMat<T>::nn || j<1 || j>NRMat<T>::mm) laerror("index out of range");
		if (!NRMat<T>::v) laerror("unallocated matrix");
	#endif
		NOT_GPU(*this);
	#ifdef MATPTR
		return NRMat<T>::v[i - 1][j - 1];
	#else
		return NRMat<T>::v[(i-1)*(size_t)NRMat<T>::mm+j-1];
	#endif
	}

	inline  T& operator() (const int i, const int j) {
	#ifdef DEBUG
		if (_LA_count_check && *NRMat<T>::count != 1) laerror("matrix with *count > 1 used as l-value");
	        if (i<1 || i>NRMat<T>::nn || j<1 || j>NRMat<T>::mm) laerror("index out of range");
		if (!NRMat<T>::v) laerror("unallocated matrix");
	#endif
		NOT_GPU(*this);
	#ifdef MATPTR
		return NRMat<T>::v[i-1][j-1];
	#else
		return NRMat<T>::v[(i-1)*NRMat<T>::mm+j-1];
	#endif
        }

	inline const T get_ij(const int i, const int j) const {
		T ret;
	#ifdef DEBUG
		if (i<1 || i>NRMat<T>::nn || j<1 || j>NRMat<T>::mm) laerror("index out of range");
		if (!NRMat<T>::v) laerror("unallocated matrix");
	#endif
	#ifdef CUDALA
		if(NRMat<T>::location == cpu){
	#endif
	#ifdef MATPTR
			return NRMat<T>::v[i - 1][j - 1];
	#else
			return NRMat<T>::v[(size_t)(i-1)*NRMat<T>::mm + (j-1)];
	#endif
	#ifdef CUDALA
		}else{
			const size_t pozice = (size_t)(i-1)*NRMat<T>::mm + (j-1);
			gpuget(1, sizeof(T), NRMat<T>::v + pozice, &ret);
			return ret;
		}
	#endif
	}
};

/***************************************************************************//**
 * compute Hadamard (component-wise) product with a given matrix \f$A\f$
 * @param[in] rhs matrix \f$A\f$
 * @see count, operator*
 * @return reference to the multiplied matrix
 ******************************************************************************/
template<typename T>
NRMat<T>& NRMat<T>::operator^=(const NRMat<T> &rhs){
#ifdef DEBUG
	if (nn != rhs.nn || mm != rhs.mm) laerror("incompatible matrices");
#endif
	SAME_LOC(*this, rhs);
	NOT_GPU(*this);

	copyonwrite();// ensure that *count == 1
#ifdef MATPTR
	for (register size_t i=0; i< (size_t)nn*mm; i++) v[0][i] *= rhs.v[0][i];
#else
	const size_t Dim = (size_t)nn*mm;
	for(register size_t i=0; i<Dim; i++) v[i] *= rhs.v[i];
#endif
	return *this;
}


/***************************************************************************//**
 * performs memory movements in CUDA mode
 * @param[in] dest memory destination
 * @see count, location
 ******************************************************************************/
#ifdef CUDALA
template<typename T>
void NRMat<T>::moveto(const GPUID dest) {
	if(location == dest) return;// no operation is necessary
	/*
	   currently, only movements between CPU and GPU are implemented
	   CUBLAS seems to lack support for multiple GPUs
	*/
	CPU_GPU(location, dest);
	location = dest;

	if(v && !count) laerror("internal inconsistency of reference counting 1");
	if (!count) return;

	if(v && *count==0) laerror("internal inconsistency of reference counting 2");
	if(!v) return;

	T *vold = v;

	if(dest == cpu){ //moving from GPU to CPU
		v = new T[(size_t)nn*mm];
		gpuget((size_t)nn*mm, sizeof(T), vold, v);
		if(*count == 1){ gpufree(vold); }
		else{ --(*count); count = new int(1); }

	}else{ //moving from CPU to GPU
		v = (T *) gpualloc((size_t)nn*mm*sizeof(T));
		gpuput((size_t)nn*mm, sizeof(T), vold, v);
		if(*count == 1) delete[] vold;
		else{ --(*count); count = new int(1);}
	}
}
#endif

/***************************************************************************//**
 * generate operators: Mat + a, a + Mat, Mat * a
 * corresponding macro is defined in vec.h
 ******************************************************************************/
NRVECMAT_OPER(Mat, +)
NRVECMAT_OPER(Mat, -)
NRVECMAT_OPER(Mat, *)

/***************************************************************************//**
 * generate Mat + Mat, Mat - Mat
 * corresponding macro is defined in vec.h
 ******************************************************************************/
NRVECMAT_OPER2(Mat, +)
NRVECMAT_OPER2(Mat, -)

}//end of the LA-namespace
#endif/* _LA_MAT_H_ */

