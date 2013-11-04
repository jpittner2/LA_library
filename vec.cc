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
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include "vec.h"
#include "qsort.h"
#include <unistd.h>


namespace LA {


/***************************************************************************//**
 * conversion constructor interpreting a given matrix with \f$N\f$ rows and 
 * \f$M\f$ columns of general type <code>T</code> as a vector of \f$N\times{}M\f$
 * elements
 * @param[in] rhs matrix being converted
 * @see NRMat<T>::NRMat()
 ******************************************************************************/
#ifndef MATPTR
template <typename T>
NRVec<T>::NRVec(const NRMat<T> &rhs) {
#ifdef CUDALA
	location = rhs.location;
#endif
	nn = rhs.nn*rhs.mm;
	v = rhs.v;
	count = rhs.count;
	(*count)++;
}
#endif

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

/***************************************************************************//**
 * routine for formatted output via lawritemat
 * @param[in] file pointer to <tt>FILE</tt> structure representing the output file
 * @param[in] format format specification in standard printf-like form
 * @param[in] modulo
 * @see lawritemat()
 ******************************************************************************/
template<typename T>
void NRVec<T>::fprintf(FILE *file, const char *format, const int modulo) const {
	NOT_GPU(*this);

	lawritemat(file, v, 1, nn, format, 1, modulo, 0);
}

/***************************************************************************//**
 * routine for formatted input via fscanf
 * @param[in] f pointer to <tt>FILE</tt> structure representing the input file
 * @param[in] format format specification in standard printf-like form
 ******************************************************************************/
template <typename T>
void NRVec<T>::fscanf(FILE *f, const char *format) {
	int n(0);
	NOT_GPU(*this);

	if(::fscanf(f, "%d", &n) != 1) laerror("can not read vector dimension");
	resize(n);
	for(register int i=0; i<n; i++){
		if (::fscanf(f, format, v + i) != 1){
			laerror("can not read the vector element");
		}
	}
}

/***************************************************************************//**
 * unary minus operator in case of real double-precision vector
 * @return the modified vector by value
 ******************************************************************************/
template<>
const NRVec<double> NRVec<double>::operator-() const {
	NRVec<double> result(*this);
	result.copyonwrite();
#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_dscal(nn, -1.0, v, 1);
#ifdef CUDALA
	}else{
		cublasDscal(nn, -1.0, v, 1);
		TEST_CUBLAS("cublasDscal");
	}
#endif
	return result;
}

/***************************************************************************//**
 * unary minus operator in case of complex double-precision vector
 * @return the modified vector by value
 ******************************************************************************/
template<>
const NRVec<complex<double> > NRVec<complex<double> >::operator-() const {
	NRVec<complex<double> > result(*this);
	result.copyonwrite();
#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_zdscal(nn, -1.0, v, 1);
#ifdef CUDALA
	}else{
		cublasZdscal(nn, -1.0, (cuDoubleComplex*)v, 1);
		TEST_CUBLAS("cublasZdscal");
	}
#endif
	return result;
}

/***************************************************************************//**
 * unary minus operator for vector of general type
 * @return the modified vector
 ******************************************************************************/
template <typename T>
const NRVec<T> NRVec<T>::operator-() const {
	NOT_GPU(*this);
	NRVec<T> result(nn, getlocation());
	for(register int i=0; i<nn; i++) result.v[i] = -v[i];
	return result;
}

/***************************************************************************//**
 * comparison operator (lexicographical order)
 * @param[in] rhs vector intended for comparison
 * @return 
 *	\li \c true current vector is bigger than vector \c rhs
 *	\li \c false current vector is smaller than vector \c rhs
 ******************************************************************************/
template <typename T>
const bool NRVec<T>::operator>(const NRVec &rhs) const {
	int n(nn);

	SAME_LOC(*this, rhs);
	NOT_GPU(*this);

	if(rhs.nn < n) n = rhs.nn;
	for(register int i=0; i<n;++i){
		if(LA_traits<T>::bigger(v[i], rhs.v[i])) return true;
		if(LA_traits<T>::smaller(v[i], rhs.v[i])) return false;
	}
	return nn>rhs.nn;
}

/***************************************************************************//**
 * comparison operator (lexicographical order)
 * @param[in] rhs vector intended for comparison
 * @return 
 *	\li \c false current vector is bigger than vector \c rhs
 *	\li \c true current vector is smaller than vector \c rhs
 ******************************************************************************/
template <typename T>
const bool NRVec<T>::operator<(const NRVec &rhs) const {
	int n(nn);
	
	SAME_LOC(*this, rhs);
	NOT_GPU(*this);

	if(rhs.nn < n) n = rhs.nn;
	for(register int i=0; i<n;++i){
		if(LA_traits<T>::smaller(v[i], rhs.v[i])) return true;
		if(LA_traits<T>::bigger(v[i], rhs.v[i])) return false;
	}
	return nn<rhs.nn;
}

/***************************************************************************//**
 * fill the real vector with pseudorandom numbers generated using uniform distribution
 * @param[in] x specification of the interval \f$[0,x]\f$ for the random number generator
 ******************************************************************************/
template<>
void NRVec<double>::randomize(const double &x){
	NOT_GPU(*this);

	for(register int i=0; i<nn; ++i){
		v[i] = x*(2.*random()/(1. + RAND_MAX) - 1.);
	}
}

/***************************************************************************//**
 * fill the complex vector with pseudorandom numbers generated using uniform distribution
 * the real and imaginary parts are generated independently
 * @param[in] x specification of the interval \f$[0,x]\f$ for the random number generator
 * @return 
 *	\li \c false current vector is bigger than vector \c rhs
 *	\li \c true current vector is smaller than vector \c rhs
 ******************************************************************************/
template<>
void NRVec<complex<double> >::randomize(const double &x) {
	NOT_GPU(*this);

	for(register int i=0; i<nn; ++i){
		v[i] = complex<double>(x*(2.*random()/(1. + RAND_MAX) - 1.), x*(2.*random()/(1. + RAND_MAX) - 1.));
	}
}

/***************************************************************************//**
 * constructor creating complex vector from a real one
 * @param[in] rhs the real vector being converted into the complex one
 * @param[in] imagpart 
 *	\li \c true vector \c rhs is interpreted as the imaginary part of the new complex vector
 *	\li \c false vector \c rhs is interpreted as the real part of the new complex vector
 * @return 
 *	\li \c false current vector is bigger than vector \c rhs
 *	\li \c true current vector is smaller than vector \c rhs
 ******************************************************************************/
template<>
NRVec<complex<double> >::NRVec(const NRVec<double> &rhs, bool imagpart): nn(rhs.size()){

	 count = new int;
	*count = 1;
#ifdef CUDALA
	location = rhs.getlocation();
	if(location == cpu){
#endif
		v = (complex<double>*)new complex<double>[nn];
		memset(v, 0, nn*sizeof(complex<double>));
		cblas_dcopy(nn, &rhs[0], 1, ((double *)v) + (imagpart?1:0), 2);
#ifdef CUDALA
	}else{
		v = (complex<double>*) gpualloc(nn*sizeof(complex<double>));

		cublasZscal(nn, CUZERO, (cuDoubleComplex*)v, 1);
		TEST_CUBLAS("cublasZscal");

		cublasDcopy(nn, &rhs[0], 1, ((double *)v) + (imagpart?1:0), 2);
		TEST_CUBLAS("cublasDcopy");
	}
#endif
}

/***************************************************************************//**
 * perform the <b>axpy</b> operation on the current real vector \f$\vec{v}\f$, i.e.
 * \f[ \vec{v} \leftarrow \vec{v} + \alpha\vec{x} \f]
 * @param[in] alpha double-precision real parameter \f$\alpha\f$
 * @param[in] x double-precision real vector \f$\vec{x}\f$
 ******************************************************************************/
template<>
void NRVec<double>::axpy(const double alpha, const NRVec<double> &x) {
#ifdef DEBUG
	if (nn != x.nn) laerror("incompatible vectors");
#endif
	SAME_LOC(*this, x);
	copyonwrite();

#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_daxpy(nn, alpha, x.v, 1, v, 1);
#ifdef CUDALA
	}else{
		cublasDaxpy(nn, alpha, x.v, 1, v, 1);
		TEST_CUBLAS("cublasDaxpy");
	}
#endif
}

/***************************************************************************//**
 * perform the <b>axpy</b> operation on the current complex vector \f$\vec{v}\f$, i.e.
 * \f[ \vec{v} \leftarrow \vec{v} + \alpha\vec{x} \f]
 * @param[in] alpha \f$\alpha\f$ parameter
 * @param[in] x complex vector \f$\vec{x}\f$
 ******************************************************************************/
template<>
void NRVec<complex<double> >::axpy(const complex<double> alpha, const NRVec<complex<double> > &x){
#ifdef DEBUG
	if (nn != x.nn) laerror("incompatible vectors");
#endif
	SAME_LOC(*this, x);
	copyonwrite();

#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_zaxpy(nn, &alpha, x.v, 1, v, 1);
#ifdef CUDALA
	}else{
		const cuDoubleComplex _alpha = make_cuDoubleComplex(alpha.real(), alpha.imag());
		cublasZaxpy(nn, _alpha, (cuDoubleComplex*)x.v, 1, (cuDoubleComplex*)v, 1);
		TEST_CUBLAS("cublasZaxpy");
	}
#endif

}

/***************************************************************************//**
 * perform the <b>axpy</b> operation on the current real vector \f$\vec{v}\f$, i.e.
 * \f[ \vec{v} \leftarrow \vec{v} + \alpha\vec{x} \f]
 * @param[in] alpha \f$\alpha\f$ parameter
 * @param[in] x pointer to double-precision real data
 * @param[in] stride sets the stride
 ******************************************************************************/
template<>
void NRVec<double>::axpy(const double alpha, const double *x, const int stride){
	NOT_GPU(*this);

	copyonwrite();
	cblas_daxpy(nn, alpha, x, stride, v, 1);
}

/***************************************************************************//**
 * perform the <b>axpy</b> operation on the current complex vector \f$\vec{v}\f$, i.e.
 * \f[ \vec{v} \leftarrow \vec{v} + \alpha\vec{x} \f]
 * @param[in] alpha double-precision complex parameter \f$\alpha\f$
 * @param[in] x pointer to double-precision complex data
 * @param[in] stride sets the stride
 ******************************************************************************/
template<>
void NRVec<complex<double> >::axpy(const complex<double> alpha, const complex<double> *x, const int stride){
	NOT_GPU(*this);

	copyonwrite();
	cblas_zaxpy(nn, &alpha, x, stride, v, 1);
}

/***************************************************************************//**
 * assign real scalar value to every element of the current vector
 * @param[in] a scalar value to be assigned
 * @return reference to the modified vector
 ******************************************************************************/
template<>
NRVec<double>& NRVec<double>::operator=(const double &a){
copyonwrite();

#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_dcopy(nn, &a, 0, v, 1);
#ifdef CUDALA
	}else{
		smart_gpu_set(nn, (double)0, v);
	}
#endif	
	return *this;
}

/***************************************************************************//**
 * assign complex scalar value to every element of the current vector
 * @param[in] a scalar value to be assigned
 * @return reference to the modified vector
 ******************************************************************************/
template<>
NRVec<complex<double> >& NRVec<complex<double> >::operator=(const complex<double> &a){
copyonwrite();

#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_zcopy(nn, &a, 0, v, 1);
#ifdef CUDALA
	}else{
		smart_gpu_set(nn, (complex<double>)0, v);
	}
#endif	
	return *this;
}

/***************************************************************************//**
 * assign scalar value to every element of the current vector of general type <code>T</code>
 * @param[in] a scalar value to be assigned
 * @return reference to the modified vector
 ******************************************************************************/
template <typename T>
NRVec<T>& NRVec<T>::operator=(const T &a){
	NOT_GPU(*this);
	copyonwrite();

	if(a != (T)0){
		for(register int i=0; i<nn; i++) v[i] = a;
	}else{
		memset(v, 0, nn*sizeof(T));
	}
	return *this;
}

/***************************************************************************//**
 * normalize current real vector (in the Euclidean norm)
 * @param[in] norm if not NULL, the norm of this vector is stored into *norm
 * @return reference to the modified vector
 ******************************************************************************/
template<>
NRVec<double>& NRVec<double>::normalize(double *norm){
	double tmp(0.0);
#ifdef CUDALA
	if(location == cpu){
#endif
		tmp = cblas_dnrm2(nn, v, 1);
		if(norm) *norm = tmp;
		#ifdef DEBUG
			if(!tmp) laerror("attempt to normalize zero vector");
		#endif
		copyonwrite();
		tmp = 1.0 / tmp;
		cblas_dscal(nn, tmp, v, 1);
#ifdef CUDALA
	}else{
		tmp = cublasDnrm2(nn, v, 1);
		TEST_CUBLAS("cublasDnrm2");

		if(norm) *norm = tmp;
		#ifdef DEBUG
			if(!tmp) laerror("attempt to normalize zero vector");
		#endif
		copyonwrite();
		tmp = 1.0 / tmp;
		cublasDscal(nn, tmp, v, 1);
		TEST_CUBLAS("cublasDscal");
	}
#endif
	return *this;
}

/***************************************************************************//**
 * normalize current complex vector (in the Euclidean norm)
 * @param[in] norm if not NULL, the norm of this vector is stored into *norm
 * @return reference to the modified vector
 ******************************************************************************/
template<>
NRVec<complex<double> > & NRVec<complex<double> >::normalize(double *norm){
	double tmp(0.0);
#ifdef CUDALA
	if(location == cpu){
#endif
		tmp = cblas_dznrm2(nn, v, 1);
		if(norm) *norm = tmp;
		#ifdef DEBUG
		if(tmp == 0.0) laerror("attempt to normalize zero vector");
		#endif
		copyonwrite();
		tmp = 1.0 / tmp;
		cblas_zdscal(nn, tmp, v, 1);
#ifdef CUDALA
	}else{
		tmp = cublasDznrm2(nn, (cuDoubleComplex*)v, 1);
		TEST_CUBLAS("cublasDznrm2");

		if(norm) *norm = tmp;
		#ifdef DEBUG
			if(tmp == 0.0) laerror("attempt to normalize zero vector");
		#endif
		copyonwrite();

		tmp = 1.0 / tmp;
		cublasZdscal(nn, tmp, (cuDoubleComplex*)v, 1);
		TEST_CUBLAS("cublasZdscal");
	}
#endif		
	return *this;
}

/***************************************************************************//**
 * perform the \b gemv operation on this real vector \f$\vec{y}\f$, i.e.
 * \f[\vec{y}\leftarrow \alpha\operatorname{op}(A)\cdot\vec{x}+\beta\vec{y}\f]
 * @param[in] beta real parameter \f$\beta\f$
 * @param[in] A real matrix \f$A\f$
 * @param[in] trans if <code>trans == 'n'</code> use \f$A\f$ directly, otherwise \f$\operatorname{op}(A)\equiv{}A^\mathrm{T}\f$
 * @param[in] alpha real parameter \f$\alpha\f$
 * @param[in] x real vector \f$\vec{x}\f$
 * @see NRMat<T>::gemm
 ******************************************************************************/
template<>
void NRVec<double>::gemv(const double beta, const NRMat<double> &A, 
		const char trans, const double alpha, const NRVec &x) {
#ifdef DEBUG
	if((tolower(trans) == 'n'?A.ncols():A.nrows()) != x.size()){ laerror("incompatible vectors"); }
#endif
	SAME_LOC3(*this, x, A);
	copyonwrite();

#ifdef CUDALA
	if(location==cpu){
#endif
		cblas_dgemv(CblasRowMajor, (tolower(trans)=='n' ? CblasNoTrans:CblasTrans), A.nrows(), A.ncols(), alpha, A, A.ncols(), x.v, 1, beta, v, 1);
#ifdef CUDALA
	}else{
		cublasDgemv((tolower(trans)=='n'?'T':'N'), A.ncols(), A.nrows(), alpha, A, A.ncols(), x.v, 1, beta, v, 1);
		TEST_CUBLAS("cublasDgemv");
	}
#endif
}

/***************************************************************************//**
 * perform the \b gemv operation on this complex vector \f$\vec{y}\f$, i.e.
 * \f[\vec{y}\leftarrow \alpha\operatorname{op}(A)\cdot\vec{x}+\beta\vec{y}\f]
 * @param[in] beta real parameter \f$\beta\f$
 * @param[in] A <b>real</b> matrix \f$A\f$
 * @param[in] trans if <tt>trans == 'n'</tt> use \f$A\f$ directly, otherwise \f$\operatorname{op}(A)\equiv{}A^\mathrm{T}\f$
 * @param[in] alpha real parameter \f$\alpha\f$
 * @param[in] x real vector \f$\vec{x}\f$
 * @see gemm
 ******************************************************************************/
template<>
void NRVec<complex<double> >::gemv(const double beta, const NRMat<double> &A,
                const char trans, const double alpha, const NRVec<complex<double> > &x) {
#ifdef DEBUG
	if ((tolower(trans) == 'n'?A.ncols():A.nrows()) != x.size()){ laerror("incompatible vectors"); }
#endif
	SAME_LOC3(*this, x, A);
	copyonwrite();

#ifdef CUDALA
	if(location==cpu){
#endif
		cblas_dgemv(CblasRowMajor, (tolower(trans)=='n'?CblasNoTrans:CblasTrans),
				A.nrows(), A.ncols(), alpha, A, A.ncols(), (double *)x.v, 2, beta, (double *)v, 2);
		cblas_dgemv(CblasRowMajor, (tolower(trans)=='n'?CblasNoTrans:CblasTrans),
				A.nrows(), A.ncols(), alpha, A, A.ncols(), ((double *)x.v) + 1, 2, beta, ((double *)v)+1, 2);
#ifdef CUDALA
	}else{
		cublasDgemv((tolower(trans)=='n'?'T':'N'), A.ncols(), A.nrows(), alpha, A, A.ncols(), (double*)(x.v), 2, beta, (double *)v, 2);
		TEST_CUBLAS("cublasDgemv");

		cublasDgemv((tolower(trans)=='n'?'T':'N'), A.ncols(), A.nrows(), alpha, A, A.ncols(), ((double *)x.v) + 1, 2, beta, ((double *)v)+1, 2);
		TEST_CUBLAS("cublasDgemv");
	}
#endif
}


/***************************************************************************//**
 * perform the \b gemv operation on this complex vector \f$\vec{y}\f$, i.e.
 * \f[\vec{y}\leftarrow \alpha\operatorname{op}(A)\cdot\vec{x}+\beta\vec{y}\f]
 * @param[in] beta complex parameter \f$\beta\f$
 * @param[in] A <b>complex</b> matrix \f$A\f$
 * @param[in] trans if <code>trans == 'n'</code> use \f$A\f$ directly, otherwise \f$\operatorname{op}(A)\equiv{}A^\mathrm{T}\f$
 * @param[in] alpha complex parameter \f$\alpha\f$
 * @param[in] x real vector \f$\vec{x}\f$
 * @see gemm
 ******************************************************************************/
template<>
void NRVec<complex<double> >::gemv(const complex<double> beta,
		const NRMat<complex<double> > &A, const char trans, 
		const complex<double> alpha, const NRVec<complex<double> > &x) {
#ifdef DEBUG
	if ((tolower(trans) == 'n'?A.ncols():A.nrows()) != x.size()){ laerror("incompatible vectors"); }
#endif
	SAME_LOC3(*this, x, A);
	copyonwrite();
#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_zgemv(CblasRowMajor, (tolower(trans)=='n'?CblasNoTrans:CblasTrans), 
				A.nrows(), A.ncols(), &alpha, A, A.ncols(), x.v, 1, &beta, v, 1);
#ifdef CUDALA
	}else{

		const cuDoubleComplex _alpha = make_cuDoubleComplex(alpha.real(), alpha.imag());
		const cuDoubleComplex  _beta = make_cuDoubleComplex(beta.real(), beta.imag());

		cublasZgemv((tolower(trans)=='n'?'T':'N'), A.ncols(), A.nrows(),
				_alpha, (cuDoubleComplex*)(A[0]), A.ncols(), (cuDoubleComplex*)(x.v), 1, _beta, (cuDoubleComplex*)v, 1);
		TEST_CUBLAS("cublasZgemv");
	}
#endif
}

/***************************************************************************//**
 * perform the \b gemv operation on this real vector \f$\vec{y}\f$, i.e.
 * \f[\vec{y}\leftarrow \alpha\operatorname{op}(A)\cdot\vec{x}+\beta\vec{y}\f]
 * @param[in] beta real parameter \f$\beta\f$
 * @param[in] A real symmetric matrix \f$A\f$ stored in packed form
 * @param[in] trans if <code>trans == 'n'</code> use \f$A\f$ directly, otherwise \f$\operatorname{op}(A)\equiv{}A^\mathrm{T}\f$
 * @param[in] alpha real parameter \f$\alpha\f$
 * @param[in] x real vector \f$\vec{x}\f$
 * @see gemm, NRSMat<T>
 ******************************************************************************/
template<>
void NRVec<double>::gemv(const double beta, const NRSMat<double> &A,
		const char trans, const double alpha, const NRVec &x) {
#ifdef DEBUG
	if(A.ncols() != x.size()){ laerror("incompatible dimensions"); }
#endif
	SAME_LOC3(*this, A, x);
	copyonwrite();

#ifdef CUDALA
	if(location==cpu){
#endif
		cblas_dspmv(CblasRowMajor, CblasLower, A.ncols(), alpha, A, x.v, 1, beta, v, 1);
#ifdef CUDALA
	}else{
		cublasDspmv('U', A.ncols(), alpha, A, x.v, 1, beta, v, 1);
		TEST_CUBLAS("cublasDspmv");
	}
#endif
}


/***************************************************************************//**
 * perform the \c gemv operation on this complex vector \f$\vec{y}\f$, i.e.
 * \f[\vec{y}\leftarrow \alpha\operatorname{op}(A)\cdot\vec{x}+\beta\vec{y}\f]
 * @param[in] beta real parameter \f$\beta\f$
 * @param[in] A <b>real symmetric</b> matrix \f$A\f$ stored in packed form
 * @param[in] trans if <code>trans == 'n'</code> use \f$A\f$ directly, otherwise \f$\operatorname{op}(A)\equiv{}A^\mathrm{T}\f$
 * @param[in] alpha real parameter \f$\alpha\f$
 * @param[in] x complex vector \f$\vec{x}\f$
 * @see gemm, NRSMat<T>
 ******************************************************************************/
template<>
void NRVec<complex<double> >::gemv(const double beta, const NRSMat<double> &A,
		const char trans, const double alpha, const NRVec<complex<double> > &x) {
#ifdef DEBUG
	if(A.ncols() != x.size()){ laerror("incompatible dimensions"); }
#endif
	SAME_LOC3(*this, A, x);
	copyonwrite();

#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_dspmv(CblasRowMajor, CblasLower, A.ncols(), alpha, A, (double *)x.v, 2, beta, (double *)v, 2);
		cblas_dspmv(CblasRowMajor, CblasLower, A.ncols(), alpha, A, ((double *)x.v) + 1, 2, beta, ((double *)v) + 1, 2);
#ifdef CUDALA
	}else{
		cublasDspmv('U', A.ncols(), alpha, A, (double*)(x.v), 2, beta, (double*)v, 2);
		TEST_CUBLAS("cublasDspmv");
		
		cublasDspmv('U', A.ncols(), alpha, A, ((double*)(x.v)) + 1, 2, beta, ((double*)v) + 1, 2);
		TEST_CUBLAS("cublasDspmv");
	}
#endif
}

/***************************************************************************//**
 * perform the \b gemv operation on this complex vector \f$\vec{y}\f$, i.e.
 * \f[\vec{y}\leftarrow \alpha\operatorname{op}(A)\cdot\vec{x}+\beta\vec{y}\f]
 * @param[in] beta complex parameter \f$\beta\f$
 * @param[in] A <b>complex Hermitian</b> matrix \f$A\f$ stored in packed form
 * @param[in] trans not used
 * @param[in] alpha complex parameter \f$\alpha\f$
 * @param[in] x complex vector \f$\vec{x}\f$
 * @see gemm, NRSMat<T>
 ******************************************************************************/
template<>
void NRVec<complex<double> >::gemv(const complex<double> beta,
                const NRSMat<complex<double> > &A, const char trans,
                const complex<double> alpha, const NRVec<complex<double> > &x){
#ifdef DEBUG
	if(A.ncols() != x.size()) laerror("incompatible dimensions");
#endif
	SAME_LOC3(*this, A, x);
	copyonwrite();

#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_zhpmv(CblasRowMajor, CblasLower, A.ncols(), &alpha, A, x.v, 1, &beta, v, 1);
#ifdef CUDALA
	}else{
		const cuDoubleComplex _alpha = make_cuDoubleComplex(alpha.real(), alpha.imag());
		const cuDoubleComplex  _beta = make_cuDoubleComplex(beta.real(), beta.imag());

		cublasZhpmv('U', A.ncols(), _alpha, (cuDoubleComplex*)((const complex<double>*)A), (cuDoubleComplex*)(x.v), 1, _beta, (cuDoubleComplex*)(this->v), 1);
		TEST_CUBLAS("cublasZhpmv");
	}
#endif
}

/***************************************************************************//**
 * computes the outer product of this real vector \f$\vec{a}\f$ with given
 * real vector \f$\vec{b}\f$ and scales the resulting matrix with factor \f$\alpha\f$, i.e.
 * the matrix elements of the final matrix \f$A\f$ can be expressed as
 * \f[A_{i,j} = \alpha\cdot\vec{a}_i\vec{b}_j\f]
 * @param[in] b real vector \f$\vec{b}\f$
 * @param[in] conj not used
 * @param[in] scale real factor \f$\alpha\f$
 ******************************************************************************/
template<>
const NRMat<double> NRVec<double>::otimes(const NRVec<double> &b,const bool conj, const double &scale) const {

	SAME_LOC(*this, b);
	NRMat<double> result(0.0, nn, b.nn, this->getlocation());
#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_dger(CblasRowMajor, nn, b.nn, scale, v, 1, b.v, 1, result, b.nn);
#ifdef CUDALA
	}else{
		cublasDger(b.nn, nn, scale, b.v, 1, v, 1, result[0], b.nn);
		TEST_CUBLAS("cublasDger");
	}
#endif
	return result;
}

/***************************************************************************//**
 * computes the outer product of this complex vector \f$\vec{a}\f$ with given
 * complex vector \f$\vec{b}\f$ and scales the resulting matrix with factor \f$\alpha\f$, i.e.
 * the matrix elements of the final matrix \f$A\f$ can be expressed as
 * \f[A_{i,j} = \alpha\cdot\vec{a}_i\vec{b}_j\f]
 * in case <code>conj = true</code>, the result is
 * \f[A_{i,j} = \alpha\cdot\vec{a}_i\vec{b}_j^{*}\f]
 * @param[in] b complex vector \f$\vec{b}\f$
 * @param[in] conj determines whther the vector \f$\vec{b}\f$ is conjugated
 * @param[in] scale complex scaling factor \f$\alpha\f$
 ******************************************************************************/
template<>
const NRMat<complex<double> > 
NRVec<complex<double> >::otimes(const NRVec<complex<double> > &b, const bool conj, const complex<double> &scale) const {
	
	SAME_LOC(*this, b);
	NRMat<complex<double> > result(0., nn, b.nn, this->getlocation());

#ifdef CUDALA
	if(location == cpu){
#endif
		if(conj){
			cblas_zgerc(CblasRowMajor, nn, b.nn, &scale, v, 1, b.v, 1, result, b.nn);
		}else{
			cblas_zgeru(CblasRowMajor, nn, b.nn, &scale, v, 1, b.v, 1, result, b.nn);
		}
#ifdef CUDALA
	}else{
		if(conj){
			const cuDoubleComplex alpha = make_cuDoubleComplex(scale.real(), -scale.imag());

			cublasZgerc(b.nn, nn, alpha, (cuDoubleComplex*)(b.v), 1, (cuDoubleComplex*)(v), 1, (cuDoubleComplex*)(result[0]), b.nn);
			TEST_CUBLAS("cublasZgerc");

			result.conjugateme();
		}else{
			const cuDoubleComplex alpha = make_cuDoubleComplex(scale.real(), +scale.imag());

			cublasZgeru(b.nn, nn, alpha, (cuDoubleComplex*)(b.v), 1, (cuDoubleComplex*)(v), 1, (cuDoubleComplex*)(result[0]), b.nn);
			TEST_CUBLAS("cublasZgeru");
		}
	}
#endif
	return result;
}

template<typename T>
int NRVec<T>::sort(int direction, int from, int to, int *perm) {
	NOT_GPU(*this);

	copyonwrite();
	if(to == -1) to = nn - 1;
	if(direction) return memqsort<1, NRVec<T>, int, int>(*this, perm, from, to);
	else return memqsort<0, NRVec<T>, int, int>(*this, perm, from, to);
}

template<>
NRVec<complex<double> > complexify(const NRVec<double> &rhs) {
	NRVec<complex<double> > r(rhs.size(), rhs.getlocation());

#ifdef CUDALA
	if(rhs.getlocation() == cpu){
#endif
		cblas_dcopy(rhs.size(), &rhs[0], 1, (double *)(&r[0]), 2);
#ifdef CUDALA
	}else{
		r = 0;
		cublasDcopy(rhs.size(), rhs.v, 1, (double*)(r.v), 2);
		TEST_CUBLAS("cublasDcopy");
	}
#endif
	return r;
}

/***************************************************************************//**
 * forced instantization in the corespoding object file
 ******************************************************************************/
/*
 Commented out by Roman for ICC

#define INSTANTIZE(T) \
template void NRVec<T>::put(int fd, bool dim, bool transp) const; \
template void NRVec<T>::get(int fd, bool dim, bool transp); \

INSTANTIZE(double)
INSTANTIZE(complex<double>)
INSTANTIZE(char)
INSTANTIZE(short)
INSTANTIZE(int)
INSTANTIZE(long)
INSTANTIZE(long long)
INSTANTIZE(unsigned char)
INSTANTIZE(unsigned short)
INSTANTIZE(unsigned int)
INSTANTIZE(unsigned long)
INSTANTIZE(unsigned long long)
*/

#define INSTANTIZE_DUMMY(T) \
template<> void NRVec<T>::gemv(const T beta, const NRMat<T> &a, const char trans,  const T alpha, const NRVec<T> &x) { laerror("gemv on unsupported types"); } \
template<> void NRVec<T>::gemv(const T beta, const NRSMat<T> &a, const char trans,  const T alpha, const NRVec<T> &x) { laerror("gemv on unsupported types"); } \
template<> void NRVec<T>::gemv(const T beta, const SparseMat<T> &a, const char trans,  const T alpha, const NRVec<T> &x, bool s) { laerror("gemv on unsupported types"); } \
template<> void NRVec<T>::gemv(const LA_traits_complex<T>::Component_type beta, const  LA_traits_complex<T>::NRMat_Noncomplex_type  &a, const char trans,  const  LA_traits_complex<T>::Component_type alpha, const NRVec<T> &x) { laerror("gemv on unsupported types"); } \
template<> void NRVec<T>::gemv(const  LA_traits_complex<T>::Component_type beta, const  LA_traits_complex<T>::NRSMat_Noncomplex_type  &a, const char trans,  const  LA_traits_complex<T>::Component_type alpha, const NRVec<T> &x) { laerror("gemv on unsupported types"); } \
template<> NRVec<T> & NRVec<T>::normalize(LA_traits<T>::normtype *) {laerror("normalize() impossible for integer types"); return *this;} \
template<> const NRMat<T> NRVec<T>::otimes(const NRVec<T> &b,const bool conj, const T &scale) const {laerror("otimes presently implemented only for double and complex double"); return NRMat<T> ();}

// Roman
// following gemv are not implemented
template<> void NRVec<double>::gemv(const double beta, const SparseMat<double> &a, const char trans,  const double alpha, const NRVec<double> &x, bool s) { laerror("gemv on unsupported types"); } 
template<> void NRVec< complex<double> >::gemv(const complex<double> beta, const SparseMat< complex<double> > &a, const char trans,  const complex<double> alpha, const NRVec< complex<double> > &x, bool s) { laerror("gemv on unsupported types"); } 


INSTANTIZE_DUMMY(char)
INSTANTIZE_DUMMY(short)
INSTANTIZE_DUMMY(int)
INSTANTIZE_DUMMY(long)
INSTANTIZE_DUMMY(long long)
INSTANTIZE_DUMMY(unsigned char)
INSTANTIZE_DUMMY(unsigned short)
INSTANTIZE_DUMMY(unsigned int)
INSTANTIZE_DUMMY(unsigned long)
INSTANTIZE_DUMMY(unsigned long long)
INSTANTIZE_DUMMY(complex<char>)
INSTANTIZE_DUMMY(complex<short>)
INSTANTIZE_DUMMY(complex<int>)
INSTANTIZE_DUMMY(complex<long>)
INSTANTIZE_DUMMY(complex<long long>)
INSTANTIZE_DUMMY(complex<unsigned char>)
INSTANTIZE_DUMMY(complex<unsigned short>)
INSTANTIZE_DUMMY(complex<unsigned int>)
INSTANTIZE_DUMMY(complex<unsigned long>)
INSTANTIZE_DUMMY(complex<unsigned long long>)
INSTANTIZE_DUMMY(complex<complex<double> >)
INSTANTIZE_DUMMY(complex<complex<float> >)

template class NRVec<double>;
template class NRVec<complex<double> >;
template class NRVec<char>;
template class NRVec<short>;
template class NRVec<int>;
template class NRVec<long>;
template class NRVec<long long>;
template class NRVec<unsigned char>;
template class NRVec<unsigned short>;
template class NRVec<unsigned int>;
template class NRVec<unsigned long>;
template class NRVec<unsigned long long>;

}//namespace
