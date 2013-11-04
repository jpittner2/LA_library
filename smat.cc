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

#include "smat.h"
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>


namespace LA {

/***************************************************************************//**
 * routine for raw output 
 * @param[in] fd file descriptor for output
 * @param[in] dim number of elements intended for output
 * @param[in] transp reserved
 * @see NRMat<T>::get(), NRSMat<T>::copyonwrite()
 ******************************************************************************/
template <typename T>
void NRSMat<T>::put(int fd, bool dim, bool transp) const {
#ifdef CUDALA
	if(location != cpu){
		NRSMat<T> tmp= *this;
		tmp.moveto(cpu);
		tmp.put(fd,dim,transp);
		return;
	}
#endif
	errno = 0;
	if(dim){
		if(sizeof(int) != write(fd,&nn,sizeof(int))) laerror("cannot write");
		if(sizeof(int) != write(fd,&nn,sizeof(int))) laerror("cannot write");
	}
	LA_traits<T>::multiput((size_t)nn*(nn+1)/2,fd,v,dim);
}

/***************************************************************************//**
 * routine for raw input 
 * @param[in] fd file descriptor for input
 * @param[in] dim number of elements intended for input
 * @param[in] transp reserved
 * @see NRSMat<T>::put(), NRSMat<T>::copyonwrite()
 ******************************************************************************/
template <typename T>
void NRSMat<T>::get(int fd, bool dim, bool transp) {
#ifdef CUDALA
	if(location != cpu){
		NRSMat<T> tmp;
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
		if(2*sizeof(int) != read(fd,&nn0,2*sizeof(int))) laerror("cannot read");
		resize(nn0[0]);
	}else{
		copyonwrite();
	}
	LA_traits<T>::multiget((size_t)nn*(nn+1)/2,fd,v,dim);
}


/***************************************************************************//**
 * constructor symmetrizing given matrix \f$A\f$ of general type <code>T</code> yielding \f$(A+A^\mathrm{T})/2\f$
 * @param[in] rhs matrix \f$A\f$
 ******************************************************************************/
template <typename T>
NRSMat<T>::NRSMat(const NRMat<T> &rhs) {
	NOT_GPU(rhs);

	nn = rhs.nrows();
#ifdef DEBUG
	if(nn != rhs.ncols()) laerror("attempt to convert nonsquare NRMat<T> to NRSMat<T>");
#endif
#ifdef CUDALA
	location = rhs.getlocation();
#endif
	count = new int;
	*count = 1;
	v = new T[NN2];
	int i, j, k(0);
	for(i=0; i<nn; i++){
		for(j=0; j<=i; j++){
			v[k++] = (rhs[i][j] + rhs[j][i])/((T)2);
		}
	}
}

/***************************************************************************//**
 * zero out this symmetric matrix of general type <code>T</code> and then set
 * the diagonal elements to prescribed value
 * @param[in] a scalar value to be assigned to the diagonal
 * @return reference to the modified matrix
 ******************************************************************************/
template <typename T>
NRSMat<T> & NRSMat<T>::operator=(const T &a) {
	NOT_GPU(*this);
	copyonwrite();
	memset(v, 0, NN2*sizeof(T));
	for(register int i=0; i<nn; i++) v[(size_t)i*(i+1)/2 + i] = a;
	return *this;
}

/***************************************************************************//**
 * get or divide by the diagonal of real symmetric double-precision matrix
 * @param[in, out] r vector for storing the diagonal
 * @param[in] divide
 * 	\li \c false save the diagonal to vector r
 *	\li \c true divide the vector r by the diagonal elements element-wise
 * @param[in] cache reserved
 * @return
 * 	\li <tt>divide == true</tt> NULL
 *	\li <tt>divide == false</tt> pointer to the first element of r
 ******************************************************************************/
template <typename T>
const T* NRSMat<T>::diagonalof(NRVec<T> &r, const bool divide, bool cache) const {
#ifdef DEBUG
	if(r.size() != nn) laerror("incompatible vector in const T* NRSMat<T>::diagonalof(NRVec<T> &, const bool, bool)");
#endif
	NOT_GPU(*this);
	SAME_LOC(*this, r);

	r.copyonwrite();

	if(divide){
		for(register int i=0; i<nn; i++){
			const T a = v[(size_t)i*(i+1)/2+i];
			if(a != 0.) r[i] /= a;
		}
	}else{
	        for(register int i=0; i<nn; i++) r[i] = v[(size_t)i*(i+1)/2+i];
	}

	return divide?NULL:&r[0];
}


/***************************************************************************//**
 * implements unary minus operator for this symmetric
 * matrix of general type <code>T</code>
 * @return modified copy of this matrix
 ******************************************************************************/
template <typename T>
const NRSMat<T> NRSMat<T>::operator-() const {
	NOT_GPU(*this);

	NRSMat<T> result(nn, getlocation());
	for(register size_t i = 0; i<NN2; i++) result.v[i]= -v[i];
	return result;
}

/***************************************************************************//**
 * implements unary minus operator for this real symmetric matrix
 * @return modified copy of this matrix
 ******************************************************************************/
template <>
const NRSMat<double> NRSMat<double>::operator-() const {
	NRSMat<double> result(nn, getlocation());
#ifdef CUDALA
	if(location == cpu){
#endif
		memcpy(result.v, v, NN2*sizeof(double));
		cblas_dscal(NN2, -1., result.v, 1);
#ifdef CUDALA
	}else{
		cublasDcopy(NN2, v, 1, result.v, 1);
		TEST_CUBLAS("cublasDcopy");

		cublasDscal(NN2, -1., result.v, 1);
		TEST_CUBLAS("cublasDscal");
	}
#endif
	return result;
}

/***************************************************************************//**
 * implements unary minus operator for this hermitian matrix
 * @return modified copy of this matrix
 ******************************************************************************/
template <>
const NRSMat<complex<double> > NRSMat<complex<double> >::operator-() const {
	NRSMat<complex<double> > result(nn, getlocation());
#ifdef CUDALA
        if(location == cpu) {
#endif
		memcpy(result.v, v, NN2*sizeof(complex<double>));
		cblas_zscal(NN2, &CMONE, result.v, 1);

#ifdef CUDALA
        }else{
		cublasZcopy(NN2, (cuDoubleComplex*)v, 1, (cuDoubleComplex*)result.v, 1);
		TEST_CUBLAS("cublasZcopy");

		cublasZscal(NN2, CUMONE, (cuDoubleComplex*)result.v, 1);
		TEST_CUBLAS("cublasZscal");
	}
#endif
	return result;
}

/***************************************************************************//**
 * @return the sum of the diagonal elements
 ******************************************************************************/
template <typename T>
const T NRSMat<T>::trace() const {
	NOT_GPU(*this);

	T tmp = 0;
	for(register int i=0; i<nn; i++) tmp += v[(size_t)i*(i+1)/2+i];
	return tmp;
}

/***************************************************************************//**
 * fill this real symmetric matrix with
 * pseudorandom numbers generated from uniform distribution
 ******************************************************************************/
template<>
void NRSMat<double>::randomize(const double &x) {
	NOT_GPU(*this);

	for(size_t i=0; i<NN2; ++i){
		v[i] = x*(2.*random()/(1.+RAND_MAX) -1.);
	}
}

/***************************************************************************//**
 * Fill this hermitian matrix with pseudorandom numbers generated from uniform 
 * distribution. The real and imaginary parts are generated independently.
 ******************************************************************************/
template<>
void NRSMat<complex<double> >::randomize(const double &x) {
	for(register size_t i=0; i<NN2; ++i) v[i].real() = x*(2.*random()/(1. + RAND_MAX) -1.);
	for(register size_t i=0; i<NN2; ++i) v[i].imag() = x*(2.*random()/(1. + RAND_MAX) -1.);
	for(register int i=0; i<nn; ++i){
		for(register int j=0; j<=i; ++j){
		       	if(i == j) v[i*(size_t)(i+1)/2+j].imag() = 0; //hermitean
		}
	}
}

/***************************************************************************//**
 * routine for formatted output via lawritemat
 * @param[in] file pointer to <tt>FILE</tt> structure representing the output file
 * @param[in] format format specification in standard printf-like form
 * @param[in] modulo
 * @see lawritemat()
 ******************************************************************************/
template <typename T>
void NRSMat<T>::fprintf(FILE *file, const char *format, const int modulo) const {
	NOT_GPU(*this);

	lawritemat(file, (const T *)(*this) ,nn, nn, format, 2, modulo, 1);
}


/***************************************************************************//**
 * routine for formatted input via fscanf
 * @param[in] f pointer to <tt>FILE</tt> structure representing the input file
 * @param[in] format format specification in standard printf-like form
 ******************************************************************************/
template <typename T>
void NRSMat<T>::fscanf(FILE *f, const char *format) {
	int n, m;
	NOT_GPU(*this);

	if (::fscanf(f,"%d %d", &n, &m) != 2)
		laerror("cannot read matrix dimensions in NRSMat<T>::fscanf(FILE *, const char *)");
	if (n != m) laerror("different dimensions in NRSMat<T>::fscanf(FILE *, const char *)");
	resize(n);
	for (int i=0; i<n; i++) 
		for (int j=0; j<n; j++)
			if (::fscanf(f,format,&((*this)(i,j))) != 1)
				laerror("NRSMat<T>::fscanf(FILE *, const char *) - unable to read matrix element");
}


/***************************************************************************//**
 * multiply this real double-precision symmetric matrix \f$S\f$ stored in packed form
 * with real double-precision dense matrix \f$A\f$
 * @param[in] rhs real double-precision matrix \f$A\f$
 * @return matrix produt \f$S\times{}A\f$
 ******************************************************************************/
template<>
const NRMat<double> NRSMat<double>::operator*(const NRMat<double> &rhs) const {
#ifdef DEBUG
	if(nn != rhs.nrows()) laerror("incompatible dimensions in NRMat<double> NRSMat<double>::operator*(const NRMat<double> &)");
#endif
	SAME_LOC(*this, rhs);
	NRMat<double> result(nn, rhs.ncols(), getlocation());
#ifdef CUDALA
	if(location == cpu){
#endif
		for(register int k = 0; k<rhs.ncols(); k++){
			cblas_dspmv(CblasRowMajor, CblasLower, nn, 1.0, v, rhs[0] + k, rhs.ncols(), 0.0, result[0] + k, rhs.ncols());
		}
#ifdef CUDALA
	}else{
		for(register int k = 0; k<rhs.ncols(); k++){
			cublasDspmv('U', nn, 1.0, v, rhs[0] + k, rhs.ncols(), 0.0, result[0] + k, rhs.ncols());
			TEST_CUBLAS("cublasDspmv");
		}
	}
#endif
	return result;
}


/***************************************************************************//**
 * multiply this real double-precision symmetric matrix \f$S\f$ stored in packed form
 * with real double-precision dense matrix \f$A\f$
 * @param[in] rhs real double-precision matrix \f$A\f$
 * @return matrix produt \f$S\times{}A\f$
 ******************************************************************************/
template<>
const NRMat<complex<double> >
NRSMat<complex<double> >::operator*(const NRMat<complex<double> > &rhs) const {
#ifdef DEBUG
	if (nn != rhs.nrows()) laerror("incompatible dimensions in NRSMat<complex<double> >::operator*(const NRMat<complex<double> > &)");
#endif
	SAME_LOC(*this, rhs);
	NRMat<complex<double> > result(nn, rhs.ncols(), getlocation());
#ifdef CUDALA
	if(location == cpu){
#endif
		for(register int k=0; k<rhs.ncols(); k++){
			cblas_zhpmv(CblasRowMajor, CblasLower, nn, &CONE, v, rhs[0]+k, rhs.ncols(), &CZERO, result[0]+k, rhs.ncols());
		}
#ifdef CUDALA
	}else{
		for(register int k = 0; k<rhs.ncols(); k++){
			cublasZhpmv('U', nn, 
					CUONE, (cuDoubleComplex*)v, (cuDoubleComplex*)(rhs[0] + k), rhs.ncols(),
					CUZERO, (cuDoubleComplex*)(result[0] + k), rhs.ncols());

			TEST_CUBLAS("cublasDspmv");
		}
	}
#endif
	return result;
}

/***************************************************************************//**
 * multiply this real double-precision symmetric matrix \f$S\f$ stored in packed form
 * with real double-precision symmetric matrix \f$T\f$
 * @return matrix produt \f$S\times{}T\f$ (not necessarily symmetric)
 ******************************************************************************/
template<>
const NRMat<double> NRSMat<double>::operator*(const NRSMat<double> &rhs) const {
#ifdef DEBUG
	if (nn != rhs.nn) laerror("incompatible dimensions in NRMat<double> NRSMat<double>::operator*(const NRSMat<double> &)");
#endif
	NRMat<double> result(0.0, nn, nn);
	double *p, *q;

	p = v;
	for (int i=0; i<nn;i++) {
		q = rhs.v;
		for (int k=0; k<=i; k++) {
			cblas_daxpy(k+1, *p++, q, 1, result[i], 1);
			q += k+1;
		}
	}

	p = v;
	for (int i=0; i<nn;i++) {
		q = rhs.v+1;
		for (int j=1; j<nn; j++) {
			result[i][j] += cblas_ddot(i+1<j ? i+1 : j, p, 1, q, 1);
			q += j+1;
		}
		p += i+1;
	}

	p = v; 
	q = rhs.v;
	for (int i=0; i<nn; i++) {
		cblas_dger(CblasRowMajor, i, i+1, 1., p, 1, q, 1, result, nn);
		p += i+1;
		q += i+1;
	}
	
	q = rhs.v+3;
	for (int j=2; j<nn; j++) {
		p = v+1;
		for (int i=1; i<j; i++) {
			cblas_daxpy(i, *++q, p, 1, result[0]+j, nn);
			p += i+1;
		}
		q += 2;
	}

	return result;
}


/***************************************************************************//**
 * multiply this complex double-precision symmetric matrix \f$G\f$ stored in packed form
 * with complex double-precision symmetric matrix \f$H\f$
 * @return matrix produt \f$G\times{}H\f$ (not necessarily symmetric)
 ******************************************************************************/
template<>
const NRMat<complex<double> > 
NRSMat<complex<double> >::operator*(const NRSMat<complex<double> > &rhs) const {
#ifdef DEBUG
	if (nn != rhs.nn) laerror("incompatible dimensions in NRSMat<complex<double> >::operator*(const NRSMat<complex<double> > &)");
#endif
	SAME_LOC(*this, rhs);
	NRMat<complex<double> > result(nn, nn, getlocation());
	NRMat<complex<double> > rhsmat(rhs);
	result = *this * rhsmat;
	return result;
}


/***************************************************************************//**
 * compute inner product of this real symmetric matrix \f$A\f$ with given real symmetric matrix \f$B\f$
 * i.e. determine the value of
 * \f[\sum_{i,j}A_{i,j}B_{i,j}\f]
 * @param[in] rhs matrix \f$B\f$
 * @return computed inner product
 ******************************************************************************/
template<>
const double NRSMat<double>::dot(const NRSMat<double> &rhs) const {
	double ret(0.);
#ifdef DEBUG
	if (nn != rhs.nn) laerror("incompatible dimensions in double NRSMat<double>::dot(const NRSMat<double> &)");
#endif
	SAME_LOC(*this, rhs);
#ifdef CUDALA
	if(location == cpu){
#endif
		ret = cblas_ddot(NN2, v, 1, rhs.v, 1);
#ifdef CUDALA
	}else{
		ret = cublasDdot(NN2, v, 1, rhs.v, 1);
	}
#endif
	return ret;
}


/***************************************************************************//**
 * compute inner product of this complex symmetric matrix \f$A\f$ with given complex symmetric matrix \f$B\f$
 * i.e. determine the value of
 * \f[\sum_{i,j}\overbar{A_{i,j}}B_{i,j}\f]
 * @param[in] rhs matrix \f$B\f$
 * @return computed inner product
 ******************************************************************************/
template<>
const complex<double> NRSMat<complex<double> >::dot(const NRSMat<complex<double> > &rhs) const {
#ifdef DEBUG
	if (nn != rhs.nn) laerror("incompatible dimensions in complex<double> NRSMat<complex<double> >::dot(const NRSMat<complex<double> > &)");
#endif
	complex<double> dot(0., 0.);
	SAME_LOC(*this, rhs);

#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_zdotc_sub(NN2, v, 1, rhs.v, 1, &dot);
#ifdef CUDALA
	}else{
		const cuDoubleComplex _dot = cublasZdotc(NN2, (cuDoubleComplex*)v, 1, (cuDoubleComplex*)(rhs.v), 1);
		dot = complex<double>(cuCreal(_dot), cuCimag(_dot));
		TEST_CUBLAS("cublasZdotc");
	}
#endif
	return dot;
}


/***************************************************************************//**
 * compute inner product of this real double-precision symmetric matrix \f$S\f$ of order \f$n\f$
 * with given real double-precision vector \f$\vec{v}\f$ of length \f$n(n+1)/2\f$
 * @param[in] rhs real double-precision vector \f$\vec{v}\f$
 * @return computed inner product
 ******************************************************************************/
template<>
const double NRSMat<double>::dot(const NRVec<double> &rhs) const {
	double ret(0.0);
#ifdef DEBUG
	if(NN2 != rhs.nn) laerror("incompatible dimensions in double NRSMat<double>::dot(const NRVec<double> &)");
#endif
	SAME_LOC(*this, rhs);
#ifdef CUDALA
	if(location == cpu){
#endif
		ret = cblas_ddot(NN2, v, 1, rhs.v, 1);
#ifdef CUDALA
	}else{
		ret = cublasDdot(NN2, v, 1, rhs.v, 1);
		TEST_CUBLAS("cublasDdot");
	}
#endif
}


/***************************************************************************//**
 * compute inner product of this complex double-precision hermitian matrix \f$H\f$ of order \f$n\f$
 * with given complex double-precision vector \f$\vec{v}\f$ of length \f$n(n+1)/2\f$
 * @param[in] rhs complex double-precision vector \f$\vec{v}\f$
 * @return computed inner product
 ******************************************************************************/
template<>
const complex<double> 
NRSMat<complex<double> >::dot(const NRVec<complex<double> > &rhs) const {
#ifdef DEBUG
	if(NN2 != rhs.nn) laerror("incompatible dimensions in complex<double>  NRSMat<complex<double> >::dot(const NRVec<complex<double> > &)");
#endif
	complex<double> dot(0., 0.);
	SAME_LOC(*this, rhs);
#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_zdotc_sub(NN2, v, 1, rhs.v, 1, &dot);
#ifdef CUDALA
	}else{
		const cuDoubleComplex _dot = cublasZdotc(NN2, (cuDoubleComplex*)v, 1, (cuDoubleComplex*)rhs.v, 1);
		TEST_CUBLAS("cublasZdotc");
		dot = complex<double>(cuCreal(_dot), cuCimag(_dot));
	}
#endif
	return dot;
}

/***************************************************************************//**
 * compute the Frobenius norm of this real double-precision symmetric  matrix
 * @param[in] scalar subtract this scalar value from the diagonal elements before the norm computation
 ******************************************************************************/
template<>
const double NRSMat<double>::norm(const double scalar) const {
	if(!scalar){
		double ret(0.);
#ifdef CUDALA
		if(location == cpu){
#endif
			ret = cblas_dnrm2(NN2, v, 1);
#ifdef CUDALA
		}else{
			ret = cublasDnrm2(NN2, v, 1);
			TEST_CUBLAS("cublasDnrm2");
		}
#endif
		return ret;
	}

	NOT_GPU(*this);

	double sum(0.);
	int k(0);
	for(register int i=0; i<nn; ++i){
		for(register int j=0; j<=i; ++j) {
			register double tmp = v[k++];
			if(i == j) tmp -= scalar;
			sum += tmp*tmp;
		}
	}
	return std::sqrt(sum);
}

/***************************************************************************//**
 * compute the Frobenius norm of this complex double-precision hermitian  matrix
 * @param[in] scalar subtract this scalar value from the diagonal elements before the norm computation
 ******************************************************************************/
template<>
const double NRSMat< complex<double> >::norm(const complex<double> scalar) const {
	if(!(scalar.real()) && !(scalar.imag())){
		double ret(0.);
#ifdef CUDALA
		if(location == cpu){
#endif
			ret = cblas_dznrm2(NN2, v, 1);
#ifdef CUDALA
		}else{
			ret = cublasDznrm2(NN2, (cuDoubleComplex*)v, 1);
			TEST_CUBLAS("cublasDznrm2");
		}
#endif
		return ret;
	}

	int k(0);
	double sum(0.);
	complex<double> tmp;

	for(register int i=0; i<nn; ++i){
		for(register int j=0; j<=i; ++j){
			tmp = v[k++];
			if (i == j) tmp -= scalar;
			sum += tmp.real()*tmp.real() + tmp.imag()*tmp.imag();
		}
	}
	return std::sqrt(sum);
}


/***************************************************************************//**
 * for this real double-precision symmetric matrix \f$S\f$ stored in packed form,
 * real scalar value \f$\alpha\f$ and real double-precision symmetric matrix \f$T\f$, compute
 * \f[S \leftarrow \alpha T + S\f]
 ******************************************************************************/
template<>
void NRSMat<double>::axpy(const double alpha, const NRSMat<double> &x) {
#ifdef DEBUG
	if(nn != x.nn) laerror("incompatible dimensions in void NRSMat<double>::axpy(const double, const NRSMat<double>&)");
#endif
	SAME_LOC(*this, x);
	copyonwrite();
#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_daxpy(NN2, alpha, x.v, 1, v, 1);
#ifdef CUDALA
	}else{
		cublasDaxpy(NN2, alpha, x.v, 1, v, 1);
		TEST_CUBLAS("cublasDaxpy");
	}
#endif
}


/***************************************************************************//**
 * for this complex double-precision hermitian matrix \f$H\f$ stored in packed form,
 * complex scalar value \f$\alpha\f$ and complex double-precision hermitian matrix \f$G\f$, compute
 * \f[H \leftarrow \alpha G + H\f]
 ******************************************************************************/
template<>
void NRSMat<complex<double> >::axpy(const complex<double> alpha, const NRSMat<complex<double> > & x) {
#ifdef DEBUG
	if(nn != x.nn) laerror("incompatible dimensions in void NRSMat<complex<double> >::axpy(const complex<double> , const NRSMat<complex<double> >&)");
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
		cublasZaxpy(NN2, _alpha, (cuDoubleComplex*)x.v, 1, (cuDoubleComplex*)v, 1);
		TEST_CUBLAS("cublasZaxpy");
	}
#endif
	
}

/***************************************************************************//**
 * create hermitian matrix \f$H\f$ from given real double-precision symmetric
 * matrix \f$S\f$
 * @param[in] rhs real double-precision symmetric matrix \f$S\f$
 * @param[in] imagpart flag determining whether \f$S\f$ should correspond to the real or imaginary part of \f$H\f$
 ******************************************************************************/
template<>
NRSMat<complex<double> >::NRSMat(const NRSMat<double> &rhs, bool imagpart): nn(rhs.nrows()), count(new int(1)) {
	//inconsistent in general case?
	const int nnp1 = nn*(nn + 1)/2;
#ifdef CUDALA
	location = rhs.getlocation();
	if(location == cpu){
#endif
		v = new complex<double>[nnp1];
		memset(v, 0, nnp1*sizeof(complex<double>));
		cblas_dcopy(nnp1, &rhs(0, 0), 1, ((double *)v) + (imagpart?1:0), 2);
#ifdef CUDALA
	}else{
		v = (complex<double>*) gpualloc(nnp1*sizeof(complex<double>));

		complex<double> *_val = gpuputcomplex(CZERO);
		cublasZcopy(nnp1, (cuDoubleComplex*)_val, 0, (cuDoubleComplex*)v, 1);
		TEST_CUBLAS("cublasZcopy");
		gpufree(_val);

		cublasDcopy(nnp1, (double*)(&rhs(0,0)), 1, ((double*)v) + (imagpart?1:0), 2);
		TEST_CUBLAS("cublasDcopy"); 
	}
#endif
}

/***************************************************************************//**
 * forced instantization in the corresponding object file
 ******************************************************************************/
template class NRSMat<double>;
template class NRSMat<complex<double> >;

template class NRSMat<long long>;
template class NRSMat<long>;
template class NRSMat<int>;
template class NRSMat<short>;
template class NRSMat<char>;
template class NRSMat<unsigned char>;
template class NRSMat<unsigned short>;
template class NRSMat<unsigned int>;
template class NRSMat<unsigned long>;
template class NRSMat<unsigned long long>;

}//namespace
