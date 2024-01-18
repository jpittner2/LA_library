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

#include "mat.h"
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <math.h>
#include "nonclass.h"


namespace LA {

/***************************************************************************//**
 * implements direct sum with a given matrix \f$B\f$ via storesubmatrix()
 * @param[in] rhs input matrix \f$B\f$
 * @return result of the computation (new instance of NRMat<T>)
 * @see submatrix()
 ******************************************************************************/
template <typename T>
const NRMat<T> NRMat<T>::oplus(const NRMat<T> &rhs) const {

	// special cases
	if(nn == 0 && mm == 0) return rhs;
	if(rhs.nn == 0 && rhs.mm == 0) return *this;

	SAME_LOC(*this, rhs);
	NRMat<T> ret(nn + rhs.nn, mm + rhs.mm, getlocation());

	ret.clear();
	ret.storesubmatrix(0, 0, *this);
	ret.storesubmatrix(nn, mm, rhs);
	return ret;
}

/***************************************************************************//**
 * implements direct product with a given matrix \f$B\f$
 * @param[in] rhs input matrix \f$B\f$
 * @return result of the computation (new instance of NRMat<T>)
 ******************************************************************************/
template <typename T>
const NRMat<T> NRMat<T>::otimes(const NRMat<T> &rhs, bool reversecolumns) const {

	// special cases
	if(nn == 0 && mm == 0) return *this;
	if(rhs.nn == 0 && rhs.mm == 0) return rhs;

	NRMat<T> r((T)0, nn*rhs.nn, mm*rhs.mm);

	int i,j,k,l;

	if(reversecolumns){
		for(i=0;i<nn;i++) for(j=0;j<mm;j++)
		{
			T c = (*this)(i,j);
			for(k=0;k<rhs.nn;k++) for(l=0;l<rhs.mm;l++)
				r( i*(size_t)rhs.nn + k, l*mm + j ) = c*rhs(k,l);
		}
	}else{
		for(i=0;i<nn;i++) for(j=0;j<mm;j++)
		{
			T c=(*this)(i,j);
			for(k=0;k<rhs.nn;k++) for(l=0;l<rhs.mm;l++)
				r( i*(size_t)rhs.nn+k, j*(size_t)rhs.mm+l ) = c *rhs(k,l);
		}
	}

	return r;
}

/***************************************************************************//**
 * extract given row of this matrix of general type <code>T</code>
 * @param[in] i row index starting from zero
 * @param[in] l consider this value as the count of columns
 * @return extracted elements as a NRVec<T> object
 ******************************************************************************/
template <typename T>
const NRVec<T> NRMat<T>::row(const int i, int l) const {
#ifdef DEBUG
	if(i < 0 || i >= nn) laerror("illegal index");
#endif
	if(l < 0) l = mm;
	NRVec<T> r(l);
	LA_traits<T>::copy(&r[0],
#ifdef MATPTR
	        v[i]
#else
		v + i*(size_t)l
#endif
		, l);
	return r;
}

/***************************************************************************//**
 * routine for raw output
 * @param[in] fd file descriptor for output
 * @param[in] dim number of elements intended for output
 * @param[in] transp reserved
 * @see NRVec<T>::put()
 ******************************************************************************/
template <typename T>
void NRMat<T>::put(int fd, bool dim, bool transp) const {
#ifdef CUDALA
	if(location != cpu) {
		NRMat<T> tmp = *this;
		tmp.moveto(cpu);
		tmp.put(fd, dim, transp);
		return;
	}
#endif
	errno = 0;
	if(dim){
		if(sizeof(int) != write(fd,&(transp?mm:nn),sizeof(int))) laerror("write failed");
		if(sizeof(int) != write(fd,&(transp?nn:mm),sizeof(int))) laerror("write failed");
	}

	if(transp){  //not particularly efficient
	        for(int j=0; j<mm; ++j){
		       for(int i=0; i<nn; ++i){
	                        LA_traits<T>::put(fd, 
				#ifdef MATPTR
	                                v[i][j]
				#else
	                                v[i*(size_t)mm+j]
				#endif
					,dim ,transp);
			}
		}
        }else{
		LA_traits<T>::multiput((size_t)nn*(size_t)mm,fd,
		#ifdef MATPTR
		        v[0]
		#else
		        v
		#endif
			,dim);
	}
}

/***************************************************************************//**
 * routine for raw input 
 * @param[in] fd file descriptor for input
 * @param[in] dim number of elements intended for input, for dim=0 perform copyonwrite
 * @param[in] transp reserved
 * @see NRVec<T>::get(), copyonwrite()
 ******************************************************************************/
template <typename T>
void NRMat<T>::get(int fd, bool dim, bool transp){
#ifdef CUDALA
	if(location != cpu){
		NRMat<T> tmp;
		tmp.moveto(cpu);
		tmp.get(fd, dim, transp);
		tmp.moveto(getlocation());
		*this = tmp;
		return;
	}
#endif
	int nn0, mm0;
	errno = 0;
	if(dim){
		if(sizeof(int) != read(fd, &nn0, sizeof(int))) laerror("read failed");
		if(sizeof(int) != read(fd, &mm0, sizeof(int))) laerror("read failed");
		if(transp) resize(mm0, nn0); else resize(nn0, mm0);
	}else{
		copyonwrite();
	}

	if(transp){
		for(register int j=0; j<mm; ++j){
			for(register int i=0; i<nn; ++i){
				LA_traits<T>::get(fd,
				#ifdef MATPTR
					v[i][j]
				#else
					v[i*(size_t)mm+j]
				#endif
				,dim,transp);
			}
		}
	}else{
		LA_traits<T>::multiget((size_t)nn*(size_t)mm,fd,
		#ifdef MATPTR
		        v[0]
		#else
		        v
		#endif
			,dim);
	}
}

/***************************************************************************//**
 * assigns a scalar value of general type <code>T</code> to the diagonal elements of this
 * matrix of general type <code>T</code>
 * @param[in] a scalar value of type <code>T</code>
 * @return reference to the modified matrix
 ******************************************************************************/
template <typename T>
NRMat<T>& NRMat<T>::operator=(const T &a) {
	NOT_GPU(*this);
	const int n2 = nn*nn;

	copyonwrite();
#ifdef DEBUG
	if(nn != mm) laerror("nonsquare matrix");
#endif
#ifdef MATPTR
	memset(v[0], 0, n2*sizeof(T));
	for(register int i=0; i < nn; i++) v[i][i] = a;
#else
	memset(v, 0, n2*sizeof(T));
	for(register int i=0; i < n2; i += nn + 1) v[i] = a;
#endif
	return *this;
}

/***************************************************************************//**
 * assigns a double-precision real scalar value to the diagonal elements of this
 * double-precision real matrix
 * @param[in] a double-precision real scalar value
 * @return reference to the modified matrix
 ******************************************************************************/
template <>
NRMat<double>& NRMat<double>::operator=(const double &a){
	const int n2 = nn*nn;
	copyonwrite();
#ifdef DEBUG
	if(nn != mm) laerror("nonsquare matrix");
#endif
#ifdef CUDALA
	if(location == cpu){
#endif
#ifdef MATPTR
		memset(v[0], 0, n2*sizeof(double));
		for(register int i=0; i< nn; i++) v[i][i] = a;
#else
		const double n = 0.;
		//set all matrix elements equal to zero
		cblas_dcopy(n2, &n, 0, v, 1);
		//update the diagonal elements
		cblas_dcopy(nn, &a, 0, v, nn + 1);
#endif
#ifdef CUDALA
	}else{
		smart_gpu_set(n2, 0.0, v, 1);
		smart_gpu_set(nn, a, v, nn + 1);
	}
#endif
         return *this;
}

/***************************************************************************//**
 * assigns a double-precision complex scalar value to the diagonal elements of this
 * double-precision complex matrix
 * @param[in] a double-precision complex scalar value
 * @return reference to the modified matrix
 ******************************************************************************/
template <>
NRMat<std::complex<double> >& NRMat<std::complex<double> >::operator=(const std::complex<double> &a){
	const int n2 = nn*nn;
	copyonwrite();
#ifdef DEBUG
	if(nn != mm) laerror("nonsquare matrix");
#endif
#ifdef CUDALA
	if(location == cpu){
#endif
#ifdef MATPTR
		memset(v[0], 0, n2*sizeof(std::complex<double>));
		for(register int i=0; i< nn; i++) v[i][i] = a;
#else
		//set all matrix elements equal to zero
		cblas_zcopy(n2, &CZERO, 0, v, 1);
		//update the diagonal elements
		cblas_zcopy(nn, &a, 0, v, nn + 1);
#endif
#ifdef CUDALA
	}else{
		smart_gpu_set(n2, CZERO, v, 1);
		smart_gpu_set(nn, a, v, nn + 1);
	}
#endif
         return *this;
}
/***************************************************************************//**
 * adds a double-precision real scalar value to the diagonal elements of this
 * double-precision real matrix
 * @param[in] a double-precision real scalar value
 * @return reference to the modified matrix
 ******************************************************************************/
template <>
NRMat<double> & NRMat<double>::operator+=(const double& a) {
        copyonwrite();
#ifdef DEBUG
	if(nn != mm) laerror("nonsquare matrix");
#endif
#ifdef CUDALA
	if(location == cpu){
#endif
#ifdef MATPTR
		for(register int i=0; i < nn; i++) v[i][i] += a;
#else
		cblas_daxpy(nn, 1.0, &a, 0, *this, nn + 1);
#endif
#ifdef CUDALA
	}else{
		double *d = gpuputdouble(a);
		cublasDaxpy(nn, 1.0, d, 0, *this, nn+1);
		TEST_CUBLAS("cublasDaxpy");
		gpufree(d);
	}
#endif
        return *this;
}

/***************************************************************************//**
 * adds a double-precision complex scalar value to the diagonal elements of this
 * double-precision complex matrix
 * @param[in] a double-precision complex scalar value
 * @return reference to the modified matrix
 ******************************************************************************/
template <>
NRMat<std::complex<double> > & NRMat<std::complex<double> >::operator+=(const std::complex<double>& a) {
        copyonwrite();
#ifdef DEBUG
	if(nn != mm) laerror("nonsquare matrix");
#endif
#ifdef CUDALA
	if(location == cpu){
#endif
#ifdef MATPTR
		for(register int i=0; i < nn; i++) v[i][i] += a;
#else
		cblas_zaxpy(nn, &CONE, &a, 0, *this, nn + 1);
#endif
#ifdef CUDALA
	}else{
		std::complex<double>* d = gpuputcomplex(a);
		cublasZaxpy(nn, CUMONE, (cuDoubleComplex*)d, 0, (cuDoubleComplex*)v, nn+1);
		TEST_CUBLAS("cublasDaxpy");
		gpufree(d);
	}
#endif
        return *this;
}

/***************************************************************************//**
 * subtracts a double-precision real scalar value from the diagonal elements of this
 * double-precision real matrix
 * @param[in] a double-precision real scalar value
 * @return reference to the modified matrix
 ******************************************************************************/
template <>
NRMat<double>& NRMat<double>::operator-=(const double& a) {
        copyonwrite();
#ifdef DEBUG
        if(nn != mm) laerror("nonsquare matrix");
#endif
#ifdef CUDALA
	if(location == cpu){
#endif
#ifdef MATPTR
		for(register int i=0; i< nn; i++) v[i][i] -= a;
#else
		cblas_daxpy(nn, -1.0, &a, 0, *this, nn+1);
#endif
#ifdef CUDALA
	}else{
		double *d = gpuputdouble(a);
		cublasDaxpy(nn, -1.0, d, 0, *this, nn+1);
		TEST_CUBLAS("cublasDaxpy");
		gpufree(d);
	}
#endif
        return *this;
}

/***************************************************************************//**
 * subtracts a double-precision complex scalar value from the diagonal elements of this
 * double-precision complex matrix
 * @param[in] a double-precision complex scalar value
 * @return reference to the modified matrix
 ******************************************************************************/
template <>
NRMat<std::complex<double> >& NRMat<std::complex<double> >::operator-=(const std::complex<double>& a) {
        copyonwrite();
#ifdef DEBUG
	if(nn != mm) laerror("nonsquare matrix");
#endif
#ifdef CUDALA
	if(location == cpu){
#endif
#ifdef MATPTR
		for(register int i=0; i < nn; i++) v[i][i] -= a;
#else
		cblas_zaxpy(nn, &CMONE, &a, 0, *this, nn + 1);
#endif
#ifdef CUDALA
	}else{
		std::complex<double>* d = gpuputcomplex(a);
		cublasZaxpy(nn, CUMONE, (cuDoubleComplex*)d, 0, (cuDoubleComplex*)v, nn+1);
		TEST_CUBLAS("cublasDaxpy");
		gpufree(d);
	}
#endif
        return *this;
}
/***************************************************************************//**
 * add a scalar value of type <code>T</code> to the diagonal elements of this
 * matrix of general type <code>T</code>
 * @return reference to the modified matrix
 ******************************************************************************/
template <typename T>
NRMat<T>& NRMat<T>::operator+=(const T &a) {
	NOT_GPU(*this);

	copyonwrite();
#ifdef DEBUG
	if(nn != mm) laerror("nonsquare matrix");
#endif
#ifdef MATPTR
	for(register int i=0; i < nn; i++) v[i][i] += a;
#else
	for(register int i=0; i < nn*nn; i += nn+1) v[i] += a;
#endif
	return *this;
}

/***************************************************************************//**
 * subtracts a scalar value of type <code>T</code> from the diagonal elements of this
 * matrix of general type <code>T</code>
 * @return reference to the modified matrix
 ******************************************************************************/
template <typename T>
NRMat<T> & NRMat<T>::operator-=(const T &a) {
	NOT_GPU(*this);

	copyonwrite();
#ifdef DEBUG
	if(nn != mm) laerror("nonsquare matrix");
#endif
#ifdef MATPTR
	for(register int i=0; i< nn; i++) v[i][i] -= a;
#else
	for(register int i=0; i< nn*nn; i+=nn+1) v[i] -= a;
#endif
	return *this;
}

/***************************************************************************//**
 * implements unary minus operator for this double-recision real matrix
 * @return modified copy of this matrix
 ******************************************************************************/
template <>
const NRMat<double> NRMat<double>::operator-() const {
	const size_t nm = (size_t)nn*mm;
        NRMat<double> result(nn, mm, getlocation());
#ifdef CUDALA
        if(location == cpu) {
#endif
#ifdef MATPTR
	        for(register size_t i=0; i<nm; i++) result.v[0][i] = -v[0][i];
#else
		memcpy(result.v, v, nm*sizeof(double));
		cblas_dscal(nm, -1., result.v, 1);
#endif
#ifdef CUDALA
        }else{
		cublasDcopy(nm, v, 1, result.v, 1);
		TEST_CUBLAS("cublasDcopy");

		cublasDscal(nm, -1., result.v, 1);
		TEST_CUBLAS("cublasDscal");
	}
#endif
        return result;
}


/***************************************************************************//**
 * implements unary minus operator for this double-precision complex matrix
 * @return modified copy of this matrix
 ******************************************************************************/
template <>
const NRMat<std::complex<double> > NRMat<std::complex<double> >::operator-() const {
	const size_t nm = (size_t)nn*mm;
        NRMat<std::complex<double> > result(nn, mm, getlocation());
#ifdef CUDALA
        if(location == cpu) {
#endif
#ifdef MATPTR
	        for(register size_t i=0; i<nm; i++) result.v[0][i]= -v[0][i];
#else
		memcpy(result.v, v, nm*sizeof(std::complex<double>));
		cblas_zscal(nm, &CMONE, result.v, 1);
#endif
#ifdef CUDALA
        }else{
		cublasZcopy(nm, (cuDoubleComplex*)v, 1, (cuDoubleComplex*)result.v, 1);
		TEST_CUBLAS("cublasZcopy");

		cublasZscal(nm, CUMONE, (cuDoubleComplex*)result.v, 1);
		TEST_CUBLAS("cublasZscal");
	}
#endif
        return result;
}

/***************************************************************************//**
 * implements unary minus operator for this matrix of general type <code>T</code>
 * @return modified copy of this matrix
 ******************************************************************************/
template <typename T>
const NRMat<T> NRMat<T>::operator-() const {
	NOT_GPU(*this);

	NRMat<T> result(nn, mm, getlocation());
#ifdef MATPTR
	for(register size_t i=0; i<(size_t)nn*mm; i++) result.v[0][i] = -v[0][i];
#else
	for(register size_t i=0; i<(size_t)nn*mm; i++) result.v[i] = -v[i];
#endif
	return result;
}


// direct sum
template <typename T>
const NRMat<T> NRMat<T>::operator&(const NRMat<T> &b) const {
	SAME_LOC(*this, b);
	NRMat<T> result((T)0, nn + b.nn, mm + b.mm, getlocation());
	if(!LA_traits<T>::is_plaindata()) laerror("only implemented for plain data");
#ifdef CUDALA
	if(location == cpu){
#endif
		for(register int i=0; i<nn; i++){ memcpy(result[i], (*this)[i], sizeof(T)*mm); }
		for(register int i=0; i<b.nn; i++){ memcpy(result[nn + i] + mm, b[i], sizeof(T)*b.mm); }
#ifdef CUDALA
	}else{
		if(sizeof(T)%sizeof(float) != 0) laerror("memory alignment problem");

		for(register int i=0; i<nn; i++){
			cublasScopy(mm*sizeof(T)/sizeof(float), (float*)(v + i*(size_t)mm), 1, (float*)(result.v + i*(size_t)(mm + b.mm)), 1);
			TEST_CUBLAS("cublasScopy");
		}
		for(register int i=0; i<b.nn; i++){
			cublasScopy(mm*sizeof(T)/sizeof(float), (float*)(b.v + i*(size_t)b.mm), 1, (float*)(result.v + (nn + i)*(mm + b.mm)), 1);
			TEST_CUBLAS("cublasScopy");
		}
	}
#endif
	return result;
}

// direct product 
template <typename T>
const NRMat<T> NRMat<T>::operator|(const NRMat<T> &b) const {
	NRMat<T> result(nn*b.nn, mm*b.mm);
	for (int i=0; i<nn; i++)
		for (int j=0; j<mm; j++)
			for (int k=0; k<b.nn; k++)
				for (int l=0; l<b.mm; l++)
					result[i*(size_t)b.nn+k][j*(size_t)b.mm+l] = (*this)[i][j]*b[k][l];
	return result;
}

/***************************************************************************//**
 * sum up the columns of the current matrix of general type <code>T</code>
 * @return summed columns in a form of a vector
 ******************************************************************************/
template <typename T>
const NRVec<T> NRMat<T>::csum() const {
	NOT_GPU(*this);
	NRVec<T> result(nn, getlocation());
	T sum;
	
	for(register int i=0; i<nn; i++) { 
		sum = (T)0;
		for(int j=0; j<mm; j++) sum += (*this)[i][j];
		result[i] = sum;
	}
	return result;
}

/***************************************************************************//**
 * sum up the columns of the current double-precision real matrix
 * @return summed columns in a form of a vector
 ******************************************************************************/
template <>
const NRVec<double> NRMat<double>::csum() const {
	NRVec<double> result(nn, getlocation());
	result = 0.0;
#ifdef CUDALA
	if(location == cpu){
#endif
		for(register int i=0; i<mm; i++){
			cblas_daxpy(nn, 1.0, &((*this)(0, i)), nn, result.v, 1);
		}
#ifdef CUDALA
	}else{
		for(register int i=0; i<mm; i++){
			cublasDaxpy(nn, 1.0, v + i, nn, result.v, 1);
			TEST_CUBLAS("cublasDaxpy");
		}
	}
#endif
	return result;
}

/***************************************************************************//**
 * sum up the columns of the current double-precision complex matrix
 * @return summed columns in a form of a vector
 ******************************************************************************/
template <>
const NRVec<std::complex<double> > NRMat<std::complex<double> >::csum() const {
	NRVec<std::complex<double> > result(nn, getlocation());
	result = 0.0;
#ifdef CUDALA
	if(location == cpu){
#endif
		for(register int i=0;i<mm;i++){
			cblas_zaxpy(nn, &CONE, &((*this)(0, i)), nn, result.v, 1);
		}
#ifdef CUDALA
	}else{
		for(register int i=0;i<nn;i++){
			cublasZaxpy(nn, CUONE, (cuDoubleComplex*)(v + i), nn, (cuDoubleComplex*)(result.v), 1);
			TEST_CUBLAS("cublasZaxpy");
		}
	}
#endif
	return result;
}

/***************************************************************************//**
 * sum up the rows of the current matrix of general type <code>T</code>
 * @return summed rows in a form of a vector
 ******************************************************************************/
template <typename T>
const NRVec<T> NRMat<T>::rsum() const {
	NOT_GPU(*this);
	NRVec<T> result(mm, getlocation());
	T sum;
	
	for(register int i=0; i<mm; i++) { 
		sum = (T)0;
		for(int j=0; j<nn; j++) sum += (*this)[j][i];
		result[i] = sum;
	}
	return result;
}

/***************************************************************************//**
 * sum up the rows of the current double-precision real matrix
 * @return summed rows in a form of a vector
 ******************************************************************************/
template <>
const NRVec<double> NRMat<double>::rsum() const {
	NRVec<double> result(mm, getlocation());
	result = 0.0;
#ifdef CUDALA
	if(location == cpu){
#endif
		for(register int i=0;i<nn;i++){
			cblas_daxpy(mm, 1.0, (*this)[i], 1, result.v, 1);
		}
#ifdef CUDALA
	}else{
		for(register int i=0;i<nn;i++){
			cublasDaxpy(mm, 1.0, v + i*(size_t)mm, 1, result.v, 1);
			TEST_CUBLAS("cublasDaxpy");
		}
	}
#endif
	return result;
}

/***************************************************************************//**
 * sum up the rows of the current double-precision complex matrix
 * @return summed rows in a form of a vector
 ******************************************************************************/
template <>
const NRVec<std::complex<double> > NRMat<std::complex<double> >::rsum() const {
	NRVec<std::complex<double> > result(mm, getlocation());
	result = 0.0;
#ifdef CUDALA
	if(location == cpu){
#endif
		for(register int i=0;i<nn;i++){
			cblas_zaxpy(mm, &CONE, (*this)[i], 1, result.v, 1);
		}
#ifdef CUDALA
	}else{
		for(register int i=0;i<nn;i++){
			cublasZaxpy(mm, CUONE, (cuDoubleComplex*)(v + i*(size_t)mm), 1, (cuDoubleComplex*)(result.v), 1);
			TEST_CUBLAS("cublasZaxpy");
		}
	}
#endif
	return result;
}

/***************************************************************************//**
 * extract block submatrix
 * @param[in] fromrow starting row
 * @param[in] torow final row
 * @param[in] fromcol starting column
 * @param[in] tocol final column
 * @return extracted block submatrix
 ******************************************************************************/
template <typename T>
const NRMat<T> NRMat<T>::submatrix(const int fromrow, const int torow, const int fromcol, const int tocol) const {
#ifdef DEBUG
	if(fromrow<0 || fromrow>=nn|| torow<0 || torow>=nn || fromcol<0 || fromcol>=mm || tocol<0 || tocol>=mm || fromrow>torow || fromcol>tocol){
		laerror("invalid submatrix specification");
	}
#endif
	const int n = torow - fromrow + 1;
	const int m = tocol - fromcol + 1;
	NRMat<T> r(n, m, getlocation());
	if(!LA_traits<T>::is_plaindata()) laerror("only implemented for plain data");

#ifdef CUDALA
	if(location == cpu){
#endif
		for(register int i=fromrow; i<=torow; ++i){
			#ifdef MATPTR
			memcpy(r.v[i - fromrow], v[i] + fromcol, m*sizeof(T));
			#else
			memcpy(r.v+(i - fromrow)*m, v + i*(size_t)mm + fromcol, m*sizeof(T));
			#endif
		}
#ifdef CUDALA
	}else{
		if(sizeof(T)%sizeof(float) != 0) laerror("cpu memcpy alignment problem");
		for(register int i=fromrow; i<=torow; ++i){
			cublasScopy(m*sizeof(T)/sizeof(float), (const float *)(v + i*(size_t)mm + fromcol), 1, (float*)(r.v + (i - fromrow)*m), 1);
			TEST_CUBLAS("cublasScopy");
		}
	}
#endif	
	return r;
}


template <typename T>
const NRMat<T> NRMat<T>::submatrix(const NRVec<int> &rows, const NRVec<int> &cols) const
{
	NOT_GPU(*this);
        const int n = rows.size();
        const int m = cols.size();
        NRMat<T> r(n,m);

        for(int i=0; i<n; ++i)
		{
		int ii=rows[i];
		if(ii<0||ii>=nn) laerror("bad row index in submatrix");
		for(int j=0; j<m; ++j)
			{
			int jj=cols[j];
			if(jj<0||jj>=mm) laerror("bad col index in submatrix");
			r(i,j) = (*this)(ii,jj);
			}
                }
return r;
}


/***************************************************************************//**
 * places given matrix as submatrix at given position
 * @param[in] fromrow row-coordinate of top left corner
 * @param[in] fromcol col-coordinate of top left corner
 * @param[in] rhs input matrix
 ******************************************************************************/
template <typename T>
void NRMat<T>::storesubmatrix(const int fromrow, const int fromcol, const NRMat &rhs) {
	const int tocol = fromcol + rhs.ncols() - 1;
	const int torow = fromrow + rhs.nrows() - 1;
#ifdef DEBUG
	if(fromrow<0 || fromrow>=nn || torow>=nn || fromcol<0 || fromcol>=mm || tocol>=mm) laerror("bad indices in storesubmatrix");
#endif
	SAME_LOC(*this, rhs);
	if(!LA_traits<T>::is_plaindata()) laerror("only implemented for plain data");

	const int m = tocol - fromcol + 1;
	for(register int i = fromrow; i <= torow; ++i){
	#ifdef CUDALA
		if(location == cpu){
	#endif
		#ifdef MATPTR
			memcpy(v[i] + fromcol, rhs.v[i - fromrow], m*sizeof(T));
		#else
			memcpy(v + i*(size_t)mm + fromcol, rhs.v + (i - fromrow)*m, m*sizeof(T));
		#endif

	#ifdef CUDALA
		}else{
			if(sizeof(T)%sizeof(float) != 0) laerror("cpu memcpy alignment problem");
			cublasScopy(m*sizeof(T)/sizeof(float), (const float *) (rhs.v + (i - fromrow)*m), 1, (float *)(v + i*(size_t)mm + fromcol), 1);
		}
	#endif
	}
}

template <typename T>
void NRMat<T>::storesubmatrix(const NRVec<int> &rows, const NRVec<int> &cols, const NRMat &rhs)
{
        NOT_GPU(*this);
        const int n = rows.size();
        const int m = cols.size();
	if(rhs.nrows()!=n || rhs.ncols()!=m) laerror("incompatible dimensions in storesubmatrix");
                
        for(int i=0; i<n; ++i)
                {
                int ii=rows[i];
                if(ii<0||ii>=nn) laerror("bad row index in storesubmatrix");
                for(int j=0; j<m; ++j)
                        {
                        int jj=cols[j];
                        if(jj<0||jj>=mm) laerror("bad col index in storesubmatrix");
                        (*this)(ii,jj) = rhs(i,j);
                        }
                }       
}

/***************************************************************************//**
 * compute matrix transposition for a principal leading minor
 * @param[in] _n order of the leading minor
 * @return reference to the modified matrix
 ******************************************************************************/
template <typename T>
NRMat<T>& NRMat<T>::transposeme(const int _n) {
	const int n = (n <= 0)?nn:_n;//!< transpose the entire matrix
#ifdef DEBUG
	if (n==nn && nn != mm || n>mm || n>nn ) laerror("NRMat<T>::transposeme() - invalid parameter n. Non-square matrix?");
#endif
#ifdef CUDALA
	if(location == cpu){
#endif
		copyonwrite();
		for(register int i=1; i<n; i++){
			for(register int j=0; j<i; j++){
			#ifdef MATPTR
				T tmp = v[i][j]; 
				v[i][j] = v[j][i]; 
				v[j][i] = tmp;
			#else
				register int a, b; 
				a = i*(size_t)mm + j;
				b = j*(size_t)mm + i;
				T tmp = v[a];
				v[a] = v[b];
				v[b] = tmp;
			#endif
			}
		}
#ifdef CUDALA
	}else{
laerror("transposeme not implemented on GPU yet");

	}
#endif
	return *this;
}


/***************************************************************************//**
 * compute matrix non-symmetry
 ******************************************************************************/
template <typename T>
const typename  LA_traits<T>::normtype  NRMat<T>::nonsymmetry() const {
#ifdef DEBUG
	if (nn != mm) laerror("NRMat<T>:nonsymmetry() invalid for non-square matrix");
#endif
typename  LA_traits<T>::normtype sum = 0;
#ifdef CUDALA
	if(location == cpu){
#endif
		for(register int i=1; i<nn; i++){
			for(register int j=0; j<i; j++){
			#ifdef MATPTR
				sum += (v[i][j]-v[j][i])*(v[i][j]-v[j][i]); 
			#else
				register int a, b; 
				a = i*(size_t)mm + j;
				b = j*(size_t)mm + i;
				sum += (v[a] - v[b])*(v[a] - v[b]);
			#endif
			}
		}
#ifdef CUDALA
	}else{
laerror("nonsymmetry not implemented on GPU yet");

	}
#endif
	return sum;
}

/***************************************************************************//**
 * compute matrix non-hermiticity
 ******************************************************************************/
template <>
const double NRMat<std::complex<double> >::nonhermiticity() const {
#ifdef DEBUG
	if (nn != mm) laerror("NRMat<T>:nonsymmetry() invalid for non-square matrix");
#endif
double sum = 0;
std::complex<double> tmp;
#ifdef CUDALA
	if(location == cpu){
#endif
		for(register int i=1; i<nn; i++){
			for(register int j=0; j<=i; j++){
			#ifdef MATPTR
				tmp = std::complex<double> (v[i][j].real()-v[j][i].real(),v[i][j].imag()+v[j][i].imag()); 
			#else
				register int a, b; 
				a = i*(size_t)mm + j;
				b = j*(size_t)mm + i;
				tmp = std::complex<double> (v[a].real() - v[b].real(), v[a].imag()+v[b].imag());
			#endif
			sum += tmp.real()*tmp.real()+tmp.imag()*tmp.imag();
			}
		}
#ifdef CUDALA
	}else{
laerror("nonsymmetry not implemented on GPU yet");

	}
#endif
	return sum;
}

template <>
const double NRMat<std::complex<double> >::nonsymmetry() const {
#ifdef DEBUG
        if (nn != mm) laerror("NRMat<T>:nonsymmetry() invalid for non-square matrix");
#endif
double sum = 0;
std::complex<double> tmp;
#ifdef CUDALA
        if(location == cpu){
#endif
                for(register int i=1; i<nn; i++){
                        for(register int j=0; j<i; j++){
                        #ifdef MATPTR
                                tmp = v[i][j]-v[j][i];
                        #else
                                register int a, b;
                                a = i*(size_t)mm + j;
                                b = j*(size_t)mm + i;
                                tmp = v[a] - v[b];
                        #endif
                        sum += tmp.real()*tmp.real()+tmp.imag()*tmp.imag();
                        }
                }
#ifdef CUDALA
        }else{
laerror("nonsymmetry not implemented on GPU yet");

        }
#endif
        return sum;
}



/***************************************************************************//**
 * create complex double-precision matrix from real double-precision matrix \f$A\f$
 * @param[in] rhs real double-precision matrix \f$A\f$
 * @param[in] imagpart flag indicating whether the matrix \f$A\f$ should be considered as a real
 *  or imaginary part of the complex matrix being created
 ******************************************************************************/
template<>
NRMat<std::complex<double> >::NRMat(const NRMat<double> &rhs, bool imagpart): nn(rhs.nrows()), mm(rhs.ncols()), count(new int(1)) {
	const size_t nn_mm = (size_t)nn*mm;
#ifdef CUDALA
	if(location == cpu){
#endif
	#ifdef MATPTR
	        v = new std::complex<double>*[n];
	        v[0] = new std::complex<double>[nn_mm];
	        for(register int i=1; i<n; i++) v[i] = v[i-1] + m;
	
		memset(v[0], 0, nn_mm*sizeof(std::complex<double>));
	        cblas_dcopy(nn_mm, &rhs[0][0], 1, ((double *)v[0]) + (imagpart?1:0), 2);
	#else
	        v = new std::complex<double>[nn_mm];
	        memset(v, 0, nn_mm*sizeof(std::complex<double>));
	
		cblas_dcopy(nn_mm, &rhs[0][0], 1, ((double *)v) + (imagpart?1:0), 2);
	#endif
#ifdef CUDALA
	}else{
		v = (std::complex<double>*)gpualloc(sizeof(std::complex<double>)*nn_mm);
		std::complex<double> *_val = gpuputcomplex(CZERO);
		cublasZcopy(nn_mm, (cuDoubleComplex*)_val, 0, (cuDoubleComplex*)v, 1);
		TEST_CUBLAS("cublasZcopy");
		gpufree(_val);

		cublasDcopy(nn_mm, (double*)(&rhs[0][0]), 1, ((double*)v) + (imagpart?1:0), 2);
		TEST_CUBLAS("cublasDcopy");
	}
#endif
}



/***************************************************************************//**
 * create double-precision matrix from complex double-precision matrix \f$A\f$
 * @param[in] rhs complex double-precision matrix \f$A\f$
 * @param[in] imagpart flag indicating whether the matrix \f$A\f$ should be taken as the real
 *  or imaginary part of the input complex matrix 
 ******************************************************************************/
template<>
NRMat<double>::NRMat(const NRMat<std::complex<double> > &rhs, bool imagpart): nn(rhs.nrows()), mm(rhs.ncols()), count(new int(1)) {
	const size_t nn_mm = (size_t) nn*mm;
#ifdef CUDALA
	if(location == cpu){
#endif
	#ifdef MATPTR
	        v = new double*[n];
	        v[0] = new double[nn_mm];
	        for(register int i=1; i<n; i++) v[i] = v[i-1] + m;
	
	        cblas_dcopy(nn_mm, ((double *)&rhs[0][0]) + (imagpart?1:0), 2, v[0], 1);
	#else
	        v = new double[nn_mm];
		cblas_dcopy(nn_mm, ((double *) &rhs[0][0]) + (imagpart?1:0), 2, v , 1);
	#endif
#ifdef CUDALA
	}else{
		v = (double *)gpualloc(sizeof(double)*nn_mm);
		cublasDcopy(nn_mm, ((double*)&rhs[0][0])+ (imagpart?1:0), 2, v , 1);
		TEST_CUBLAS("cublasDcopy");
	}
#endif
}





/***************************************************************************//**
 * output of a matrix of general type via lawritemat
 ******************************************************************************/
template <typename T>
void NRMat<T>::fprintf(FILE *file, const char *format, const int modulo) const {
#ifdef CUDALA
	if(location == cpu){
#endif
		lawritemat(file, (const T*)(*this), nn, mm, format, 2, modulo, 0);
#ifdef CUDALA
	}else{
		NRMat<T> tmp = *this;
		tmp.moveto(cpu);
		lawritemat(file, (const T*)(tmp), nn, mm, format, 2, modulo, 0);
	}
#endif
	
}

/***************************************************************************//**
 * input of a matrix of general type via lawritemat
 ******************************************************************************/
template <typename T>
void NRMat<T>::fscanf(FILE *f, const char *format) {
	T *p;
	NRMat<T> *tmp;

	int n(0), m(0);
	if (::fscanf(f, "%d %d", &n, &m) != 2) laerror("cannot read matrix dimensions");

#ifdef CUDALA
	if(location == cpu){
		p = *this;
	}else{
		tmp = new NRMat<T>(n, m, this->location);
		p = *tmp;
	}
#endif
	resize(n, m);
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			if(::fscanf(f,format, p++) != 1) laerror("cannot read matrix element");
		}
	}

#ifdef CUDALA
	if(location != cpu){
		delete tmp;
	}
#endif
}

//-----------------------------------------------------------------------------
// BLAS specializations for double and complex<double> types
//-----------------------------------------------------------------------------

/***************************************************************************//**
 * for a given real matrix \f$A\f$ compute \f$A^\mathrm{T}A\f$
 * @return real NRSMat object because of the symmetry of \f$A^\mathrm{T}A\f$
 ******************************************************************************/
template<>
const NRSMat<double> NRMat<double>::transposedtimes() const {
	int i(0), j(0);
	NRSMat<double> r(mm, getlocation());//!< resulting matrix has mm rows
#ifdef CUDALA
	if(location == cpu){
#endif
		for(i=0; i<mm; ++i){
			for(j=0; j<=i; ++j){
			#ifdef MATPTR
				r(i, j) = cblas_ddot(nn, v[0] + i, mm, v[0] + j, mm);
			#else
				r(i, j) = cblas_ddot(nn, v + i, mm, v + j, mm);
			#endif
			}
		}
#ifdef CUDALA
	}else{
		for(i=0; i<mm; ++i){
			for(j=0; j<=i; ++j){
				r(i, j) = cublasDdot(nn, v + i, mm, v + j, mm);
				TEST_CUBLAS("cublasDdot");
			}
		}
		r.moveto(this->location);
	}
#endif
	return r;
}

/***************************************************************************//**
 * for a given complex matrix \f$A\f$ compute \f$A^\dagger{}A\f$
 * @return complex NRSMat object because of the hermiticity of \f$A^\dagger{}A\f$
 ******************************************************************************/
template<>
const NRSMat<std::complex<double> >  NRMat<std::complex<double> >::transposedtimes() const {
	int i(0), j(0);
	NRSMat<std::complex<double> > r(mm, getlocation());
#ifdef CUDALA
	if(location == cpu){
#endif
		for(i=0; i<mm; ++i){
			for(j=0; j<=i; ++j){
			#ifdef MATPTR
				cblas_zdotc_sub(nn, v[0] + i , mm, v[0] + j, mm, &r(i,j));
			#else
				cblas_zdotc_sub(nn, v + i , mm, v + j, mm, &r(i,j));
			#endif
			}
		}
#ifdef CUDALA
	}else{
		for(i=0; i<mm; ++i){
			for(j=0; j<=i; ++j){
				cuDoubleComplex val = cublasZdotc(nn, (const cuDoubleComplex*)(v + i), mm, (const cuDoubleComplex*)(v + j), mm);
				TEST_CUBLAS("cublasZdotc");
				r(i, j) = *(reinterpret_cast<std::complex<double>*> (&val));
			}
		}
		r.moveto(this->location);
	}
#endif
	return r;
}

/***************************************************************************//**
 * for a given matrix \f$A\f$ (general type) compute \f$A^\mathrm{T}A\f$
 * @return NRSMat<T> object because of the symmetry of the result
 ******************************************************************************/
template <typename T>
const NRSMat<T>  NRMat<T>::transposedtimes() const {
	int i(0), j(0);
	NOT_GPU(*this);

	NRSMat<T> r(mm, getlocation());
	for(i=0; i<mm; ++i){
		for(j=0; j<=i; ++j){
			T s =(T)0;
			for(int k=0; k<nn; ++k){
				s += (*this)(k,i) * (*this)(k,j);
			}
			r(i,j) = s;
		}
	}
	return r;
}


/***************************************************************************//**
 * for a given real matrix \f$A\f$ compute \f$AA^\mathrm{T}\f$
 * @return real NRSMat object because of the symmetry of \f$AA^\mathrm{T}\f$
 ******************************************************************************/
template<>
const NRSMat<double>  NRMat<double>::timestransposed() const {
	int i(0), j(0);
	NRSMat<double> r(nn, getlocation());//!< resulting matrix has nn rows
#ifdef CUDALA
	if(location == cpu){
#endif
		for(i=0; i<nn; ++i){
			for(j=0; j<=i; ++j){
			#ifdef MATPTR
				r(i, j) = cblas_ddot(mm, v[i], 1, v[j], 1);
			#else
				r(i, j) = cblas_ddot(mm, v + i*(size_t)mm, 1, v + j*(size_t)mm, 1);
			#endif
			}
		}
#ifdef CUDALA
	}else{
		for(i=0; i<nn; ++i){
			for(j=0; j<=i; ++j){
				r(i, j) = cublasDdot(nn, v + i*(size_t)mm, 1, v + j*(size_t)mm, 1);
				TEST_CUBLAS("cublasDdot");
			}
		}
		r.moveto(this->location);
	}
#endif
	return r;
}

/***************************************************************************//**
 * for a given complex matrix \f$A\f$ compute \f$AA^\dagger{}\f$
 * @return complex NRSMat object because of the hermiticity of \f$AA^\dagger{}\f$
 ******************************************************************************/
template<>
const NRSMat<std::complex<double> >  NRMat<std::complex<double> >::timestransposed() const {
	int i(0), j(0);
	NRSMat<std::complex<double> > r(nn, getlocation());
#ifdef CUDALA
	if(location == cpu){
#endif
		for(i=0; i<nn; ++i){
			for(j=0; j<=i; ++j){
			#ifdef MATPTR
				cblas_zdotc_sub(nn, v[i], 1, v[j], 1, &r(i,j));
			#else
				cblas_zdotc_sub(nn, v + i*(size_t)mm, 1, v + j*(size_t)mm, 1, &r(i,j));
			#endif
			}
		}
#ifdef CUDALA
	}else{
		for(i=0; i<mm; ++i){
			for(j=0; j<=i; ++j){
				cuDoubleComplex val = cublasZdotc(nn, (const cuDoubleComplex *)(v + i*(size_t)mm), 1, (const cuDoubleComplex *)(v + j*(size_t)mm), 1);
				TEST_CUBLAS("cublasZdotc");
				r(i, j) = *(reinterpret_cast<std::complex<double>*> (&val));
			}
		}
		r.moveto(this->location);
	}
#endif
	return r;
}


/***************************************************************************//**
 * for a given matrix \f$A\f$ (general type) compute \f$A^\mathrm{T}A\f$
 * @return NRSMat<T> object because of the symmetry of the result
 ******************************************************************************/
template <typename T>
const NRSMat<T> NRMat<T>::timestransposed() const {
	int i(0), j(0);
	NOT_GPU(*this);

	NRSMat<T> r(nn);
	for(i=0; i<nn; ++i){
		for(j=0; j<=i; ++j){
			T s = (T)0;
			for(int k=0; k<mm; ++k) s += (*this)(i,k) * (*this)(j,k);
			r(i,j)=s;
		}
	}
	return r;
}


/***************************************************************************//**
 * fill given real matrix with random numbers
 * @param[in] x generate random numbers from the interval [0, x]
 ******************************************************************************/
template<>
void NRMat<double>::randomize(const double &x) {
#ifdef CUDALA
	if(location == cpu){
#endif
		for(register int i=0; i<nn; ++i){
			for(register int j=0; j<mm; ++j){
				(*this)(i,j) = x*RANDDOUBLESIGNED();
			}
		}
#ifdef CUDALA
	}else{
		NRMat<double> tmp(nn, mm, cpu);
		double *tmp_data = tmp;
		for(register size_t i=0; i<(size_t)nn*mm; ++i){
			tmp_data[i] = x*RANDDOUBLESIGNED();
		}
		tmp.moveto(this->location);
		*this |= tmp;
	}
#endif
}


/***************************************************************************//**
 * fill given complex matrix with random numbers
 * real/imaginary components are generated independently
 * @param[in] x generate random numbers from the interval [0, x]
 ******************************************************************************/
template<>
void NRMat<std::complex<double> >::randomize(const double &x) {
#ifdef CUDALA
	if(location == cpu){
#endif
		for(register int i=0; i<nn; ++i){
			for(register int j=0; j<mm; ++j){
				const double re = x*RANDDOUBLESIGNED();
				const double im = x*RANDDOUBLESIGNED();
				(*this)(i,j) = std::complex<double>(re, im);
			}
		}
#ifdef CUDALA
	}else{
		NRMat<std::complex<double> > tmp(nn, mm, cpu);
		std::complex<double> *tmp_data = tmp;
		for(register size_t i=0; i<(size_t)nn*mm; ++i){
			const double re = x*RANDDOUBLESIGNED();
			const double im = x*RANDDOUBLESIGNED();
			tmp_data[i] = std::complex<double>(re, im);
		}
		tmp.moveto(this->location);
		*this |= tmp;
	}
#endif
}


/***************************************************************************//**
 * scale real matrix with a real factor
 * @param[in] a scaling factor
 * @return reference to the modified matrix
 ******************************************************************************/
template<>
NRMat<double>& NRMat<double>::operator*=(const double &a) {
	copyonwrite();
#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_dscal((size_t)nn*mm, a, *this, 1);
#ifdef CUDALA
	}else{
		cublasDscal((size_t)nn*mm, a, v, 1);
		TEST_CUBLAS("cublasDscal");
	}
#endif
	return *this;
}


/***************************************************************************//**
 * scale complex matrix with a complex factor
 * @param[in] a scaling factor
 * @return reference to the modified matrix
 ******************************************************************************/
template<>
NRMat<std::complex<double> > &
NRMat<std::complex<double> >::operator*=(const std::complex<double> &a) {
	copyonwrite();
#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_zscal((size_t)nn*mm, &a, (*this)[0], 1);
#ifdef CUDALA
	}else{
		const cuDoubleComplex fac = *(reinterpret_cast<const cuDoubleComplex*> (&a));
		cublasZscal((size_t)nn*mm, fac, (cuDoubleComplex *)v, 1);
		TEST_CUBLAS("cublasZscal");
	}
#endif
	return *this;
}


/***************************************************************************//**
 * scale matrix of type T with a factor
 * @param[in] a scaling factor
 * @return reference to the modified matrix
 ******************************************************************************/
template <typename T>
NRMat<T> & NRMat<T>::operator*=(const T &a) {
	NOT_GPU(*this);
        copyonwrite();
#ifdef MATPTR
         for(register size_t i=0; i< (size_t)nn*mm; i++) v[0][i] *= a;
#else
         for(register size_t i=0; i< (size_t)nn*mm; i++) v[i] *= a;
#endif
        return *this;
}


/***************************************************************************//**
 * add a given real matrix \f$A\f$ to the current real matrix
 * @param[in] rhs matrix \f$A\f$
 * @return reference to the modified matrix
 ******************************************************************************/
template<>
NRMat<double> & NRMat<double>::operator+=(const NRMat<double>  &rhs) {
#ifdef DEBUG
	if (nn != rhs.nn || mm != rhs.mm) laerror("incompatible matrices");
#endif
	SAME_LOC(*this, rhs);
	copyonwrite();
#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_daxpy((size_t)nn*mm, 1.0, rhs, 1, *this, 1);
#ifdef CUDALA
	}else{
		cublasDaxpy((size_t)nn*mm, 1.0, rhs, 1, v, 1);
		TEST_CUBLAS("cublasDaxpy");
	}
#endif
	return *this;
}

/***************************************************************************//**
 * add a given complex matrix \f$A\f$ to the current complex matrix
 * @param[in] rhs complex matrix \f$A\f$
 * @return reference to the modified matrix
 ******************************************************************************/
template<>
NRMat<std::complex<double> > &
NRMat<std::complex<double> >::operator+=(const NRMat< std::complex<double> >  &rhs) {
#ifdef DEBUG
	if (nn != rhs.nn || mm != rhs.mm) laerror("incompatible matrices");
#endif
	SAME_LOC(*this, rhs);
	copyonwrite();
#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_zaxpy((size_t)nn*mm, &CONE, rhs[0], 1, (*this)[0], 1);
#ifdef CUDALA
	}else{
		cublasZaxpy((size_t)nn*mm, CUONE, (cuDoubleComplex*)(rhs[0]), 1, (cuDoubleComplex*)((*this)[0]), 1);
	}
#endif
	return *this;
}


/***************************************************************************//**
 * subtract a given real matrix \f$A\f$ from the current real matrix
 * @param[in] rhs matrix \f$A\f$
 * @return reference to the modified matrix
 ******************************************************************************/
template<>
NRMat<double> & NRMat<double>::operator-=(const NRMat<double>  &rhs) {
#ifdef DEBUG
	if (nn != rhs.nn || mm != rhs.mm) laerror("incompatible matrices");
#endif
	SAME_LOC(*this,rhs);
	copyonwrite();
#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_daxpy((size_t)nn*mm, -1.0, rhs, 1, *this, 1);
#ifdef CUDALA
	}else{
		cublasDaxpy((size_t)nn*mm, -1.0, rhs, 1, v, 1);
	}
#endif
	return *this;
}


/***************************************************************************//**
 * subtract a given complex matrix \f$A\f$ from the current complex matrix
 * @param[in] rhs matrix \f$A\f$
 * @return reference to the modified matrix
 ******************************************************************************/
template<>
NRMat< std::complex<double> > &
NRMat< std::complex<double> >::operator-=(const NRMat< std::complex<double> >  &rhs) {
#ifdef DEBUG
	if (nn != rhs.nn || mm != rhs.mm) laerror("incompatible matrices");
#endif
	SAME_LOC(*this, rhs);
	copyonwrite();

#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_zaxpy((size_t)nn*mm, &CMONE, rhs[0], 1, (*this)[0], 1);
#ifdef CUDALA
	}else{
		cublasZaxpy((size_t)nn*mm, CUMONE, (cuDoubleComplex*)(rhs[0]), 1, (cuDoubleComplex*)((*this)[0]), 1);
	}
#endif
	return *this;
}



/***************************************************************************//**
 * add a given sparse real matrix \f$A\f$ stored in packed form to the current
 * real matrix
 * @param[in] rhs symmetric real matrix \f$A\f$ in packed form
 * @return reference to the modified matrix
 * @see NRSMat<T>
 ******************************************************************************/
template<>
NRMat<double> & NRMat<double>::operator+=(const NRSMat<double> &rhs) {
#ifdef DEBUG
	if (nn != rhs.nn || mm != rhs.nn) laerror("incompatible matrices");
#endif
	const double *p = rhs;

	SAME_LOC(*this, rhs);
	copyonwrite();

#ifdef CUDALA
	if(location == cpu){
#endif
		for(register int i=0; i<nn; i++){
			cblas_daxpy(i + 1, 1.0, p, 1, (*this)[i], 1); 
			p += i + 1;
		}

		p = rhs; p++;
		for(int i=1; i<nn; i++){
			cblas_daxpy(i, 1.0, p, 1, (*this)[0] + i, nn);
			p += i + 1;
		}
#ifdef CUDALA
	}else{
		for(register int i=0; i<nn; i++){
			cublasDaxpy(i + 1, 1.0, p, 1, (*this)[i], 1); 
			p += i + 1;
		}

		p = rhs; p++;
		for(int i=1; i<nn; i++){
			cublasDaxpy(i, 1.0, p, 1, (*this)[0] + i, nn);
			p += i + 1;
		}
	}
#endif
	return *this;
}


/***************************************************************************//**
 * add a given sparse complex matrix \f$A\f$ stored in packed form to the current
 * complex matrix
 * @param[in] rhs symmetric complex matrix \f$A\f$ in packed form
 * @return reference to the modified matrix
 * @see NRSMat<T>
 ******************************************************************************/
template<>
NRMat< std::complex<double> > & 
NRMat< std::complex<double> >::operator+=(const NRSMat< std::complex<double> > &rhs)
{
#ifdef DEBUG
	if (nn != rhs.nn || mm != rhs.nn) laerror("incompatible matrices");
#endif
	const std::complex<double> *p = rhs;

	SAME_LOC(*this, rhs);
	copyonwrite();
#ifdef CUDALA
	if(location == cpu){
#endif
		for(register int i=0; i<nn; i++){
			cblas_zaxpy(i + 1, &CONE, p, 1, (*this)[i], 1); 
			p += i + 1;
		}
		p = rhs; p++;
		for(register int i=1; i<nn; i++){
			cblas_zaxpy(i, &CONE, p, 1, (*this)[0]+i, nn);
			p += i+1;
		}
#ifdef CUDALA
	}else{
		for(register int i=0; i<nn; i++){
			cublasZaxpy(i + 1, CUONE, (cuDoubleComplex*)p, 1, (cuDoubleComplex*)((*this)[i]), 1); 
			p += i + 1;
		}
		p = rhs; p++;
		for(register int i=1; i<nn; i++){
			cublasZaxpy(i, CUONE, (cuDoubleComplex*)p, 1, (cuDoubleComplex*)((*this)[0]+i), nn);
			p += i+1;
		}

	}
#endif
	return *this;
}


/***************************************************************************//**
 * add a given general sparse matrix \f$A\f$ stored in packed form to the current
 * general matrix (of type T)
 * @param[in] rhs symmetric general matrix \f$A\f$ in packed form
 * @return reference to the modified matrix
 * @see NRSMat<T>
 ******************************************************************************/
template <typename T>
NRMat<T> & NRMat<T>::operator+=(const NRSMat<T> &rhs) {
#ifdef DEBUG
	if (nn != rhs.nn || mm != rhs.nn) laerror("incompatible matrices");
#endif
	const T *p = rhs;

	SAME_LOC(*this, rhs);
	NOT_GPU(*this);

	copyonwrite();

	for(register int i=0; i<nn; i++) {
		for(register int j=0; j<i+1; ++j) *((*this)[i]+j) += p[j];
		p += i+1;
	}
	p = rhs; p++;
	for(register int i=1; i<nn; i++) {
		for(register int j=0; j<i; ++j) *((*this)[j]+i) += p[j];
		p += i+1;
        }
        return *this;
}


/***************************************************************************//**
 * subtract a given sparse real matrix \f$A\f$ stored in packed form from
 * the current real matrix
 * @param[in] rhs symmetric real matrix \f$A\f$ in packed form
 * @return reference to the modified matrix
 * @see NRSMat<T>
 ******************************************************************************/
template<>
NRMat<double> & NRMat<double>::operator-=(const NRSMat<double> &rhs)
{
#ifdef DEBUG
	if (nn != rhs.nn || mm != rhs.nn) laerror("incompatible matrices");
#endif
	const double *p = rhs;
	SAME_LOC(*this, rhs);
	copyonwrite();

#ifdef CUDALA
	if(location == cpu){
#endif
		for(register int i=0; i<nn; i++) {
			cblas_daxpy(i+1, -1.0, p, 1, (*this)[i], 1); 
			p += i+1;
		}
		p = rhs; p++;
		for(register int i=1; i<nn; i++) {
			cblas_daxpy(i, -1.0, p, 1, (*this)[0]+i, nn);
			p += i+1;
		}
#ifdef CUDALA
	}else{
		for(register int i=0; i<nn; i++) {
			cublasDaxpy(i+1, -1.0, p, 1, (*this)[i], 1); 
			p += i+1;
		}
		p = rhs; p++;
		for(register int i=1; i<nn; i++) {
			cublasDaxpy(i, -1.0, p, 1, (*this)[0]+i, nn);
			p += i+1;
		}
	}
#endif
	return *this;
}

/***************************************************************************//**
 * subtract a given sparse complex matrix \f$A\f$ stored in packed form from
 * the current complex matrix
 * @param[in] rhs symmetric complex matrix \f$A\f$ in packed form
 * @return reference to the modified matrix
 * @see NRSMat<T>
 ******************************************************************************/
template<>
NRMat<std::complex<double> > & 
NRMat<std::complex<double> >::operator-=(const NRSMat<std::complex<double> > &rhs) {
#ifdef DEBUG
	if (nn != rhs.nn || mm != rhs.nn) laerror("incompatible matrices");
#endif
	const std::complex<double> *p = rhs;

	SAME_LOC(*this, rhs);
	copyonwrite();
#ifdef CUDALA
	if(location == cpu){
#endif
		for(register int i=0; i<nn; i++){
			cblas_zaxpy(i + 1, &CMONE, p, 1, (*this)[i], 1); 
			p += i + 1;
		}
		p = rhs; p++;
		for(register int i=1; i<nn; i++){
			cblas_zaxpy(i, &CMONE, p, 1, (*this)[0]+i, nn);
			p += i + 1;
		}
#ifdef CUDALA
	}else{
		for(register int i=0; i<nn; i++){
			cublasZaxpy(i + 1, CUMONE, (cuDoubleComplex*)p, 1, (cuDoubleComplex*)((*this)[i]), 1); 
			p += i + 1;
		}
		p = rhs; p++;
		for(register int i=1; i<nn; i++){
			cublasZaxpy(i, CUMONE, (cuDoubleComplex*)p, 1, (cuDoubleComplex*)((*this)[0]+i), nn);
			p += i + 1;
		}
	}
#endif
	return *this;
}

/***************************************************************************//**
 * subtract a given general sparse matrix \f$A\f$ stored in packed form from
 * the current general matrix (of type T)
 * @param[in] rhs symmetric general matrix \f$A\f$ in packed form
 * @return reference to the modified matrix
 * @see NRSMat<T>
 ******************************************************************************/
template <typename T>
NRMat<T> & NRMat<T>::operator-=(const NRSMat<T> &rhs) {
#ifdef DEBUG
	if (nn != rhs.nn || mm != rhs.nn) laerror("incompatible matrices");
#endif
	SAME_LOC(*this, rhs);
	NOT_GPU(*this);

	const T *p = rhs;
	copyonwrite();
	for(register int i=0; i<nn; i++){
		for(register int j=0; j<i+1; ++j) *((*this)[i]+j) -= p[j];
		p += i+1;
	}
	p = rhs; p++;
	for(register int i=1; i<nn; i++){
		for(register int j=0; j<i; ++j) *((*this)[j]+i) -= p[j];
		p += i+1;
        }
        return *this;
}


/***************************************************************************//**
 * compute scalar product of this matrix \f$A\f$ with given real matrix \f$B\f$
 * i.e. determine \f$\sum_{i,j}A_{i,j}B_{i,j}\f$
 * @param[in] rhs matrix \f$B\f$
 * @return computed scalar product
 ******************************************************************************/
template<>
const double NRMat<double>::dot(const NRMat<double> &rhs) const {
#ifdef DEBUG
	if(nn != rhs.nn || mm != rhs.mm) laerror("incompatible matrices in NRMat<double>::dot(const NRMat<double>&)");
#endif
	double ret(0.0);
	SAME_LOC(*this, rhs);
#ifdef CUDALA
	if(location == cpu){
#endif
		ret = cblas_ddot((size_t)nn*mm, (*this)[0], 1, rhs[0], 1);
#ifdef CUDALA
	}else{
		ret = cublasDdot((size_t)nn*mm, v, 1, rhs.v, 1);
	}
#endif
	return ret;
}



/***************************************************************************//**
 * compute scalar product of this matrix \f$A\f$ with given complex matrix \f$B\f$
 * i.e. determine \f$\sum_{i,j}A^{*}_{i,j}B_{i,j}\f$
 * @param[in] rhs matrix \f$B\f$
 * @return computed scalar product
 ******************************************************************************/
template<>
const std::complex<double>
NRMat<std::complex<double> >::dot(const NRMat<std::complex<double> > &rhs) const {
#ifdef DEBUG
	if(nn != rhs.nn || mm != rhs.mm) laerror("incompatible matrices in NRMat<std::complex<double> >::dot(const NRMat<std::complex<double> >&)");
#endif

	std::complex<double> ret(0.0, 0.0);
#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_zdotc_sub((size_t)nn*mm, (*this)[0], 1, rhs[0], 1, &ret);
#ifdef CUDALA
	}else{
		cuDoubleComplex val = cublasZdotc((size_t)nn*mm, (cuDoubleComplex*)v, 1, (cuDoubleComplex*)(rhs.v), 1);
		ret = *(reinterpret_cast<std::complex<double>*> (&val));
	}
#endif
	return ret;
}


template<>
const NRMat<int> NRMat<int>::operator*(const NRMat<int> &rhs) const {
#ifdef DEBUG            
        if(mm != rhs.nn) laerror("incompatible matrices in NRMat<int>::operator*(const NRMat<int>&)");
        if(mm<=0 ||rhs.mm <= 0) laerror("illegal matrix dimension in gemm");
#endif  
	NOT_GPU(rhs);
	NOT_GPU(*this);
        NRMat<int> result(nn, rhs.mm);
	result.clear();
	for(int i=0;i<nn;++i)
		for(int j=0; j<mm; ++j)
			for(int k=0; k<rhs.mm; ++k)
				result(i,k) += (*this)(i,j)*rhs(j,k);
        return result;
}               


/***************************************************************************//**
 * compute product of this matrix \f$A\f$ with given real matrix \f$B\f$
 * @param[in] rhs matrix \f$B\f$
 * @return computed product by value
 ******************************************************************************/
template<>
const NRMat<double> NRMat<double>::operator*(const NRMat<double> &rhs) const {
#ifdef DEBUG
	if(mm != rhs.nn) laerror("incompatible matrices in NRMat<double>::operator*(const NRMat<double>&)");
	if(rhs.mm <= 0) laerror("illegal matrix dimension in gemm");
#endif
	SAME_LOC(*this, rhs);
	NRMat<double> result(nn, rhs.mm, getlocation());
#ifdef CUDALA
	if(location == cpu){
#endif
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nn, rhs.mm, mm, 1.0,
		*this, mm, rhs, rhs.mm, 0.0, result, rhs.mm);
#ifdef CUDALA
	}else{
		cublasDgemm('N', 'N', rhs.mm, nn, mm, 1.0, rhs, rhs.mm, *this, mm, 0.0, result, rhs.mm);
	}
#endif
	return result;
}

/***************************************************************************//**
 * compute product of this matrix \f$A\f$ with given complex matrix \f$B\f$
 * @param[in] rhs matrix \f$B\f$
 * @return computed product by value
 ******************************************************************************/
template<>
const NRMat< std::complex<double> > 
NRMat< std::complex<double> >::operator*(const NRMat< std::complex<double> > &rhs) const {
#ifdef DEBUG
	if(mm != rhs.nn) laerror("incompatible matrices in NRMat<std::complex<double> >::operator*(const NRMat<std::complex<double> >&)");
	if(rhs.mm <= 0) laerror("illegal matrix dimension in gemm");
#endif
	SAME_LOC(*this, rhs);
	NRMat<std::complex<double> > result(nn, rhs.mm, getlocation());
#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nn, rhs.mm, mm,
			&CONE, (*this)[0], mm, rhs[0], rhs.mm, &CZERO, result[0], rhs.mm);
#ifdef CUDALA
	}else{
		cublasZgemm('N', 'N', rhs.mm, nn, mm, CUONE, 
				(cuDoubleComplex*)rhs.v, rhs.mm, (cuDoubleComplex*)(this->v), mm, CUZERO, (cuDoubleComplex*)result.v, rhs.mm);
	}
#endif
	return result;
}

/***************************************************************************//**
 * multiply this real matrix \f$A\f$ by diagonal real matrix \f$D\f$ from left
 * because of cuBlas implementation, \f$D\f$ is required to be placed in CPU memory
 * @param[in] rhs real vector represeting the diagonal of matrix \f$D\f$
 ******************************************************************************/
template<>
void NRMat<double>::diagmultl(const NRVec<double> &rhs) {
#ifdef DEBUG
	if(nn != rhs.size()) laerror("incompatible matrices in NRMat<double>::diagmultl(const NRVec<double>&)");
#endif
	NOT_GPU(rhs);
	copyonwrite();

#ifdef CUDALA
	if(location == cpu){
#endif		
		for(register int i=0; i<nn; i++){ cblas_dscal(mm, rhs[i], (*this)[i], 1); }
#ifdef CUDALA
	}else{
		for(register int i=0; i<nn; i++){ cublasDscal(mm, rhs[i], v + i*(size_t)mm, 1); }
	}
#endif
}

/***************************************************************************//**
 * multiply this complex matrix \f$A\f$ by diagonal complex matrix \f$D\f$ from left
 * because of cuBlas implementation, \f$D\f$ is required to be placed in CPU memory
 * @param[in] rhs complex vector represeting the diagonal of matrix \f$D\f$
 ******************************************************************************/
template<>
void NRMat< std::complex<double> >::diagmultl(const NRVec< std::complex<double> > &rhs) {
#ifdef DEBUG
	if (nn != rhs.size()) laerror("incompatible matrices in NRMat<std::complex<double> >::diagmultl(const NRVec<std::complex<double> >&)");
#endif
	NOT_GPU(rhs);
	copyonwrite();

#ifdef CUDALA
	if(location == cpu){
#endif		
		for(register int i=0; i<nn; i++){ cblas_zscal(mm, &(rhs[i]), (*this)[i], 1); }
#ifdef CUDALA
	}else{
		for(register int i=0; i<nn; i++){
			const cuDoubleComplex alpha = make_cuDoubleComplex(rhs[i].real(), rhs[i].imag());
			cublasZscal(mm, alpha, (cuDoubleComplex*)(v + i*(size_t)mm), 1);
		}
	}
#endif
}

/***************************************************************************//**
 * multiply this real matrix \f$A\f$ by diagonal real matrix \f$D\f$ from right
 * because of cuBlas implementation, \f$D\f$ is required to be placed in CPU memory
 * @param[in] rhs real vector represeting the diagonal of matrix \f$D\f$
 ******************************************************************************/
template<>
void NRMat<double>::diagmultr(const NRVec<double> &rhs) {
#ifdef DEBUG
	if(mm != rhs.size()) laerror("incompatible matrices in NRMat<double>::diagmultr(const NRVec<double>&)");
#endif
	NOT_GPU(rhs);
	copyonwrite();

#ifdef CUDALA
	if(location == cpu){
#endif		
		for(register int i=0; i<mm; i++){ cblas_dscal(nn, rhs[i], &(*this)(0, i), mm); }
#ifdef CUDALA
	}else{
		for(register int i=0; i<mm; i++){ cublasDscal(mm, rhs[i], v + i, mm); }
	}
#endif
}


/***************************************************************************//**
 * multiply this complex matrix \f$A\f$ by diagonal complex matrix \f$D\f$ from left
 * @param[in] rhs complex vector represeting the diagonal of matrix \f$D\f$
 ******************************************************************************/
template<>
void NRMat< std::complex<double> >::diagmultr(const NRVec< std::complex<double> > &rhs) {
#ifdef DEBUG
	if(mm != rhs.size()) laerror("incompatible matrices in NRMat<std::complex<double> >::diagmultr(const NRVec<std::complex<double> >&)");
#endif
	NOT_GPU(rhs);
	copyonwrite();

#ifdef CUDALA
	if(location == cpu){
#endif		
		for(register int i=0; i<mm; i++){ cblas_zscal(nn, &(rhs[i]), &(*this)(0, i), mm); }
#ifdef CUDALA
	}else{
		for(register int i=0; i<mm; i++){
			const cuDoubleComplex alpha = make_cuDoubleComplex(rhs[i].real(), rhs[i].imag());
			cublasZscal(nn, alpha, (cuDoubleComplex*)(v + i), mm);
		}
	}
#endif
}

/***************************************************************************//**
 * multiply this real matrix \f$A\f$ by symmetric matrix \f$S\f$
 * \f$S\f$ is stored in packed form, therefore dspmv routine is used
 * @param[in] rhs real symmetric matrix \f$S\f$ stored in packed form
 * @return \f$A\times\S\f$ by value
 ******************************************************************************/
template<>
const NRMat<double> 
NRMat<double>::operator*(const NRSMat<double> &rhs) const {
#ifdef DEBUG
	if(mm != rhs.nrows()) laerror("incompatible matrices int NRMat<double>::operator*(const NRSMat<double> &)");
#endif
	SAME_LOC(*this, rhs);
	const int rhs_ncols = rhs.ncols();
	NRMat<double> result(nn, rhs_ncols, getlocation());

#ifdef CUDALA
	if(location == cpu){
#endif
		for(register int i=0; i<nn; i++){
			cblas_dspmv(CblasRowMajor, CblasLower, mm, 1.0, &rhs[0], 
					(*this)[i], 1, 0.0, result[i], 1);
		}
#ifdef CUDALA
	}else{
		for(register int i=0; i<nn; i++){
			cublasDspmv('U', mm, 1.0, rhs.v, v + i*(size_t)mm, 1, 0.0, result.v + i*(size_t)rhs_ncols, 1);
		}
	}
#endif
	return result;
}


/***************************************************************************//**
 * multiply this complex matrix \f$A\f$ by symmetric complex matrix \f$S\f$
 * \f$S\f$ is stored in packed form, therefore zhpmv routine is used
 * @param[in] rhs complex symmetric matrix \f$S\f$ stored in packed form
 * @return \f$A\times\S\f$ by value
 ******************************************************************************/
template<>
const NRMat< std::complex<double> >
NRMat< std::complex<double> >::operator*(const NRSMat< std::complex<double> > &rhs) const {
#ifdef DEBUG
	if(mm != rhs.nrows()) laerror("incompatible matrices int NRMat<std::complex<double> >::operator*(const NRSMat<std::complex<double> > &)");
#endif
	SAME_LOC(*this, rhs);
	const int rhs_ncols = rhs.ncols();
	NRMat<std::complex<double> > result(nn, rhs_ncols, getlocation());

#ifdef CUDALA
	if(location == cpu){
#endif
		for(register int i=0; i<nn; i++){
			cblas_zhpmv(CblasRowMajor, CblasLower, mm, &CONE, &rhs[0],
					(*this)[i], 1, &CZERO, result[i], 1);
		}
#ifdef CUDALA
	}else{
		for(register int i=0; i<nn; i++){
			cublasZhpmv('U', mm, CUONE, (cuDoubleComplex*)rhs.v, (cuDoubleComplex*)(v + i*(size_t)mm), 1, CUZERO, (cuDoubleComplex*)(result.v + i*(size_t)rhs_ncols), 1);
		}
	}
#endif
	return result;
}


/***************************************************************************//**
 * conjugate this non-complex matrix \f$A\f$, i.e. do nothing :-)
 * @return reference to the (unmodified) matrix
 ******************************************************************************/
template<typename T>
NRMat<T>& NRMat<T>::conjugateme() {
	return *this;
}

/***************************************************************************//**
 * conjugate this complex matrix \f$A\f$, or leading minor of size n
 * @return reference to the modified matrix
 ******************************************************************************/
template<>
NRMat<std::complex<double> >& NRMat<std::complex<double> >::conjugateme() {
	copyonwrite();
#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_dscal((size_t)mm*nn, -1.0, (double *)((*this)[0]) + 1, 2);
#ifdef CUDALA
	}else{
		cublasDscal((size_t)mm*nn, -1.0, (double *)(this->v) + 1, 2);
	}
#endif
	return *this;
}

template<>
NRMat<std::complex<float> >& NRMat<std::complex<float> >::conjugateme() {
        copyonwrite();
#ifdef CUDALA
        if(location == cpu){
#endif
                cblas_sscal((size_t)mm*nn, -1.0, (float *)((*this)[0]) + 1, 2);
#ifdef CUDALA
        }else{
                cublasSscal((size_t)mm*nn, -1.0, (float *)(this->v) + 1, 2);
        }
#endif
        return *this;
}


/***************************************************************************//**
 * compute transpose (optionally conjugated) of this real matrix \f$A\f$
 * @param[in] conj conjugation flag, unused for real matrices
 * @return transposed (conjugated) matrix by value
 ******************************************************************************/
template<>
const NRMat<double> NRMat<double>::transpose(bool conj) const {

	NRMat<double> result(mm, nn, getlocation());
#ifdef CUDALA
	if(location == cpu){
#endif
		for(register int i=0; i<nn; i++) cblas_dcopy(mm, (*this)[i], 1, result[0] + i, nn);
#ifdef CUDALA
	}else{
		for(register int i=0; i<nn; i++){
			cublasDcopy(mm, (*this)[i], 1, result[0] + i, nn);
		}
	}
#endif
	return result;
}

/***************************************************************************//**
 * compute transpose (optionally conjugated) of this real matrix \f$A\f$
 * @param[in] conj conjugation flag
 * @return transposed (conjugated) matrix by value
 ******************************************************************************/
template<>
const NRMat<std::complex<double> >
NRMat<std::complex<double> >::transpose(bool conj) const {
	NRMat<std::complex<double> > result(mm, nn, getlocation());
#ifdef CUDALA
	if(location == cpu){
#endif
		for(register int i=0; i<nn; i++){
			cblas_zcopy(mm, (*this)[i], 1, (result[0] + i), nn);
		}
		if(conj){ cblas_dscal(mm*nn, -1.0, (double *)(result[0]) + 1, 2); }
#ifdef CUDALA
	}else{
		for(register int i=0; i<nn; i++){
			cublasZcopy(mm, (cuDoubleComplex*)((*this)[i]), 1, (cuDoubleComplex*)(result[0] + i), nn);
		}
		if(conj){ cublasDscal(mm*nn, -1.0, (double *)(result.v) + 1, 2); }
	}
#endif
	return result;
}

/***************************************************************************//**
 * perform the gemm operation for this real matrix \f$M\f$, i.e. compute
 * \f[M\leftarrow\alpha\times\operatorname{op}(A)\times\operatorname{op}(B) + \beta\times{}M\f]
 * @param[in] beta \f$\beta\f$
 * @param[in] a real matrix \f$A\f$
 * @param[in] transa transposition flag of matrix \f$A\f$
 * @param[in] b real matrix \f$B\f$
 * @param[in] transb transposition flag of matrix \f$B\f$
 * @param[in] alpha \f$\alpha\f$
 ******************************************************************************/
template<>
void NRMat<double>::gemm(const double &beta, const NRMat<double> &a,
		const char transa, const NRMat<double> &b, const char transb, 
		const double &alpha) {

	int k(tolower(transa)=='n'?a.mm:a.nn);

#ifdef DEBUG
	int l(tolower(transa)=='n'?a.nn:a.mm);
	int kk(tolower(transb)=='n'?b.nn:b.mm);
	int ll(tolower(transb)=='n'?b.mm:b.nn);
	if (l!=nn || ll!=mm || k!=kk) laerror("incompatible matrices in NRMat<double>::gemm(...)");
	if(b.mm <=0 || mm<=0) laerror("illegal matrix dimension in gemm");
#endif

	SAME_LOC3(*this, a, b);

	if (alpha==0.0 && beta==1.0) return;

	copyonwrite();
#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_dgemm(CblasRowMajor, (tolower(transa)=='n' ? CblasNoTrans : CblasTrans),
				(tolower(transb)=='n' ? CblasNoTrans : CblasTrans), nn, mm, k, alpha, a,
				a.mm, b , b.mm, beta, *this , mm);
#ifdef CUDALA
	}else{
		cublasDgemm(transb, transa, mm, nn, k, alpha, b, b.mm, a, a.mm, beta, *this, mm);
	}
#endif
}


template<>
void NRMat<std::complex<double> >::gemm(const std::complex<double> & beta,
		const NRMat<std::complex<double> > & a, const char transa, 
		const NRMat<std::complex<double> > & b, const char transb, 
		const std::complex<double> & alpha)
{
	int k(tolower(transa)=='n'?a.mm:a.nn);

#ifdef DEBUG
	int l(tolower(transa)=='n'?a.nn:a.mm);
	int kk(tolower(transb)=='n'?b.nn:b.mm);
	int ll(tolower(transb)=='n'?b.mm:b.nn);
	if (l!=nn || ll!=mm || k!=kk) laerror("incompatible matrices in NRMat<std::complex<double> >::gemm(...)");
#endif
	if (alpha==CZERO && beta==CONE) return;

	copyonwrite();
	cblas_zgemm(CblasRowMajor,
			(tolower(transa)=='n' ? CblasNoTrans : (tolower(transa)=='c'?CblasConjTrans:CblasTrans)), 
			(tolower(transb)=='n' ? CblasNoTrans : (tolower(transb)=='c'?CblasConjTrans:CblasTrans)),
			nn, mm, k, &alpha, a , a.mm, b , b.mm, &beta, *this , mm);
}

/***************************************************************************//**
 * compute the Frobenius norm of the current real matrix \f$A\f$, i.e.
 * \f[ \sqrt{\sum_{i=1}^{N}\sum_{j=1}^{M}\left|A_{i,j}\right|^2} \f]
 * where \f$N\f$ and \f$M\f$ is the number of rows and columns, respectively
 * @param[in] scalar real value subtracted from the diagonal elements
 * @return computed norm
 ******************************************************************************/
template<>
const double  NRMat<double>::norm(const double scalar) const {
	if(!scalar){
#ifdef CUDALA
		if(location == cpu){
#endif
			return cblas_dnrm2((size_t)nn*mm, (*this)[0], 1);
#ifdef CUDALA
		}else{
			return cublasDnrm2((size_t)nn*mm, v, 1);
		}
#endif
	}

	NOT_GPU(*this);

	double sum(0.0);
	for(register int i=0; i<nn; i++)
		for(register int j=0; j<mm; j++) {
			register double tmp(0.0);
#ifdef MATPTR
			tmp = v[i][j];
#else
			tmp = v[i*(size_t)mm+j];
#endif
			if(i == j) tmp -= scalar;
			sum += tmp*tmp;
		}
	return std::sqrt(sum);
}


/***************************************************************************//**
 * compute the Frobenius norm of the current complex matrix \f$A\f$, i.e.
 * \f[ \sqrt{\sum_{i=1}^{N}\sum_{j=1}^{M}\left|A_{i,j}\right|^2} \f]
 * where \f$N\f$ and \f$M\f$ is the number of rows and columns, respectively
 * @param[in] scalar complex value subtracted from the diagonal elements
 * @return computed norm
 ******************************************************************************/
template<>
const double NRMat<std::complex<double> >::norm(const std::complex<double> scalar) const {
	if(scalar == CZERO){
#ifdef CUDALA
		if(location == cpu){
#endif
			return cblas_dznrm2((size_t)nn*mm, (*this)[0], 1);
#ifdef CUDALA
		}else{
			return cublasDznrm2((size_t)nn*mm, (cuDoubleComplex*)v, 1);
		}
#endif
	}

	NOT_GPU(*this);
	double sum(0.0);
	for(register int i=0; i<nn; i++)
		for(register int j=0; j<mm; j++) {
			register std::complex<double> tmp(0.0, 0.0);
#ifdef MATPTR
			tmp = v[i][j];
#else
			tmp = v[i*(size_t)mm+j];
#endif
			if(i == j) tmp -= scalar;
			const double re = tmp.real();
			const double im = tmp.imag();
			sum += re*re + im*im;
		}
	return std::sqrt(sum);
}

/***************************************************************************//**
 * perform the <b>axpy</b> operation on the current real matrix \f$A\f$, i.e.
 * \f[ A \leftarrow A + \alpha{}B \f]
 * for real matrix \f$B\f$
 * @param[in] alpha \f$\alpha\f$ parameter
 * @param[in] mat real matrix \f$B\f$
 ******************************************************************************/
template<>
void NRMat<double>::axpy(const double alpha, const NRMat<double> &mat) {
#ifdef DEBUG
	if (nn != mat.nn || mm != mat.mm) laerror("incompatible matrices in NRMat<double>::axpy(...)");
#endif
	SAME_LOC(*this, mat);
	copyonwrite();
#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_daxpy((size_t)nn*mm, alpha, mat, 1, *this, 1);
#ifdef CUDALA
	}else{
		cublasDaxpy((size_t)nn*mm, alpha, mat, 1, *this, 1);
	}
#endif
}

/***************************************************************************//**
 * perform the <b>axpy</b> operation on the current complex matrix \f$A\f$, i.e.
 * \f[ A \leftarrow A + \alpha{}B \f]
 * for real matrix \f$B\f$
 * @param[in] alpha complex parameter \f$\alpha\f$
 * @param[in] mat complex matrix \f$B\f$
 ******************************************************************************/
template<>
void NRMat<std::complex<double> >::axpy(const std::complex<double> alpha, 
		const NRMat<std::complex<double> > & mat) {
#ifdef DEBUG
	if (nn != mat.nn || mm != mat.mm) laerror("incompatible matrices in NRMat<std::complex<double> >::axpy(...)");
#endif
	SAME_LOC(*this, mat);
	copyonwrite();
#ifdef CUDALA
	if(location == cpu){
#endif
		cblas_zaxpy((size_t)nn*mm, &alpha, mat, 1, (*this)[0], 1);
#ifdef CUDALA
	}else{
		const cuDoubleComplex _alpha = make_cuDoubleComplex(alpha.real(), alpha.imag());
		cublasZaxpy(nn*mm, _alpha, (cuDoubleComplex*)(mat[0]), 1, (cuDoubleComplex*)(this->v), 1);
	}
#endif
}

/***************************************************************************//**
 * compute the trace of current genenal square matrix \f$A\f$, i.e.
 * \f[ \sum_{i=1}^{N} A_{i,i} \f]
 * where \f$N\f$ is the order of the matrix
 ******************************************************************************/
template <typename T>
const T NRMat<T>::trace() const {
#ifdef DEBUG
	if (nn != mm) laerror("nonsquare matrix in NRMat<T>::trace()");
#endif
	NOT_GPU(*this);
	T sum(0);
#ifdef MATPTR
	for(register int i=0; i<nn; ++i) sum += v[i][i];
#else
	for(register size_t i=0; i<(size_t)nn*nn; i += (nn+1)) sum += v[i];
#endif
	return sum;
}


/***************************************************************************//**
 * get or divide by the diagonal of real double-precision matrix
 * in case of nonsquare matrix \f$A\f$, use the diagonal of \f$A^\mathrm{T}A\f$
 * @param[in, out] r vector for storing the diagonal
 * @param[in] divide
 * 	\li \c false save the diagonal to vector r
 *	\li \c true divide the vector r by the diagonal elements element-wise
 * @param[in] cache reserved
 * @return
 * 	\li <tt>divide == true</tt> NULL
 *	\li <tt>divide == false</tt> pointer to the first element of r
 ******************************************************************************/
template<>
const double* NRMat<double>::diagonalof(NRVec<double> &r, const bool divide, bool cache) const {
	double *ret(NULL);
#ifdef DEBUG
	if(r.size() != mm) laerror("incompatible vector in NRMat<double>::diagonalof(...)");
#endif

	double a(0.0);//!< temporary variable for storing the scaling factor

	SAME_LOC(*this, r);
	if(divide){
		NOT_GPU(*this);
	}

	r.copyonwrite();
#ifdef CUDALA
	if(location == cpu){
#endif
		if(nn==mm){
		#ifdef MATPTR
			if(divide){
				for(int i=0; i< nn; i++) if((a=v[i][i])) r[i] /= a;
			}else{
				for(int i=0; i< nn; i++) r[i] = v[i][i];
			}
		#else
			if(divide){
				int i(0),j(0);
				for(i=j=0; j<nn; ++j, i+=nn+1){
					if((a=v[i])) r[j] /= a;
				}
			}else{
				/*
				 int i(0),j(0);
				 for(i=j=0; j< nn; ++j, i+=nn+1) r[j] = v[i];
				*/
				cblas_dcopy(nn, v, nn+1, r.v, 1);
			}
		#endif
		}else{//non-square matrix
			for(register int i=0; i< mm; i++){
			#ifdef MATPTR
				a = cblas_ddot(nn, v[0]+i, mm, v[0]+i, mm);
			#else
				a = cblas_ddot(nn, v+i, mm, v+i, mm);
			#endif
				if(divide){ if(a) r[i] /= a;}
				else{ r[i] = a; }
			}
		}
		ret = divide?NULL:&r[0];
#ifdef CUDALA
	}else{
		if(nn == mm){
			cublasDcopy(nn, v, nn+1, r.v, 1);
		}else{
			NRVec<double> tmp(mm, cpu);
			for(int i=0;i<mm;i++){
				const double x = cublasDdot(nn, v + i, 1, v + i, 1);
				tmp[i] = x;
			}
			tmp.moveto(location);
			r |= tmp;
		}
		ret = NULL;
	}
#endif
	return ret;
}

/***************************************************************************//**
 * sets the diagonal of real matrix to a given real vector
 * @param[in] r real vector which is supposed to be assigned to the diagonal 
 * @return void
 ******************************************************************************/
template<>
void NRMat<double>::diagonalset(const NRVec<double> &r) {
int nnmin= nn<=mm?nn:mm;
#ifdef DEBUG
	if(r.size() != nnmin) laerror("incompatible vectors int NRMat<double>::diagonalset(...)");
#endif

	SAME_LOC(*this, r);
	copyonwrite();

#ifdef CUDALA
	if(location == cpu){
#endif

#ifdef MATPTR
	for (int i=0; i<nnmin; i++) v[i][i] = r[i];
#else
	cblas_dcopy(nnmin, r.v, 1, v, mm+1); //{int i,j; for (i=j=0; j< nnmin; ++j, i+=mm+1) v[i] = r[j];}
#endif

#ifdef CUDALA
	}else{
		cublasDcopy(nnmin, r.v, 1, v, mm+1);
	}
#endif
}

/***************************************************************************//**
 * sets the diagonal of complex matrix to a given complex vector
 * @param[in] r complex vector which is supposed to be assigned to the diagonal 
 * @return void
 ******************************************************************************/
template<>
void NRMat<std::complex<double> >::diagonalset(const NRVec<std::complex<double> > &r) {
int nnmin= nn<=mm?nn:mm;
#ifdef DEBUG
	if(r.size() != nnmin) laerror("incompatible vectors int NRMat<std::complex<double> >::diagonalset(...)");
#endif
	SAME_LOC(*this, r);
	copyonwrite();

#ifdef CUDALA
	if(location == cpu){
#endif
	#ifdef MATPTR
		for (int i=0; i<nnmin; i++) v[i][i] = r[i];
	#else
		cblas_zcopy(nnmin, r.v, 1, v, mm+1);//{int i,j; for (i=j=0; j<nnmin; ++j, i+=mm+1) v[i] = r[j];}
	#endif
#ifdef CUDALA
	}else{
		cublasZcopy(nnmin, (cuDoubleComplex*)(r.v), 1, (cuDoubleComplex*)(this->v), mm+1);
	}
#endif
}

/***************************************************************************//**
 * perform straightforward orthonormalization via modified Gram-Schmidt process
 * @param[in] rowcol flag regarding the interpretation of the current matrix
 *	\li \c true the vectors being orthonormalized are stored as rows 
 *	\li \c false the vectors being orthonormalized are stored as columns
 * @param[in] metric pointer to real symmetric matrix stored in packed form which
 *	is used in computing the inner products in the process, the standard inner product 
 *	is taken into account for <tt>metric == NULL</tt>
 * @return void
 ******************************************************************************/
template<>
void NRMat<double>::orthonormalize(const bool rowcol, const NRSMat<double> *metric) {

	SAME_LOC(*this, *metric);
	if(metric){
		if(rowcol){ //vectors are stored in rows
			if((*metric).nrows() != mm) laerror("incompatible metric in NRMat<double>::orthonormalize(rowcol = true)");

#ifdef CUDALA
			if(location == cpu){
#endif
				for(register int j=0; j<nn; ++j){
					for(register int i=0; i<j; ++i){
						NRVec<double> tmp = *metric * (*this).row(i);
						const double fact = cblas_ddot(mm,(*this)[j],1,tmp,1);
						cblas_daxpy(mm,-fact,(*this)[i],1,(*this)[j],1);
					}
					const NRVec<double> tmp = *metric * (*this).row(j);
					const double norm = cblas_ddot(mm,(*this)[j],1,tmp,1);
					if(norm <= 0.) laerror("zero vector or nonpositive metric in NRMat<double>::orthonormalize(...)");
					cblas_dscal(mm,1./std::sqrt(norm),(*this)[j],1);
				}
#ifdef CUDALA
			}else{
				for(register int j=0; j<nn; ++j){
					for(register int i=0; i<j; ++i){
						NRVec<double> tmp(mm, location);
					        tmp = *metric * (*this).row(i);
						const double fact = cublasDdot(mm, (*this)[j], 1, tmp, 1);
						cublasDaxpy(mm, -fact, (*this)[i], 1, (*this)[j], 1);
					}
					NRVec<double> tmp(mm, location);
					tmp = *metric * (*this).row(j);
					const double norm = cublasDdot(mm, (*this)[j], 1, tmp, 1);
					if(norm <= 0.) laerror("zero vector or nonpositive metric in NRMat<double>::orthonormalize(...)");
					cublasDscal(mm, 1./std::sqrt(norm), (*this)[j], 1);
				}

			}
#endif			
		}else{ //vectors are stored in columns
#ifdef CUDALA
			if(location = cpu){
#endif
				if((*metric).nrows() != nn) laerror("incompatible metric in NRMat<double>::orthonormalize(rowcol = false)");
				for(register int j=0; j<mm; ++j){
					for(register int i=0; i<j; ++i){
						NRVec<double> tmp = *metric * (*this).column(i);
						double fact = cblas_ddot(nn, &(*this)[0][j], mm, tmp, 1);
						cblas_daxpy(nn, -fact, &(*this)[0][i], mm, &(*this)[0][j], mm);
					}
					NRVec<double> tmp = *metric * (*this).column(j);
					double norm = cblas_ddot(nn, &(*this)[0][j], mm, tmp, 1);
					if(norm <= 0.) laerror("zero vector or nonpositive metric in NRMat<double>::orthonormalize(...)");
			                cblas_dscal(nn, 1./std::sqrt(norm), &(*this)[0][j], mm);
				}
#ifdef CUDALA
			}else{
				if((*metric).nrows() != nn) laerror("incompatible metric in NRMat<double>::orthonormalize(rowcol = false)");
				for(register int j=0; j<mm; ++j){
					for(register int i=0; i<j; ++i){
						NRVec<double> tmp(nn, location);
						tmp = *metric * (*this).column(i);
						double fact = cublasDdot(nn, &(*this)[0][j], mm, tmp, 1);
						cublasDaxpy(nn, -fact, &(*this)[0][i], mm, &(*this)[0][j], mm);
					}
					NRVec<double> tmp(nn, location);
					tmp = *metric * (*this).column(j);
					double norm = cublasDdot(nn, &(*this)[0][j], mm, tmp, 1);
					if(norm <= 0.) laerror("zero vector or nonpositive metric in NRMat<double>::orthonormalize(...)");
			                cublasDscal(nn, 1./std::sqrt(norm), &(*this)[0][j], mm);
				}
			}
#endif
		}
	}else{ //unit metric (standard inner product) will be used
		if(rowcol){
#ifdef CUDALA
			if(location == cpu){
#endif			
				for(register int j=0; j<nn; ++j){
					for(register int i=0; i<j; ++i){
						const double fact = cblas_ddot(mm,(*this)[j],1,(*this)[i],1);
						cblas_daxpy(mm,-fact,(*this)[i],1,(*this)[j],1);
					}
					const double norm = cblas_dnrm2(mm,(*this)[j],1);
					if(norm <= 0.) laerror("zero vector or nonpositive metric in NRMat<double>::orthonormalize(...)");
					cblas_dscal(mm,1./norm,(*this)[j],1);
				}
#ifdef CUDALA
			}else{
				for(register int j=0; j<nn; ++j){
					for(register int i=0; i<j; ++i){
						const double fact = cublasDdot(mm, (*this)[j], 1, (*this)[i], 1);
						cublasDaxpy(mm, -fact, (*this)[i], 1, (*this)[j], 1);
					}
					const double norm = cublasDnrm2(mm, (*this)[j], 1);
					if(norm <= 0.) laerror("zero vector or nonpositive metric in NRMat<double>::orthonormalize(...)");
					cublasDscal(mm, 1./norm, (*this)[j], 1);
				}
			}
#endif
		}else{ // vectors are stored in columns
#ifdef CUDALA
			if(location == cpu){
#endif
				for(register int j=0; j<mm; ++j){
					for(register int i=0; i<j; ++i){
						const double fact = cblas_ddot(nn, &(*this)[0][j], mm, &(*this)[0][i], mm);
						cblas_daxpy(nn, -fact, &(*this)[0][i], mm, &(*this)[0][j], mm);
					}
					const double norm = cblas_dnrm2(nn, &(*this)[0][j], mm);
					if(norm <= 0.) laerror("zero vector or nonpositive metric in NRMat<double>::orthonormalize(...)");
					cblas_dscal(nn, 1./norm, &(*this)[0][j], mm);
				}
#ifdef CUDALA
			}else{
				for(register int j=0; j<mm; ++j){
					for(register int i=0; i<j; ++i){
						const double fact = cublasDdot(nn, &(*this)[0][j], mm, &(*this)[0][i], mm);
						cublasDaxpy(nn, -fact, &(*this)[0][i], mm, &(*this)[0][j], mm);
					}
					const double norm = cublasDnrm2(nn, &(*this)[0][j], mm);
					if(norm <= 0.) laerror("zero vector or nonpositive metric in NRMat<double>::orthonormalize(...)");
					cublasDscal(nn, 1./norm, &(*this)[0][j], mm);
				}
			}
#endif
		}
	} //end of the unit-metric branch
}

/***************************************************************************//**
 * interchange the order of the rows of the current (real) matrix
 * @return reference to the modified matrix
 ******************************************************************************/
template<>
NRMat<double>& NRMat<double>::swap_rows(){
        copyonwrite();
        const int n_pul = this->nn >> 1;

#ifdef CUDALA
	if(location == cpu){
#endif		
	        for(register int i=0; i<n_pul; i++){
		        cblas_dswap(mm, (*this)[i], 1, (*this)[nn - i - 1], 1);
	        }
#ifdef CUDALA
	}else{
	        for(register int i=0; i<n_pul; i++){
		        cublasDswap(mm, v + i*(size_t)mm, 1, v + (nn - i - 1)*mm, 1);
			TEST_CUBLAS("cublasDswap");
	        }
	}
#endif	
        return *this;
}

template<>
NRMat<double>& NRMat<double>::swap_rows(const int i, const int j){
        copyonwrite();

#ifdef CUDALA
	if(location == cpu){
#endif		
		        cblas_dswap(mm, (*this)[i], 1, (*this)[j], 1);
#ifdef CUDALA
	}else{
		        cublasDswap(mm, v + i*(size_t)mm, 1, v + j*mm, 1);
			TEST_CUBLAS("cublasDswap");
	}
#endif	
        return *this;
}

/***************************************************************************//**
 * interchange the order of the rows of the current (complex) matrix
 * @return reference to the modified matrix
 ******************************************************************************/
template<>
NRMat<std::complex<double> >& NRMat<std::complex<double> >::swap_rows(){
        copyonwrite();
        const int n_pul = this->nn >> 1;

#ifdef CUDALA
	if(location == cpu){
#endif
		for(register int i=0; i<n_pul; i++){
		        cblas_zswap(mm, (*this)[i], 1, (*this)[nn - i - 1], 1);
		}
#ifdef CUDALA
	}else{
		for(register int i=0; i<n_pul; i++){
		        cublasZswap(mm, (cuDoubleComplex*)(v + i*(size_t)mm), 1, (cuDoubleComplex*)(v + (nn - i - 1)*mm), 1);
			TEST_CUBLAS("cublasZswap");
		}
	}
#endif
        return *this;
}

template<>
NRMat<std::complex<double> >& NRMat<std::complex<double> >::swap_rows(const int i, const int j){
        copyonwrite();

#ifdef CUDALA
	if(location == cpu){
#endif
		        cblas_zswap(mm, (*this)[i], 1, (*this)[j], 1);
#ifdef CUDALA
	}else{
		        cublasZswap(mm, (cuDoubleComplex*)(v + i*(size_t)mm), 1, (cuDoubleComplex*)(v + j*(size_t)mm), 1);
			TEST_CUBLAS("cublasZswap");
	}
#endif
        return *this;
}

/***************************************************************************//**
 * interchange the order of the rows of the current general matrix of type T
 * for GPU computations, the condition sizeof(T)%sizeof(float) is required
 * @return reference to the modified matrix
 ******************************************************************************/
template<typename T>
NRMat<T>& NRMat<T>::swap_rows(){
	T tmp;
        copyonwrite();
        const int n_pul = this->nn >> 1;

#ifdef CUDALA
	if(location == cpu){
#endif		
	        for(register int i=0; i<n_pul; i++){
			for(register int j=0; j<mm; j++){
			        tmp = (*this)[i][j];
			       	(*this)[i][j] = (*this)[nn - i - 1][j];
			       	(*this)[nn - i - 1][j] = tmp;
			}
	        }
#ifdef CUDALA
	}else{
		if(sizeof(T)%sizeof(float) != 0) laerror("cpu memcpy alignment problem in NRMat<T>::swap_rows");
	        for(register int i=0; i<n_pul; i++){
		        cublasSswap(mm*sizeof(T)/sizeof(float), (float *)(v + i*(size_t)mm), 1, (float *)(v + (nn - i - 1)*mm), 1);
			TEST_CUBLAS("cublasSswap");
	        }
	}
#endif	
        return *this;
}


/***************************************************************************//**
 * interchange the order of the columns of the current (real) matrix
 * @return reference to the modified matrix
 ******************************************************************************/
template<>
NRMat<double>& NRMat<double>::swap_cols(){
        copyonwrite();
        const int m_pul = mm >> 1;

#ifdef CUDALA
	if(location == cpu){
#endif
		for(register int i=0; i<m_pul; i++){
			cblas_dswap(nn, &((*this)(0, i)), mm, &((*this)(0, mm - i - 1)), mm);
		}
#ifdef CUDALA
	}else{
		for(register int i=0; i<m_pul; i++){
			cublasDswap(nn, v + i, mm, v + (mm - i - 1), mm);
			TEST_CUBLAS("cublasDswap");
		}
	}
#endif	
	return *this;
}

template<>
NRMat<double>& NRMat<double>::swap_cols(const int i, const int j){
        copyonwrite();

#ifdef CUDALA
	if(location == cpu){
#endif
			cblas_dswap(nn, &((*this)(0, i)), mm, &((*this)(0, j)), mm);
#ifdef CUDALA
	}else{
			cublasDswap(nn, v + i, mm, v + j, mm);
			TEST_CUBLAS("cublasDswap");
	}
#endif	
	return *this;
}

/***************************************************************************//**
 * interchange the order of the columns of the current (complex) matrix
 * @return reference to the modified matrix
 ******************************************************************************/
template<>
NRMat<std::complex<double> >& NRMat<std::complex<double> >::swap_cols(){
        copyonwrite();
        const int m_pul = mm >> 1;

#ifdef CUDALA
	if(location == cpu){
#endif
		for(register int i=0; i<m_pul; i++){
			cblas_zswap(nn, &((*this)(0, i)), mm, &((*this)(0, mm - i - 1)), mm);
		}
#ifdef CUDALA
	}else{
		for(register int i=0; i<m_pul; i++){
			cublasZswap(nn, (cuDoubleComplex*)(v + i), mm, (cuDoubleComplex*)(v + (mm - i - 1)), mm);
			TEST_CUBLAS("cublasZswap");
		}
	}
#endif	
	return *this;
}

template<>
NRMat<std::complex<double> >& NRMat<std::complex<double> >::swap_cols(const int i, const int j){
        copyonwrite();

#ifdef CUDALA
	if(location == cpu){
#endif
			cblas_zswap(nn, &((*this)(0, i)), mm, &((*this)(0, j)), mm);
#ifdef CUDALA
	}else{
			cublasZswap(nn, (cuDoubleComplex*)(v + i), mm, (cuDoubleComplex*)(v + j), mm);
			TEST_CUBLAS("cublasZswap");
	}
#endif	
	return *this;
}

/***************************************************************************//**
 * interchange the order of the columns of the current general matrix of type T
 * because of the cuBlas implementation, the GPU version requires that
 * sizeof(T)%sizeof(float)==0
 * @return reference to the modified matrix
 ******************************************************************************/
template<typename T>
NRMat<T>& NRMat<T>::swap_cols(){
	T tmp;
        copyonwrite();
        const int m_pul = mm >> 1;
#ifdef CUDALA
	if(location == cpu){
#endif
		for(register int i=0; i<m_pul; i++){
			for(register int j=0;j<nn;j++){
				tmp = (*this)(j, i);
				(*this)(j, i) = (*this)(j, mm - i - 1);
				(*this)(j, mm - i - 1) = tmp;
			}
		}
#ifdef CUDALA
	}else{
		if(sizeof(T)%sizeof(float) != 0) laerror("cpu memcpy alignment problem in NRMat<T>::swap_cols");
	        for(register int i=0; i<m_pul; i++){
		        cublasSswap(nn*sizeof(T)/sizeof(float), 
					(float *)(v + i), mm*sizeof(T)/sizeof(float), 
					(float *)(v + (mm - i - 1)), mm*sizeof(T)/sizeof(float) );
			TEST_CUBLAS("cublasSswap");
	        }
	}
#endif	
        return *this;
}

/*interchange two columns*/
template<typename T>
NRMat<T>& NRMat<T>::swap_cols(const int a, const int b){
	T tmp;
        copyonwrite();
#ifdef CUDALA
	if(location == cpu){
#endif
			for(register int j=0;j<nn;j++){
				tmp = (*this)(j, a);
				(*this)(j, a) = (*this)(j,b);
				(*this)(j,b) = tmp;
			}
#ifdef CUDALA
	}else{
		if(sizeof(T)%sizeof(float) != 0) laerror("cpu memcpy alignment problem in NRMat<T>::swap_cols");
		        cublasSswap(nn*sizeof(T)/sizeof(float), 
					(float *)(v + a), mm*sizeof(T)/sizeof(float), 
					(float *)(v + b), mm*sizeof(T)/sizeof(float) );
			TEST_CUBLAS("cublasSswap");
	}
#endif	
        return *this;
}

/*interchange two rows*/
template<typename T>
NRMat<T>& NRMat<T>::swap_rows(const int a, const int b){
	T tmp;
        copyonwrite();
#ifdef CUDALA
	if(location == cpu){
#endif
			for(register int j=0;j<mm;j++){
				tmp = (*this)(a,j);
				(*this)(a,j) = (*this)(b,j);
				(*this)(b,j) = tmp;
			}
#ifdef CUDALA
	}else{
		if(sizeof(T)%sizeof(float) != 0) laerror("cpu memcpy alignment problem in NRMat<T>::swap_rows");
		        cublasSswap(nn*sizeof(T)/sizeof(float), 
					(float *)(v + a*mm), sizeof(T)/sizeof(float), 
					(float *)(v + b*mm), sizeof(T)/sizeof(float) );
			TEST_CUBLAS("cublasSswap");
	}
#endif	
        return *this;
}


/*rotate rows or columns of a matrix - general implementation, more efficient version could be done with BLAS scal and axpy operations
 *  but it would require allocation of temporary storage
 */

template<typename T>
NRMat<T>& NRMat<T>::rotate_rows(const int a, const int b, const T phi){
        T tmp1,tmp2;
        copyonwrite();
	T c=cos(phi);
	T s=sin(phi);
#ifdef CUDALA
        if(location == cpu){
#endif
                        for(register int j=0;j<mm;j++){
                                tmp1 = (*this)(a,j);
                                tmp2 = (*this)(b,j);
                                (*this)(a,j) = c*tmp1 + s*tmp2;
                                (*this)(b,j) = c*tmp2 - s*tmp1;
                        }
#ifdef CUDALA
        }else{
		laerror("rotate_rows not implemented on gpu");
        }
#endif
        return *this;
}

template<typename T>
NRMat<T>& NRMat<T>::rotate_cols(const int a, const int b, const T phi){
        T tmp1,tmp2;
        copyonwrite();
        T c=cos(phi);
        T s=sin(phi);
#ifdef CUDALA
        if(location == cpu){
#endif
                        for(register int j=0;j<nn;j++){
                                tmp1 = (*this)(j,a);
                                tmp2 = (*this)(j,b);
                                (*this)(j,a) = c*tmp1 + s*tmp2;
                                (*this)(j,b) = c*tmp2 - s*tmp1;
                        }
#ifdef CUDALA
        }else{
                laerror("rotate_rows not implemented on gpu");
        }
#endif
        return *this;
}





/***************************************************************************//**
 * interchange the order of the rows and columns of the current
 * real matrix \f$A\f$ of type T, i.e. perform the operation
 * \f[A_{i,j}\leftarrow A_{nn-1-i, mm-1-j}\f]
 * where \f$0\leq{}i\le{}nn\f$ and \f$0\leq{}j\le{}mm\f$
 * @return reference to the modified matrix
 ******************************************************************************/
template<>
NRMat<double>& NRMat<double>::swap_rows_cols(){
	const int n_pul = nn >> 1;
	const int m_pul = mm >> 1;
	double tmp(0.0);

        copyonwrite();
#ifdef CUDALA
	if(location == cpu){
#endif	
		for(register int i=0; i<n_pul; i++){
			std::cout << "swapping row " << i << " and " << nn-i-1 << std::endl;

			std::cout << "elements: " << *((*this)[i]) << std::endl;
			std::cout << "elements: " << *((*this)[nn - i - 1] + mm - 1) << std::endl;

			cblas_dswap(mm, (*this)[i], 1, (*this)[nn - i - 1] + mm - 1, -1);
		}
		return *this;

		if(nn & 1){ // odd number of rows
			for(register int i=0; i<=m_pul; i++){

				tmp = (*this)(n_pul, i);
				(*this)(n_pul, i) = (*this)(n_pul, mm-i-1);
				(*this)(n_pul, mm-i-1) = tmp;
			}
		}
#ifdef CUDALA
	}else{
		for(register int i=0; i<n_pul; i++){
			cublasDswap(mm, v + i*(size_t)mm, 1, v + (nn - i - 1)*mm + mm - 1, -1);
			TEST_CUBLAS("cublasDswap");
		}

		if(nn & 1){
			void *gpu_ptr = gpualloc(sizeof(double)*mm);
			cublasDswap(mm, v + n_pul*mm + mm - 1, -1, (double *)gpu_ptr, 1);
			cublasDcopy(mm, (double *)gpu_ptr, 1, v + n_pul*mm, 1);
			gpufree(gpu_ptr);
		}
        }
#endif
        return *this;
}

/***************************************************************************//**
 * interchange the order of the rows and columns of the current
 * complex matrix \f$A\f$ of type T, i.e. perform the operation
 * \f[A_{i,j}\leftarrow A_{nn-1-i, mm-1-j}\f]
 * where \f$0\leq{}i\le{}nn\f$ and \f$0\leq{}j\le{}mm\f$
 * @return reference to the modified matrix
 ******************************************************************************/
template<>
NRMat<std::complex<double> >& NRMat<std::complex<double> >::swap_rows_cols(){
	const int n_pul = nn >> 1;
	const int m_pul = mm >> 1;

	std::complex<double> tmp(0.0, 0.0);

        copyonwrite();
#ifdef CUDALA
	if(location == cpu){
#endif	
		for(register int i=0;i<n_pul;i++){
			cblas_zswap(mm, (*this)[i], 1, (*this)[nn - i - 1] + mm - 1, -1);
		}

		if(nn & 1){
			for(register int i=0; i<=m_pul; i++){ // odd number of rows
				tmp = (*this)(n_pul, i);
				(*this)(n_pul, i) = (*this)(n_pul, mm-i-1);
				(*this)(n_pul, mm-i-1) = tmp;
			}
		}
#ifdef CUDALA
	}else{
		for(register int i=0;i<n_pul;i++){
			cublasZswap(mm, (cuDoubleComplex*)(v + i*(size_t)mm), 1, (cuDoubleComplex*)(v + (nn - i - 1)*mm + mm - 1), -1);
			TEST_CUBLAS("cublasZswap");
		}
		if(nn & 1){
			void *gpu_ptr = gpualloc(sizeof(std::complex<double>)*mm);
			cublasZswap(mm, (cuDoubleComplex*)(v + n_pul*mm + mm - 1), -1, (cuDoubleComplex*)gpu_ptr, 1);
			cublasZcopy(mm, (cuDoubleComplex*)gpu_ptr, 1, (cuDoubleComplex*)(v + n_pul*mm), 1);
			gpufree(gpu_ptr);
		}
        }
#endif
        return *this;
}

/***************************************************************************//**
 * interchange the order of the rows and columns of the current
 * general matrix \f$A\f$ of type T, i.e. perform the operation
 * \f[A_{i,j}\leftarrow A_{nn-1-i, mm-1-j}\f]
 * where \f$0\leq{}i\le{}nn\f$ and \f$0\leq{}j\le{}mm\f$
 * @return reference to the modified matrix
 ******************************************************************************/
template<typename T>
NRMat<T>& NRMat<T>::swap_rows_cols(){
	const int n_pul = nn >> 1;
	const int m_pul = mm >> 1;
	const size_t dim = (size_t)nn*mm;

	T *data_ptr;
	T tmp;
        copyonwrite();

#ifdef CUDALA
	if(location == cpu){
#endif
		data_ptr = (*this)[0];	
		const int dim_pul = dim >> 1;
		for(register int i=0; i<=dim_pul; i++){
			tmp = data_ptr[i];
			data_ptr[i] = data_ptr[dim - i - 1];
			data_ptr[dim - i - 1] = tmp;
		}
#ifdef CUDALA
	}else{
		if(sizeof(T)%sizeof(float) != 0) laerror("cpu memcpy alignment problem in NRMat<T>::swap_rows_cols");
	        for(register int i=0; i<n_pul; i++){
		        cublasSswap(mm*sizeof(T)/sizeof(float), (float *)(v + i*(size_t)mm), 1, (float *)(v + (nn - i - 1)*mm) - 1, -1);
			TEST_CUBLAS("cublasSswap");
	        }

		if(nn & 1){
			void *gpu_ptr = gpualloc(mm*sizeof(T));

			cublasSswap(mm*sizeof(T)/sizeof(float), (float *)(v + (n_pul + 1)*mm) - 1, -1, (float *)gpu_ptr, 1);
			TEST_CUBLAS("cublasSswap");

			cublasScopy(mm*sizeof(T)/sizeof(float), (float *)gpu_ptr, 1, (float *)( v + n_pul*mm ), 1);
			TEST_CUBLAS("cublasScopy");

			gpufree(gpu_ptr);
		}
	}
#endif
        return *this;
}

//permutation matrix
template<typename T>
void NRMat<T>::axpy(const T alpha, const NRPerm<int> &p, const bool direction)
{
int n=p.size();
for(int i=0; i<n; ++i)
	{
	if(direction) (*this)(i,p[i+1]-1) +=alpha;
	else (*this)(p[i+1]-1,i) += alpha;
	}
}

template<typename T>
NRMat<T>::NRMat(const NRPerm<int> &p, const bool direction, const bool parity)
{
int n=p.size();
nn=mm=0; count=0; v=0; resize(n,n);
clear();
T alpha= parity? p.parity():1;
axpy(alpha,p,direction);
}

template<typename T>
NRMat<T>::NRMat(const WeightPermutation<int,T> &wp, const bool direction)
{
int n=wp.size();
nn=mm=0; count=0; v=0; resize(n,n);
clear(); 
axpy(wp.weight,wp.perm,direction);
}

template<typename T>
NRMat<T>::NRMat(const PermutationAlgebra<int,T> &ap, const bool direction , int nforce)
{
int na= ap.size();
if(na<=0 && nforce<=0) laerror("cannot deduce matrix size from empty PermutationAlgebra");
int n= nforce>0?nforce:ap[0].size();
nn=mm=0; count=0; v=0; resize(n,n);
clear();
for(int i=0; i<na; ++i) axpy(ap[i].weight,ap[i].perm,direction);
}




//apply permutations
template<typename T>
const NRMat<T> NRMat<T>::permuted_rows(const NRPerm<int> &p, const bool inverse) const
{
#ifdef DEBUG
if(!p.is_valid()) laerror("invalid permutation of matrix");
#endif
int n=p.size();
if(n!=nn) laerror("incompatible permutation and matrix");
#ifdef CUDALA
        if(this->getlocation() != cpu || p.getlocation() != cpu ) laerror("permutations can be done only in CPU memory");
#endif
NRMat<T> r(nn,mm);
if(inverse) for(int i=1; i<=n; ++i) {int pi=p[i]-1; for(int j=0; j<mm; ++j)  r(i-1,j) = (*this)(pi,j);}
else for(int i=1; i<=n; ++i) {int pi=p[i]-1; for(int j=0; j<mm; ++j)  r(pi,j) = (*this)(i-1,j);}
return r;
}

template<typename T>
const NRMat<T> NRMat<T>::permuted_cols(const NRPerm<int> &p, const bool inverse) const
{
#ifdef DEBUG
if(!p.is_valid()) laerror("invalid permutation of matrix");
#endif
int n=p.size();
if(n!=mm) laerror("incompatible permutation and matrix");
#ifdef CUDALA
        if(this->getlocation() != cpu || p.getlocation() != cpu ) laerror("permutations can be done only in CPU memory");
#endif
NRMat<T> r(nn,mm);
if(inverse) for(int i=1; i<=n; ++i) {int pi=p[i]-1; for(int j=0; j<nn; ++j)  r(j,i-1) = (*this)(j,pi);}
else for(int i=1; i<=n; ++i) {int pi=p[i]-1; for(int j=0; j<nn; ++j)  r(j,pi) = (*this)(j,i-1);}
return r;
}

template<typename T>
const NRMat<T> NRMat<T>::permuted_both(const NRPerm<int> &p, const NRPerm<int> &q, const bool inverse) const
{
#ifdef DEBUG
if(!p.is_valid() || !q.is_valid() ) laerror("invalid permutation of matrix");
#endif
int n=p.size();
int m=q.size();
if(n!=nn ||m!=mm) laerror("incompatible permutation and matrix");
#ifdef CUDALA
        if(this->getlocation() != cpu || p.getlocation() != cpu ) laerror("permutations can be done only in CPU memory");
#endif
NRMat<T> r(nn,mm);
if(inverse) for(int i=1; i<=n; ++i) {int pi=p[i]-1; for(int j=1; j<=m; ++j)  r(i-1,j-1) = (*this)(pi,q[j]-1);}
else for(int i=1; i<=n; ++i) {int pi=p[i]-1; for(int j=1; j<=m; ++j)  r(pi,q[j]-1) = (*this)(i-1,j-1);}
return r;
}


template<typename T>
void NRMat<T>::permuteme_rows(const CyclePerm<int> &p)
{
#ifdef DEBUG
if(!p.is_valid()) laerror("invalid permutation of matrix");
#endif
if(p.max()>nn) laerror("incompatible permutation and matrix");
#ifdef CUDALA
        if(this->getlocation() != cpu || p.getlocation() != cpu ) laerror("permutations can be done only in CPU memory");
#endif
copyonwrite();
T *tmp = new T[mm];
for(int cycle=1; cycle<=p.size(); ++cycle)
        {
        int length= p[cycle].size();
        if(length<=1) continue; //trivial cycle
        for(int j=0; j<mm; ++j) tmp[j] = (*this)(p[cycle][length]-1,j);
        for(int i=length; i>1; --i) 
		for(int j=0; j<mm; ++j) (*this)(p[cycle][i]-1,j)=(*this)(p[cycle][i-1]-1,j);
	for(int j=0; j<mm; ++j) (*this)(p[cycle][1]-1,j)=tmp[j];
        }
delete[] tmp;
}

template<typename T>
void NRMat<T>::permuteme_cols(const CyclePerm<int> &p)
{
#ifdef DEBUG
if(!p.is_valid()) laerror("invalid permutation of matrix");
#endif
if(p.max()>mm) laerror("incompatible permutation and matrix");
#ifdef CUDALA
        if(this->getlocation() != cpu || p.getlocation() != cpu ) laerror("permutations can be done only in CPU memory");
#endif
copyonwrite();
T *tmp = new T[nn];
for(int cycle=1; cycle<=p.size(); ++cycle)
        {
        int length= p[cycle].size();
        if(length<=1) continue; //trivial cycle
        for(int j=0; j<nn; ++j) tmp[j] = (*this)(j,p[cycle][length]-1);
        for(int i=length; i>1; --i) 
                for(int j=0; j<nn; ++j) (*this)(j,p[cycle][i]-1)=(*this)(j,p[cycle][i-1]-1);
        for(int j=0; j<nn; ++j) (*this)(j,p[cycle][1]-1)=tmp[j];
        }
delete[] tmp;
}


//double and complex specialization
template<>
void NRMat<double>::scale_row(const int i, const double f)
{
#ifdef DEBUG
if(i<0||i>=nn) laerror("index out of range in scale_row");
#endif
copyonwrite();
#ifdef CUDALA
        if(location == cpu) {
#endif
                cblas_dscal(mm, f, &(*this)(i,0), 1);
#ifdef CUDALA
        }else{
                cublasDscal(mm, f, v+i*mm, 1);
                TEST_CUBLAS("cublasDscal");
        }
#endif
}

template<>
void NRMat<double>::scale_col(const int i, const double f)
{
#ifdef DEBUG
if(i<0||i>=mm) laerror("index out of range in scale_col");
#endif
copyonwrite();
#ifdef CUDALA
        if(location == cpu) {
#endif
                cblas_dscal(nn, f, &(*this)(0,i), mm);
#ifdef CUDALA
        }else{
                cublasDscal(nn, f, v+i, mm);
                TEST_CUBLAS("cublasDscal");
        }
#endif
}


template<>
void NRMat<std::complex<double> >::scale_row(const int i, const std::complex<double> f)
{
#ifdef DEBUG
if(i<0||i>=nn) laerror("index out of range in scale_row");
#endif
copyonwrite();
#ifdef CUDALA
        if(location == cpu) {
#endif
                cblas_zscal(mm, &f, &(*this)(i,0), 1);
#ifdef CUDALA
        }else{
		const cuDoubleComplex fac = *(reinterpret_cast<const cuDoubleComplex*> (&f));
                cublasZscal(mm, &fac, v+i*mm, 1);
                TEST_CUBLAS("cublasDscal");
        }
#endif
}

template<>
void NRMat<std::complex<double> >::scale_col(const int i, const std::complex<double> f)
{
#ifdef DEBUG
if(i<0||i>=mm) laerror("index out of range in scale_col");
#endif
copyonwrite();
#ifdef CUDALA
        if(location == cpu) {
#endif
                cblas_zscal(nn, &f, &(*this)(0,i), mm);
#ifdef CUDALA
        }else{
		const cuDoubleComplex fac = *(reinterpret_cast<const cuDoubleComplex*> (&f));
                cublasZscal(nn, &fac, v+i, mm);
                TEST_CUBLAS("cublasDscal");
        }
#endif
}



template<>
NRMat<double> NRMat<double>::inverse()
{
NRMat<double> tmp(*this);
return calcinverse(tmp);
}

template<>
NRMat<std::complex<double> > NRMat<std::complex<double> >::inverse()
{
NRMat<std::complex<double> >  tmp(*this);
return calcinverse(tmp);
}       






//general version
template<typename T>
void NRMat<T>::scale_row(const int i, const T f)
{
#ifdef DEBUG
if(i<0||i>=nn) laerror("index out of range in scale_row");
#endif
copyonwrite();
for(int j=0; j<mm; ++j) (*this)(i,j) *= f;
}


template<typename T>
void NRMat<T>::scale_col(const int i, const T f)
{
#ifdef DEBUG
if(i<0||i>=mm) laerror("index out of range in scale_col");
#endif
copyonwrite();
for(int j=0; j<nn; ++j) (*this)(j,i) *= f;
}






/***************************************************************************//**
 * forced instantization in the corresponding object file
 ******************************************************************************/
template class NRMat<double>;
template class NRMat<std::complex<double> >;
//template class NRMat<float>;
//template class NRMat<std::complex<float> >;
template class NRMat<long long>;
template class NRMat<long>;
template class NRMat<int>;
template class NRMat<short>;
template class NRMat<char>;
template class NRMat<unsigned char>;
template class NRMat<unsigned short>;
template class NRMat<unsigned int>;
template class NRMat<unsigned long>;
template class NRMat<unsigned long long>;

}//namespace
