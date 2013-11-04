/* vim: set ts=8 sw=8 sts=8 noexpandtab cindent: */
/*******************************************************************************
    LA: linear algebra C++ interface library
    Copyright (C) 2008 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>

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

#include "noncblas.h"
#include "laerror.h"
#include "mat.h"
#include "fortran.h"


#ifdef NONCBLAS
//Level 1 - straightforward wrappers

extern "C" double FORNAME(ddot) (const FINT *n, const double *x, const FINT *incx, const double *y, const FINT *incy);
double cblas_ddot(const int N, const double *X, const int incX, const double *Y, const int incY){
#ifdef FORINT
	const FINT ntmp=N;
	const FINT incxtmp=incX;
	const FINT incytmp=incY;
	return FORNAME(ddot)(&ntmp,X,&incxtmp,Y,&incytmp);
#else
	return FORNAME(ddot)(&N,X,&incX,Y,&incY);
#endif
}

extern "C" void FORNAME(dscal) (const FINT *n, const double *a, double *x, const FINT *incx);
void cblas_dscal(const int N, const double alpha, double *X, const int incX){
#ifdef FORINT
	const FINT ntmp=N;
	const FINT incxtmp=incX;
	FORNAME(dscal) (&ntmp,&alpha,X,&incxtmp);
#else
	FORNAME(dscal) (&N,&alpha,X,&incX);
#endif
}

extern "C" void FORNAME(dcopy) (const FINT *n, const double *x, const FINT *incx, double *y, const FINT *incy);
void cblas_dcopy(const int N, const double *X, const int incX, double *Y, const int incY){
#ifdef FORINT
	const FINT ntmp=N;
	const FINT incxtmp=incX;
	const FINT incytmp=incY;
	FORNAME(dcopy) (&ntmp,X,&incxtmp,Y,&incytmp);
#else
	FORNAME(dcopy) (&N,X,&incX,Y,&incY);
#endif
}

extern "C" void FORNAME(daxpy) (const FINT *n, const double *a, const double *x, const FINT *incx, double *y, const FINT *incy);
void cblas_daxpy(const int N, const double alpha, const double *X, const int incX, double *Y, const int incY){
#ifdef FORINT
	const FINT ntmp=N;
	const FINT incxtmp=incX;
	const FINT incytmp=incY;
	FORNAME(daxpy) (&ntmp,&alpha,X,&incxtmp,Y,&incytmp);
#else
	FORNAME(daxpy) (&N,&alpha,X,&incX,Y,&incY);
#endif
}

extern "C" double FORNAME(dnrm2) (const FINT *n, const double *x, const FINT *incx);
double cblas_dnrm2(const int N, const double *X, const int incX){
#ifdef FORINT
	const FINT ntmp=N;
	const FINT incxtmp=incX;
	return FORNAME(dnrm2) (&ntmp,X,&incxtmp);
#else
	return FORNAME(dnrm2) (&N,X,&incX);
#endif
}

extern "C" double FORNAME(dasum) (const FINT *n, const double *x, const FINT *incx);
double cblas_dasum(const int N, const double *X, const int incX){
#ifdef FORINT
	const FINT ntmp=N;
	const FINT incxtmp=incX;
	return FORNAME(dasum) (&ntmp,X,&incxtmp);
#else
	return FORNAME(dasum) (&N,X,&incX);
#endif
}

extern "C" void FORNAME(zcopy) (const FINT *n, const void *x, const FINT *incx, void *y, const FINT *incy);
void cblas_zcopy(const int N, const void *X, const int incX, void *Y, const int incY){
#ifdef FORINT
	const FINT ntmp=N;
	const FINT incxtmp=incX;
	const FINT incytmp=incY;
	FORNAME(zcopy) (&ntmp,X,&incxtmp,Y,&incytmp);
#else
	FORNAME(zcopy) (&N,X,&incX,Y,&incY);
#endif
}

extern "C" void FORNAME(zaxpy) (const FINT *n, const void *a, const void *x, const FINT *incx, void *y, const FINT *incy);
void cblas_zaxpy(const int N, const void *alpha, const void *X, const int incX, void *Y, const int incY){
#ifdef FORINT
	const FINT ntmp=N;
	const FINT incxtmp=incX;
	const FINT incytmp=incY;
	FORNAME(zaxpy) (&ntmp,alpha,X,&incxtmp,Y,&incytmp);
#else
	FORNAME(zaxpy) (&N,alpha,X,&incX,Y,&incY);
#endif
}

extern "C" void FORNAME(zscal) (const FINT *n, const void *a, void *x, const FINT *incx);
void cblas_zscal(const int N, const void *alpha, void *X, const int incX){
#ifdef FORINT
	const FINT ntmp=N;
	const FINT incxtmp=incX;
	FORNAME(zscal)(&ntmp,alpha,X,&incxtmp);
#else
	FORNAME(zscal)(&N,alpha,X,&incX);
#endif
}

extern "C" void FORNAME(zdscal) (const FINT *n, const double *a, void *x, const FINT *incx);
void cblas_zdscal(const int N, const double alpha, void *X, const int incX){
#ifdef FORINT
	const FINT ntmp=N;
	const FINT incxtmp=incX;
	FORNAME(zdscal)(&ntmp,&alpha,X,&incxtmp);
#else
	FORNAME(zdscal)(&N,&alpha,X,&incX);
#endif
}


extern "C" double FORNAME(dznrm2) (const FINT *n, const void *x, const FINT *incx);
double cblas_dznrm2(const int N, const void *X, const int incX){
#ifdef FORINT
	const FINT ntmp=N;
	const FINT incxtmp=incX;
	return FORNAME(dznrm2) (&ntmp,X,&incxtmp);
#else
	return FORNAME(dznrm2) (&N,X,&incX);
#endif
}


//the following ones are f2c-compatible, but is it truly portable???
extern "C" void FORNAME(zdotu) (void *retval, const FINT *n, const void *x, const FINT *incx, const void *y, const FINT *incy);
void cblas_zdotu_sub(const int N, const void *X, const int incX, const void *Y, const int incY, void *dotu){
#ifdef FORINT
	const FINT ntmp=N;
	const FINT incxtmp=incX;
	const FINT incytmp=incY;
	FORNAME(zdotu) (dotu,&ntmp,X,&incxtmp,Y,&incytmp);
#else
	FORNAME(zdotu) (dotu,&N,X,&incX,Y,&incY);
#endif
}


extern "C" void FORNAME(zdotc) (void *retval, const FINT *n, const void *x, const FINT *incx, const void *y, const FINT *incy);
void cblas_zdotc_sub(const int N, const void *X, const int incX, const void *Y, const int incY, void *dotc){
#ifdef FORINT
	const FINT ntmp=N;
	const FINT incxtmp=incX;
	const FINT incytmp=incY;
	FORNAME(zdotc) (dotc,&ntmp,X,&incxtmp,Y,&incytmp);
#else
	FORNAME(zdotc) (dotc,&N,X,&incX,Y,&incY);
#endif
}


//Level 2 and Level 3 on symmetric/hermitian packed matrices - straightforward
//enum CBLAS_UPLO  {CblasUpper=121, CblasLower=122};
//enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102 };
//enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, AtlasConj=114};


extern "C" void FORNAME(dspmv) (const char *uplo, const FINT *n, const double *alpha, const double *ap, const double *x, const FINT *incx, const double *beta, double *y, const FINT *incy);
void cblas_dspmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
		const int N, const double alpha, const double *Ap,
		const double *X, const int incX,
		const double beta, double *Y, const int incY)
{
	if(Order!=CblasRowMajor) laerror("CblasRowMajor order asserted");
	if(Uplo!=CblasLower) laerror("CblasLower uplo asserted");
	char U = BLAS_FORTRANCASE('u');
#ifdef FORINT
	const FINT ntmp=N;
	const FINT incxtmp=incX;
	const FINT incytmp=incY;
	FORNAME(dspmv) (&U,&ntmp, &alpha, Ap, X, &incxtmp, &beta, Y, &incytmp);
#else
	FORNAME(dspmv) (&U,&N, &alpha, Ap, X, &incX, &beta, Y, &incY);
#endif
}


extern "C" void FORNAME(zhpmv) (const char *uplo, const FINT *n, const void *alpha, const void *ap, const void *x, const FINT *incx, const void *beta, void *y, const FINT *incy);
void cblas_zhpmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
		const int N, const void *alpha, const void *Ap,
		const void *X, const int incX,
		const void *beta, void *Y, const int incY)
{
	if(Order!=CblasRowMajor) laerror("CblasRowMajor order asserted");
	if(Uplo!=CblasLower) laerror("CblasLower uplo asserted");
	char U = BLAS_FORTRANCASE('u');
#ifdef FORINT
	const FINT ntmp=N;
	const FINT incxtmp=incX;
	const FINT incytmp=incY;
	FORNAME(zhpmv) (&U,&ntmp, alpha, Ap, X, &incxtmp, beta, Y, &incytmp);
#else
	FORNAME(zhpmv) (&U,&N, alpha, Ap, X, &incX, beta, Y, &incY);
#endif
}


//Level 2 and Level 3 on general matrices - take into account the transposed storage of matrices in Fortran and C

extern "C" void FORNAME(dger) (const FINT *m, const FINT *n, const double *alpha, const double *x, const FINT *incx, const double *y, const FINT *incy, double *a, const FINT *lda);
void cblas_dger(const enum CBLAS_ORDER Order, const int M, const int N,
		const double alpha, const double *X, const int incX,
		const double *Y, const int incY, double *A, const int lda)
{
	if(Order!=CblasRowMajor) laerror("CblasRowMajor order asserted");
	//swap m-n, y-x
#ifdef FORINT
	const FINT mtmp=M;
	const FINT ntmp=N;
	const FINT incxtmp=incX;
	const FINT incytmp=incY;
	const FINT ldatmp=lda;
	FORNAME(dger) (&ntmp, &mtmp, &alpha, Y, &incytmp, X, &incxtmp, A, &ldatmp);
#else
	FORNAME(dger) (&N, &M, &alpha, Y, &incY, X, &incX, A, &lda);
#endif
}

extern "C" void FORNAME(zgerc)  (const FINT *m, const FINT *n, const void *alpha, const void *x, const FINT *incx, const void *y, const FINT *incy, void *a, const FINT *lda);
void cblas_zgerc(const enum CBLAS_ORDER Order, const int M, const int N,
		const void *alpha, const void *X, const int incX,
		const void *Y, const int incY, void *A, const int lda)
{
	if(Order!=CblasRowMajor) laerror("CblasRowMajor order asserted");
//conjugate y
complex<double> *p;
p= (complex<double> *) Y; for(int i=0; i<N; ++i) {p->imag() = -p->imag(); p+= incY;}
#ifdef FORINT
	const FINT mtmp=M;
        const FINT ntmp=N;
	const FINT incxtmp=incX;
	const FINT incytmp=incY;
	const FINT ldatmp=lda;
	FORNAME(zgerc) (&ntmp, &mtmp, alpha, Y, &incytmp, X, &incxtmp, A, &ldatmp);
#else
	FORNAME(zgerc)(&N, &M, alpha, Y, &incY, X, &incX, A, &lda);
#endif
//conjugate y back
p= (complex<double> *) Y; for(int i=0; i<N; ++i) {p->imag() = -p->imag(); p+= incY;}
}


extern "C" void FORNAME(zgeru)  (const FINT *m, const FINT *n, const void *alpha, const void *x, const FINT *incx, const void *y, const FINT *incy, void *a, const FINT *lda);
void cblas_zgeru(const enum CBLAS_ORDER Order, const int M, const int N,
		const void *alpha, const void *X, const int incX,
		const void *Y, const int incY, void *A, const int lda)
{
	if(Order!=CblasRowMajor) laerror("CblasRowMajor order asserted");
#ifdef FORINT
	const FINT mtmp=M;
        const FINT ntmp=N;
	const FINT incxtmp=incX;
	const FINT incytmp=incY;
	const FINT ldatmp=lda;
	FORNAME(zgeru) (&ntmp, &mtmp, alpha, Y, &incytmp, X, &incxtmp, A, &ldatmp);
#else
	FORNAME(zgeru)(&N, &M, alpha, Y, &incY, X, &incX, A, &lda);
#endif
}


extern "C" void FORNAME(dgemm) (const char *transa, const char *transb, const FINT *m, const FINT *n, const FINT *k, const double *alpha, const double *a, const FINT *lda, const double *b, const FINT *ldb, const double *beta, double *c, const FINT *ldc);
void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
		const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
		const int K, const double alpha, const double *A,
		const int lda, const double *B, const int ldb,
		const double beta, double *C, const int ldc)
{
	if(Order!=CblasRowMajor) laerror("CblasRowMajor order asserted");
	//swap a-b, m-n
	char transb = BLAS_FORTRANCASE(TransB==CblasNoTrans?'N':'T');
	char transa = BLAS_FORTRANCASE(TransA==CblasNoTrans?'N':'T');
#ifdef FORINT
	const FINT mtmp=M;
	const FINT ntmp=N;
	const FINT ktmp=K;
	const FINT ldatmp=lda;
	const FINT ldbtmp=ldb;
	const FINT ldctmp=ldc;
	FORNAME(dgemm) (&transb,&transa,
			&ntmp, &mtmp, &ktmp, &alpha, B, &ldbtmp, A, &ldatmp, &beta, C, &ldctmp);
#else
	FORNAME(dgemm) (&transb,&transa,
			&N, &M, &K, &alpha, B, &ldb, A, &lda, &beta, C, &ldc);
#endif
}

extern "C" void FORNAME(zgemm) (const char *transa, const char *transb, const FINT *m, const FINT *n, const FINT *k, const void *alpha, const void *a, const FINT *lda, const void *b, const FINT *ldb, const void *beta, void *c, const FINT *ldc);
void cblas_zgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
		const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
		const int K, const void *alpha, const void *A,
		const int lda, const void *B, const int ldb,
		const void *beta, void *C, const int ldc)
{
	if(Order!=CblasRowMajor) laerror("CblasRowMajor order asserted");
	//swap a-b, m-n
	char transb =  BLAS_FORTRANCASE(TransB==CblasConjTrans?'C':(TransB==CblasNoTrans?'N':'T'));
	char transa =  BLAS_FORTRANCASE(TransA==CblasConjTrans?'C':(TransA==CblasNoTrans?'N':'T'));
#ifdef FORINT
	const FINT mtmp=M;
	const FINT ntmp=N;
	const FINT ktmp=K;
	const FINT ldatmp=lda;
	const FINT ldbtmp=ldb;
	const FINT ldctmp=ldc;
	FORNAME(zgemm) (&transb,&transa,
			&ntmp, &mtmp, &ktmp, alpha, B, &ldbtmp, A, &ldatmp, beta, C, &ldctmp);
#else
	FORNAME(zgemm) ( &transb,&transa,
			&N, &M, &K, alpha, B, &ldb, A, &lda, beta, C, &ldc);
#endif
}


extern "C" void FORNAME(dgemv)  (const char *TRANS, const FINT *M, const FINT *N, const double *ALPHA, const double *A,  const FINT *LDA,  const double *X,  const FINT *INCX, const double *BETA, double *Y, const FINT *INCY);
void cblas_dgemv(const enum CBLAS_ORDER Order,
		const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
		const double alpha, const double *A, const int lda,
		const double *X, const int incX, const double beta,
		double *Y, const int incY)
{
	char transa =  BLAS_FORTRANCASE(TransA==CblasNoTrans?'N':'T');
	char transax =  BLAS_FORTRANCASE(TransA==CblasNoTrans?'T':'N');
#ifdef FORINT
	const FINT mtmp=M;
	const FINT ntmp=N;
	const FINT ldatmp=lda;
	const FINT incxtmp=incX;
	const FINT incytmp=incY;
	if(Order!=CblasRowMajor) FORNAME(dgemv) (&transa, &ntmp, &mtmp, &alpha, A, &ldatmp, X, &incxtmp, &beta, Y, &incytmp );
	//swap n-m and toggle transposition
	else FORNAME(dgemv) (&transax, &ntmp, &mtmp, &alpha, A, &ldatmp, X, &incxtmp, &beta, Y, &incytmp );
#else
	if(Order!=CblasRowMajor) FORNAME(dgemv) (&transa, &N, &M, &alpha, A, &lda, X, &incX, &beta, Y, &incY );
	//swap n-m and toggle transposition
	else FORNAME(dgemv) (&transax, &N, &M, &alpha, A, &lda, X, &incX, &beta, Y, &incY );
#endif
}


extern "C" void FORNAME(zgemv)  (const char *TRANS, const FINT *M, const FINT *N, const void *ALPHA, const void *A,  const FINT *LDA,  const void *X,  const FINT *INCX, const void *BETA, void *Y, const FINT *INCY);
void cblas_zgemv(const enum CBLAS_ORDER Order,
		const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
		const void *alpha, const void *A, const int lda,
		const void *X, const int incX, const void *beta,
		void *Y, const int incY)
{
	if(Order!=CblasRowMajor) laerror("CblasRowMajor order asserted");
	if(TransA == CblasConjTrans) laerror("zgemv with CblasConjTrans not supportted");
	//swap n-m and toggle transposition
	char transa =  BLAS_FORTRANCASE(TransA==CblasNoTrans?'T':'N');
#ifdef FORINT
	const FINT mtmp=M;
	const FINT ntmp=N;
	const FINT ldatmp=lda;
	const FINT incxtmp=incX;
	const FINT incytmp=incY;
	FORNAME(zgemv) (&transa, &ntmp, &mtmp, alpha, A, &ldatmp, X, &incxtmp, beta, Y, &incytmp );
#else
	FORNAME(zgemv) (&transa, &N, &M, alpha, A, &lda, X, &incX, beta, Y, &incY );
#endif
}

extern "C" FINT FORNAME(idamax) (const FINT *N, const double *DX, const FINT *INCX);
CBLAS_INDEX cblas_idamax(const int N, const double *X, const int incX) {
#ifdef FORINT
	const FINT ntmp=N;
	const FINT incxtmp=incX;
	return (CBLAS_INDEX)FORNAME(idamax)(&ntmp,X,&incxtmp);
#else
	return (CBLAS_INDEX)FORNAME(idamax)(&N,X,&incX);
#endif
}

extern "C" FINT FORNAME(izamax) (const FINT *N, const void *DX, const FINT *INCX);
CBLAS_INDEX cblas_izamax(const int N, const void *X, const int incX) {
#ifdef FORINT
	const FINT ntmp=N;
	const FINT incxtmp=incX;
	return (CBLAS_INDEX)FORNAME(izamax)(&ntmp, X, &incxtmp);
#else
	return (CBLAS_INDEX)FORNAME(izamax)(&N, X, &incX);
#endif
}

/*
extern "C" FINT FORNAME(idamin) (const FINT *N, const double *DX, const FINT *INCX);
CBLAS_INDEX cblas_idamin(const int N, const double *X, const int incX) {
#ifdef FORINT
	const FINT ntmp=N;
	const FINT incxtmp=incX;
	return (CBLAS_INDEX)FORNAME(idamin)(&ntmp,X,&incxtmp);
#else
	return (CBLAS_INDEX)FORNAME(idamin)(&N,X,&incX);
#endif
}

extern "C" FINT FORNAME(izamin) (const FINT *N, const void *DX, const FINT *INCX);
CBLAS_INDEX cblas_izamin(const int N, const void *X, const int incX) {
#ifdef FORINT
	const FINT ntmp=N;
	const FINT incxtmp=incX;
	return (CBLAS_INDEX)FORNAME(izamin)(&ntmp, X, &incxtmp);
#else
	return (CBLAS_INDEX)FORNAME(izamin)(&N, X, &incX);
#endif
}
*/
#endif


#ifdef NONCLAPACK
//clapack_dgesv
//allocate auxiliary storage and transpose input and output quantities to fortran/C order
extern "C" void FORNAME(dgesv) (const FINT *N, const FINT *NRHS, double *A, const FINT *LDA, FINT *IPIV, double *B, const FINT *LDB, FINT *INFO);

int clapack_dgesv(const enum CBLAS_ORDER Order, const int N, const int NRHS,
		double *A, const int lda, int *ipiv,
		double *B, const int ldb)
{
	FINT INFO=0;
	if(Order!=CblasRowMajor) laerror("CblasRowMajor order asserted");
	//B should be in the same physical order, just transpose A in place and the LU result on output
	for(int i=1; i<N; ++i) for(int j=0; j<i; ++j)  {double t=A[j*lda+i]; A[j*lda+i]=A[i*lda+j]; A[i*lda+j]=t;}
#ifdef FORINT
	const FINT ntmp=N;
	const FINT nrhstmp=NRHS;
	const FINT ldatmp=lda;
	const FINT ldbtmp=ldb;
	FINT ipivtmp=*ipiv;
	FORNAME(dgesv) (&ntmp,&nrhstmp,A,&ldatmp,&ipivtmp,B,&ldbtmp,&INFO);
#else
	FORNAME(dgesv) (&N,&NRHS,A,&lda,ipiv,B,&ldb,&INFO);
#endif
	for(int i=1; i<N; ++i) for(int j=0; j<i; ++j)  {double t=A[j*lda+i]; A[j*lda+i]=A[i*lda+j]; A[i*lda+j]=t;}
	return (int)INFO;
}

#endif


extern "C" void FORNAME(dswap)(const FINT *N, double *X, const FINT *incX, double *Y, const FINT *incY);
void cblas_dswap(const int N, double *X, const int incX, double *Y, const int incY){
#ifdef FORINT
	const FINT N_tmp = N;
	const FINT incX_tmp = incX;
	const FINT incY_tmp = incY;
	FORNAME(dswap)(&N_tmp, X, &incX_tmp, Y, &incY_tmp);
#else
	FORNAME(dswap)(&N, X, &incX, Y, &incY);
#endif
}

extern "C" void FORNAME(zswap)(const FINT *N, void *X, const FINT *incX, void *Y, const FINT *incY);
void cblas_zswap(const int N, void *X, const int incX, void *Y, const int incY){
#ifdef FORINT
	const FINT N_tmp = N;
	const FINT incX_tmp = incX;
	const FINT incY_tmp = incY;
	FORNAME(zswap)(&N_tmp, X, &incX_tmp, Y, &incY_tmp);
#else
	FORNAME(zswap)(&N, X, &incX, Y, &incY);
#endif
}
