/*
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
*/


#include "noncblas.h"
#include "laerror.h"
#include "mat.h"


#ifdef FORTRAN_
#define FORNAME(x) x##_
#else
#define FORNAME(x) x
#endif

#ifdef NONCBLAS
//Level 1 - straightforward wrappers

extern "C" double FORNAME(ddot) (const int *n, const double *x, const int *incx, const double *y, const int *incy);
double cblas_ddot(const int N, const double *X, const int incX,
                  const double *Y, const int incY)
{
return FORNAME(ddot)(&N,X,&incX,Y,&incY);
}

extern "C" void FORNAME(dscal) (const int *n, const double *a, double *x, const int *incx);
void cblas_dscal(const int N, const double alpha, double *X, const int incX)
{
FORNAME(dscal) (&N,&alpha,X,&incX);
}

extern "C" void FORNAME(dcopy) (const int *n, const double *x, const int *incx, double *y, const int *incy);
void cblas_dcopy(const int N, const double *X, const int incX,
                 double *Y, const int incY)
{
FORNAME(dcopy) (&N,X,&incX,Y,&incY);
}

extern "C" void FORNAME(daxpy) (const int *n, const double *a, const double *x, const int *incx, double *y, const int *incy);
void cblas_daxpy(const int N, const double alpha, const double *X,
                 const int incX, double *Y, const int incY)
{
FORNAME(daxpy) (&N,&alpha,X,&incX,Y,&incY);
}

extern "C" double FORNAME(dnrm2) (const int *n, const double *x, const int *incx);
double cblas_dnrm2(const int N, const double *X, const int incX)
{
return FORNAME(dnrm2) (&N,X,&incX);
}

extern "C" double FORNAME(dasum) (const int *n, const double *x, const int *incx);
double cblas_dasum(const int N, const double *X, const int incX)
{
return FORNAME(dasum) (&N,X,&incX);
}

extern "C" void FORNAME(zcopy) (const int *n, const void *x, const int *incx, void *y, const int *incy);
void cblas_zcopy(const int N, const void *X, const int incX,
                 void *Y, const int incY)
{
FORNAME(zcopy) (&N,X,&incX,Y,&incY);
}

extern "C" void FORNAME(zaxpy) (const int *n, const void *a, const void *x, const int *incx, void *y, const int *incy);
void cblas_zaxpy(const int N, const void *alpha, const void *X,
                 const int incX, void *Y, const int incY)
{
FORNAME(zaxpy) (&N,alpha,X,&incX,Y,&incY);
}

extern "C" void FORNAME(zscal) (const int *n, const void *a, void *x, const int *incx);
void cblas_zscal(const int N, const void *alpha, void *X, const int incX)
{
FORNAME(zscal)(&N,alpha,X,&incX);
}

extern "C" void FORNAME(zdscal) (const int *n, const double *a, void *x, const int *incx);
void cblas_zdscal(const int N, const double alpha, void *X, const int incX)
{
FORNAME(zdscal)(&N,&alpha,X,&incX);
}


extern "C" double FORNAME(dznrm2) (const int *n, const void *x, const int *incx);
double cblas_dznrm2(const int N, const void *X, const int incX)
{
return FORNAME(dznrm2) (&N,X,&incX);
}


//the following ones are f2c-compatible, but is it truly portable???

extern "C" void FORNAME(zdotu) (void *retval, const int *n, const void *x, const int *incx, const void *y, const int *incy);

void   cblas_zdotu_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotu)
{
FORNAME(zdotu) (dotu,&N,X,&incX,Y,&incY);
}


extern "C" void FORNAME(zdotc) (void *retval, const int *n, const void *x, const int *incx, const void *y, const int *incy);
void   cblas_zdotc_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotc)
{
FORNAME(zdotc) (dotc,&N,X,&incX,Y,&incY);
}


//Level 2 and Level 3 on symmetric/hermitian packed matrices - straightforward
//enum CBLAS_UPLO  {CblasUpper=121, CblasLower=122};
//enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102 };
//enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, AtlasConj=114};


extern "C" void FORNAME(dspmv) (const char *uplo, const int *n, const double *alpha, const double *ap, const double *x, const int *incx, const double *beta, double *y, const int *incy);
void cblas_dspmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const int N, const double alpha, const double *Ap,
                 const double *X, const int incX,
                 const double beta, double *Y, const int incY)
{
if(Order!=CblasRowMajor) LA::laerror("CblasRowMajor order asserted");
if(Uplo!=CblasLower) LA::laerror("CblasLower uplo asserted");
FORNAME(dspmv) ("U",&N, &alpha, Ap, X, &incX, &beta, Y, &incY);
}


extern "C" void FORNAME(zhpmv) (const char *uplo, const int *n, const void *alpha, const void *ap, const void *x, const int *incx, const void *beta, void *y, const int *incy);
void cblas_zhpmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const int N, const void *alpha, const void *Ap,
                 const void *X, const int incX,
                 const void *beta, void *Y, const int incY)
{
if(Order!=CblasRowMajor) LA::laerror("CblasRowMajor order asserted");
if(Uplo!=CblasLower) LA::laerror("CblasLower uplo asserted");
FORNAME(zhpmv) ("U",&N, alpha, Ap, X, &incX, beta, Y, &incY);
}


//Level 2 and Level 3 on general matrices - take into account the transposed storage of matrices in Fortran and C

extern "C" void FORNAME(dger) (const int *m, const int *n, const double *alpha, const double *x, const int *incx, const double *y, const int *incy, double *a, const int *lda);
void cblas_dger(const enum CBLAS_ORDER Order, const int M, const int N,
                const double alpha, const double *X, const int incX,
                const double *Y, const int incY, double *A, const int lda)
{
if(Order!=CblasRowMajor) LA::laerror("CblasRowMajor order asserted");
//swap m-n, y-x
FORNAME(dger) (&N, &M, &alpha, Y, &incY, X, &incX, A, &lda);
}

void cblas_zgerc(const enum CBLAS_ORDER Order, const int M, const int N,
                 const void *alpha, const void *X, const int incX,
                 const void *Y, const int incY, void *A, const int lda)
{
LA::laerror("cblas_zgerc cannot be simply converted to fortran order");
}

void cblas_zgeru(const enum CBLAS_ORDER Order, const int M, const int N,
                 const void *alpha, const void *X, const int incX,
                 const void *Y, const int incY, void *A, const int lda)
{
LA::laerror("cblas_zgeru cannot be simply converted to fortran order");
}

extern "C" void FORNAME(dgemm) (const char *transa, const char *transb, const int *m, const int *n, const int *k, const double *alpha, const double *a, const int *lda, const double *b, const int *ldb, const double *beta, double *c, const int *ldc);
void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb,
                 const double beta, double *C, const int ldc)
{
if(Order!=CblasRowMajor) LA::laerror("CblasRowMajor order asserted");
//swap a-b, m-n
FORNAME(dgemm) (TransB==CblasNoTrans?"N":"T", TransA==CblasNoTrans?"N":"T",
	 &N, &M, &K, &alpha, B, &ldb, A, &lda, &beta, C, &ldc);
}

extern "C" void FORNAME(zgemm) (const char *transa, const char *transb, const int *m, const int *n, const int *k, const void *alpha, const void *a, const int *lda, const void *b, const int *ldb, const void *beta, void *c, const int *ldc);
void cblas_zgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const void *alpha, const void *A,
                 const int lda, const void *B, const int ldb,
                 const void *beta, void *C, const int ldc)
{
if(Order!=CblasRowMajor) LA::laerror("CblasRowMajor order asserted");
//swap a-b, m-n
FORNAME(zgemm) ( TransB==CblasConjTrans?"C":(TransB==CblasNoTrans?"N":"T"),
		 TransA==CblasConjTrans?"C":(TransB==CblasNoTrans?"N":"T"),
         &N, &M, &K, alpha, B, &ldb, A, &lda, beta, C, &ldc);
}


extern "C" void FORNAME(dgemv)  (const char *TRANS, const int *M, const int *N, const double *ALPHA, const double *A,  const int *LDA,  const double *X,  const int *INCX, const double *BETA, double *Y, const int *INCY);
void cblas_dgemv(const enum CBLAS_ORDER Order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 const double *X, const int incX, const double beta,
                 double *Y, const int incY)
{
if(Order!=CblasRowMajor) FORNAME(dgemv) (TransA==CblasNoTrans?"N":"T", &N, &M, &alpha, A, &lda, X, &incX, &beta, Y, &incY );
//swap n-m and toggle transposition
else FORNAME(dgemv) (TransA==CblasNoTrans?"T":"N", &N, &M, &alpha, A, &lda, X, &incX, &beta, Y, &incY );
}


extern "C" void FORNAME(zgemv)  (const char *TRANS, const int *M, const int *N, const void *ALPHA, const void *A,  const int *LDA,  const void *X,  const int *INCX, const void *BETA, void *Y, const int *INCY);
void cblas_zgemv(const enum CBLAS_ORDER Order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const void *alpha, const void *A, const int lda,
                 const void *X, const int incX, const void *beta,
                 void *Y, const int incY)
{
if(Order!=CblasRowMajor) LA::laerror("CblasRowMajor order asserted");
if(TransA == CblasConjTrans) LA::laerror("zgemv with CblasConjTrans not supportted");
//swap n-m and toggle transposition
FORNAME(zgemv) (TransA==CblasNoTrans?"T":"N", &N, &M, alpha, A, &lda, X, &incX, beta, Y, &incY );
}

extern "C" int FORNAME(idamax) (const int *N, const double *DX, const int *INCX);

CBLAS_INDEX cblas_idamax(const int N, const double *X, const int incX)
{
return FORNAME(idamax)(&N,X,&incX);
}


#endif

#ifdef  NONCLAPACK
//clapack_dgesv
//allocate auxiliary storage and transpose input and output quantities to fortran/C order
extern "C" void FORNAME(dgesv) (const int *N, const int *NRHS, double *A, const int *LDA, int *IPIV, double *B, const int *LDB, int *INFO);

int clapack_dgesv(const enum CBLAS_ORDER Order, const int N, const int NRHS,
                  double *A, const int lda, int *ipiv,
                  double *B, const int ldb)
{
int INFO=0;
if(Order!=CblasRowMajor) LA::laerror("CblasRowMajor order asserted");
//B should be in the same physical order, just transpose A in place and the LU result on output
for(int i=1; i<N; ++i) for(int j=0; j<i; ++j)  {double t=A[j*lda+i]; A[j*lda+i]=A[i*lda+j]; A[i*lda+j]=t;}
FORNAME(dgesv) (&N,&NRHS,A,&lda,ipiv,B,&ldb,&INFO);
for(int i=1; i<N; ++i) for(int j=0; j<i; ++j)  {double t=A[j*lda+i]; A[j*lda+i]=A[i*lda+j]; A[i*lda+j]=t;}
return INFO;
}

#endif


