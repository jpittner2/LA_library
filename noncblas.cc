#ifdef NONCBLAS

#ifdef FORTRAN_
#define FORNAME(x) x##_
#else
#define FORNAME(x) x
#endif

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






//Level 2 and Level 3 - take into account the transposed storage of matrices in Fortran and C

#endif
