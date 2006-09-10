#ifndef _LA_NONCLASS_H_
#define _LA_NONCLASS_H_

#include "vec.h"
#include "smat.h"
#include "mat.h"
#include "la_traits.h"


//MISC
export template <class T>
const NRSMat<T> twoside_transform(const NRSMat<T> &S, const NRMat<T> &C, bool transp=0) //calculate C^dagger S C
{
if(transp)
	{
	NRMat<T> tmp =  C * S;
	NRMat<T> result(C.nrows(),C.nrows());
	result.gemm((T)0,tmp,'n',C,'t',(T)1);
	return NRSMat<T>(result);
	}
NRMat<T> tmp = S * C;
NRMat<T> result(C.ncols(),C.ncols());
result.gemm((T)0,C,'t',tmp,'n',(T)1);
return NRSMat<T>(result);
}




export template <class T>
const NRMat<T> diagonalmatrix(const NRVec<T> &x)
{
int n=x.size();
NRMat<T> result((T)0,n,n);
T *p = result[0];
for(int j=0; j<n; j++) {*p  = x[j]; p+=(n+1);}
return result;
}


//more efficient commutator for a special case of full matrices
template<class T>
inline const NRMat<T> commutator ( const NRMat<T> &x, const NRMat<T> &y, const bool trx=0, const bool tryy=0)
{
NRMat<T> r(trx?x.ncols():x.nrows(), tryy?y.nrows():y.ncols());
r.gemm((T)0,x,trx?'t':'n',y,tryy?'t':'n',(T)1);
r.gemm((T)1,y,tryy?'t':'n',x,trx?'t':'n',(T)-1);
return r;
}

//more efficient commutator for a special case of full matrices
template<class T>
inline const NRMat<T> anticommutator ( const NRMat<T> &x, const NRMat<T> &y, const bool trx=0, const bool tryy=0)
{
NRMat<T> r(trx?x.ncols():x.nrows(), tryy?y.nrows():y.ncols());
r.gemm((T)0,x,trx?'t':'n',y,tryy?'t':'n',(T)1);
r.gemm((T)1,y,tryy?'t':'n',x,trx?'t':'n',(T)1);
return r;
}




//////////////////////
// LAPACK interface //
//////////////////////

#define declare_la(T) \
extern const  NRVec<T> diagofproduct(const NRMat<T> &a, const NRMat<T> &b,\
		bool trb=0, bool conjb=0); \
extern T trace2(const NRMat<T> &a, const NRMat<T> &b, bool trb=0); \
extern T trace2(const NRSMat<T> &a, const NRSMat<T> &b, const bool diagscaled=0);\
extern T trace2(const NRSMat<T> &a, const NRMat<T> &b, const bool diagscaled=0);\
extern void linear_solve(NRMat<T> &a, NRMat<T> *b, double *det=0,int n=0); \
extern void linear_solve(NRSMat<T> &a, NRMat<T> *b, double *det=0, int n=0); \
extern void linear_solve(NRMat<T> &a, NRVec<T> &b, double *det=0, int n=0); \
extern void linear_solve(NRSMat<T> &a, NRVec<T> &b, double *det=0, int n=0); \
extern void diagonalize(NRMat<T> &a, NRVec<T> &w, const bool eivec=1, const bool corder=1, int n=0, NRMat<T> *b=NULL, const int itype=1); \
extern void diagonalize(NRSMat<T> &a, NRVec<T> &w, NRMat<T> *v, const bool corder=1, int n=0, NRSMat<T> *b=NULL, const int itype=1);\
extern void singular_decomposition(NRMat<T> &a, NRMat<T> *u, NRVec<T> &s,\
		NRMat<T> *v, const bool corder=1, int m=0, int n=0);

/*NOTE!!! all versions of diagonalize DESTROY A and generalized diagonalize also B matrix */

declare_la(double)
declare_la(complex<double>)

// Separate declarations
//general nonsymmetric matrix and generalized diagonalization
extern void gdiagonalize(NRMat<double> &a, NRVec<double> &wr, NRVec<double> &wi,
		NRMat<double> *vl, NRMat<double> *vr, const bool corder=1, int n=0, const int sorttype=0, const bool biorthonormalize=0,
		NRMat<double> *b=NULL, NRVec<double> *beta=NULL);
extern void gdiagonalize(NRMat<double> &a, NRVec< complex<double> > &w,
		 NRMat< complex<double> >*vl, NRMat< complex<double> > *vr,
		 const bool corder=1, int n=0, const int sorttype=0, const bool biorthonormalize=0,
		NRMat<double> *b=NULL, NRVec<double> *beta=NULL);
extern NRMat<double> matrixfunction(NRSMat<double> a, double (*f) (double));
extern NRMat<double> realmatrixfunction(NRMat<double> a, double (*f) (double)); //a has to by in fact symmetric
extern NRMat<complex<double> > complexmatrixfunction(NRMat<double> a, double (*fre) (double), double (*fim) (double)); //a has to by in fact symmetric
extern NRMat<double> matrixfunction(NRMat<double> a, complex<double> (*f)(const complex<double> &),const bool adjust=0);

extern complex<double> sqrtinv(const complex<double> &);
extern double sqrtinv(const double);

//functions on matrices
inline NRMat<double>  sqrt(const NRSMat<double> &a) { return matrixfunction(a,&sqrt); }
inline NRMat<double>  sqrtinv(const NRSMat<double> &a) { return matrixfunction(a,&sqrtinv); }
inline NRMat<double>  realsqrt(const NRMat<double> &a) { return realmatrixfunction(a,&sqrt); }
inline NRMat<double>  realsqrtinv(const NRMat<double> &a) { return realmatrixfunction(a,&sqrtinv); }
inline NRMat<double>  log(const NRSMat<double> &a) { return matrixfunction(a,&log); }
extern NRMat<double> log(const NRMat<double> &a);
extern NRMat<double> exp0(const NRMat<double> &a);


extern const NRMat<double> realpart(const NRMat< complex<double> >&);
extern const NRMat<double> imagpart(const NRMat< complex<double> >&);
extern const NRMat< complex<double> > realmatrix (const NRMat<double>&);
extern const NRMat< complex<double> > imagmatrix (const NRMat<double>&);

//inverse by means of linear solve, preserving rhs intact
template<typename T>
const NRMat<T> inverse(NRMat<T> a, T *det=0)
{
#ifdef DEBUG
	if(a.nrows()!=a.ncols()) laerror("inverse() for non-square matrix");
#endif
	NRMat<T> result(a.nrows(),a.nrows());
	result = (T)1.;
	linear_solve(a, &result, det);
	return result;
}

//general determinant
template<class MAT>
const typename LA_traits<MAT>::elementtype determinant(MAT a)//passed by value
{
typename LA_traits<MAT>::elementtype det;
if(a.nrows()!=a.ncols()) laerror("determinant of non-square matrix");
linear_solve(a,NULL,&det);
return det;
}

//general determinant destructive on input 
template<class MAT>
const typename LA_traits<MAT>::elementtype determinant_destroy(MAT &a) //passed by reference 
{
typename LA_traits<MAT>::elementtype det;
if(a.nrows()!=a.ncols()) laerror("determinant of non-square matrix");
linear_solve(a,NULL,&det);
return det;
}


//general submatrix, INDEX will typically be NRVec<int> or even int*
//NOTE: in order to check consistency between nrows and rows in rows is a NRVec
//some advanced metaprogramming would be necessary
//application: e.g. ignoresign=true, equalsigns=true, indexshift= -1 ... elements of Slater overlaps for RHF

template<class MAT, class INDEX>
const NRMat<typename LA_traits<MAT>::elementtype> submatrix(const MAT a, const int nrows, const INDEX rows, const int ncols, const INDEX cols, int indexshift=0, bool ignoresign=false, bool equalsigns=false)
{
NRMat<typename LA_traits<MAT>::elementtype> r(nrows,ncols);

if(equalsigns) //make the element zero if signs of both indices are opposite
{
if(ignoresign)
{
for(int i=0; i<nrows; ++i)
        for(int j=0; j<ncols; ++j)
                r(i,j) = rows[i]*cols[j]<0?0.:a(abs(rows[i])+indexshift,abs(cols[j])+indexshift);
}
else
{
for(int i=0; i<nrows; ++i)
        for(int j=0; j<ncols; ++j)
                r(i,j) = rows[i]*cols[j]<0?0.:a(rows[i]+indexshift,cols[j]+indexshift);
}
}
else
{
if(ignoresign)
{
for(int i=0; i<nrows; ++i)
        for(int j=0; j<ncols; ++j)
                r(i,j) = a(abs(rows[i])+indexshift,abs(cols[j])+indexshift);
}
else
{
for(int i=0; i<nrows; ++i)
	for(int j=0; j<ncols; ++j)
		r(i,j) = a(rows[i]+indexshift,cols[j]+indexshift);
}
}

return r;
}


//auxiliary routine to adjust eigenvectors to guarantee real logarithm
extern void adjustphases(NRMat<double> &v);

template<class T>
T abssqr(const complex<T> &x)
{
return x.real()*x.real()+x.imag()*x.imag();
}

//declaration of template interface to cblas routines with full options available
//just to facilitate easy change between float, double, complex in a user code
//very incomplete, add new ones as needed
template<class T> inline void xcopy(int n, const T *x, int incx, T *y, int incy);
template<class T> inline void xaxpy(int n, const T &a, const T *x, int incx, T *y, int incy);
template<class T> inline T xdot(int n, const T *x, int incx, const T *y, int incy);

//specialized definitions have to be in the header file to be inlineable, eliminating any runtime overhead

template<>
inline void xcopy<double> (int n, const double *x, int incx, double *y, int incy)
{
cblas_dcopy(n, x, incx, y, incy);
}

template<>
inline void xaxpy<double>(int n, const double &a, const double *x, int incx, double *y, int incy)
{
cblas_daxpy(n, a, x, incx, y, incy);
}

template<>
inline double xdot<double>(int n, const double *x, int incx, const double *y, int incy)
{
return cblas_ddot(n,x,incx,y,incy);
}


#endif
