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
#ifndef _LA_NONCLASS_H_
#define _LA_NONCLASS_H_
#include "vec.h"
#include "smat.h"
#include "mat.h"
#include "la_traits.h"

namespace LA {

//MISC


//positive power for a general class
template<typename T>
T positive_power(const T &x, const int n)
{
if(n<=0) laerror("zero or negative n in positive_power");
int i=n;
if(i==1) return x;
T y,z;
z=x;
while(!(i&1))
        {
        z = z*z;
        i >>= 1;
        }
y=z;
while((i >>= 1)/*!=0*/)
                {
                z = z*z;
                if(i&1) y = y*z;
                }
return y;
}


//general integer power for a class providing identity() and inverse()
template<typename T>
T power(const T &x, const int n)
{
int i=n;
if(i==0) {T r(x); r.identity(); return r;}
T z;
if(i>0) z=x;
else {z=x.inverse(); i= -i;}
if(i==1) return z;
return positive_power(z,i);
}




template <class T>
const NRSMat<T> twoside_transform(const NRSMat<T> &S, const NRMat<T> &C, bool transp=0) //calculate C^dagger S C for symmetric/hermitian S
{
char tchar='t';
if(LA_traits<T>::is_complex()) tchar='c';
if(transp)
	{
	NRMat<T> tmp =  C * S;
	NRMat<T> result(C.nrows(),C.nrows());
	result.gemm((T)0,tmp,'n',C,tchar,(T)1);
	return NRSMat<T>(result);
	}
NRMat<T> tmp = S * C;
NRMat<T> result(C.ncols(),C.ncols());
result.gemm((T)0,C,tchar,tmp,'n',(T)1);
return NRSMat<T>(result);
}


template <class T>
const NRMat<T> twoside_transform(const NRMat<T> &S, const NRMat<T> &C, bool transp=0) //calculate C^dagger S C for non-symmetric S
{
char tchar='t';
if(LA_traits<T>::is_complex()) tchar='c';
if(transp)
	{
	NRMat<T> tmp =  C * S;
	NRMat<T> result(C.nrows(),C.nrows());
	result.gemm((T)0,tmp,'n',C,tchar,(T)1);
	return result;
	}
NRMat<T> tmp = S * C;
NRMat<T> result(C.ncols(),C.ncols());
result.gemm((T)0,C,tchar,tmp,'n',(T)1);
return result;
}




template <class T>
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
extern T trace2(const NRMat<T> &a, const NRSMat<T> &b, const bool diagscaled=0);\
extern void linear_solve(NRMat<T> &a, NRMat<T> *b, T *det=0,int n=0); /*solve Ax^T=b^T (b is nrhs x n) */ \
extern void linear_solve(NRSMat<T> &a, NRMat<T> *b, T *det=0, int n=0); /*solve Ax^T=b^T (b is nrhs x n) */\
extern void linear_solve(NRMat<T> &a, NRVec<T> &b, double *det=0, int n=0); \
extern void linear_solve(NRSMat<T> &a, NRVec<T> &b, double *det=0, int n=0); \
extern void diagonalize(NRMat<T> &a, NRVec<LA_traits<T>::normtype> &w, const bool eivec=1, const bool corder=1, int n=0, NRMat<T> *b=NULL, const int itype=1); \
extern void diagonalize(NRSMat<T> &a, NRVec<LA_traits<T>::normtype> &w, NRMat<T> *v, const bool corder=1, int n=0, NRSMat<T> *b=NULL, const int itype=1);\
extern void singular_decomposition(NRMat<T> &a, NRMat<T> *u, NRVec<LA_traits<T>::normtype> &s, NRMat<T> *v, const bool vnotdagger=0, int m=0, int n=0); \

/*NOTE!!! all versions of diagonalize DESTROY A and generalized diagonalize also B matrix */

declare_la(double)
declare_la(std::complex<double>)



//svd_solve without contamination from nullspace, b is n x nrhs, result returns in b, A is destroyed
template<typename T>
NRVec<T> svd_solve(NRMat<T> &a, const NRVec<T> &b, double thres=1e-12)
{
if(b.size()!=a.nrows()) laerror("size mismatch in svd_solve");
NRVec<double> w(a.ncols());
NRMat<T> u(a.nrows(),a.ncols());
NRMat<T> vt(a.ncols(),a.ncols());
singular_decomposition(a,&u,w,&vt,false);
NRVec<T> utb = b*u;
for(int i=0; i<w.size(); ++i) utb[i] =  w[i]>thres ? utb[i]/w[i] : 0.;
return utb*vt;
}


template<typename T>
NRMat<T> svd_solve(NRMat<T> &a, const NRMat<T> &b, double thres=1e-12)
{
if(b.nrows()!=a.nrows()) laerror("size mismatch in svd_solve");
NRVec<double> w(a.ncols());
NRMat<T> u(a.nrows(),a.ncols());
NRMat<T> vt(a.ncols(),a.ncols());
singular_decomposition(a,&u,w,&vt,false);
NRVec<T> ww(w.size());
for(int i=0; i<w.size(); ++i) ww[i] =  w[i]>thres ? 1./w[i] : 0.;
NRMat<T> utb(a.ncols(),b.ncols());
utb.gemm((T)0,u,'c',b,'n',(T)1);
utb.diagmultl(ww);
NRMat<T> res(a.ncols(),b.ncols());
res.gemm((T)0,vt,'c',utb,'n',(T)1);
return res;
}



// Separate declarations
//general nonsymmetric matrix and generalized diagonalization
//corder =0 ... C rows are eigenvectors, =1 ... C columns are eigenvectors
extern void gdiagonalize(NRMat<double> &a, NRVec<double> &wr, NRVec<double> &wi,
		NRMat<double> *vl=NULL, NRMat<double> *vr=NULL, const bool corder=1, int n=0, const int sorttype=0, const int biorthonormalize=0,
		NRMat<double> *b=NULL, NRVec<double> *beta=NULL); //this used real storage of eigenvectors like dgeev

template<typename T>
extern void gdiagonalize(NRMat<T> &a, NRVec< std::complex<double> > &w,
		 NRMat< std::complex<double> >*vl=NULL, NRMat< std::complex<double> > *vr=NULL,
		 const bool corder=1, int n=0, const int sorttype=0, const int biorthonormalize=0,
		NRMat<T> *b=NULL, NRVec<T> *beta=NULL); //eigenvectors are stored in complex matrices for T both double and complex

//for compatibility in davidson
extern void gdiagonalize(NRMat<std::complex<double> > &a, NRVec<double> &wr, NRVec<double> &wi,
                NRMat<std::complex<double> > *vl=NULL, NRMat<std::complex<double> > *vr=NULL, const bool corder=1, int n=0, const int sorttype=0, const int biorthonormalize=0,
                NRMat<std::complex<double> > *b=NULL, NRVec<std::complex<double> > *beta=NULL);

//diagonalization of symmetric real tridiagonal matrix
extern void diagonalize3(NRVec<double> &d, NRVec<double> &d1, NRMat<double> *v, const bool corder=1, int n0=0);

//complex,real,imaginary parts of various entities
template<typename T>
extern const typename LA_traits<T>::realtype realpart(const T&);
template<typename T>
extern const typename LA_traits<T>::realtype imagpart(const T&);
template<typename T>
extern const typename LA_traits<T>::complextype realmatrix (const T&);
template<typename T>
extern const typename LA_traits<T>::complextype imagmatrix (const T&);
template<typename T>
extern const typename LA_traits<T>::complextype complexmatrix (const T&, const T&);

//Cholesky decomposition
extern void cholesky(NRMat<double> &a, bool upper=1);
extern void cholesky(NRMat<std::complex<double> > &a, bool upper=1);

//inverse by means of linear solve, preserving rhs intact
template<typename T>
const NRMat<T> calcinverse(NRMat<T> a, T *det=NULL)
{
#ifdef DEBUG
	if(a.nrows()!=a.ncols()) laerror("inverse() for non-square matrix");
#endif
	NRMat<T> result(a.nrows(),a.nrows());
	result = (T)1.;
	a.copyonwrite();
	linear_solve(a, &result, det);
	result.transposeme(); //tested with noncblas
	return result;
}

//several matrix norms
template<class MAT>
typename LA_traits<MAT>::normtype MatrixNorm(const MAT  &A, const char norm);

//condition number
template<class MAT>
typename LA_traits<MAT>::normtype CondNumber(const MAT  &A, const char norm);


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


//------------------------------------------------------------------------------
// solves set of linear equations using gesvx
// input:
//	A		double precision matrix of dimension nn x mm, where min(nn, mm) >= n
//	B		double prec. array dimensioned as nrhs x n
//	rhsCount	nrhs - count of right hand sides
//	eqCount	n - count of equations
//	eq		use equilibration of matrix A before solving
//	saveA		if set, do no overwrite A if equilibration in effect
//	rcond		if not NULL, store the returned rcond value from dgesvx
// output:
//	solution is stored in B
//	the info parameter of gesvx is returned (see man dgesvx)
//------------------------------------------------------------------------------
template<class T>
int linear_solve_x(NRMat<T> &A, T *B, const int rhsCount, const int eqCount, const bool eq, const bool saveA, double *rcond);


//------------------------------------------------------------------------------
// for given square matrices A, B computes X = AB^{-1} as follows
// 	XB = A => B^TX^T = A^T
// input:
//	_A		double precision matrix of dimension nn x nn
//	_B		double prec. matrix of dimension nn x nn
//	_useEq		use equilibration suitable for badly conditioned matrices
//	_rcond		if not NULL, store the returned value of rcond fromd dgesvx
// output:
//	solution is stored in _B
//	the info parameter of dgesvx is returned (see man dgesvx)
//------------------------------------------------------------------------------
template<class T>
int multiply_by_inverse(NRMat<T> &A, NRMat<T> &B, bool useEq=false, double *rcond=NULL);


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
                r(i,j) = rows[i]*cols[j]<0?0.:a(std::abs(rows[i])+indexshift,std::abs(cols[j])+indexshift);
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
                r(i,j) = a(std::abs(rows[i])+indexshift,std::abs(cols[j])+indexshift);
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



//debugging aid: reconstruct an explicit matrix from the implicit version
//which provides gemv only
template<typename M, typename T>
NRMat<T> reconstructmatrix(const M &implicitmat)
{
NRMat<T> r(implicitmat.nrows(),implicitmat.ncols());
NRVec<T> rhs(0.,implicitmat.ncols());
NRVec<T> tmp(implicitmat.nrows());
for(int i=0; i<implicitmat.ncols(); ++i)
	{
	rhs[i]=1.;
	implicitmat.gemv(0.,tmp,'n',1.,rhs);
	for(int j=0; j<implicitmat.nrows(); ++j) r(j,i)=tmp[j];
	rhs[i]=0.;
	}
return r;
}


//matrix functions via diagonalization

extern NRMat<double> realmatrixfunction(NRMat<double> a, double (*f) (double)); //a has to by in fact symmetric
extern NRMat<std::complex<double> > complexmatrixfunction(NRMat<double> a, double (*fre) (double), double (*fim) (double)); //a has to by in fact symmetric

template<typename T> 
NRMat<T> matrixfunction(NRSMat<T> a, double (*f) (double)) //of symmetric/hermitian matrix
{
	int n = a.nrows();
	NRVec<double> w(n);
	NRMat<T> v(n, n);
	diagonalize(a, w, &v, 0);

	for (int i=0; i<a.nrows(); i++) w[i] = (*f)(w[i]);
	NRMat<T> u = v;
	NRVec<T> ww=w; //diagmultl needs same type
	v.diagmultl(ww);
	NRMat<T> r(n, n);
	r.gemm(0.0, u, 't', v, 'n', 1.0); //gemm will use 'c' for complex ones 
	return r;
}


template<typename T>
extern NRMat<T> matrixfunction(NRMat<T> a, std::complex<double> (*f)(const std::complex<double> &)) //of a general real/complex matrix
{
	int n = a.nrows();
	NRVec<std::complex<double> > w(n);
	NRMat<std::complex<double> > u(n,n),v(n,n);

#ifdef debugmf
NRMat<std::complex<double> > a0=a;
#endif

        gdiagonalize<T>(a, w, &u, &v, false,n,0,false,NULL,NULL);//a gets destroyed, eigenvectors are rows
        NRVec< std::complex<double> > z = diagofproduct(u, v, 1, 1);

#ifdef debugmf
std::cout <<"TEST matrixfunction\n"<<w<<u<<v<<z;
std::cout <<"TEST matrixfunction1 "<< u*a0 - diagonalmatrix(w)*u<<std::endl;
std::cout <<"TEST matrixfunction2 "<< a0*v.transpose(1) - v.transpose(1)*diagonalmatrix(w)<<std::endl;
std::cout <<"TEST matrixfunction3 "<< u*v.transpose(1)<<diagonalmatrix(z)<<std::endl;
#endif

NRVec< std::complex<double> > wz(n);
for (int i=0; i<a.nrows(); i++) wz[i] = w[i]/z[i];

#ifdef debugmf
std::cout <<"TEST matrixfunction4 "<< a0<< v.transpose(true)*diagonalmatrix(wz)*u<<std::endl;
#endif

        for (int i=0; i<a.nrows(); i++) w[i] = (*f)(w[i])/z[i];
        u.diagmultl(w);

        NRMat< std::complex<double> > r(n, n);
        r.gemm(0.0, v, 'c', u, 'n', 1.0);
	return (NRMat<T>) r; //convert back to real if applicable by the explicit decomplexifying constructor; it is NOT checked to which accuracy the imaginary part is actually zero
}




extern std::complex<double> sqrtinv(const std::complex<double> &);
extern double sqrtinv(const double);

//functions on matrices
inline NRMat<double>  sqrt(const NRSMat<double> &a) { return matrixfunction(a,&std::sqrt); }
inline NRMat<double>  sqrtinv(const NRSMat<double> &a) { return matrixfunction(a,&sqrtinv); }
inline NRMat<double>  realsqrt(const NRMat<double> &a) { return realmatrixfunction(a,&std::sqrt); }
inline NRMat<double>  realsqrtinv(const NRMat<double> &a) { return realmatrixfunction(a,&sqrtinv); }
inline NRMat<double>  log(const NRSMat<double> &a) { return matrixfunction(a,&std::log); }
extern NRMat<double> log(const NRMat<double> &a);
extern NRMat<std::complex<double> > log(const NRMat<std::complex<double> > &a);
extern NRMat<std::complex<double> > exp0(const NRMat<std::complex<double> > &a);
extern NRMat<std::complex<double> > copytest(const NRMat<std::complex<double> > &a);
extern NRMat<double> copytest(const NRMat<double> &a);
extern NRMat<double> exp0(const NRMat<double> &a);


}//namespace
#endif
