#include "vec.h"
#include "smat.h"
#include "mat.h"

//MISC
template <class T> extern const NRMat<T> diagonalmatrix(const NRVec<T> &x);
template <class T> extern const NRVec<T> lineof(const NRMat<T> &x, const int i); 
template <class T> extern const NRVec<T> columnof(const NRMat<T> &x, const int i);
template <class T> extern const NRVec<T> diagonalof(const NRMat<T> &x); 

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
extern void linear_solve(NRMat<T> &a, NRMat<T> *b, double *det=0); \
extern void linear_solve(NRSMat<T> &a, NRMat<T> *b, double *det=0); \
extern void diagonalize(NRMat<T> &a, NRVec<T> &w, const bool eivec=1,\
		const bool corder=1); \
extern void diagonalize(NRSMat<T> &a, NRVec<T> &w, NRMat<T> *v, const bool corder=1);\
extern void singular_decomposition(NRMat<T> &a, NRMat<T> *u, NRVec<T> &s,\
		NRMat<T> *v, const bool corder=1);

declare_la(double)
declare_la(complex<double>)

// Separate declarations
extern void gdiagonalize(NRMat<double> &a, NRVec<double> &wr, NRVec<double> &wi,
		NRMat<double> *vl, NRMat<double> *vr, const bool corder=1);
extern void gdiagonalize(NRMat<double> &a, NRVec< complex<double> > &w,
		 NRMat< complex<double> >*vl, NRMat< complex<double> > *vr);
extern NRMat<double> matrixfunction(NRSMat<double> a, double (*f) (double));
extern NRMat<double> matrixfunction(NRMat<double> a, complex<double> (*f)(const complex<double> &),const bool adjust=0);

//functions on matrices
inline NRMat<double>  sqrt(const NRSMat<double> &a) { return matrixfunction(a,&sqrt); }
inline NRMat<double>  log(const NRSMat<double> &a) { return matrixfunction(a,&log); }
extern NRMat<double> log(const NRMat<double> &a);


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

