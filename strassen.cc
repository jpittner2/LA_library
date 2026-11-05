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


#include "mat.h"

namespace LA {
/*Strassen algorithm in the Winograd variant (7 products, 15 additions), recursion on strided quadrants of the original matrices,
leaf products by BLAS gemm
V. Strassen, Gaussian elimination is not optimal, Numer. Math. 13 (1969) 354-356
S. Winograd, On multiplication of 2x2 matrices, Linear Algebra Appl. 4 (1971) 381-388
schedule with 2 temporaries per level: B. Boyer, J.-G. Dumas, C. Pernet, W. Zhou, Memory efficient scheduling of
	Strassen-Winograd's matrix multiplication algorithm, Proc. ISSAC 2009, 55-62
odd dimensions by dynamic peeling: S. Huss-Lederman, E. M. Jacobson, J. R. Johnson, A. Tsao, T. Turnbull,
	Implementation of Strassen's algorithm for matrix multiplication, Proc. Supercomputing '96
*/

//device-dependent primitives: leaf gemm, quadrant addition, workspace (GPU version: cublas?gemm, cublas?geam, gpualloc/gpufree)
static inline CBLAS_TRANSPOSE s_trans(const char t) {return t=='n'?CblasNoTrans:(t=='c'?CblasConjTrans:CblasTrans);}

static inline void s_gemm(const char ta, const char tb, const int m, const int n, const int k, const double alpha, const double *a, const int lda, const double *b, const int ldb, const double beta, double *c, const int ldc)
{cblas_dgemm(CblasRowMajor,s_trans(ta),s_trans(tb),m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);}
static inline void s_gemm(const char ta, const char tb, const int m, const int n, const int k, const float alpha, const float *a, const int lda, const float *b, const int ldb, const float beta, float *c, const int ldc)
{cblas_sgemm(CblasRowMajor,s_trans(ta),s_trans(tb),m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);}
static inline void s_gemm(const char ta, const char tb, const int m, const int n, const int k, const std::complex<double> alpha, const std::complex<double> *a, const int lda, const std::complex<double> *b, const int ldb, const std::complex<double> beta, std::complex<double> *c, const int ldc)
{cblas_zgemm(CblasRowMajor,s_trans(ta),s_trans(tb),m,n,k,&alpha,a,lda,b,ldb,&beta,c,ldc);}
static inline void s_gemm(const char ta, const char tb, const int m, const int n, const int k, const std::complex<float> alpha, const std::complex<float> *a, const int lda, const std::complex<float> *b, const int ldb, const std::complex<float> beta, std::complex<float> *c, const int ldc)
{cblas_cgemm(CblasRowMajor,s_trans(ta),s_trans(tb),m,n,k,&alpha,a,lda,b,ldb,&beta,c,ldc);}

//z = x+y or x-y, r x c blocks with leading dimensions; z may coincide with x or y (one pass, memory bound)
template<typename T>
static void s_add(const int r, const int c, T *z, const int ldz, const T *x, const int ldx, const T *y, const int ldy, const bool minus)
{
for(int i=0; i<r; ++i)
	{
	T *zz=z+(size_t)i*ldz; const T *xx=x+(size_t)i*ldx; const T *yy=y+(size_t)i*ldy;
	if(minus) for(int j=0; j<c; ++j) zz[j]=xx[j]-yy[j];
	else for(int j=0; j<c; ++j) zz[j]=xx[j]+yy[j];
	}
}

template<typename T> static T *s_alloc(const size_t n) {return n? new T[n] : NULL;}
template<typename T> static void s_free(T *p) {if(p) delete[] p;}

//operand op(M) of the product, op = identity ('n'), transposition ('t') or conjugate transposition ('c'),
//M stored row-major at p with leading dimension ld; the recursion is formulated in op() space and never transposes data:
//- at(i,j) is the address of element (i,j) of op(M), i.e. M(i,j) for 'n' and M(j,i) for 't','c';
//  a block of op(M) starting at (i,j) is then the block of M starting at at(i,j) with the same ld and flag
//  (quadrant (op M)_{IJ} = op(M_{JI}), the stored quadrant being swapped for 't','c' and conjugated by gemm for 'c')
//- sums of quadrants are formed elementwise in the stored layout (S stored = M_{JI} + M_{LK} for 't','c'), which is correct
//  because op(X+Y)=op(X)+op(Y), so the temporaries keep the flag of their operand and gemm applies op() at the leaves only
template<typename T>
struct s_op
	{
	const T *p; int ld; char t;
	const T *at(const int i, const int j) const {return t=='n' ? p+(size_t)i*ld+j : p+(size_t)j*ld+i;}
	};

static inline bool s_split(const int m, const int n, const int k, const int cutoff) {return m>cutoff && n>cutoff && k>cutoff;}

//workspace needed for all levels below (m,n,k)
static size_t s_work(const int m, const int n, const int k, const int cutoff)
{
if(!s_split(m,n,k,cutoff)) return 0;
size_t mh=m/2, nh=n/2, kh=k/2;
return (mh*kh>mh*nh ? mh*kh : mh*nh) + kh*nh + s_work(mh,nh,kh,cutoff);
}

//C(m x n, ldc) = alpha op(A) op(B)
template<typename T>
static void s_rec(const int m, const int n, const int k, const T alpha, const s_op<T> &A, const s_op<T> &B, T *C, const int ldc, T *ws, const int cutoff)
{
if(!s_split(m,n,k,cutoff)) {s_gemm(A.t,B.t,m,n,k,alpha,A.p,A.ld,B.p,B.ld,(T)0,C,ldc); return;}
const int mh=m/2, nh=n/2, kh=k/2;
//stored dimensions of the quadrants of A and B
const int ar= A.t=='n'?mh:kh, ac= A.t=='n'?kh:mh;
const int br= B.t=='n'?kh:nh, bc= B.t=='n'?nh:kh;
const T *A11=A.at(0,0), *A12=A.at(0,kh), *A21=A.at(mh,0), *A22=A.at(mh,kh);
const T *B11=B.at(0,0), *B12=B.at(0,nh), *B21=B.at(kh,0), *B22=B.at(kh,nh);
T *C11=C, *C12=C+nh, *C21=C+(size_t)mh*ldc, *C22=C21+nh;
const size_t xs= (size_t)mh*kh>(size_t)mh*nh ? (size_t)mh*kh : (size_t)mh*nh;
T *X=ws, *Y=ws+xs, *wn=Y+(size_t)kh*nh;
const s_op<T> sx={X,ac,A.t}, sy={Y,bc,B.t};	//X, Y hold sums of quadrants of A, B in their stored layout, contiguous (ld = stored columns), same flag
const s_op<T> a11={A11,A.ld,A.t}, a12={A12,A.ld,A.t}, a22={A22,A.ld,A.t};
const s_op<T> b11={B11,B.ld,B.t}, b21={B21,B.ld,B.t}, b22={B22,B.ld,B.t};
s_add(ar,ac,X,ac,A11,A.ld,A21,A.ld,true);	//S3=A11-A21
s_add(br,bc,Y,bc,B22,B.ld,B12,B.ld,true);	//T3=B22-B12
s_rec(mh,nh,kh,alpha,sx,sy,C21,ldc,wn,cutoff);	//P7
s_add(ar,ac,X,ac,A21,A.ld,A22,A.ld,false);	//S1=A21+A22
s_add(br,bc,Y,bc,B12,B.ld,B11,B.ld,true);	//T1=B12-B11
s_rec(mh,nh,kh,alpha,sx,sy,C22,ldc,wn,cutoff);	//P5
s_add(ar,ac,X,ac,X,ac,A11,A.ld,true);		//S2=S1-A11
s_add(br,bc,Y,bc,B22,B.ld,Y,bc,true);		//T2=B22-T1
s_rec(mh,nh,kh,alpha,sx,sy,C12,ldc,wn,cutoff);	//P6
s_add(ar,ac,X,ac,A12,A.ld,X,ac,true);		//S4=A12-S2
s_rec(mh,nh,kh,alpha,sx,b22,C11,ldc,wn,cutoff);	//P3
s_rec(mh,nh,kh,alpha,a11,b11,X,nh,wn,cutoff);	//P1 in X
s_add(mh,nh,C12,ldc,X,nh,C12,ldc,false);	//U2=P1+P6
s_add(mh,nh,C21,ldc,C12,ldc,C21,ldc,false);	//U3=U2+P7
s_add(mh,nh,C12,ldc,C12,ldc,C22,ldc,false);	//U4=U2+P5
s_add(mh,nh,C22,ldc,C21,ldc,C22,ldc,false);	//C22=U3+P5
s_add(mh,nh,C12,ldc,C12,ldc,C11,ldc,false);	//C12=U4+P3
s_add(br,bc,Y,bc,Y,bc,B21,B.ld,true);		//T4=T2-B21
s_rec(mh,nh,kh,alpha,a22,sy,C11,ldc,wn,cutoff);	//P4
s_add(mh,nh,C21,ldc,C21,ldc,C11,ldc,true);	//C21=U3-P4
s_rec(mh,nh,kh,alpha,a12,b21,C11,ldc,wn,cutoff);	//P2
s_add(mh,nh,C11,ldc,X,nh,C11,ldc,false);	//C11=P1+P2
//odd dimensions
const int me=2*mh, ne=2*nh, ke=2*kh;
if(k>ke) s_gemm(A.t,B.t,me,ne,1,alpha,A.at(0,ke),A.ld,B.at(ke,0),B.ld,(T)1,C,ldc);
if(n>ne) s_gemm(A.t,B.t,me,1,k,alpha,A.p,A.ld,B.at(0,ne),B.ld,(T)0,C+ne,ldc);
if(m>me) s_gemm(A.t,B.t,1,n,k,alpha,A.at(me,0),A.ld,B.p,B.ld,(T)0,C+(size_t)me*ldc,ldc);
}

//C = beta C + alpha op(A) op(B), row-major m x n
template<typename T>
static void s_top(const int m, const int n, const int k, const T alpha, const T *a, const int lda, const char ta, const T *b, const int ldb, const char tb, const T beta, T *c, const int cutoff)
{
const s_op<T> A={a,lda,ta}, B={b,ldb,tb};
if(!s_split(m,n,k,cutoff)) {s_gemm(ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,n); return;}
const size_t w=s_work(m,n,k,cutoff);
const bool acc= beta!=(T)0;
T *ws=s_alloc<T>(w+(acc?(size_t)m*n:0));
T *cc= acc? ws+w : c;
s_rec(m,n,k,alpha,A,B,cc,n,ws,cutoff);
if(acc) for(size_t i=0; i<(size_t)m*n; ++i) c[i]=beta*c[i]+cc[i];
s_free(ws);
}

static inline char s_flag(const char t, const bool cmplx) {char tt=tolower(t); return tt=='n'?'n':(tt=='c'&&cmplx?'c':'t');}

template<typename T>
static void s_strassen(NRMat<T> &c, const T &beta, const NRMat<T> &a, const char transa, const NRMat<T> &b, const char transb, const T &alpha, const int cutoff)
{
const char ta=s_flag(transa,LA_traits<T>::is_complex()), tb=s_flag(transb,LA_traits<T>::is_complex());
int k(ta=='n'?a.ncols():a.nrows());
#ifdef DEBUG
int l(ta=='n'?a.nrows():a.ncols());
int kk(tb=='n'?b.nrows():b.ncols());
int ll(tb=='n'?b.ncols():b.nrows());
if (l!=c.nrows() || ll!=c.ncols() || k!=kk) laerror("incompatible matrices in NRMat<T>::strassen(...)");
if(b.ncols() <=0 || c.ncols()<=0) laerror("illegal matrix dimension in strassen");
#endif
SAME_LOC3(c, a, b);
NOT_GPU(c);
if(cutoff<1) laerror("illegal cutoff in strassen");
if(alpha==(T)0 && beta==(T)1) return;
c.copyonwrite();
s_top(c.nrows(),c.ncols(),k,alpha,(const T *)a,a.ncols(),ta,(const T *)b,b.ncols(),tb,beta,(T *)c,cutoff);
}

#define INSTANTIZE(T) \
template<> \
void NRMat<T>::strassen(const T &beta, const NRMat<T> &a, const char transa, const NRMat<T> &b, const char transb, const T &alpha, const int cutoff) \
{s_strassen(*this,beta,a,transa,b,transb,alpha,cutoff);}

INSTANTIZE(double)
INSTANTIZE(float)
INSTANTIZE(std::complex<double>)
INSTANTIZE(std::complex<float>)

}//namespace
