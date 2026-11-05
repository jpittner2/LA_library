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

//this can be safely included since it contains ifdefs NONCBLAS and NONCLAPACK anyway
#include "la_traits.h"
#include "noncblas.h"
#include "vec.h"
#include "smat.h"
#include "mat.h"
#include "nonclass.h"
#include "qsort.h"
#include "fortran.h"

//#define IPIV_DEBUG


namespace LA {


#define INSTANTIZE(T) \
template void lawritemat(FILE *file,const T *a,int r,int c,const char *form0, \
		int nodim,int modulo, int issym);
INSTANTIZE(double)
INSTANTIZE(std::complex<double>)
INSTANTIZE(float)
INSTANTIZE(std::complex<float>)
INSTANTIZE(int)
INSTANTIZE(short)
INSTANTIZE(char)
INSTANTIZE(long)
INSTANTIZE(long long)
INSTANTIZE(unsigned char)
INSTANTIZE(unsigned short)
INSTANTIZE(unsigned int)
INSTANTIZE(unsigned long)
INSTANTIZE(unsigned long long)


template <typename T>
void lawritemat(FILE *file,const T *a,int r,int c,const char *form0,
		int nodim,int modulo, int issym)
{
	int i,j;
	const char *f;

	/*print out title before %*/
	f=form0;
	skiptext:
	while (*f && *f !='%' ) {fputc(*f++,file);}
	if (*f=='%' && f[1]=='%') {
		fputc(*f,file); f+=2; 
		goto skiptext;
	}
	/* this has to be avoided when const arguments should be allowed *f=0; */
	/*use the rest as a format for numbers*/

	if (modulo) nodim=0;
	if (nodim==2) fprintf(file,"%d %d\n",r,c);
	if (nodim==1) fprintf(file,"%d\n",c);
	if (modulo) {
		int n1, n2, l, m;
		char ff[32];
		/* prepare integer format for column numbering */
		if (sscanf(f+1,"%d",&l) != 1) l=128/modulo;
		l -= 2;
		m = l/2;
		l = l-m;
		sprintf(ff,"%%%ds%%3d%%%ds", l, m);
		n1 = 1;
		while(n1 <= c) {
			n2=n1+modulo-1;
			if (n2 > c) n2 = c;

			/*write block between columns n1 and n2 */
			fprintf(file,"\n    ");
			for (i=n1; i<=n2; i++) fprintf(file,ff," ",i," ");
			fprintf(file,"\n\n");

			for (i=1; i<=r; i++) {
				fprintf(file, "%3d ", i);
				for (j=n1; j<=n2; j++) {
					if(issym) {
						int ii,jj;
						if (i >= j) {
							ii=i; 
							jj=j;
						} else {
							ii=j; 
							jj=i;
						}
						const std::complex<double> x = (std::complex<double>)a[(ii-1)*ii/2+jj-1]; //packed lower triangle, 1-based ii,jj
						fprintf(file, f, x.real(), i>=j ? x.imag() : -x.imag()); //hermitian for complex
					} else fprintf(file, f, ((std::complex<double>)a[(i-1)*c+j-1]).real(), ((std::complex<double>)a[(i-1)*c+j-1]).imag());
					if (j < n2) fputc(' ',file);
				}
				fprintf(file, "\n");
			}
			n1 = n2+1;
		}
	} else {
		for (i=1; i<=r; i++) {
			for (j=1; j<=c; j++) {
				if (issym) {
					int ii,jj;
					if (i >= j) {
						ii=i; 
						jj=j;
					} else {
						ii=j; 
						jj=i;
					}
					const std::complex<double> x = (std::complex<double>)a[(ii-1)*ii/2+jj-1]; //packed lower triangle, 1-based ii,jj
					fprintf(file, f, x.real(), i>=j ? x.imag() : -x.imag()); //hermitian for complex
				} else fprintf(file,f,((std::complex<double>)a[(i-1)*c+j-1]).real(), ((std::complex<double>)a[(i-1)*c+j-1]).imag());
				putc(j<c?' ':'\n',file);
			}
		}
	}
}

//////////////////////
// LAPACK interface //
//////////////////////

// A will be overwritten, B will contain the solutions, A is nxn, B is rhs x n
static void linear_solve_do(NRMat<double> &A, double *B, const int nrhs, const int ldb, double *det, int n)
{
	int r, *ipiv;
	int iswap=0;

	if(nrhs==0) std::cout<<"Warning: some dgesv implementations might skip LU decomposition when nrhs==0\n";	
	if (n==A.nrows() && A.nrows() != A.ncols()) laerror("linear_solve() call for non-square matrix");
	A.copyonwrite();
	ipiv = new int[A.nrows()];
#ifdef IPIV_DEBUG
	for(int i=0; i<A.nrows(); ++i) ipiv[i]=123456789;
	std::cout <<"canary ipiv initialized\n";
	std::cout <<"A before clapack_dgesv = "<<A<<std::endl;
	std::cout <<"Active dimension n= "<<n<<std::endl;
#endif
	r = clapack_dgesv(CblasRowMajor, n, nrhs, &A(0,0), A.ncols(), ipiv, B , ldb);
#ifdef IPIV_DEBUG
	std::cout <<"A after clapack_dgesv = "<<A<<std::endl;
	std::cout <<"ipiv = ";
	for (int i=0; i<n; ++i) std::cout <<ipiv[i]<<" ";
	std::cout <<std::endl;
#endif
	if (r < 0) {
		delete[] ipiv;
		laerror("illegal argument in lapack_gesv");
	}
	if (det && r==0) {
		*det = 1.;
		//take into account some numerical instabilities in dgesv for singular matrices
		for (int i=0; i<n; ++i)  {double t=A[i][i]; if(!finite(t) || std::abs(t) < LA_precision<double>::det_tiny() ) {*det=0.; break;} else *det *=t;}
		//find out whether ipiv are numbered from 0 or from 1
		int shift=1;
	        for (int i=0; i<n; ++i)
			{
			if(ipiv[i]==0) shift=0;
			if(ipiv[i]<0 || ipiv[i]>n) 
				{
				std::cout <<"IPIV["<<i<<"] = "<<ipiv[i]<<std::endl;
				laerror("problem with ipiv in clapack_dgesv");
				}
			}
#ifdef IPIV_DEBUG
       std::cout <<"shift = "<<shift<<std::endl;
#endif
		//change sign of det by parity of ipiv permutation
		if(*det) for (int i=0; i<n; ++i) if(i+shift != ipiv[i]) {*det = -(*det); ++iswap;}
	}
	if(det && r>0) *det = 0;
#ifdef IPIV_DEBUG
       std::cout <<"iswap = "<<iswap<<std::endl;
#endif

	delete [] ipiv;
	if (r>0 && B) laerror("singular matrix in lapack_gesv");
}

static void linear_solve_do(NRMat<float> &A, float *B, const int nrhs, const int ldb, float *det, int n)
{
	int r, *ipiv;
	int iswap=0;

	if(nrhs==0) std::cout<<"Warning: some dgesv implementations might skip LU decomposition when nrhs==0\n";	
	if (n==A.nrows() && A.nrows() != A.ncols()) laerror("linear_solve() call for non-square matrix");
	A.copyonwrite();
	ipiv = new int[A.nrows()];
#ifdef IPIV_DEBUG
	for(int i=0; i<A.nrows(); ++i) ipiv[i]=123456789;
	std::cout <<"canary ipiv initialized\n";
	std::cout <<"A before clapack_sgesv = "<<A<<std::endl;
	std::cout <<"Active dimension n= "<<n<<std::endl;
#endif
	r = clapack_sgesv(CblasRowMajor, n, nrhs, &A(0,0), A.ncols(), ipiv, B , ldb);
#ifdef IPIV_DEBUG
	std::cout <<"A after clapack_sgesv = "<<A<<std::endl;
	std::cout <<"ipiv = ";
	for (int i=0; i<n; ++i) std::cout <<ipiv[i]<<" ";
	std::cout <<std::endl;
#endif
	if (r < 0) {
		delete[] ipiv;
		laerror("illegal argument in lapack_gesv");
	}
	if (det && r==0) {
		*det = 1.;
		//take into account some numerical instabilities in dgesv for singular matrices
		for (int i=0; i<n; ++i)  {float t=A[i][i]; if(!finite(t) || std::abs(t) < LA_precision<float>::det_tiny() ) {*det=0.; break;} else *det *=t;}
		//find out whether ipiv are numbered from 0 or from 1
		int shift=1;
	        for (int i=0; i<n; ++i)
			{
			if(ipiv[i]==0) shift=0;
			if(ipiv[i]<0 || ipiv[i]>n) 
				{
				std::cout <<"IPIV["<<i<<"] = "<<ipiv[i]<<std::endl;
				laerror("problem with ipiv in clapack_sgesv");
				}
			}
#ifdef IPIV_DEBUG
       std::cout <<"shift = "<<shift<<std::endl;
#endif
		//change sign of det by parity of ipiv permutation
		if(*det) for (int i=0; i<n; ++i) if(i+shift != ipiv[i]) {*det = -(*det); ++iswap;}
	}
	if(det && r>0) *det = 0;
#ifdef IPIV_DEBUG
       std::cout <<"iswap = "<<iswap<<std::endl;
#endif

	delete [] ipiv;
	if (r>0 && B) laerror("singular matrix in lapack_gesv");
}

void linear_solve(NRMat<double> &A, NRMat<double> *B, double *det, int n)
{
if(n<=0) n=A.nrows(); //default - whole matrix
if (n==A.nrows() && B && A.nrows() != B->ncols() || B && n>B->ncols() ||n>A.nrows()) laerror("incompatible matrices in linear_solve()");
if(B) B->copyonwrite();
linear_solve_do(A,B?(*B)[0]:NULL,B?B->nrows() : 0,  B?B->ncols():A.nrows(), det,n);
}

void linear_solve(NRMat<float> &A, NRMat<float> *B, float *det, int n)
{
if(n<=0) n=A.nrows(); //default - whole matrix
if (n==A.nrows() && B && A.nrows() != B->ncols() || B && n>B->ncols() ||n>A.nrows()) laerror("incompatible matrices in linear_solve()");
if(B) B->copyonwrite();
linear_solve_do(A,B?(*B)[0]:NULL,B?B->nrows() : 0,  B?B->ncols():A.nrows(), det,n);
}

void linear_solve(NRMat<double> &A, NRVec<double> &B, double *det, int n)
{
if(n<=0) n=A.nrows(); //default - whole matrix
if(n==A.nrows() && A.nrows() != B.size() || n>B.size()||n>A.nrows() ) laerror("incompatible matrices in linear_solve()");
B.copyonwrite();
linear_solve_do(A,&B[0],1,A.nrows(),det,n);
}

void linear_solve(NRMat<float> &A, NRVec<float> &B, float *det, int n)
{
if(n<=0) n=A.nrows(); //default - whole matrix
if(n==A.nrows() && A.nrows() != B.size() || n>B.size()||n>A.nrows() ) laerror("incompatible matrices in linear_solve()");
B.copyonwrite();
linear_solve_do(A,&B[0],1,A.nrows(),det,n);
}


// Next routines are not available in clapack, fotran ones will be used with an
// additional swap/transpose of outputs when needed

extern "C" void FORNAME(dspsv)(const char *UPLO, const FINT *N, const FINT *NRHS,
		double *AP, FINT *IPIV, double *B, const FINT *LDB, FINT *INFO);
extern "C" void FORNAME(sspsv)(const char *UPLO, const FINT *N, const FINT *NRHS,
		float *AP, FINT *IPIV, float *B, const FINT *LDB, FINT *INFO);

static void linear_solve_do(NRSMat<double> &a, double *b, const int nrhs, const int ldb, double *det, int n)
{
 if(nrhs==0) std::cout<<"Warning: some dspsv implementations might skip LU decomposition when nrhs==0\n";
	FINT r, *ipiv;
	a.copyonwrite();
	ipiv = new FINT[n];
	char U = LAPACK_FORTRANCASE('u');
#ifdef FORINT
        const FINT ntmp=n;
        const FINT nrhstmp=nrhs;
	const FINT ldbtmp=ldb;
	FORNAME(dspsv)(&U, &ntmp, &nrhstmp, a, ipiv, b, &ldbtmp,&r);
#else	
	FORNAME(dspsv)(&U, &n, &nrhs, a, ipiv, b, &ldb,&r);
#endif	
//	std::cout <<"A after dspsv = "<<a<<std::endl;
	if (r < 0) {
		delete[] ipiv;
		laerror("illegal argument in spsv() call of linear_solve()");
	}
	if (det && r == 0) {
		*det = 1.;
		for (int i=0; i<n; i++)
			{
			double t;
			if(ipiv[i]<0) {t = a(i,i)*a(i+1,i+1)-a(i,i+1)*a(i,i+1); ++i;} //2x2 block of D (dsptrf with UPLO='U')
			else t=a(i,i);
			if(!finite(t) || std::abs(t) < LA_precision<double>::det_tiny() ) {*det=0.; break;} else *det *= t;
			}
		//ipiv is needed only for the block structure of D, the permutation matrix occurs twice in the decomposition and signs thus cancel (man dspsv)
	}
        if (det && r>0)  *det = 0;
	delete[] ipiv;
	if (r > 0 && b) laerror("singular matrix in linear_solve(SMat&, Mat*, double*");
}

static void linear_solve_do(NRSMat<float> &a, float *b, const int nrhs, const int ldb, float *det, int n)
{
 if(nrhs==0) std::cout<<"Warning: some dspsv implementations might skip LU decomposition when nrhs==0\n";
	FINT r, *ipiv;
	a.copyonwrite();
	ipiv = new FINT[n];
	char U = LAPACK_FORTRANCASE('u');
#ifdef FORINT
        const FINT ntmp=n;
        const FINT nrhstmp=nrhs;
	const FINT ldbtmp=ldb;
	FORNAME(sspsv)(&U, &ntmp, &nrhstmp, a, ipiv, b, &ldbtmp,&r);
#else	
	FORNAME(sspsv)(&U, &n, &nrhs, a, ipiv, b, &ldb,&r);
#endif	
//	std::cout <<"A after dspsv = "<<a<<std::endl;
	if (r < 0) {
		delete[] ipiv;
		laerror("illegal argument in spsv() call of linear_solve()");
	}
	if (det && r == 0) {
		*det = 1.;
		for (int i=0; i<n; i++)
			{
			float t;
			if(ipiv[i]<0) {t = a(i,i)*a(i+1,i+1)-a(i,i+1)*a(i,i+1); ++i;} //2x2 block of D (dsptrf with UPLO='U')
			else t=a(i,i);
			if(!finite(t) || std::abs(t) < LA_precision<float>::det_tiny() ) {*det=0.; break;} else *det *= t;
			}
		//ipiv is needed only for the block structure of D, the permutation matrix occurs twice in the decomposition and signs thus cancel (man dspsv)
	}
        if (det && r>0)  *det = 0;
	delete[] ipiv;
	if (r > 0 && b) laerror("singular matrix in linear_solve(SMat&, Mat*, float*");
}


void linear_solve(NRSMat<double> &a, NRMat<double> *B, double *det, int n)
{
if(n<=0) n=a.nrows();
	if (n==a.nrows() && B && a.nrows() != B->ncols() || B && n>B->ncols() || n>a.nrows())
		laerror("incompatible matrices in symmetric linear_solve()");
	if (B) B->copyonwrite();
linear_solve_do(a,B?(*B)[0]:NULL,B?B->nrows() : 0, B?B->ncols():a.nrows(),det,n);
}

void linear_solve(NRSMat<float> &a, NRMat<float> *B, float *det, int n)
{
if(n<=0) n=a.nrows();
	if (n==a.nrows() && B && a.nrows() != B->ncols() || B && n>B->ncols() || n>a.nrows())
		laerror("incompatible matrices in symmetric linear_solve()");
	if (B) B->copyonwrite();
linear_solve_do(a,B?(*B)[0]:NULL,B?B->nrows() : 0, B?B->ncols():a.nrows(),det,n);
}


void linear_solve(NRSMat<double> &a, NRVec<double> &B, double *det, int n)
{
if(n<=0) n=a.nrows();
	if (n==a.nrows()  && a.nrows()!= B.size() || n>B.size() || n>a.nrows())
		laerror("incompatible matrices in symmetric linear_solve()");
	B.copyonwrite();
linear_solve_do(a,&B[0],1,a.nrows(),det,n);
}

void linear_solve(NRSMat<float> &a, NRVec<float> &B, float *det, int n)
{
if(n<=0) n=a.nrows();
	if (n==a.nrows()  && a.nrows()!= B.size() || n>B.size() || n>a.nrows())
		laerror("incompatible matrices in symmetric linear_solve()");
	B.copyonwrite();
linear_solve_do(a,&B[0],1,a.nrows(),det,n);
}


// complex hermitian version: lower triangle stored, upper triangle is its complex conjugate
extern "C" void FORNAME(zhpsv)(const char *UPLO, const FINT *N, const FINT *NRHS,
		std::complex<double> *AP, FINT *IPIV, std::complex<double> *B, const FINT *LDB, FINT *INFO);
extern "C" void FORNAME(chpsv)(const char *UPLO, const FINT *N, const FINT *NRHS,
		std::complex<float> *AP, FINT *IPIV, std::complex<float> *B, const FINT *LDB, FINT *INFO);

static void linear_solve_do(NRSMat<std::complex<double> > &a, std::complex<double> *b, const int nrhs, const int ldb, std::complex<double> *det, int n)
{
 if(nrhs==0) std::cout<<"Warning: some zhpsv implementations might skip LU decomposition when nrhs==0\n";
	FINT r, *ipiv;
	a.copyonwrite();
	ipiv = new FINT[n];
	char U = LAPACK_FORTRANCASE('u');
	//the packed C-order lower triangle is the fortran upper triangle of the transposed = complex conjugated hermitian matrix,
	//solve conj(A) conj(x) = conj(b)
	for(int i=0; i<nrhs; ++i) for(int j=0; j<n; ++j) b[i*ldb+j] = std::conj(b[i*ldb+j]);
        const FINT ntmp=n;
        const FINT nrhstmp=nrhs;
	const FINT ldbtmp=ldb;
	FORNAME(zhpsv)(&U, &ntmp, &nrhstmp, a, ipiv, b, &ldbtmp, &r);
	for(int i=0; i<nrhs; ++i) for(int j=0; j<n; ++j) b[i*ldb+j] = std::conj(b[i*ldb+j]);
	if (r < 0) {
		delete[] ipiv;
		laerror("illegal argument in hpsv() call of linear_solve()");
	}
	if (det && r == 0) {
		*det = 1.;
		for (int i=0; i<n; i++)
			{
			double t; //determinant of a hermitian matrix is real, det(conj(A)) = det(A)
			if(ipiv[i]<0) {t = a(i,i).real()*a(i+1,i+1).real()-std::norm(a(i,i+1)); ++i;} //2x2 block of D (zhptrf with UPLO='U')
			else t=a(i,i).real();
			if(!finite(t) || std::abs(t) < LA_precision<double>::det_tiny() ) {*det=0.; break;} else *det *= t;
			}
		//ipiv is needed only for the block structure of D, the permutation matrix occurs twice in the decomposition and signs thus cancel (man zhpsv)
	}
        if (det && r>0)  *det = 0;
	delete[] ipiv;
	if (r > 0 && b) laerror("singular matrix in linear_solve(SMat&, Mat*, complex*");
}

static void linear_solve_do(NRSMat<std::complex<float> > &a, std::complex<float> *b, const int nrhs, const int ldb, std::complex<float> *det, int n)
{
 if(nrhs==0) std::cout<<"Warning: some zhpsv implementations might skip LU decomposition when nrhs==0\n";
	FINT r, *ipiv;
	a.copyonwrite();
	ipiv = new FINT[n];
	char U = LAPACK_FORTRANCASE('u');
	//the packed C-order lower triangle is the fortran upper triangle of the transposed = complex conjugated hermitian matrix,
	//solve conj(A) conj(x) = conj(b)
	for(int i=0; i<nrhs; ++i) for(int j=0; j<n; ++j) b[i*ldb+j] = std::conj(b[i*ldb+j]);
        const FINT ntmp=n;
        const FINT nrhstmp=nrhs;
	const FINT ldbtmp=ldb;
	FORNAME(chpsv)(&U, &ntmp, &nrhstmp, a, ipiv, b, &ldbtmp, &r);
	for(int i=0; i<nrhs; ++i) for(int j=0; j<n; ++j) b[i*ldb+j] = std::conj(b[i*ldb+j]);
	if (r < 0) {
		delete[] ipiv;
		laerror("illegal argument in hpsv() call of linear_solve()");
	}
	if (det && r == 0) {
		*det = 1.;
		for (int i=0; i<n; i++)
			{
			float t; //determinant of a hermitian matrix is real, det(conj(A)) = det(A)
			if(ipiv[i]<0) {t = a(i,i).real()*a(i+1,i+1).real()-std::norm(a(i,i+1)); ++i;} //2x2 block of D (zhptrf with UPLO='U')
			else t=a(i,i).real();
			if(!finite(t) || std::abs(t) < LA_precision<float>::det_tiny() ) {*det=0.; break;} else *det *= t;
			}
		//ipiv is needed only for the block structure of D, the permutation matrix occurs twice in the decomposition and signs thus cancel (man zhpsv)
	}
        if (det && r>0)  *det = 0;
	delete[] ipiv;
	if (r > 0 && b) laerror("singular matrix in linear_solve(SMat&, Mat*, complex*");
}


void linear_solve(NRSMat<std::complex<double> > &a, NRMat<std::complex<double> > *B, std::complex<double> *det, int n)
{
if(n<=0) n=a.nrows();
	if (n==a.nrows() && B && a.nrows() != B->ncols() || B && n>B->ncols() || n>a.nrows())
		laerror("incompatible matrices in hermitian linear_solve()");
	if (B) B->copyonwrite();
linear_solve_do(a,B?(*B)[0]:NULL,B?B->nrows() : 0, B?B->ncols():a.nrows(),det,n);
}

void linear_solve(NRSMat<std::complex<float> > &a, NRMat<std::complex<float> > *B, std::complex<float> *det, int n)
{
if(n<=0) n=a.nrows();
	if (n==a.nrows() && B && a.nrows() != B->ncols() || B && n>B->ncols() || n>a.nrows())
		laerror("incompatible matrices in hermitian linear_solve()");
	if (B) B->copyonwrite();
linear_solve_do(a,B?(*B)[0]:NULL,B?B->nrows() : 0, B?B->ncols():a.nrows(),det,n);
}


void linear_solve(NRSMat<std::complex<double> > &a, NRVec<std::complex<double> > &B, std::complex<double> *det, int n)
{
if(n<=0) n=a.nrows();
	if (n==a.nrows()  && a.nrows()!= B.size() || n>B.size() || n>a.nrows())
		laerror("incompatible matrices in hermitian linear_solve()");
	B.copyonwrite();
linear_solve_do(a,&B[0],1,a.nrows(),det,n);
}

void linear_solve(NRSMat<std::complex<float> > &a, NRVec<std::complex<float> > &B, std::complex<float> *det, int n)
{
if(n<=0) n=a.nrows();
	if (n==a.nrows()  && a.nrows()!= B.size() || n>B.size() || n>a.nrows())
		laerror("incompatible matrices in hermitian linear_solve()");
	B.copyonwrite();
linear_solve_do(a,&B[0],1,a.nrows(),det,n);
}


// Roman, complex version of linear_solve()
extern "C" void FORNAME(zgesv)(const FINT *N, const FINT *NRHS, std::complex<double> *A, const FINT *LDA,
           FINT *IPIV, std::complex<double> *B, const FINT *LDB, FINT *INFO);
extern "C" void FORNAME(cgesv)(const FINT *N, const FINT *NRHS, std::complex<float> *A, const FINT *LDA,
           FINT *IPIV, std::complex<float> *B, const FINT *LDB, FINT *INFO);

// A will be overwritten, B will contain the solutions, A is nxn, B is rhs x n
// Fortran zgesv sees the row-major A transposed, so A is transposed in place before and after, like in clapack_dgesv in noncblas.cc
static void linear_solve_do(NRMat<std::complex<double> > &A, std::complex<double> *B, const int nrhs, const int ldb, std::complex<double> *det, int n)
{
	FINT r, *ipiv;
	int iswap=0;

	if(nrhs==0) std::cout<<"Warning: some zgesv implementations might skip LU decomposition when nrhs==0\n";
	if (n==A.nrows() && A.nrows() != A.ncols()) laerror("linear_solve() call for non-square matrix");
	A.copyonwrite();
	ipiv = new FINT[n];
	const int lda=A.ncols();
	std::complex<double> *a= &A(0,0);
	for(int i=1; i<n; ++i) for(int j=0; j<i; ++j)  {std::complex<double> t=a[j*lda+i]; a[j*lda+i]=a[i*lda+j]; a[i*lda+j]=t;}
	const FINT ntmp=n;
	const FINT nrhstmp=nrhs;
	const FINT ldatmp=lda;
	const FINT ldbtmp=ldb;
	FORNAME(zgesv)(&ntmp, &nrhstmp, a, &ldatmp, ipiv, B, &ldbtmp, &r);
	for(int i=1; i<n; ++i) for(int j=0; j<i; ++j)  {std::complex<double> t=a[j*lda+i]; a[j*lda+i]=a[i*lda+j]; a[i*lda+j]=t;}
	if (r < 0) {
		delete[] ipiv;
		laerror("illegal argument in lapack_gesv");
	}
	if (det && r==0) {
		*det = 1.;
		//take into account some numerical instabilities in zgesv for singular matrices
		for (int i=0; i<n; ++i)  {std::complex<double> t=A[i][i]; if(!std::isfinite(t.real()) || !std::isfinite(t.imag()) || std::abs(t) < LA_precision<double>::det_tiny() ) {*det=0.; break;} else *det *=t;}
		//find out whether ipiv are numbered from 0 or from 1
		int shift=1;
	        for (int i=0; i<n; ++i)
			{
			if(ipiv[i]==0) shift=0;
			if(ipiv[i]<0 || ipiv[i]>n) 
				{
				std::cout <<"IPIV["<<i<<"] = "<<ipiv[i]<<std::endl;
				laerror("problem with ipiv in zgesv");
				}
			}
#ifdef IPIV_DEBUG
       std::cout <<"shift = "<<shift<<std::endl;
#endif
		//change sign of det by parity of ipiv permutation
		if(*det!=0.) for (int i=0; i<n; ++i) if(i+shift != ipiv[i]) {*det = -(*det); ++iswap;}
	}
	if(det && r>0) *det = 0;
#ifdef IPIV_DEBUG
       std::cout <<"iswap = "<<iswap<<std::endl;
#endif

	delete [] ipiv;
	if (r>0 && B) laerror("singular matrix in zgesv");
}

static void linear_solve_do(NRMat<std::complex<float> > &A, std::complex<float> *B, const int nrhs, const int ldb, std::complex<float> *det, int n)
{
	FINT r, *ipiv;
	int iswap=0;

	if(nrhs==0) std::cout<<"Warning: some zgesv implementations might skip LU decomposition when nrhs==0\n";
	if (n==A.nrows() && A.nrows() != A.ncols()) laerror("linear_solve() call for non-square matrix");
	A.copyonwrite();
	ipiv = new FINT[n];
	const int lda=A.ncols();
	std::complex<float> *a= &A(0,0);
	for(int i=1; i<n; ++i) for(int j=0; j<i; ++j)  {std::complex<float> t=a[j*lda+i]; a[j*lda+i]=a[i*lda+j]; a[i*lda+j]=t;}
	const FINT ntmp=n;
	const FINT nrhstmp=nrhs;
	const FINT ldatmp=lda;
	const FINT ldbtmp=ldb;
	FORNAME(cgesv)(&ntmp, &nrhstmp, a, &ldatmp, ipiv, B, &ldbtmp, &r);
	for(int i=1; i<n; ++i) for(int j=0; j<i; ++j)  {std::complex<float> t=a[j*lda+i]; a[j*lda+i]=a[i*lda+j]; a[i*lda+j]=t;}
	if (r < 0) {
		delete[] ipiv;
		laerror("illegal argument in lapack_gesv");
	}
	if (det && r==0) {
		*det = 1.;
		//take into account some numerical instabilities in zgesv for singular matrices
		for (int i=0; i<n; ++i)  {std::complex<float> t=A[i][i]; if(!std::isfinite(t.real()) || !std::isfinite(t.imag()) || std::abs(t) < LA_precision<float>::det_tiny() ) {*det=0.; break;} else *det *=t;}
		//find out whether ipiv are numbered from 0 or from 1
		int shift=1;
	        for (int i=0; i<n; ++i)
			{
			if(ipiv[i]==0) shift=0;
			if(ipiv[i]<0 || ipiv[i]>n) 
				{
				std::cout <<"IPIV["<<i<<"] = "<<ipiv[i]<<std::endl;
				laerror("problem with ipiv in zgesv");
				}
			}
#ifdef IPIV_DEBUG
       std::cout <<"shift = "<<shift<<std::endl;
#endif
		//change sign of det by parity of ipiv permutation
		if(*det!=0.f) for (int i=0; i<n; ++i) if(i+shift != ipiv[i]) {*det = -(*det); ++iswap;}
	}
	if(det && r>0) *det = 0;
#ifdef IPIV_DEBUG
       std::cout <<"iswap = "<<iswap<<std::endl;
#endif

	delete [] ipiv;
	if (r>0 && B) laerror("singular matrix in zgesv");
}

void linear_solve(NRMat< std::complex<double> > &A, NRMat< std::complex<double> > *B, std::complex<double> *det, int n)
{
if(n<=0) n=A.nrows(); //default - whole matrix
if (n==A.nrows() && B && A.nrows() != B->ncols() || B && n>B->ncols() ||n>A.nrows()) laerror("incompatible matrices in linear_solve()");
if(B) B->copyonwrite();
linear_solve_do(A,B?(*B)[0]:NULL,B?B->nrows() : 0,  B?B->ncols():A.nrows(), det,n);
}

void linear_solve(NRMat< std::complex<float> > &A, NRMat< std::complex<float> > *B, std::complex<float> *det, int n)
{
if(n<=0) n=A.nrows(); //default - whole matrix
if (n==A.nrows() && B && A.nrows() != B->ncols() || B && n>B->ncols() ||n>A.nrows()) laerror("incompatible matrices in linear_solve()");
if(B) B->copyonwrite();
linear_solve_do(A,B?(*B)[0]:NULL,B?B->nrows() : 0,  B?B->ncols():A.nrows(), det,n);
}

void linear_solve(NRMat< std::complex<double> > &A, NRVec< std::complex<double> > &B, std::complex<double> *det, int n)
{
if(n<=0) n=A.nrows(); //default - whole matrix
if(n==A.nrows() && A.nrows() != B.size() || n>B.size()||n>A.nrows() ) laerror("incompatible matrices in linear_solve()");
B.copyonwrite();
linear_solve_do(A,&B[0],1,A.nrows(),det,n);
}

void linear_solve(NRMat< std::complex<float> > &A, NRVec< std::complex<float> > &B, std::complex<float> *det, int n)
{
if(n<=0) n=A.nrows(); //default - whole matrix
if(n==A.nrows() && A.nrows() != B.size() || n>B.size()||n>A.nrows() ) laerror("incompatible matrices in linear_solve()");
B.copyonwrite();
linear_solve_do(A,&B[0],1,A.nrows(),det,n);
}



//other version of linear solver based on gesvx

//------------------------------------------------------------------------------
extern "C" void FORNAME(zgesvx)(const char *fact, const char *trans, const FINT *n, const FINT *nrhs, std::complex<double> *A,  const FINT *lda, std::complex<double> *AF, const FINT *ldaf, const FINT *ipiv, char *equed, double *R,double *C, std::complex<double> *B, const FINT *ldb, std::complex<double> *X, const FINT *ldx, double *rcond, double *ferr, double *berr, std::complex<double> *work, double *rwork, FINT *info);
extern "C" void FORNAME(cgesvx)(const char *fact, const char *trans, const FINT *n, const FINT *nrhs, std::complex<float> *A,  const FINT *lda, std::complex<float> *AF, const FINT *ldaf, const FINT *ipiv, char *equed, float *R,float *C, std::complex<float> *B, const FINT *ldb, std::complex<float> *X, const FINT *ldx, float *rcond, float *ferr, float *berr, std::complex<float> *work, float *rwork, FINT *info);
extern "C" void FORNAME(dgesvx)(const char *fact, const char *trans, const FINT *n, const FINT *nrhs, double *A,  const FINT *lda, double *AF, const FINT *ldaf, const FINT *ipiv, char *equed, double *R,double *C, double *B, const FINT *ldb, double *X, const FINT *ldx, double *rcond, double *ferr, double *berr, double *work, FINT *iwork, FINT *info);
extern "C" void FORNAME(sgesvx)(const char *fact, const char *trans, const FINT *n, const FINT *nrhs, float *A,  const FINT *lda, float *AF, const FINT *ldaf, const FINT *ipiv, char *equed, float *R,float *C, float *B, const FINT *ldb, float *X, const FINT *ldx, float *rcond, float *ferr, float *berr, float *work, FINT *iwork, FINT *info);
//------------------------------------------------------------------------------
// solves set of linear equations using dgesvx
// input:
//	_A		double precision matrix of dimension nn x mm, where min(nn, mm) >= n
//	_B		double prec. array dimensioned as nrhs x n
//	_rhsCount	nrhs - count of right hand sides
//	_eqCount	n - count of equations
//	_eq		use equilibration of matrix A before solving
//	_saveA		if set, do no overwrite A if equilibration in effect
//	_rcond		if not NULL, store the returned rcond value from dgesvx
// output:
//	solution is stored in _B
//	the info parameter of dgesvx is returned (see man dgesvx)
//------------------------------------------------------------------------------
template<>
int linear_solve_x<double>(NRMat<double> &_A, double *_B, const int _rhsCount, const int _eqCount, const bool _eq, const bool _saveA, double *_rcond){
	const int A_rows = _A.nrows();
	const int A_cols = _A.ncols();

	const char fact  = LAPACK_FORTRANCASE(_eq?'E':'N');
	const char trans = LAPACK_FORTRANCASE('T');//because of c-order
	char equed = LAPACK_FORTRANCASE('B');//if fact=='N' then equed is an output argument, therefore not declared as const

	if(_eqCount < 0 || _eqCount > A_rows || _eqCount > A_cols || _rhsCount < 0){
		laerror("linear_solve_x: invalid input matrices");
	}

	double *A;
	if(_eq && !_saveA) _A.copyonwrite(); //A will be overwritten by the equilibrated matrix, detach before taking the pointer
	double * const _A_data = (double*)_A;

	FINT info;
	const FINT nrhs	= _rhsCount;
	const FINT n	= _eqCount;
	FINT lda	= A_cols;
	const FINT ldaf	= lda;

	double rcond;
	double ferr[nrhs], berr[nrhs], work[4*n];
	double R[n], C[n];

	FINT *const iwork = new FINT[n];
	FINT *const ipiv  = new FINT[n];

	double *X  = new double[n*nrhs];
	double *AF = new double[ldaf*n];

	A = _A_data;
	if(_eq){
		if(_saveA){//store the corresponding submatrix of _A (not needed provided fact=='N')
			A = new double[n*n];
			int offset1 = 0;int offset2 = 0;
			for(register int i=0;i<n;i++){
				cblas_dcopy(n, _A_data + offset1, 1, A + offset2, 1);
				offset1 += A_cols;
				offset2 += n;
			}
			lda = n;//!!!
		}
	}

	FORNAME(dgesvx)(&fact, &trans, &n, &nrhs, A, &lda, AF, &ldaf, &ipiv[0], &equed, &R[0], &C[0], _B, &n, X, &n, &rcond, ferr, berr, work, iwork, &info);
	
	if(_rcond)*_rcond = rcond;
	cblas_dcopy(n*nrhs, X, 1, _B, 1);//store the solution

	delete[] iwork;delete[] ipiv;
	delete[] AF;delete[] X;
	if(_eq && _saveA){//the copy is made only with equilibration
		delete[] A;
	}
	return (int)info;
}

template<>
int linear_solve_x<float>(NRMat<float> &_A, float *_B, const int _rhsCount, const int _eqCount, const bool _eq, const bool _saveA, float *_rcond){
	const int A_rows = _A.nrows();
	const int A_cols = _A.ncols();

	const char fact  = LAPACK_FORTRANCASE(_eq?'E':'N');
	const char trans = LAPACK_FORTRANCASE('T');//because of c-order
	char equed = LAPACK_FORTRANCASE('B');//if fact=='N' then equed is an output argument, therefore not declared as const

	if(_eqCount < 0 || _eqCount > A_rows || _eqCount > A_cols || _rhsCount < 0){
		laerror("linear_solve_x: invalid input matrices");
	}

	float *A;
	if(_eq && !_saveA) _A.copyonwrite(); //A will be overwritten by the equilibrated matrix, detach before taking the pointer
	float * const _A_data = (float*)_A;

	FINT info;
	const FINT nrhs	= _rhsCount;
	const FINT n	= _eqCount;
	FINT lda	= A_cols;
	const FINT ldaf	= lda;

	float rcond;
	float ferr[nrhs], berr[nrhs], work[4*n];
	float R[n], C[n];

	FINT *const iwork = new FINT[n];
	FINT *const ipiv  = new FINT[n];

	float *X  = new float[n*nrhs];
	float *AF = new float[ldaf*n];

	A = _A_data;
	if(_eq){
		if(_saveA){//store the corresponding submatrix of _A (not needed provided fact=='N')
			A = new float[n*n];
			int offset1 = 0;int offset2 = 0;
			for(register int i=0;i<n;i++){
				cblas_scopy(n, _A_data + offset1, 1, A + offset2, 1);
				offset1 += A_cols;
				offset2 += n;
			}
			lda = n;//!!!
		}
	}

	FORNAME(sgesvx)(&fact, &trans, &n, &nrhs, A, &lda, AF, &ldaf, &ipiv[0], &equed, &R[0], &C[0], _B, &n, X, &n, &rcond, ferr, berr, work, iwork, &info);
	
	if(_rcond)*_rcond = rcond;
	cblas_scopy(n*nrhs, X, 1, _B, 1);//store the solution

	delete[] iwork;delete[] ipiv;
	delete[] AF;delete[] X;
	if(_eq && _saveA){//the copy is made only with equilibration
		delete[] A;
	}
	return (int)info;
}
//------------------------------------------------------------------------------
// solves set of linear equations using zgesvx
// input:
//	_A		double precision complex matrix of dimension nn x mm, where min(nn, mm) >= n
//	_B		double prec. complex array dimensioned as nrhs x n
//	_rhsCount	nrhs - count of right hand sides
//	_eqCount	n - count of equations
//	_eq		use equilibration
//	_saveA		if set, do no overwrite A if equilibration in effect
//	_rcond		if not NULL, store the returned rcond value from dgesvx
// output:
//	solution is stored in _B
//	the info parameter of dgesvx is returned (see man dgesvx)
//------------------------------------------------------------------------------
template<>
int linear_solve_x<std::complex<double> >(NRMat<std::complex<double> > &_A, std::complex<double> *_B, const int _rhsCount, const int _eqCount, const bool _eq, const bool _saveA, double *_rcond){
	const int A_rows = _A.nrows();
	const int A_cols = _A.ncols();

	const char fact  = LAPACK_FORTRANCASE(_eq?'E':'N');
	const char trans = LAPACK_FORTRANCASE('T');//because of c-order
	char equed = LAPACK_FORTRANCASE('B');//if fact=='N' then equed is an output argument, therefore not declared as const

	if(_eqCount < 0 || _eqCount > A_rows || _eqCount > A_cols || _rhsCount < 0){
		laerror("linear_solve_x: invalid input matrices");
	}

	std::complex<double> *A;
	if(_eq && !_saveA) _A.copyonwrite(); //A will be overwritten by the equilibrated matrix, detach before taking the pointer
	std::complex<double> * const _A_data = (std::complex<double>*)_A;

	FINT info;
	const FINT nrhs	= _rhsCount;
	const FINT n	= _eqCount;
	FINT lda	= A_cols;
	const FINT ldaf	= lda;

	double rcond;
	double ferr[nrhs], berr[nrhs];
	double R[n], C[n], rwork[2*n];
	std::complex<double> work[2*n];

	FINT *const ipiv = new FINT[n];

	std::complex<double> *X  = new std::complex<double>[n*nrhs];
	std::complex<double> *AF = new std::complex<double>[ldaf*n];

	A = _A_data;
	if(_eq){
		if(_saveA){//store the corresponding submatrix of _A (not needed provided fact=='N')
			A = new std::complex<double>[n*n];
			int offset1 = 0;int offset2 = 0;
			for(register int i=0;i<n;i++){
				cblas_zcopy(n, _A_data + offset1, 1, A + offset2, 1);
				offset1 += A_cols;
				offset2 += n;
			}
			lda = n;//!!!
		}
	}

	FORNAME(zgesvx)(&fact, &trans, &n, &nrhs, A, &lda, AF, &ldaf, &ipiv[0], &equed, &R[0], &C[0], _B, &n, X, &n, &rcond, ferr, berr, work, rwork, &info);


	if(_rcond)*_rcond = rcond;
	cblas_zcopy(n*nrhs, X, 1, _B, 1);//store the solution

	delete[] ipiv;
	delete[] AF;delete[] X;
	if(_eq && _saveA){//the copy is made only with equilibration
		delete[] A;
	}
	return (int)info;
}

template<>
int linear_solve_x<std::complex<float> >(NRMat<std::complex<float> > &_A, std::complex<float> *_B, const int _rhsCount, const int _eqCount, const bool _eq, const bool _saveA, float *_rcond){
	const int A_rows = _A.nrows();
	const int A_cols = _A.ncols();

	const char fact  = LAPACK_FORTRANCASE(_eq?'E':'N');
	const char trans = LAPACK_FORTRANCASE('T');//because of c-order
	char equed = LAPACK_FORTRANCASE('B');//if fact=='N' then equed is an output argument, therefore not declared as const

	if(_eqCount < 0 || _eqCount > A_rows || _eqCount > A_cols || _rhsCount < 0){
		laerror("linear_solve_x: invalid input matrices");
	}

	std::complex<float> *A;
	if(_eq && !_saveA) _A.copyonwrite(); //A will be overwritten by the equilibrated matrix, detach before taking the pointer
	std::complex<float> * const _A_data = (std::complex<float>*)_A;

	FINT info;
	const FINT nrhs	= _rhsCount;
	const FINT n	= _eqCount;
	FINT lda	= A_cols;
	const FINT ldaf	= lda;

	float rcond;
	float ferr[nrhs], berr[nrhs];
	float R[n], C[n], rwork[2*n];
	std::complex<float> work[2*n];

	FINT *const ipiv = new FINT[n];

	std::complex<float> *X  = new std::complex<float>[n*nrhs];
	std::complex<float> *AF = new std::complex<float>[ldaf*n];

	A = _A_data;
	if(_eq){
		if(_saveA){//store the corresponding submatrix of _A (not needed provided fact=='N')
			A = new std::complex<float>[n*n];
			int offset1 = 0;int offset2 = 0;
			for(register int i=0;i<n;i++){
				cblas_ccopy(n, _A_data + offset1, 1, A + offset2, 1);
				offset1 += A_cols;
				offset2 += n;
			}
			lda = n;//!!!
		}
	}

	FORNAME(cgesvx)(&fact, &trans, &n, &nrhs, A, &lda, AF, &ldaf, &ipiv[0], &equed, &R[0], &C[0], _B, &n, X, &n, &rcond, ferr, berr, work, rwork, &info);


	if(_rcond)*_rcond = rcond;
	cblas_ccopy(n*nrhs, X, 1, _B, 1);//store the solution

	delete[] ipiv;
	delete[] AF;delete[] X;
	if(_eq && _saveA){//the copy is made only with equilibration
		delete[] A;
	}
	return (int)info;
}
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
template<>
int multiply_by_inverse<double>(NRMat<double> &_A, NRMat<double> &_B, bool _useEq, double *_rcond){
	
	const FINT n = _A.nrows();
	const FINT m = _A.ncols();
	if(n != m || n != _B.nrows() || n != _B.ncols()){
		laerror("multiply_by_inverse: incompatible matrices");
	}
	
	const char fact  = _useEq?'E':'N';
	const char trans = 'N';//because of c-order
	char equed = 'B';//if fact=='N' then equed is an output argument, therefore not declared as const
	const int n2 = n*n;
	
	_B.copyonwrite();//even if fact='N', call copyonwrite because the solution is going to be stored in _B; before taking the pointer
	double * const B = (double*)_B;
	//_A is the right hand side for gesvx, which scales it if equilibrated: work on a copy then, _A is input only
	double * const A = _useEq ? new double[n2] : (double*)_A;
	if(_useEq) memcpy(A,(double*)_A,n2*sizeof(double));

	FINT info;
	double rcond;
	double ferr[n], berr[n], work[4*n];
	double R[n], C[n];

	FINT *const iwork = new FINT[n];
	FINT *const ipiv  = new FINT[n];

	double *X  = new double[n2];
	double *AF = new double[n2];

	FORNAME(dgesvx)(&fact, &trans, &n, &n, B, &n, AF, &n, &ipiv[0], &equed, &R[0], &C[0], A, &n, X, &n, &rcond, ferr, berr, work, iwork, &info);


	if(_rcond)*_rcond = rcond;
	cblas_dcopy(n2, X, 1, B, 1);//store the solution

	delete[] iwork;delete[] ipiv;
	delete[] AF;delete[] X;
	if(_useEq) delete[] A;
	
	return (int)info;
}

template<>
int multiply_by_inverse<float>(NRMat<float> &_A, NRMat<float> &_B, bool _useEq, float *_rcond){
	
	const FINT n = _A.nrows();
	const FINT m = _A.ncols();
	if(n != m || n != _B.nrows() || n != _B.ncols()){
		laerror("multiply_by_inverse: incompatible matrices");
	}
	
	const char fact  = _useEq?'E':'N';
	const char trans = 'N';//because of c-order
	char equed = 'B';//if fact=='N' then equed is an output argument, therefore not declared as const
	const int n2 = n*n;
	
	_B.copyonwrite();//even if fact='N', call copyonwrite because the solution is going to be stored in _B; before taking the pointer
	float * const B = (float*)_B;
	//_A is the right hand side for gesvx, which scales it if equilibrated: work on a copy then, _A is input only
	float * const A = _useEq ? new float[n2] : (float*)_A;
	if(_useEq) memcpy(A,(float*)_A,n2*sizeof(float));

	FINT info;
	float rcond;
	float ferr[n], berr[n], work[4*n];
	float R[n], C[n];

	FINT *const iwork = new FINT[n];
	FINT *const ipiv  = new FINT[n];

	float *X  = new float[n2];
	float *AF = new float[n2];

	FORNAME(sgesvx)(&fact, &trans, &n, &n, B, &n, AF, &n, &ipiv[0], &equed, &R[0], &C[0], A, &n, X, &n, &rcond, ferr, berr, work, iwork, &info);


	if(_rcond)*_rcond = rcond;
	cblas_scopy(n2, X, 1, B, 1);//store the solution

	delete[] iwork;delete[] ipiv;
	delete[] AF;delete[] X;
	if(_useEq) delete[] A;
	
	return (int)info;
}
//------------------------------------------------------------------------------
// for given square matrices A, B computes X = AB^{-1} as follows
// 	XB = A => B^TX^T = A^T
// input:
//	_A		double precision matrix of dimension nn x nn
//	_B		double prec. matrix of dimension nn x nn
//	_useEq		use equilibration suitable for badly conditioned matrices
//	_rcond		if not NULL, store the returned value of rcond fromd zgesvx
// output:
//	solution is stored in _B
//	the info parameter of zgesvx is returned (see man zgesvx)
//------------------------------------------------------------------------------
template<>
int multiply_by_inverse<std::complex<double> >(NRMat<std::complex<double> > &_A, NRMat<std::complex<double> > &_B, bool _useEq, double *_rcond){
	
	const FINT n = _A.nrows();
	const FINT m = _A.ncols();
	if(n != m || n != _B.nrows() || n != _B.ncols()){
		laerror("multiply_by_inverse: incompatible matrices");
	}
	const int n2 = n*n;
	
	const char fact  = _useEq?'E':'N';
	const char trans = 'N';//because of c-order
	char equed = 'B';//if fact=='N' then equed is an output argument, therefore not declared as const
	
	_B.copyonwrite();//even if fact='N', call copyonwrite because the solution is going to be stored in _B; before taking the pointer
	std::complex<double> * const B = (std::complex<double>*)_B;
	//_A is the right hand side for gesvx, which scales it if equilibrated: work on a copy then, _A is input only
	std::complex<double> * const A = _useEq ? new std::complex<double>[n2] : (std::complex<double>*)_A;
	if(_useEq) memcpy(A,(std::complex<double>*)_A,n2*sizeof(std::complex<double>));

	FINT info;
	double rcond;
	double ferr[n], berr[n];
	double R[n], C[n], rwork[2*n];
	std::complex<double> work[2*n];

	FINT *const ipiv = new FINT[n];

	std::complex<double> *X  = new std::complex<double>[n2];
	std::complex<double> *AF = new std::complex<double>[n2];

	FORNAME(zgesvx)(&fact, &trans, &n, &n, B, &n, AF, &n, &ipiv[0], &equed, &R[0], &C[0], A, &n, X, &n, &rcond, ferr, berr, work, rwork, &info);


	if(_rcond)*_rcond = rcond;
	cblas_zcopy(n2, X, 1, B, 1);//store the solution

	delete[] ipiv;
	delete[] AF;delete[] X;
	if(_useEq) delete[] A;
	
	return (int)info;
}

template<>
int multiply_by_inverse<std::complex<float> >(NRMat<std::complex<float> > &_A, NRMat<std::complex<float> > &_B, bool _useEq, float *_rcond){
	
	const FINT n = _A.nrows();
	const FINT m = _A.ncols();
	if(n != m || n != _B.nrows() || n != _B.ncols()){
		laerror("multiply_by_inverse: incompatible matrices");
	}
	const int n2 = n*n;
	
	const char fact  = _useEq?'E':'N';
	const char trans = 'N';//because of c-order
	char equed = 'B';//if fact=='N' then equed is an output argument, therefore not declared as const
	
	_B.copyonwrite();//even if fact='N', call copyonwrite because the solution is going to be stored in _B; before taking the pointer
	std::complex<float> * const B = (std::complex<float>*)_B;
	//_A is the right hand side for gesvx, which scales it if equilibrated: work on a copy then, _A is input only
	std::complex<float> * const A = _useEq ? new std::complex<float>[n2] : (std::complex<float>*)_A;
	if(_useEq) memcpy(A,(std::complex<float>*)_A,n2*sizeof(std::complex<float>));

	FINT info;
	float rcond;
	float ferr[n], berr[n];
	float R[n], C[n], rwork[2*n];
	std::complex<float> work[2*n];

	FINT *const ipiv = new FINT[n];

	std::complex<float> *X  = new std::complex<float>[n2];
	std::complex<float> *AF = new std::complex<float>[n2];

	FORNAME(cgesvx)(&fact, &trans, &n, &n, B, &n, AF, &n, &ipiv[0], &equed, &R[0], &C[0], A, &n, X, &n, &rcond, ferr, berr, work, rwork, &info);


	if(_rcond)*_rcond = rcond;
	cblas_ccopy(n2, X, 1, B, 1);//store the solution

	delete[] ipiv;
	delete[] AF;delete[] X;
	if(_useEq) delete[] A;
	
	return (int)info;
}
//------------------------------------------------------------------------------



extern "C" void FORNAME(dsyev)(const char *JOBZ, const char *UPLO, const FINT *N,
		double *A, const FINT *LDA, double *W, double *WORK, const FINT *LWORK, FINT *INFO);
extern "C" void FORNAME(ssyev)(const char *JOBZ, const char *UPLO, const FINT *N,
		float *A, const FINT *LDA, float *W, float *WORK, const FINT *LWORK, FINT *INFO);

extern "C" void FORNAME(dsygv)(const FINT *ITYPE, const char *JOBZ, const char *UPLO, const FINT *N,
                double *A, const FINT *LDA, double *B, const FINT *LDB, double *W, double *WORK, const FINT *LWORK, FINT *INFO);
extern "C" void FORNAME(ssygv)(const FINT *ITYPE, const char *JOBZ, const char *UPLO, const FINT *N,
                float *A, const FINT *LDA, float *B, const FINT *LDB, float *W, float *WORK, const FINT *LWORK, FINT *INFO);


// a will contain eigenvectors (columns if corder==1), w eigenvalues
void diagonalize(NRMat<double> &a, NRVec<double> &w, const bool eivec, 
		const bool corder, int n, NRMat<double> *b, const int itype)
{
	FINT m = a.nrows();
	if (m != a.ncols()) laerror("diagonalize() call with non-square matrix");
	if (a.nrows() != w.size()) 
		laerror("inconsistent dimension of eigenvalue vector in diagonalize()");
	if(n==0) n=m;
	if(n<0||n>m) laerror("actual dimension out of range in diagonalize");
	if(b) if(n>b->nrows() || n> b->ncols()) laerror("wrong B matrix dimension in diagonalize");

	a.copyonwrite();
	w.copyonwrite();
	if(b) b->copyonwrite();

	FINT r = 0;
	char U =LAPACK_FORTRANCASE('u');
	char vectors = LAPACK_FORTRANCASE('v');
	if (!eivec) vectors = LAPACK_FORTRANCASE('n');
	FINT LWORK = -1;
	double WORKX;
	FINT ldb=0; if(b) ldb=b->ncols();

#ifdef FORINT
        const FINT itypetmp = itype;
	FINT ntmp = n;
	// First call is to determine size of workspace
	if(b) FORNAME(dsygv)(&itypetmp,&vectors, &U, &ntmp, a, &m, *b, &ldb, w, &WORKX, &LWORK, &r );
	else FORNAME(dsyev)(&vectors, &U, &ntmp, a, &m, w, &WORKX, &LWORK, &r );
#else
	// First call is to determine size of workspace
	if(b) FORNAME(dsygv)(&itype,&vectors, &U, &n, a, &m, *b, &ldb, w, &WORKX, &LWORK, &r );
	else FORNAME(dsyev)(&vectors, &U, &n, a, &m, w, &WORKX, &LWORK, &r );
#endif

        LWORK = (FINT)WORKX;
	double *WORK = new double[LWORK];

#ifdef FORINT
	if(b) FORNAME(dsygv)(&itypetmp,&vectors, &U, &ntmp, a, &m, *b, &ldb, w, WORK, &LWORK, &r );
        else FORNAME(dsyev)(&vectors, &U, &ntmp, a, &m, w, WORK, &LWORK, &r );
#else
        if(b) FORNAME(dsygv)(&itype,&vectors, &U, &n, a, &m, *b,&ldb, w, WORK, &LWORK, &r );
	else FORNAME(dsyev)(&vectors, &U, &n, a, &m, w, WORK, &LWORK, &r );
#endif

        delete[] WORK;
	if (LAPACK_FORTRANCASE(vectors) == LAPACK_FORTRANCASE('v') && corder) a.transposeme(n);

	if (r < 0) laerror("illegal argument in sygv/syev in diagonalize()");
	if (r > 0) laerror("convergence problem in sygv/syev in diagonalize()");
}

void diagonalize(NRMat<float> &a, NRVec<float> &w, const bool eivec, 
		const bool corder, int n, NRMat<float> *b, const int itype)
{
	FINT m = a.nrows();
	if (m != a.ncols()) laerror("diagonalize() call with non-square matrix");
	if (a.nrows() != w.size()) 
		laerror("inconsistent dimension of eigenvalue vector in diagonalize()");
	if(n==0) n=m;
	if(n<0||n>m) laerror("actual dimension out of range in diagonalize");
	if(b) if(n>b->nrows() || n> b->ncols()) laerror("wrong B matrix dimension in diagonalize");

	a.copyonwrite();
	w.copyonwrite();
	if(b) b->copyonwrite();

	FINT r = 0;
	char U =LAPACK_FORTRANCASE('u');
	char vectors = LAPACK_FORTRANCASE('v');
	if (!eivec) vectors = LAPACK_FORTRANCASE('n');
	FINT LWORK = -1;
	float WORKX;
	FINT ldb=0; if(b) ldb=b->ncols();

#ifdef FORINT
        const FINT itypetmp = itype;
	FINT ntmp = n;
	// First call is to determine size of workspace
	if(b) FORNAME(ssygv)(&itypetmp,&vectors, &U, &ntmp, a, &m, *b, &ldb, w, &WORKX, &LWORK, &r );
	else FORNAME(ssyev)(&vectors, &U, &ntmp, a, &m, w, &WORKX, &LWORK, &r );
#else
	// First call is to determine size of workspace
	if(b) FORNAME(ssygv)(&itype,&vectors, &U, &n, a, &m, *b, &ldb, w, &WORKX, &LWORK, &r );
	else FORNAME(ssyev)(&vectors, &U, &n, a, &m, w, &WORKX, &LWORK, &r );
#endif

        LWORK = (FINT)WORKX;
	float *WORK = new float[LWORK];

#ifdef FORINT
	if(b) FORNAME(ssygv)(&itypetmp,&vectors, &U, &ntmp, a, &m, *b, &ldb, w, WORK, &LWORK, &r );
        else FORNAME(ssyev)(&vectors, &U, &ntmp, a, &m, w, WORK, &LWORK, &r );
#else
        if(b) FORNAME(ssygv)(&itype,&vectors, &U, &n, a, &m, *b,&ldb, w, WORK, &LWORK, &r );
	else FORNAME(ssyev)(&vectors, &U, &n, a, &m, w, WORK, &LWORK, &r );
#endif

        delete[] WORK;
	if (LAPACK_FORTRANCASE(vectors) == LAPACK_FORTRANCASE('v') && corder) a.transposeme(n);

	if (r < 0) laerror("illegal argument in sygv/syev in diagonalize()");
	if (r > 0) laerror("convergence problem in sygv/syev in diagonalize()");
}



extern "C" void FORNAME(zheev)(const char *JOBZ, const char *UPLO, const FINT *N,
		std::complex<double> *A, const FINT *LDA, double *W, std::complex<double> *WORK, const FINT *LWORK, double *RWORK, FINT *INFO);
extern "C" void FORNAME(cheev)(const char *JOBZ, const char *UPLO, const FINT *N,
		std::complex<float> *A, const FINT *LDA, float *W, std::complex<float> *WORK, const FINT *LWORK, float *RWORK, FINT *INFO);

extern "C" void FORNAME(zhegv)(const FINT *ITYPE, const char *JOBZ, const char *UPLO, const FINT *N,
                std::complex<double> *A, const FINT *LDA, std::complex<double> *B, const FINT *LDB, double *W, std::complex<double> *WORK, const FINT *LWORK, double *RWORK, FINT *INFO);
extern "C" void FORNAME(chegv)(const FINT *ITYPE, const char *JOBZ, const char *UPLO, const FINT *N,
                std::complex<float> *A, const FINT *LDA, std::complex<float> *B, const FINT *LDB, float *W, std::complex<float> *WORK, const FINT *LWORK, float *RWORK, FINT *INFO);


// a will contain eigenvectors (columns if corder==1), w eigenvalues
void diagonalize(NRMat<std::complex<double> > &a, NRVec<double> &w, const bool eivec, 
		const bool corder, int n, NRMat<std::complex<double> > *b, const int itype)
{
	FINT m = a.nrows();
	if (m != a.ncols()) laerror("diagonalize() call with non-square matrix");
	if (a.nrows() != w.size()) 
		laerror("inconsistent dimension of eigenvalue vector in diagonalize()");
	if(n==0) n=m;
	if(n<0||n>m) laerror("actual dimension out of range in diagonalize");
	if(b) if(n>b->nrows() || n> b->ncols()) laerror("wrong B matrix dimension in diagonalize");

	a.copyonwrite();
	w.copyonwrite();
	if(b) b->copyonwrite();

	FINT r = 0;
	char U =LAPACK_FORTRANCASE('U');
	char vectors = LAPACK_FORTRANCASE('V');
	if (!eivec) vectors = LAPACK_FORTRANCASE('n');
	FINT LWORK = -1;
	std::complex<double> WORKX;
	FINT ldb=0; if(b) ldb=b->ncols();

	// First call is to determine size of workspace
	double *RWORK = new double[3*n+2]; 
#ifdef FORINT
        const FINT itypetmp = itype;
	FINT ntmp = n;
        if(b) FORNAME(zhegv)(&itypetmp,&vectors, &U, &ntmp, a, &m, *b, &ldb, w, &WORKX, &LWORK, RWORK, &r );
	else FORNAME(zheev)(&vectors, &U, &ntmp, a, &m, w, &WORKX, &LWORK, RWORK, &r );
#else
        if(b) FORNAME(zhegv)(&itype,&vectors, &U, &n, a, &m, *b, &ldb, w, &WORKX, &LWORK, RWORK, &r );
	else FORNAME(zheev)(&vectors, &U, &n, a, &m, w, &WORKX, &LWORK, RWORK, &r );
#endif	

        LWORK = (FINT)WORKX.real();
	std::complex<double> *WORK = new std::complex<double>[LWORK];

#ifdef FORINT
        if(b) FORNAME(zhegv)(&itypetmp,&vectors, &U, &ntmp, a, &m, *b, &ldb, w, WORK, &LWORK, RWORK, &r );
        else FORNAME(zheev)(&vectors, &U, &ntmp, a, &m, w, WORK, &LWORK, RWORK, &r );
#else	
        if(b) FORNAME(zhegv)(&itype,&vectors, &U, &n, a, &m, *b,&ldb, w, WORK, &LWORK, RWORK, &r );
	else FORNAME(zheev)(&vectors, &U, &n, a, &m, w, WORK, &LWORK, RWORK, &r );
#endif	

        delete[] WORK;
	delete[] RWORK;
	if (LAPACK_FORTRANCASE(vectors) == LAPACK_FORTRANCASE('v') && corder) {a.transposeme(n); a.conjugateme();}

	if (r < 0) laerror("illegal argument in hegv/heev in diagonalize()");
	if (r > 0) laerror("convergence problem in hegv/heev in diagonalize()");
}

void diagonalize(NRMat<std::complex<float> > &a, NRVec<float> &w, const bool eivec, 
		const bool corder, int n, NRMat<std::complex<float> > *b, const int itype)
{
	FINT m = a.nrows();
	if (m != a.ncols()) laerror("diagonalize() call with non-square matrix");
	if (a.nrows() != w.size()) 
		laerror("inconsistent dimension of eigenvalue vector in diagonalize()");
	if(n==0) n=m;
	if(n<0||n>m) laerror("actual dimension out of range in diagonalize");
	if(b) if(n>b->nrows() || n> b->ncols()) laerror("wrong B matrix dimension in diagonalize");

	a.copyonwrite();
	w.copyonwrite();
	if(b) b->copyonwrite();

	FINT r = 0;
	char U =LAPACK_FORTRANCASE('U');
	char vectors = LAPACK_FORTRANCASE('V');
	if (!eivec) vectors = LAPACK_FORTRANCASE('n');
	FINT LWORK = -1;
	std::complex<float> WORKX;
	FINT ldb=0; if(b) ldb=b->ncols();

	// First call is to determine size of workspace
	float *RWORK = new float[3*n+2]; 
#ifdef FORINT
        const FINT itypetmp = itype;
	FINT ntmp = n;
        if(b) FORNAME(chegv)(&itypetmp,&vectors, &U, &ntmp, a, &m, *b, &ldb, w, &WORKX, &LWORK, RWORK, &r );
	else FORNAME(cheev)(&vectors, &U, &ntmp, a, &m, w, &WORKX, &LWORK, RWORK, &r );
#else
        if(b) FORNAME(chegv)(&itype,&vectors, &U, &n, a, &m, *b, &ldb, w, &WORKX, &LWORK, RWORK, &r );
	else FORNAME(cheev)(&vectors, &U, &n, a, &m, w, &WORKX, &LWORK, RWORK, &r );
#endif	

        LWORK = (FINT)WORKX.real();
	std::complex<float> *WORK = new std::complex<float>[LWORK];

#ifdef FORINT
        if(b) FORNAME(chegv)(&itypetmp,&vectors, &U, &ntmp, a, &m, *b, &ldb, w, WORK, &LWORK, RWORK, &r );
        else FORNAME(cheev)(&vectors, &U, &ntmp, a, &m, w, WORK, &LWORK, RWORK, &r );
#else	
        if(b) FORNAME(chegv)(&itype,&vectors, &U, &n, a, &m, *b,&ldb, w, WORK, &LWORK, RWORK, &r );
	else FORNAME(cheev)(&vectors, &U, &n, a, &m, w, WORK, &LWORK, RWORK, &r );
#endif	

        delete[] WORK;
	delete[] RWORK;
	if (LAPACK_FORTRANCASE(vectors) == LAPACK_FORTRANCASE('v') && corder) {a.transposeme(n); a.conjugateme();}

	if (r < 0) laerror("illegal argument in hegv/heev in diagonalize()");
	if (r > 0) laerror("convergence problem in hegv/heev in diagonalize()");
}



extern "C" void FORNAME(dspev)(const char *JOBZ, const char *UPLO, const FINT *N,
		double *AP, double *W, double *Z, const FINT *LDZ, double *WORK, FINT *INFO);
extern "C" void FORNAME(sspev)(const char *JOBZ, const char *UPLO, const FINT *N,
		float *AP, float *W, float *Z, const FINT *LDZ, float *WORK, FINT *INFO);

extern "C" void FORNAME(dspgv)(const FINT *ITYPE, const char *JOBZ, const char *UPLO, const FINT *N,
                double *AP, double *BP, double *W, double *Z, const FINT *LDZ, double *WORK, FINT *INFO);
extern "C" void FORNAME(sspgv)(const FINT *ITYPE, const char *JOBZ, const char *UPLO, const FINT *N,
                float *AP, float *BP, float *W, float *Z, const FINT *LDZ, float *WORK, FINT *INFO);


// v will contain eigenvectors, w eigenvalues
void diagonalize(NRSMat<double> &a, NRVec<double> &w, NRMat<double> *v,
		const bool corder, int n, NRSMat<double> *b, const int itype)
{
	if(n<=0) n = a.nrows();
	if (v) if (v->nrows() != v ->ncols() || n > v->nrows() || n > a.nrows())
		laerror("diagonalize() call with inconsistent dimensions");
	if (n==a.nrows() && n != w.size() || n>w.size()) laerror("inconsistent dimension of eigenvalue vector");

	if(b) if(n>b->nrows() || n> b->ncols()) laerror("wrong B matrix dimension in diagonalize");

	a.copyonwrite();
	w.copyonwrite();
	if(v) v->copyonwrite();
	if(b) b->copyonwrite();

	FINT r = 0;
	char U = LAPACK_FORTRANCASE('u');
	char job = LAPACK_FORTRANCASE(v ? 'v' : 'n');

	double *WORK = new double[3*n];
	FINT ldv=v?v->ncols():n;
#ifdef FORINT
        const FINT itypetmp = itype;
	FINT ntmp = n;
        if(b) FORNAME(dspgv)(&itypetmp,&job, &U, &ntmp, a, *b, w, v?(*v)[0]:(double *)0, &ldv, WORK,  &r );
	else FORNAME(dspev)(&job, &U, &ntmp, a, w, v?(*v)[0]:(double *)0, &ldv, WORK,  &r );
#else
        if(b) FORNAME(dspgv)(&itype,&job, &U, &n, a, *b, w, v?(*v)[0]:(double *)0, &ldv, WORK,  &r );
	else FORNAME(dspev)(&job, &U, &n, a, w, v?(*v)[0]:(double *)0, &ldv, WORK,  &r );
#endif
	delete[] WORK;
	if (v && corder) v->transposeme(n);

	if (r < 0) laerror("illegal argument in spgv/spev in diagonalize()");
	if (r > 0) laerror("convergence problem in spgv/spev in diagonalize()");
}

void diagonalize(NRSMat<float> &a, NRVec<float> &w, NRMat<float> *v,
		const bool corder, int n, NRSMat<float> *b, const int itype)
{
	if(n<=0) n = a.nrows();
	if (v) if (v->nrows() != v ->ncols() || n > v->nrows() || n > a.nrows())
		laerror("diagonalize() call with inconsistent dimensions");
	if (n==a.nrows() && n != w.size() || n>w.size()) laerror("inconsistent dimension of eigenvalue vector");

	if(b) if(n>b->nrows() || n> b->ncols()) laerror("wrong B matrix dimension in diagonalize");

	a.copyonwrite();
	w.copyonwrite();
	if(v) v->copyonwrite();
	if(b) b->copyonwrite();

	FINT r = 0;
	char U = LAPACK_FORTRANCASE('u');
	char job = LAPACK_FORTRANCASE(v ? 'v' : 'n');

	float *WORK = new float[3*n];
	FINT ldv=v?v->ncols():n;
#ifdef FORINT
        const FINT itypetmp = itype;
	FINT ntmp = n;
        if(b) FORNAME(sspgv)(&itypetmp,&job, &U, &ntmp, a, *b, w, v?(*v)[0]:(float *)0, &ldv, WORK,  &r );
	else FORNAME(sspev)(&job, &U, &ntmp, a, w, v?(*v)[0]:(float *)0, &ldv, WORK,  &r );
#else
        if(b) FORNAME(sspgv)(&itype,&job, &U, &n, a, *b, w, v?(*v)[0]:(float *)0, &ldv, WORK,  &r );
	else FORNAME(sspev)(&job, &U, &n, a, w, v?(*v)[0]:(float *)0, &ldv, WORK,  &r );
#endif
	delete[] WORK;
	if (v && corder) v->transposeme(n);

	if (r < 0) laerror("illegal argument in spgv/spev in diagonalize()");
	if (r > 0) laerror("convergence problem in spgv/spev in diagonalize()");
}


extern "C" void FORNAME(zhpev)(const char *JOBZ, const char *UPLO, const FINT *N,
		std::complex<double> *AP, double *W, std::complex<double> *Z, const FINT *LDZ, std::complex<double> *WORK, double *RWORK, FINT *INFO);
extern "C" void FORNAME(chpev)(const char *JOBZ, const char *UPLO, const FINT *N,
		std::complex<float> *AP, float *W, std::complex<float> *Z, const FINT *LDZ, std::complex<float> *WORK, float *RWORK, FINT *INFO);

extern "C" void FORNAME(zhpgv)(const FINT *ITYPE, const char *JOBZ, const char *UPLO, const FINT *N,
                std::complex<double> *AP, std::complex<double> *BP, double *W, std::complex<double> *Z, const FINT *LDZ, std::complex<double> *WORK,  double *RWORK, FINT *INFO);
extern "C" void FORNAME(chpgv)(const FINT *ITYPE, const char *JOBZ, const char *UPLO, const FINT *N,
                std::complex<float> *AP, std::complex<float> *BP, float *W, std::complex<float> *Z, const FINT *LDZ, std::complex<float> *WORK,  float *RWORK, FINT *INFO);


// v will contain eigenvectors, w eigenvalues
void diagonalize(NRSMat<std::complex<double> > &a, NRVec<double> &w, NRMat<std::complex<double> > *v,
		const bool corder, int n, NRSMat<std::complex<double> > *b, const int itype)
{
	if(n<=0) n = a.nrows();
	if (v) if (v->nrows() != v ->ncols() || n > v->nrows() || n > a.nrows())
		laerror("diagonalize() call with inconsistent dimensions");
	if (n==a.nrows() && n != w.size() || n>w.size()) laerror("inconsistent dimension of eigenvalue vector");

	if(b) if(n>b->nrows() || n> b->ncols()) laerror("wrong B matrix dimension in diagonalize");

	a.copyonwrite();
	w.copyonwrite();
	if(v) v->copyonwrite();
	if(b) b->copyonwrite();

	FINT r = 0;
	char U = LAPACK_FORTRANCASE('u');
	char job = LAPACK_FORTRANCASE(v ? 'v' : 'n');

	std::complex<double> *WORK = new std::complex<double>[2*n];
	double *RWORK  = new double[3*n];
	FINT ldv=v?v->ncols():n;
#ifdef FORINT	
        const FINT itypetmp = itype;
        FINT ntmp = n;
	if(b) FORNAME(zhpgv)(&itypetmp,&job, &U, &ntmp, a, *b, w, v?(*v)[0]:(std::complex<double> *)0, &ldv, WORK, RWORK,  &r );
	else FORNAME(zhpev)(&job, &U, &ntmp, a, w, v?(*v)[0]:(std::complex<double> *)0, &ldv, WORK, RWORK,  &r );
#else
	if(b) FORNAME(zhpgv)(&itype,&job, &U, &n, a, *b, w, v?(*v)[0]:(std::complex<double> *)0, &ldv, WORK, RWORK,  &r );
	else FORNAME(zhpev)(&job, &U, &n, a, w, v?(*v)[0]:(std::complex<double> *)0, &ldv, WORK, RWORK,  &r );
#endif
	delete[] WORK;
	delete[] RWORK;
	if (v && corder) {v->transposeme(n); v->conjugateme();}

	if (r < 0) laerror("illegal argument in hpgv/hpev in diagonalize()");
	if (r > 0) laerror("convergence problem in hpgv/hpev in diagonalize()");
}

void diagonalize(NRSMat<std::complex<float> > &a, NRVec<float> &w, NRMat<std::complex<float> > *v,
		const bool corder, int n, NRSMat<std::complex<float> > *b, const int itype)
{
	if(n<=0) n = a.nrows();
	if (v) if (v->nrows() != v ->ncols() || n > v->nrows() || n > a.nrows())
		laerror("diagonalize() call with inconsistent dimensions");
	if (n==a.nrows() && n != w.size() || n>w.size()) laerror("inconsistent dimension of eigenvalue vector");

	if(b) if(n>b->nrows() || n> b->ncols()) laerror("wrong B matrix dimension in diagonalize");

	a.copyonwrite();
	w.copyonwrite();
	if(v) v->copyonwrite();
	if(b) b->copyonwrite();

	FINT r = 0;
	char U = LAPACK_FORTRANCASE('u');
	char job = LAPACK_FORTRANCASE(v ? 'v' : 'n');

	std::complex<float> *WORK = new std::complex<float>[2*n];
	float *RWORK  = new float[3*n];
	FINT ldv=v?v->ncols():n;
#ifdef FORINT	
        const FINT itypetmp = itype;
        FINT ntmp = n;
	if(b) FORNAME(chpgv)(&itypetmp,&job, &U, &ntmp, a, *b, w, v?(*v)[0]:(std::complex<float> *)0, &ldv, WORK, RWORK,  &r );
	else FORNAME(chpev)(&job, &U, &ntmp, a, w, v?(*v)[0]:(std::complex<float> *)0, &ldv, WORK, RWORK,  &r );
#else
	if(b) FORNAME(chpgv)(&itype,&job, &U, &n, a, *b, w, v?(*v)[0]:(std::complex<float> *)0, &ldv, WORK, RWORK,  &r );
	else FORNAME(chpev)(&job, &U, &n, a, w, v?(*v)[0]:(std::complex<float> *)0, &ldv, WORK, RWORK,  &r );
#endif
	delete[] WORK;
	delete[] RWORK;
	if (v && corder) {v->transposeme(n); v->conjugateme();}

	if (r < 0) laerror("illegal argument in hpgv/hpev in diagonalize()");
	if (r > 0) laerror("convergence problem in hpgv/hpev in diagonalize()");
}



extern "C" void FORNAME(dgesvd)(const char *JOBU,  const char *JOBVT,  const FINT *M,
		const FINT *N,  double *A, const FINT *LDA, double *S, double *U, const FINT *LDU,
		double *VT, const FINT *LDVT, double *WORK, const FINT *LWORK, FINT *INFO );
extern "C" void FORNAME(sgesvd)(const char *JOBU,  const char *JOBVT,  const FINT *M,
		const FINT *N,  float *A, const FINT *LDA, float *S, float *U, const FINT *LDU,
		float *VT, const FINT *LDVT, float *WORK, const FINT *LWORK, FINT *INFO );

//normally in v returns vtransposed for default vnotdagger=0
void singular_decomposition(NRMat<double> &a, NRMat<double> *u, NRVec<double> &s,
		NRMat<double> *v, const bool vnotdagger, int m, int n)
{
	FINT m0 = a.nrows();
	FINT n0 = a.ncols();
	if(m<=0) m=(int)m0;
	if(n<=0) n=(int)n0;
	if(n>n0 || m>m0) laerror("bad dimension in singular_decomposition");
	if (u) if (m > u->nrows() || s.size()> u->ncols())
		laerror("inconsistent dimension of U Mat in singular_decomposition()");
	if (s.size() < m && s.size() < n) 
		laerror("inconsistent dimension of S Vec in singular_decomposition()");
	if (v) if (s.size() > v->nrows() || n > v->ncols())
		laerror("inconsistent dimension of V Mat in singular_decomposition()");

	a.copyonwrite();
	s.copyonwrite();
	if (u) u->copyonwrite();
	if (v) v->copyonwrite();
	
	// C-order (transposed) input and swap u,v matrices,
	// v should be transposed at the end
        char jobu = u ? (u->nrows()==u->ncols() ? 'A' : 'S') : 'N';
        char jobv = v ? (v->nrows()==v->ncols() ? 'A' : 'S') : 'N';

	double work0;
	FINT lwork = -1;
	FINT r;

	FINT lda=a.ncols();
	FINT ldu= u ? u->ncols():0;
	FINT ldv= v ? v->ncols():0;
#ifdef FORINT
        FINT ntmp = n;
	FINT mtmp = m;
	FORNAME(dgesvd)(&jobv, &jobu, &ntmp, &mtmp, a, &lda, s, v?(*v)[0]:0, &ldv, u?(*u)[0]:0, &ldu, &work0, &lwork, &r);
#else
	FORNAME(dgesvd)(&jobv, &jobu, &n, &m, a, &lda, s, v?(*v)[0]:0, &ldv, u?(*u)[0]:0, &ldu, &work0, &lwork, &r);
#endif

	lwork = (FINT) work0;
	double *work = new double[lwork];

#ifdef FORINT
	FORNAME(dgesvd)(&jobv, &jobu, &ntmp, &mtmp, a, &lda, s, v?(*v)[0]:0, &ldv, u?(*u)[0]:0, &ldu, work, &lwork, &r);
#else
	FORNAME(dgesvd)(&jobv, &jobu, &n, &m, a, &lda, s, v?(*v)[0]:0, &ldv, u?(*u)[0]:0, &ldu, work, &lwork, &r);
#endif

	delete[] work;
	if (v && vnotdagger) v->transposeme(n);

	if (r < 0) laerror("illegal argument in gesvd() of singular_decomposition()");
	if (r > 0) laerror("convergence problem in gesvd() of singular_decomposition()");
}

void singular_decomposition(NRMat<float> &a, NRMat<float> *u, NRVec<float> &s,
		NRMat<float> *v, const bool vnotdagger, int m, int n)
{
	FINT m0 = a.nrows();
	FINT n0 = a.ncols();
	if(m<=0) m=(int)m0;
	if(n<=0) n=(int)n0;
	if(n>n0 || m>m0) laerror("bad dimension in singular_decomposition");
	if (u) if (m > u->nrows() || s.size()> u->ncols())
		laerror("inconsistent dimension of U Mat in singular_decomposition()");
	if (s.size() < m && s.size() < n) 
		laerror("inconsistent dimension of S Vec in singular_decomposition()");
	if (v) if (s.size() > v->nrows() || n > v->ncols())
		laerror("inconsistent dimension of V Mat in singular_decomposition()");

	a.copyonwrite();
	s.copyonwrite();
	if (u) u->copyonwrite();
	if (v) v->copyonwrite();
	
	// C-order (transposed) input and swap u,v matrices,
	// v should be transposed at the end
        char jobu = u ? (u->nrows()==u->ncols() ? 'A' : 'S') : 'N';
        char jobv = v ? (v->nrows()==v->ncols() ? 'A' : 'S') : 'N';

	float work0;
	FINT lwork = -1;
	FINT r;

	FINT lda=a.ncols();
	FINT ldu= u ? u->ncols():0;
	FINT ldv= v ? v->ncols():0;
#ifdef FORINT
        FINT ntmp = n;
	FINT mtmp = m;
	FORNAME(sgesvd)(&jobv, &jobu, &ntmp, &mtmp, a, &lda, s, v?(*v)[0]:0, &ldv, u?(*u)[0]:0, &ldu, &work0, &lwork, &r);
#else
	FORNAME(sgesvd)(&jobv, &jobu, &n, &m, a, &lda, s, v?(*v)[0]:0, &ldv, u?(*u)[0]:0, &ldu, &work0, &lwork, &r);
#endif

	lwork = (FINT) work0;
	float *work = new float[lwork];

#ifdef FORINT
	FORNAME(sgesvd)(&jobv, &jobu, &ntmp, &mtmp, a, &lda, s, v?(*v)[0]:0, &ldv, u?(*u)[0]:0, &ldu, work, &lwork, &r);
#else
	FORNAME(sgesvd)(&jobv, &jobu, &n, &m, a, &lda, s, v?(*v)[0]:0, &ldv, u?(*u)[0]:0, &ldu, work, &lwork, &r);
#endif

	delete[] work;
	if (v && vnotdagger) v->transposeme(n);

	if (r < 0) laerror("illegal argument in gesvd() of singular_decomposition()");
	if (r > 0) laerror("convergence problem in gesvd() of singular_decomposition()");
}






extern "C" void FORNAME(zgesvd)(const char *JOBU,  const char *JOBVT,  const FINT *M,
		const FINT *N,  std::complex<double> *A, const FINT *LDA, double *S, std::complex<double> *U, const FINT *LDU,
		std::complex<double> *VT, const FINT *LDVT, std::complex<double> *WORK, const FINT *LWORK, double *RWORK, FINT *INFO );
extern "C" void FORNAME(cgesvd)(const char *JOBU,  const char *JOBVT,  const FINT *M,
		const FINT *N,  std::complex<float> *A, const FINT *LDA, float *S, std::complex<float> *U, const FINT *LDU,
		std::complex<float> *VT, const FINT *LDVT, std::complex<float> *WORK, const FINT *LWORK, float *RWORK, FINT *INFO );

void singular_decomposition(NRMat<std::complex<double> > &a, NRMat<std::complex<double> > *u, NRVec<double> &s,
		NRMat<std::complex<double> > *v, const bool vnotdagger, int m, int n)
{
	FINT m0 = a.nrows();
	FINT n0 = a.ncols();
	if(m<=0) m=(int)m0;
	if(n<=0) n=(int)n0;
	if(n>n0 || m>m0) laerror("bad dimension in singular_decomposition");
	if (u) if (m > u->nrows() || s.size()> u->ncols())
		laerror("inconsistent dimension of U Mat in singular_decomposition()");
	if (s.size() < m && s.size() < n) 
		laerror("inconsistent dimension of S Vec in singular_decomposition()");
	if (v) if (s.size() > v->nrows() || n > v->ncols())
		laerror("inconsistent dimension of V Mat in singular_decomposition()");

	int nmin =  n<m?n:m;
	a.copyonwrite();
	s.copyonwrite();
	if (u) u->copyonwrite();
	if (v) v->copyonwrite();
	
	// C-order (transposed) input and swap u,v matrices,
	// v should be transposed at the end
	char jobu = u ? (u->nrows()==u->ncols() ? 'A' : 'S') : 'N';
	char jobv = v ? (v->nrows()==v->ncols() ? 'A' : 'S') : 'N';

	std::complex<double> work0;
	FINT lwork = -1;
	FINT r;
        double *rwork = new double[5*nmin];

        FINT lda=a.ncols();
        FINT ldu= u ? u->ncols():0;
        FINT ldv= v ? v->ncols():0;
#ifdef FORINT
        FINT ntmp = n;
	FINT mtmp = m;
	FORNAME(zgesvd)(&jobv, &jobu, &ntmp, &mtmp, a, &lda, s, v?(*v)[0]:0, &ldv, u?(*u)[0]:0, &ldu, &work0, &lwork, rwork, &r);
#else
	FORNAME(zgesvd)(&jobv, &jobu, &n, &m, a, &lda, s, v?(*v)[0]:0, &ldv, u?(*u)[0]:0, &ldu, &work0, &lwork, rwork, &r);
#endif

	lwork = (FINT) work0.real();
	std::complex<double> *work = new std::complex<double>[lwork];

#ifdef FORINT
	FORNAME(zgesvd)(&jobv, &jobu, &ntmp, &mtmp, a, &lda, s, v?(*v)[0]:0, &ldv, u?(*u)[0]:0, &ldu, work, &lwork, rwork, &r);
#else
	FORNAME(zgesvd)(&jobv, &jobu, &n, &m, a, &lda, s, v?(*v)[0]:0, &ldv, u?(*u)[0]:0, &ldu, work, &lwork, rwork, &r);
#endif

	delete[] work;
	delete[] rwork;
	if (v && vnotdagger) {v->transposeme(n); v->conjugateme();}

	if (r < 0) laerror("illegal argument in gesvd() of singular_decomposition()");
	if (r > 0) laerror("convergence problem in gesvd() of singular_decomposition()");
}

void singular_decomposition(NRMat<std::complex<float> > &a, NRMat<std::complex<float> > *u, NRVec<float> &s,
		NRMat<std::complex<float> > *v, const bool vnotdagger, int m, int n)
{
	FINT m0 = a.nrows();
	FINT n0 = a.ncols();
	if(m<=0) m=(int)m0;
	if(n<=0) n=(int)n0;
	if(n>n0 || m>m0) laerror("bad dimension in singular_decomposition");
	if (u) if (m > u->nrows() || s.size()> u->ncols())
		laerror("inconsistent dimension of U Mat in singular_decomposition()");
	if (s.size() < m && s.size() < n) 
		laerror("inconsistent dimension of S Vec in singular_decomposition()");
	if (v) if (s.size() > v->nrows() || n > v->ncols())
		laerror("inconsistent dimension of V Mat in singular_decomposition()");

	int nmin =  n<m?n:m;
	a.copyonwrite();
	s.copyonwrite();
	if (u) u->copyonwrite();
	if (v) v->copyonwrite();
	
	// C-order (transposed) input and swap u,v matrices,
	// v should be transposed at the end
	char jobu = u ? (u->nrows()==u->ncols() ? 'A' : 'S') : 'N';
	char jobv = v ? (v->nrows()==v->ncols() ? 'A' : 'S') : 'N';

	std::complex<float> work0;
	FINT lwork = -1;
	FINT r;
        float *rwork = new float[5*nmin];

        FINT lda=a.ncols();
        FINT ldu= u ? u->ncols():0;
        FINT ldv= v ? v->ncols():0;
#ifdef FORINT
        FINT ntmp = n;
	FINT mtmp = m;
	FORNAME(cgesvd)(&jobv, &jobu, &ntmp, &mtmp, a, &lda, s, v?(*v)[0]:0, &ldv, u?(*u)[0]:0, &ldu, &work0, &lwork, rwork, &r);
#else
	FORNAME(cgesvd)(&jobv, &jobu, &n, &m, a, &lda, s, v?(*v)[0]:0, &ldv, u?(*u)[0]:0, &ldu, &work0, &lwork, rwork, &r);
#endif

	lwork = (FINT) work0.real();
	std::complex<float> *work = new std::complex<float>[lwork];

#ifdef FORINT
	FORNAME(cgesvd)(&jobv, &jobu, &ntmp, &mtmp, a, &lda, s, v?(*v)[0]:0, &ldv, u?(*u)[0]:0, &ldu, work, &lwork, rwork, &r);
#else
	FORNAME(cgesvd)(&jobv, &jobu, &n, &m, a, &lda, s, v?(*v)[0]:0, &ldv, u?(*u)[0]:0, &ldu, work, &lwork, rwork, &r);
#endif

	delete[] work;
	delete[] rwork;
	if (v && vnotdagger) {v->transposeme(n); v->conjugateme();}

	if (r < 0) laerror("illegal argument in gesvd() of singular_decomposition()");
	if (r > 0) laerror("convergence problem in gesvd() of singular_decomposition()");
}


//QR decomposition
//extern "C" void FORNAME(dgeqrf)(const int *M, const int *N, double *A, const int *LDA, double *TAU, double *WORK, int *LWORK, int *INFO);


extern "C" void FORNAME(dgeev)(const char *JOBVL, const char *JOBVR, const FINT *N,
		double *A, const FINT *LDA, double *WR, double *WI, double *VL, const FINT *LDVL,
		double *VR, const FINT *LDVR, double *WORK, const FINT *LWORK, FINT *INFO );
extern "C" void FORNAME(sgeev)(const char *JOBVL, const char *JOBVR, const FINT *N,
		float *A, const FINT *LDA, float *WR, float *WI, float *VL, const FINT *LDVL,
		float *VR, const FINT *LDVR, float *WORK, const FINT *LWORK, FINT *INFO );

extern "C" void FORNAME(dggev)(const char *JOBVL, const char *JOBVR, const FINT *N,
		double *A, const FINT *LDA, double *B, const FINT *LDB, double *WR, double *WI,  double *WBETA, 
		 double *VL, const FINT *LDVL,  double *VR, const FINT *LDVR,  
		double *WORK, const FINT *LWORK, FINT *INFO );
extern "C" void FORNAME(sggev)(const char *JOBVL, const char *JOBVR, const FINT *N,
		float *A, const FINT *LDA, float *B, const FINT *LDB, float *WR, float *WI,  float *WBETA, 
		 float *VL, const FINT *LDVL,  float *VR, const FINT *LDVR,  
		float *WORK, const FINT *LWORK, FINT *INFO );

extern "C" void FORNAME(zgeev)(const char *JOBVL, const char *JOBVR, const FINT *N,
                std::complex<double> *A, const FINT *LDA, std::complex<double> *W, std::complex<double> *VL, const FINT *LDVL,
                std::complex<double> *VR, const FINT *LDVR, std::complex<double> *WORK, const FINT *LWORK,
		 double *RWORK, FINT *INFO );
extern "C" void FORNAME(cgeev)(const char *JOBVL, const char *JOBVR, const FINT *N,
                std::complex<float> *A, const FINT *LDA, std::complex<float> *W, std::complex<float> *VL, const FINT *LDVL,
                std::complex<float> *VR, const FINT *LDVR, std::complex<float> *WORK, const FINT *LWORK,
		 float *RWORK, FINT *INFO );

extern "C" void FORNAME(zggev)(const char *JOBVL, const char *JOBVR, const FINT *N,
                std::complex<double> *A, const FINT *LDA, std::complex<double> *B, const FINT *LDB, std::complex<double> *W, std::complex<double> *WBETA,
                std::complex<double> *VL, const FINT *LDVL,  std::complex<double> *VR, const FINT *LDVR,
                std::complex<double> *WORK, const FINT *LWORK, double *RWORK, FINT *INFO );
extern "C" void FORNAME(cggev)(const char *JOBVL, const char *JOBVR, const FINT *N,
                std::complex<float> *A, const FINT *LDA, std::complex<float> *B, const FINT *LDB, std::complex<float> *W, std::complex<float> *WBETA,
                std::complex<float> *VL, const FINT *LDVL,  std::complex<float> *VR, const FINT *LDVR,
                std::complex<float> *WORK, const FINT *LWORK, float *RWORK, FINT *INFO );

//tridiagonal
extern "C" void FORNAME(dsteqr)(const char *compz, const  FINT *n, double *d, double *d1, double *z, const FINT *ldz,  double *work, FINT *info);
extern "C" void FORNAME(ssteqr)(const char *compz, const  FINT *n, float *d, float *d1, float *z, const FINT *ldz,  float *work, FINT *info);

void diagonalize3(NRVec<double> &d, NRVec<double> &d1, NRMat<double> *v, const bool corder, int n0)
{
	FINT n = n0? n0: d.size();
	if(d1.size()<n-1) laerror("inconsistent dimensions in diagonalize3");
	if(v) {if(n!=v->nrows()||n!=v->ncols()) laerror("inconsistent dimensions in diagonalize3");}
	d.copyonwrite();
	d1.copyonwrite();
	if(v) v->copyonwrite();
	
	FINT ldz=n;
	FINT info;
	int worksize=2*n-2;
	if(worksize<1) worksize=1;
        double *work = new double[worksize];
	char job=v?'i':'n';
	FORNAME(dsteqr)(&job,&n,&d[0],&d1[0],(v?&(*v)(0,0):NULL),&ldz,work,&info);
	delete[] work;
	if (v && corder) v->transposeme();

	if (info < 0) laerror("illegal argument in dsteqr() fo diagonalize3");
	if (info > 0) laerror("convergence problem in dsteqr() fo diagonalize3");
}

void diagonalize3(NRVec<float> &d, NRVec<float> &d1, NRMat<float> *v, const bool corder, int n0)
{
	FINT n = n0? n0: d.size();
	if(d1.size()<n-1) laerror("inconsistent dimensions in diagonalize3");
	if(v) {if(n!=v->nrows()||n!=v->ncols()) laerror("inconsistent dimensions in diagonalize3");}
	d.copyonwrite();
	d1.copyonwrite();
	if(v) v->copyonwrite();
	
	FINT ldz=n;
	FINT info;
	int worksize=2*n-2;
	if(worksize<1) worksize=1;
        float *work = new float[worksize];
	char job=v?'i':'n';
	FORNAME(ssteqr)(&job,&n,&d[0],&d1[0],(v?&(*v)(0,0):NULL),&ldz,work,&info);
	delete[] work;
	if (v && corder) v->transposeme();

	if (info < 0) laerror("illegal argument in dsteqr() fo diagonalize3");
	if (info > 0) laerror("convergence problem in dsteqr() fo diagonalize3");
}


//statics for sorting, one set per floating type (variable templates)
static int *gdperm;
template<typename T> static T *gdwr;
template<typename T> static T *gdwi;
template<typename T> static T *gdbeta;

//compare methods
template<typename T>
static T realonly(const int i, const int j)
{
if(gdbeta<T>)
	{
	if(gdbeta<T>[i]==0. && gdbeta<T>[j]!=0) return 1.;
	if(gdbeta<T>[j]==0. && gdbeta<T>[i]!=0) return -1.;
	if(gdbeta<T>[i]==0. && gdbeta<T>[j]==0) return 0.;
	T tmp = gdwr<T>[i]/gdbeta<T>[i]-gdwr<T>[j]/gdbeta<T>[j];
	if(tmp) return tmp;
	return gdwi<T>[j]/gdbeta<T>[j]-gdwi<T>[i]/gdbeta<T>[i];
	}
//else
T tmp = gdwr<T>[i]-gdwr<T>[j];
if(tmp) return tmp;
return gdwi<T>[j]-gdwi<T>[i];
}

template<typename T>
static T realfirst(const int i, const int j)
{
if(gdwi<T>[i] && ! gdwi<T>[j]) return 1.;
if(!gdwi<T>[i] && gdwi<T>[j]) return -1.;
return realonly<T>(i,j);
}

template<typename T>
static T (* gdcompar[2])(const int, const int) = {&realonly<T>, &realfirst<T>};

//swap method
template<typename T>
static void gdswap(const int i, const int j)
{
T tmp;
int itmp;
itmp=gdperm[i]; gdperm[i]=gdperm[j]; gdperm[j]=itmp;
tmp=gdwr<T>[i]; gdwr<T>[i]=gdwr<T>[j]; gdwr<T>[j]=tmp;
tmp=gdwi<T>[i]; gdwi<T>[i]=gdwi<T>[j]; gdwi<T>[j]=tmp;
if(gdbeta<T>) {tmp=gdbeta<T>[i]; gdbeta<T>[i]=gdbeta<T>[j]; gdbeta<T>[j]=tmp;}
}



void gdiagonalize(NRMat<double> &a, NRVec<double> &wr, NRVec<double> &wi,
		NRMat<double> *vl, NRMat<double> *vr, const bool corder, int n,
		const int sorttype, const int biorthonormalize,
		NRMat<double> *b, NRVec<double> *beta)
{
	if(n<=0) {n = a.nrows(); if(a.ncols()!=a.nrows() ) laerror("gdiagonalize() call for a non-square matrix");}
	if (n > a.ncols() || n>a.nrows() ) laerror("gdiagonalize() of too big submatrix");
	if (n > wr.size()) 
		laerror("inconsistent dimension of eigen vector in gdiagonalize()");
	if (vl) if (n > vl->nrows() || n > vl->ncols())
		laerror("inconsistent dimension of vl in gdiagonalize()");
	if (vr) if (n > vr->nrows() || n > vr->ncols())
		laerror("inconsistent dimension of vr in gdiagonalize()");
	if (beta) if(n > beta ->size()) laerror("inconsistent dimension of beta in gdiagonalize()");
	if(b) if(n > b->nrows() || n > b->ncols())
		 laerror("inconsistent dimension of b in gdiagonalize()");
	if(b && !beta || beta && !b) laerror("missing array for generalized diagonalization");

	a.copyonwrite();
	wr.copyonwrite();
	wi.copyonwrite();
	if (vl) vl->copyonwrite();
	if (vr) vr->copyonwrite();
	if (beta) beta->copyonwrite();
	if (b) b->copyonwrite();
	
	char jobvl = LAPACK_FORTRANCASE(vl ? 'v' : 'n');
	char jobvr = LAPACK_FORTRANCASE(vr ? 'v' : 'n');
	double work0;
	FINT lwork = -1;
	FINT r;
	FINT lda=a.ncols();
	FINT ldb=0;
	if(b) ldb=b->ncols();
	FINT ldvl= vl?vl->ncols():lda;
	FINT ldvr= vr?vr->ncols():lda;

#ifdef FORINT
        FINT ntmp = n; 
        if(b)  FORNAME(dggev)(&jobvr, &jobvl, &ntmp, a, &lda, *b, &ldb, wr, wi, *beta, vr?(*vr)[0]:(double *)0,
                        &ldvr, vl?(*vl)[0]:(double *)0, &ldvl, &work0, &lwork, &r);
	else FORNAME(dgeev)(&jobvr, &jobvl, &ntmp, a, &lda, wr, wi, vr?(*vr)[0]:(double *)0,
			&ldvr, vl?(*vl)[0]:(double *)0, &ldvl, &work0, &lwork, &r);
#else
        if(b)  FORNAME(dggev)(&jobvr, &jobvl, &n, a, &lda, *b, &ldb, wr, wi, *beta, vr?(*vr)[0]:(double *)0,
                        &ldvr, vl?(*vl)[0]:(double *)0, &ldvl, &work0, &lwork, &r);
	else FORNAME(dgeev)(&jobvr, &jobvl, &n, a, &lda, wr, wi, vr?(*vr)[0]:(double *)0,
			&ldvr, vl?(*vl)[0]:(double *)0, &ldvl, &work0, &lwork, &r);
#endif

        lwork = (FINT) work0;
	double *work = new double[lwork];

#ifdef FORINT	
	if(b) FORNAME(dggev)(&jobvr, &jobvl, &ntmp, a, &lda, *b, &ldb, wr, wi, *beta, vr?(*vr)[0]:(double *)0,
                        &ldvr, vl?(*vl)[0]:(double *)0, &ldvl, work, &lwork, &r);
	else FORNAME(dgeev)(&jobvr, &jobvl, &ntmp, a, &lda, wr, wi, vr?(*vr)[0]:(double *)0,
			&ldvr, vl?(*vl)[0]:(double *)0, &ldvl, work, &lwork, &r);
#else
	if(b) FORNAME(dggev)(&jobvr, &jobvl, &n, a, &lda, *b, &ldb, wr, wi, *beta, vr?(*vr)[0]:(double *)0,
                        &ldvr, vl?(*vl)[0]:(double *)0, &ldvl, work, &lwork, &r);
	else FORNAME(dgeev)(&jobvr, &jobvl, &n, a, &lda, wr, wi, vr?(*vr)[0]:(double *)0,
			&ldvr, vl?(*vl)[0]:(double *)0, &ldvl, work, &lwork, &r);
#endif
	delete[] work;


	if (r < 0) laerror("illegal argument in ggev/geev in gdiagonalize()");
	if (r > 0) laerror("convergence problem in ggev/geev in gdiagonalize()");

//std::cout <<"TEST dgeev\n"<<wr<<wi<<*vr<<*vl<<std::endl;

	if(biorthonormalize && vl && vr)
		{
		if(b || beta) laerror("@@@ biorthonormalize not implemented yet for generalized non-symmetric eigenproblem");//metric b would be needed
		int i=0;
		while(i<n)
			{
			if(wi[i]==0) //real
				{
				//calculate scaling paramter
				double tmp;
				tmp=cblas_ddot(n,(*vl)[i],1,(*vr)[i], 1);
				if(biorthonormalize==1) cblas_dscal(n,1./tmp,(*vl)[i],1);
				if(biorthonormalize==2) cblas_dscal(n,1./tmp,(*vr)[i],1);
				i++;
				}
			else //complex pair
				{
				//calculate rotation parameters
				double s11,s12;
				//double s21,s22;
				s11=cblas_ddot(n,(*vl)[i],1,(*vr)[i], 1);
				s12=cblas_ddot(n,(*vl)[i],1,(*vr)[i+1], 1);
				//s21=cblas_ddot(n,(*vl)[i+1],1,(*vr)[i], 1);
                                //s22=cblas_ddot(n,(*vl)[i+1],1,(*vr)[i+1], 1);
				double t,x,y;
				t=1/(s11*s11+s12*s12);
				x=.5*t*s11;
				y=.5*t*s12;
				double alp,bet;
				t=.5*std::sqrt(t);
				alp=std::sqrt(.5*(t+x));
				bet=std::sqrt(.5*(t-x));
				if(y<0.) bet= -bet;

				//rotate left ev
				memcpy(a[i],(*vl)[i],n*sizeof(double));
				cblas_dscal(n,alp,a[i],1);
				cblas_daxpy(n,-bet,(*vl)[i+1],1,a[i],1);
				memcpy(a[i+1],(*vl)[i+1],n*sizeof(double));
				cblas_dscal(n,alp,a[i+1],1);
				cblas_daxpy(n,bet,(*vl)[i],1,a[i+1],1);
				memcpy((*vl)[i],a[i],n*sizeof(double));
				memcpy((*vl)[i+1],a[i+1],n*sizeof(double));

				//rotate right ev
				memcpy(a[i],(*vr)[i],n*sizeof(double));
                                cblas_dscal(n,alp,a[i],1);
                                cblas_daxpy(n,bet,(*vr)[i+1],1,a[i],1);
                                memcpy(a[i+1],(*vr)[i+1],n*sizeof(double));
                                cblas_dscal(n,alp,a[i+1],1);
                                cblas_daxpy(n,-bet,(*vr)[i],1,a[i+1],1);
                                memcpy((*vr)[i],a[i],n*sizeof(double));
                                memcpy((*vr)[i+1],a[i+1],n*sizeof(double));

				i+=2;
				}
			}
		}

	//LAPACK saw the transposed matrix and left and right were swapped, so for a complex pair it returns the eigenvector of w_i as Re - i Im;
	//negate Im to obtain the dgeev convention Re + i Im (for both left and right eigenvectors; biorthonormality is preserved)
	{
	int i=0;
	while(i<n)
		{
		if(wi[i]==0) {++i; continue;}
		if(vl) cblas_dscal(n,-1.,(*vl)[i+1],1);
		if(vr) cblas_dscal(n,-1.,(*vr)[i+1],1);
		i+=2;
		}
	}


	if(sorttype>0)
		{
		NRVec<int> perm(n);
		for(int i=0; i<n;++i) perm[i]=i;
		gdperm= perm;
		if(beta) gdbeta<double> = *beta; else gdbeta<double> = NULL;
		gdwr<double> = wr, gdwi<double> = wi;
		genqsort(0,n-1,gdcompar<double>[sorttype-1],gdswap<double>);
		if(vl)
			{
			for(int i=0; i<n;++i) memcpy(a[i],(*vl)[perm[i]],n*sizeof(double));
			*vl |= a;
			}
		if(vr)
			{
			for(int i=0; i<n;++i) memcpy(a[i],(*vr)[perm[i]],n*sizeof(double));
			*vr |= a;
			}
		}


	if (corder) {
		if (vl) vl->transposeme(n);
		if (vr) vr->transposeme(n);
	}

}

void gdiagonalize(NRMat<float> &a, NRVec<float> &wr, NRVec<float> &wi,
		NRMat<float> *vl, NRMat<float> *vr, const bool corder, int n,
		const int sorttype, const int biorthonormalize,
		NRMat<float> *b, NRVec<float> *beta)
{
	if(n<=0) {n = a.nrows(); if(a.ncols()!=a.nrows() ) laerror("gdiagonalize() call for a non-square matrix");}
	if (n > a.ncols() || n>a.nrows() ) laerror("gdiagonalize() of too big submatrix");
	if (n > wr.size()) 
		laerror("inconsistent dimension of eigen vector in gdiagonalize()");
	if (vl) if (n > vl->nrows() || n > vl->ncols())
		laerror("inconsistent dimension of vl in gdiagonalize()");
	if (vr) if (n > vr->nrows() || n > vr->ncols())
		laerror("inconsistent dimension of vr in gdiagonalize()");
	if (beta) if(n > beta ->size()) laerror("inconsistent dimension of beta in gdiagonalize()");
	if(b) if(n > b->nrows() || n > b->ncols())
		 laerror("inconsistent dimension of b in gdiagonalize()");
	if(b && !beta || beta && !b) laerror("missing array for generalized diagonalization");

	a.copyonwrite();
	wr.copyonwrite();
	wi.copyonwrite();
	if (vl) vl->copyonwrite();
	if (vr) vr->copyonwrite();
	if (beta) beta->copyonwrite();
	if (b) b->copyonwrite();
	
	char jobvl = LAPACK_FORTRANCASE(vl ? 'v' : 'n');
	char jobvr = LAPACK_FORTRANCASE(vr ? 'v' : 'n');
	float work0;
	FINT lwork = -1;
	FINT r;
	FINT lda=a.ncols();
	FINT ldb=0;
	if(b) ldb=b->ncols();
	FINT ldvl= vl?vl->ncols():lda;
	FINT ldvr= vr?vr->ncols():lda;

#ifdef FORINT
        FINT ntmp = n; 
        if(b)  FORNAME(sggev)(&jobvr, &jobvl, &ntmp, a, &lda, *b, &ldb, wr, wi, *beta, vr?(*vr)[0]:(float *)0,
                        &ldvr, vl?(*vl)[0]:(float *)0, &ldvl, &work0, &lwork, &r);
	else FORNAME(sgeev)(&jobvr, &jobvl, &ntmp, a, &lda, wr, wi, vr?(*vr)[0]:(float *)0,
			&ldvr, vl?(*vl)[0]:(float *)0, &ldvl, &work0, &lwork, &r);
#else
        if(b)  FORNAME(sggev)(&jobvr, &jobvl, &n, a, &lda, *b, &ldb, wr, wi, *beta, vr?(*vr)[0]:(float *)0,
                        &ldvr, vl?(*vl)[0]:(float *)0, &ldvl, &work0, &lwork, &r);
	else FORNAME(sgeev)(&jobvr, &jobvl, &n, a, &lda, wr, wi, vr?(*vr)[0]:(float *)0,
			&ldvr, vl?(*vl)[0]:(float *)0, &ldvl, &work0, &lwork, &r);
#endif

        lwork = (FINT) work0;
	float *work = new float[lwork];

#ifdef FORINT	
	if(b) FORNAME(sggev)(&jobvr, &jobvl, &ntmp, a, &lda, *b, &ldb, wr, wi, *beta, vr?(*vr)[0]:(float *)0,
                        &ldvr, vl?(*vl)[0]:(float *)0, &ldvl, work, &lwork, &r);
	else FORNAME(sgeev)(&jobvr, &jobvl, &ntmp, a, &lda, wr, wi, vr?(*vr)[0]:(float *)0,
			&ldvr, vl?(*vl)[0]:(float *)0, &ldvl, work, &lwork, &r);
#else
	if(b) FORNAME(sggev)(&jobvr, &jobvl, &n, a, &lda, *b, &ldb, wr, wi, *beta, vr?(*vr)[0]:(float *)0,
                        &ldvr, vl?(*vl)[0]:(float *)0, &ldvl, work, &lwork, &r);
	else FORNAME(sgeev)(&jobvr, &jobvl, &n, a, &lda, wr, wi, vr?(*vr)[0]:(float *)0,
			&ldvr, vl?(*vl)[0]:(float *)0, &ldvl, work, &lwork, &r);
#endif
	delete[] work;


	if (r < 0) laerror("illegal argument in ggev/geev in gdiagonalize()");
	if (r > 0) laerror("convergence problem in ggev/geev in gdiagonalize()");

//std::cout <<"TEST dgeev\n"<<wr<<wi<<*vr<<*vl<<std::endl;

	if(biorthonormalize && vl && vr)
		{
		if(b || beta) laerror("@@@ biorthonormalize not implemented yet for generalized non-symmetric eigenproblem");//metric b would be needed
		int i=0;
		while(i<n)
			{
			if(wi[i]==0) //real
				{
				//calculate scaling paramter
				float tmp;
				tmp=cblas_sdot(n,(*vl)[i],1,(*vr)[i], 1);
				if(biorthonormalize==1) cblas_sscal(n,1./tmp,(*vl)[i],1);
				if(biorthonormalize==2) cblas_sscal(n,1./tmp,(*vr)[i],1);
				i++;
				}
			else //complex pair
				{
				//calculate rotation parameters
				float s11,s12;
				//float s21,s22;
				s11=cblas_sdot(n,(*vl)[i],1,(*vr)[i], 1);
				s12=cblas_sdot(n,(*vl)[i],1,(*vr)[i+1], 1);
				//s21=cblas_sdot(n,(*vl)[i+1],1,(*vr)[i], 1);
                                //s22=cblas_sdot(n,(*vl)[i+1],1,(*vr)[i+1], 1);
				float t,x,y;
				t=1/(s11*s11+s12*s12);
				x=.5*t*s11;
				y=.5*t*s12;
				float alp,bet;
				t=.5*std::sqrt(t);
				alp=std::sqrt(.5*(t+x));
				bet=std::sqrt(.5*(t-x));
				if(y<0.) bet= -bet;

				//rotate left ev
				memcpy(a[i],(*vl)[i],n*sizeof(float));
				cblas_sscal(n,alp,a[i],1);
				cblas_saxpy(n,-bet,(*vl)[i+1],1,a[i],1);
				memcpy(a[i+1],(*vl)[i+1],n*sizeof(float));
				cblas_sscal(n,alp,a[i+1],1);
				cblas_saxpy(n,bet,(*vl)[i],1,a[i+1],1);
				memcpy((*vl)[i],a[i],n*sizeof(float));
				memcpy((*vl)[i+1],a[i+1],n*sizeof(float));

				//rotate right ev
				memcpy(a[i],(*vr)[i],n*sizeof(float));
                                cblas_sscal(n,alp,a[i],1);
                                cblas_saxpy(n,bet,(*vr)[i+1],1,a[i],1);
                                memcpy(a[i+1],(*vr)[i+1],n*sizeof(float));
                                cblas_sscal(n,alp,a[i+1],1);
                                cblas_saxpy(n,-bet,(*vr)[i],1,a[i+1],1);
                                memcpy((*vr)[i],a[i],n*sizeof(float));
                                memcpy((*vr)[i+1],a[i+1],n*sizeof(float));

				i+=2;
				}
			}
		}

	//LAPACK saw the transposed matrix and left and right were swapped, so for a complex pair it returns the eigenvector of w_i as Re - i Im;
	//negate Im to obtain the dgeev convention Re + i Im (for both left and right eigenvectors; biorthonormality is preserved)
	{
	int i=0;
	while(i<n)
		{
		if(wi[i]==0) {++i; continue;}
		if(vl) cblas_sscal(n,-1.,(*vl)[i+1],1);
		if(vr) cblas_sscal(n,-1.,(*vr)[i+1],1);
		i+=2;
		}
	}


	if(sorttype>0)
		{
		NRVec<int> perm(n);
		for(int i=0; i<n;++i) perm[i]=i;
		gdperm= perm;
		if(beta) gdbeta<float> = *beta; else gdbeta<float> = NULL;
		gdwr<float> = wr, gdwi<float> = wi;
		genqsort(0,n-1,gdcompar<float>[sorttype-1],gdswap<float>);
		if(vl)
			{
			for(int i=0; i<n;++i) memcpy(a[i],(*vl)[perm[i]],n*sizeof(float));
			*vl |= a;
			}
		if(vr)
			{
			for(int i=0; i<n;++i) memcpy(a[i],(*vr)[perm[i]],n*sizeof(float));
			*vr |= a;
			}
		}


	if (corder) {
		if (vl) vl->transposeme(n);
		if (vr) vr->transposeme(n);
	}

}



//most general complex routine
template<>
void gdiagonalize(NRMat<std::complex<double> > &a, NRVec< std::complex<double> > &w,
		NRMat< std::complex<double> >*vl, NRMat< std::complex<double> > *vr,
		const bool corder, int n, const int sorttype, const int biorthonormalize,
		NRMat<std::complex<double> > *b, NRVec<std::complex<double> > *beta)
{

	if(n<=0) {n = a.nrows(); if(a.ncols()!=a.nrows() ) laerror("gdiagonalize() call for a non-square matrix");}
	if (n > a.ncols() || n>a.nrows() ) laerror("gdiagonalize() of too big submatrix");
	if (n > w.size()) 
		laerror("inconsistent dimension of eigen vector in gdiagonalize()");
	if (vl) if (n > vl->nrows() || n > vl->ncols())
		laerror("inconsistent dimension of vl in gdiagonalize()");
	if (vr) if (n > vr->nrows() || n > vr->ncols())
		laerror("inconsistent dimension of vr in gdiagonalize()");
	if (beta) if(n > beta ->size()) laerror("inconsistent dimension of beta in gdiagonalize()");
	if(b) if(n > b->nrows() || n > b->ncols())
		 laerror("inconsistent dimension of b in gdiagonalize()");
	if(b && !beta || beta && !b) laerror("missing array for generalized diagonalization");

	a.copyonwrite();
	w.copyonwrite();
	if (vl) vl->copyonwrite();
	if (vr) vr->copyonwrite();
	if (beta) beta->copyonwrite();
	if (b) b->copyonwrite();
	
	char jobvl = LAPACK_FORTRANCASE(vl ? 'v' : 'n');
	char jobvr = LAPACK_FORTRANCASE(vr ? 'v' : 'n');
	std::complex<double> work0;
	FINT lwork = -1;
	FINT r;
	FINT lda=a.ncols();
	FINT ldb=0;
	if(b) ldb=b->ncols();
	FINT ldvl= vl?vl->ncols():lda;
	FINT ldvr= vr?vr->ncols():lda;

	double *rwork = new double[n*(b?8:2)];

#ifdef FORINT
        FINT ntmp = n; 
        if(b)  FORNAME(zggev)(&jobvr, &jobvl, &ntmp, a, &lda, *b, &ldb, w, *beta, vr?(*vr)[0]:(std::complex<double> *)0,
                        &ldvr, vl?(*vl)[0]:(std::complex<double> *)0, &ldvl, &work0, &lwork, rwork, &r);
	else FORNAME(zgeev)(&jobvr, &jobvl, &ntmp, a, &lda, w, vr?(*vr)[0]:(std::complex<double> *)0,
			&ldvr, vl?(*vl)[0]:(std::complex<double> *)0, &ldvl, &work0, &lwork, rwork, &r);
#else
        if(b)  FORNAME(zggev)(&jobvr, &jobvl, &n, a, &lda, *b, &ldb, w, *beta, vr?(*vr)[0]:(std::complex<double> *)0,
                        &ldvr, vl?(*vl)[0]:(std::complex<double> *)0, &ldvl, &work0, &lwork, rwork, &r);
	else FORNAME(zgeev)(&jobvr, &jobvl, &n, a, &lda, w, vr?(*vr)[0]:(std::complex<double> *)0,
			&ldvr, vl?(*vl)[0]:(std::complex<double> *)0, &ldvl, &work0, &lwork, rwork, &r);
#endif

        lwork = (FINT) work0.real();
	std::complex<double> *work = new std::complex<double>[lwork];

#ifdef FORINT
        if(b)  FORNAME(zggev)(&jobvr, &jobvl, &ntmp, a, &lda, *b, &ldb, w, *beta, vr?(*vr)[0]:(std::complex<double> *)0,
                        &ldvr, vl?(*vl)[0]:(std::complex<double> *)0, &ldvl, work, &lwork, rwork, &r);
	else FORNAME(zgeev)(&jobvr, &jobvl, &ntmp, a, &lda, w, vr?(*vr)[0]:(std::complex<double> *)0,
			&ldvr, vl?(*vl)[0]:(std::complex<double> *)0, &ldvl, work, &lwork, rwork, &r);
#else
        if(b)  FORNAME(zggev)(&jobvr, &jobvl, &n, a, &lda, *b, &ldb, w, *beta, vr?(*vr)[0]:(std::complex<double> *)0,
                        &ldvr, vl?(*vl)[0]:(std::complex<double> *)0, &ldvl, work, &lwork, rwork, &r);
	else FORNAME(zgeev)(&jobvr, &jobvl, &n, a, &lda, w, vr?(*vr)[0]:(std::complex<double> *)0,
			&ldvr, vl?(*vl)[0]:(std::complex<double> *)0, &ldvl, work, &lwork, rwork, &r);
#endif

	delete[] work;
	delete[] rwork;

//std::cout <<"TEST zg(g|e)ev\n"<<w<<*vr<<*vl<<std::endl;

	if (r < 0) laerror("illegal argument in ggev/geev in gdiagonalize()");
	if (r > 0) laerror("convergence problem in ggev/geev in gdiagonalize()");

	if(biorthonormalize && vl && vr)
		{
		if(b || beta) laerror("@@@ biorthonormalize not implemented yet for generalized non-hermitian eigenproblem");//metric b would be needed
		for(int i=0; i<n; ++i)
			{
			//calculate scaling paramter
			std::complex<double> tmp;
			cblas_zdotc_sub(n,(*vr)[i],1,(*vl)[i], 1, &tmp);
			tmp = 1./tmp;
			if(biorthonormalize==1) cblas_zscal(n,&tmp,(*vl)[i],1);
			if(biorthonormalize==2) cblas_zscal(n,&tmp,(*vr)[i],1);
			}
		}

	if(sorttype>0)
		{
		laerror("sorting not implemented in complex gdiagonalize");
		}


	if (corder) {
		if (vl) {vl->transposeme(n); vl->conjugateme();}
		if (vr) {vr->transposeme(n); vr->conjugateme();}
	}

}

template<>
void gdiagonalize(NRMat<std::complex<float> > &a, NRVec< std::complex<float> > &w,
		NRMat< std::complex<float> >*vl, NRMat< std::complex<float> > *vr,
		const bool corder, int n, const int sorttype, const int biorthonormalize,
		NRMat<std::complex<float> > *b, NRVec<std::complex<float> > *beta)
{

	if(n<=0) {n = a.nrows(); if(a.ncols()!=a.nrows() ) laerror("gdiagonalize() call for a non-square matrix");}
	if (n > a.ncols() || n>a.nrows() ) laerror("gdiagonalize() of too big submatrix");
	if (n > w.size()) 
		laerror("inconsistent dimension of eigen vector in gdiagonalize()");
	if (vl) if (n > vl->nrows() || n > vl->ncols())
		laerror("inconsistent dimension of vl in gdiagonalize()");
	if (vr) if (n > vr->nrows() || n > vr->ncols())
		laerror("inconsistent dimension of vr in gdiagonalize()");
	if (beta) if(n > beta ->size()) laerror("inconsistent dimension of beta in gdiagonalize()");
	if(b) if(n > b->nrows() || n > b->ncols())
		 laerror("inconsistent dimension of b in gdiagonalize()");
	if(b && !beta || beta && !b) laerror("missing array for generalized diagonalization");

	a.copyonwrite();
	w.copyonwrite();
	if (vl) vl->copyonwrite();
	if (vr) vr->copyonwrite();
	if (beta) beta->copyonwrite();
	if (b) b->copyonwrite();
	
	char jobvl = LAPACK_FORTRANCASE(vl ? 'v' : 'n');
	char jobvr = LAPACK_FORTRANCASE(vr ? 'v' : 'n');
	std::complex<float> work0;
	FINT lwork = -1;
	FINT r;
	FINT lda=a.ncols();
	FINT ldb=0;
	if(b) ldb=b->ncols();
	FINT ldvl= vl?vl->ncols():lda;
	FINT ldvr= vr?vr->ncols():lda;

	float *rwork = new float[n*(b?8:2)];

#ifdef FORINT
        FINT ntmp = n; 
        if(b)  FORNAME(cggev)(&jobvr, &jobvl, &ntmp, a, &lda, *b, &ldb, w, *beta, vr?(*vr)[0]:(std::complex<float> *)0,
                        &ldvr, vl?(*vl)[0]:(std::complex<float> *)0, &ldvl, &work0, &lwork, rwork, &r);
	else FORNAME(cgeev)(&jobvr, &jobvl, &ntmp, a, &lda, w, vr?(*vr)[0]:(std::complex<float> *)0,
			&ldvr, vl?(*vl)[0]:(std::complex<float> *)0, &ldvl, &work0, &lwork, rwork, &r);
#else
        if(b)  FORNAME(cggev)(&jobvr, &jobvl, &n, a, &lda, *b, &ldb, w, *beta, vr?(*vr)[0]:(std::complex<float> *)0,
                        &ldvr, vl?(*vl)[0]:(std::complex<float> *)0, &ldvl, &work0, &lwork, rwork, &r);
	else FORNAME(cgeev)(&jobvr, &jobvl, &n, a, &lda, w, vr?(*vr)[0]:(std::complex<float> *)0,
			&ldvr, vl?(*vl)[0]:(std::complex<float> *)0, &ldvl, &work0, &lwork, rwork, &r);
#endif

        lwork = (FINT) work0.real();
	std::complex<float> *work = new std::complex<float>[lwork];

#ifdef FORINT
        if(b)  FORNAME(cggev)(&jobvr, &jobvl, &ntmp, a, &lda, *b, &ldb, w, *beta, vr?(*vr)[0]:(std::complex<float> *)0,
                        &ldvr, vl?(*vl)[0]:(std::complex<float> *)0, &ldvl, work, &lwork, rwork, &r);
	else FORNAME(cgeev)(&jobvr, &jobvl, &ntmp, a, &lda, w, vr?(*vr)[0]:(std::complex<float> *)0,
			&ldvr, vl?(*vl)[0]:(std::complex<float> *)0, &ldvl, work, &lwork, rwork, &r);
#else
        if(b)  FORNAME(cggev)(&jobvr, &jobvl, &n, a, &lda, *b, &ldb, w, *beta, vr?(*vr)[0]:(std::complex<float> *)0,
                        &ldvr, vl?(*vl)[0]:(std::complex<float> *)0, &ldvl, work, &lwork, rwork, &r);
	else FORNAME(cgeev)(&jobvr, &jobvl, &n, a, &lda, w, vr?(*vr)[0]:(std::complex<float> *)0,
			&ldvr, vl?(*vl)[0]:(std::complex<float> *)0, &ldvl, work, &lwork, rwork, &r);
#endif

	delete[] work;
	delete[] rwork;

//std::cout <<"TEST zg(g|e)ev\n"<<w<<*vr<<*vl<<std::endl;

	if (r < 0) laerror("illegal argument in ggev/geev in gdiagonalize()");
	if (r > 0) laerror("convergence problem in ggev/geev in gdiagonalize()");

	if(biorthonormalize && vl && vr)
		{
		if(b || beta) laerror("@@@ biorthonormalize not implemented yet for generalized non-hermitian eigenproblem");//metric b would be needed
		for(int i=0; i<n; ++i)
			{
			//calculate scaling paramter
			std::complex<float> tmp;
			cblas_cdotc_sub(n,(*vr)[i],1,(*vl)[i], 1, &tmp);
			tmp = 1.f/tmp;
			if(biorthonormalize==1) cblas_cscal(n,&tmp,(*vl)[i],1);
			if(biorthonormalize==2) cblas_cscal(n,&tmp,(*vr)[i],1);
			}
		}

	if(sorttype>0)
		{
		laerror("sorting not implemented in complex gdiagonalize");
		}


	if (corder) {
		if (vl) {vl->transposeme(n); vl->conjugateme();}
		if (vr) {vr->transposeme(n); vr->conjugateme();}
	}

}


template<>
void gdiagonalize(NRMat<double> &a, NRVec< std::complex<double> > &w,
		NRMat< std::complex<double> >*vl, NRMat< std::complex<double> > *vr,
		const bool corder, int n, const int sorttype, const int biorthonormalize,
		NRMat<double> *b, NRVec<double> *beta)
{
	if(n<=0) {n = a.nrows(); if(a.ncols()!=a.nrows() ) laerror("gdiagonalize() call for a non-square matrix");}
	if(n> a.nrows() || n ==  a.nrows() && n != a.ncols()) laerror("gdiagonalize() of too big submatrix");

	NRVec<double> wr(n), wi(n);
	NRMat<double> *rvl = 0;
	NRMat<double> *rvr = 0;
	if (vl) rvl = new NRMat<double>(n, n);
	if (vr) rvr = new NRMat<double>(n, n);
	gdiagonalize(a, wr, wi, rvl, rvr, 0, n, sorttype, biorthonormalize, b, beta);
	
	//process the results into complex matrices
	int i;
	for (i=0; i<n; i++) w[i] = std::complex<double>(wr[i], wi[i]);
	if (rvl || rvr) {
		i = 0;
		while (i < n) {
			if (wi[i] == 0) {
			    if(corder)
				{
                                if (vl) for (int j=0; j<n; j++) (*vl)[j][i] = (*rvl)[i][j];
                                if (vr) for (int j=0; j<n; j++) (*vr)[j][i] = (*rvr)[i][j];
				}
			    else
				{
				if (vl) for (int j=0; j<n; j++) (*vl)[i][j] = (*rvl)[i][j];
				if (vr) for (int j=0; j<n; j++) (*vr)[i][j] = (*rvr)[i][j];
				}
				i++;
			} else {
				if (vl)
					for (int j=0; j<n; j++) {
					    if(corder)
						{
                                                (*vl)[j][i] = std::complex<double>((*rvl)[i][j], (*rvl)[i+1][j]);
                                                (*vl)[j][i+1] = std::complex<double>((*rvl)[i][j], -(*rvl)[i+1][j]);
						}
						else
						{
						(*vl)[i][j] = std::complex<double>((*rvl)[i][j], -(*rvl)[i+1][j]); //rows are the conjugated eigenvectors
						(*vl)[i+1][j] = std::complex<double>((*rvl)[i][j], (*rvl)[i+1][j]);
						}
					} 
				if (vr)
					for (int j=0; j<n; j++) {
					    if(corder)
                                                {
                                                (*vr)[j][i] = std::complex<double>((*rvr)[i][j], (*rvr)[i+1][j]);
                                                (*vr)[j][i+1] = std::complex<double>((*rvr)[i][j], -(*rvr)[i+1][j]);
                                                }
                                                else
                                                {
						(*vr)[i][j] = std::complex<double>((*rvr)[i][j], -(*rvr)[i+1][j]); //rows are the conjugated eigenvectors
						(*vr)[i+1][j] = std::complex<double>((*rvr)[i][j], (*rvr)[i+1][j]);
						}
					}
				i += 2;
			}
		}
	}
	if (rvl) delete rvl;
	if (rvr) delete rvr;
}

template<>
void gdiagonalize(NRMat<float> &a, NRVec< std::complex<float> > &w,
		NRMat< std::complex<float> >*vl, NRMat< std::complex<float> > *vr,
		const bool corder, int n, const int sorttype, const int biorthonormalize,
		NRMat<float> *b, NRVec<float> *beta)
{
	if(n<=0) {n = a.nrows(); if(a.ncols()!=a.nrows() ) laerror("gdiagonalize() call for a non-square matrix");}
	if(n> a.nrows() || n ==  a.nrows() && n != a.ncols()) laerror("gdiagonalize() of too big submatrix");

	NRVec<float> wr(n), wi(n);
	NRMat<float> *rvl = 0;
	NRMat<float> *rvr = 0;
	if (vl) rvl = new NRMat<float>(n, n);
	if (vr) rvr = new NRMat<float>(n, n);
	gdiagonalize(a, wr, wi, rvl, rvr, 0, n, sorttype, biorthonormalize, b, beta);
	
	//process the results into complex matrices
	int i;
	for (i=0; i<n; i++) w[i] = std::complex<float>(wr[i], wi[i]);
	if (rvl || rvr) {
		i = 0;
		while (i < n) {
			if (wi[i] == 0) {
			    if(corder)
				{
                                if (vl) for (int j=0; j<n; j++) (*vl)[j][i] = (*rvl)[i][j];
                                if (vr) for (int j=0; j<n; j++) (*vr)[j][i] = (*rvr)[i][j];
				}
			    else
				{
				if (vl) for (int j=0; j<n; j++) (*vl)[i][j] = (*rvl)[i][j];
				if (vr) for (int j=0; j<n; j++) (*vr)[i][j] = (*rvr)[i][j];
				}
				i++;
			} else {
				if (vl)
					for (int j=0; j<n; j++) {
					    if(corder)
						{
                                                (*vl)[j][i] = std::complex<float>((*rvl)[i][j], (*rvl)[i+1][j]);
                                                (*vl)[j][i+1] = std::complex<float>((*rvl)[i][j], -(*rvl)[i+1][j]);
						}
						else
						{
						(*vl)[i][j] = std::complex<float>((*rvl)[i][j], -(*rvl)[i+1][j]); //rows are the conjugated eigenvectors
						(*vl)[i+1][j] = std::complex<float>((*rvl)[i][j], (*rvl)[i+1][j]);
						}
					} 
				if (vr)
					for (int j=0; j<n; j++) {
					    if(corder)
                                                {
                                                (*vr)[j][i] = std::complex<float>((*rvr)[i][j], (*rvr)[i+1][j]);
                                                (*vr)[j][i+1] = std::complex<float>((*rvr)[i][j], -(*rvr)[i+1][j]);
                                                }
                                                else
                                                {
						(*vr)[i][j] = std::complex<float>((*rvr)[i][j], -(*rvr)[i+1][j]); //rows are the conjugated eigenvectors
						(*vr)[i+1][j] = std::complex<float>((*rvr)[i][j], (*rvr)[i+1][j]);
						}
					}
				i += 2;
			}
		}
	}
	if (rvl) delete rvl;
	if (rvr) delete rvr;
}


//for compatibility in davidson
void gdiagonalize(NRMat<std::complex<double> > &a, NRVec<double> &wr, NRVec<double> &wi,
                NRMat<std::complex<double> > *vl, NRMat<std::complex<double> > *vr, const bool corder, int n, const int sorttype, const int biorthonormalize,
                NRMat<std::complex<double> > *b, NRVec<std::complex<double> > *beta)
{
if(wr.size()!=wi.size()) laerror("length mismatch in gdiagonalize");
NRVec<std::complex<double> > w(wr.size());
gdiagonalize(a,w,vl,vr,corder,n,sorttype,biorthonormalize,b,beta);
wr.copyonwrite();
wi.copyonwrite();
for(int i=0; i<w.size(); ++i)
	{
	wr[i]=w[i].real();
	wi[i]=w[i].imag();
	}
}

void gdiagonalize(NRMat<std::complex<float> > &a, NRVec<float> &wr, NRVec<float> &wi,
                NRMat<std::complex<float> > *vl, NRMat<std::complex<float> > *vr, const bool corder, int n, const int sorttype, const int biorthonormalize,
                NRMat<std::complex<float> > *b, NRVec<std::complex<float> > *beta)
{
if(wr.size()!=wi.size()) laerror("length mismatch in gdiagonalize");
NRVec<std::complex<float> > w(wr.size());
gdiagonalize(a,w,vl,vr,corder,n,sorttype,biorthonormalize,b,beta);
wr.copyonwrite();
wi.copyonwrite();
for(int i=0; i<w.size(); ++i)
	{
	wr[i]=w[i].real();
	wi[i]=w[i].imag();
	}
}

template<>
const SparseMat<double> realpart<SparseMat< std::complex<double> > >(const SparseMat< std::complex<double> > &a)
{
SparseMat<double> result(a.nrows(), a.ncols());
for(SparseMat< std::complex<double> >::iterator p=a.begin(); p!=a.end(); ++p)
	{
	result.add(p->row,p->col,p->elem.real());
	}
if(a.issymmetric()) result.setsymmetric();
return result;
}

template<>
const SparseMat<double> imagpart<SparseMat< std::complex<double> > >(const SparseMat< std::complex<double> > &a)
{
SparseMat<double> result(a.nrows(), a.ncols());
for(SparseMat< std::complex<double> >::iterator p=a.begin(); p!=a.end(); ++p)
	{
	result.add(p->row,p->col,p->elem.imag());
	}
if(a.issymmetric()) result.setsymmetric();
return result;
}

template<>
const NRMat<double> realpart<NRMat< std::complex<double> > >(const NRMat< std::complex<double> > &a)
{
        NRMat<double> result(a.nrows(), a.ncols());

#ifdef CUDALA
        if(a.location == cpu){
#endif
//	NRMat<double> result(a.nrows(), a.ncols());
	cblas_dcopy(a.nrows()*a.ncols(), (const double *)a[0], 2, result, 1);
#ifdef CUDALA
        }else{
	laerror("not implemented for cuda yet");
	}
#endif
	return result;
}

template<>
const NRMat<float> realpart<NRMat< std::complex<float> > >(const NRMat< std::complex<float> > &a)
{
        NRMat<float> result(a.nrows(), a.ncols());

#ifdef CUDALA
        if(a.location == cpu){
#endif
//	NRMat<float> result(a.nrows(), a.ncols());
	cblas_scopy(a.nrows()*a.ncols(), (const float *)a[0], 2, result, 1);
#ifdef CUDALA
        }else{
	laerror("not implemented for cuda yet");
	}
#endif
	return result;
}

template<>
const NRMat<double> imagpart<NRMat< std::complex<double> > >(const NRMat< std::complex<double> > &a)
{
        NRMat<double> result(a.nrows(), a.ncols());

#ifdef CUDALA
        if(a.location == cpu){
#endif

//	NRMat<double> result(a.nrows(), a.ncols());
	cblas_dcopy(a.nrows()*a.ncols(), (const double *)a[0]+1, 2, result, 1);
#ifdef CUDALA
        }else{
        laerror("not implemented for cuda yet");
        }
#endif
	return result;
}

template<>
const NRMat<float> imagpart<NRMat< std::complex<float> > >(const NRMat< std::complex<float> > &a)
{
        NRMat<float> result(a.nrows(), a.ncols());

#ifdef CUDALA
        if(a.location == cpu){
#endif

//	NRMat<float> result(a.nrows(), a.ncols());
	cblas_scopy(a.nrows()*a.ncols(), (const float *)a[0]+1, 2, result, 1);
#ifdef CUDALA
        }else{
        laerror("not implemented for cuda yet");
        }
#endif
	return result;
}

template<>
const NRSMat<double> realpart<NRSMat< std::complex<double> > >(const NRSMat< std::complex<double> > &a)
{
        NRSMat<double> result(a.nrows());

#ifdef CUDALA
        if(a.location == cpu){
#endif
	cblas_dcopy(a.nrows()*(a.ncols()+1)/2, (const double *)(&a[0]), 2, result, 1);
#ifdef CUDALA
        }else{
	laerror("not implemented for cuda yet");
	}
#endif
	return result;
}

template<>
const NRSMat<float> realpart<NRSMat< std::complex<float> > >(const NRSMat< std::complex<float> > &a)
{
        NRSMat<float> result(a.nrows());

#ifdef CUDALA
        if(a.location == cpu){
#endif
	cblas_scopy(a.nrows()*(a.ncols()+1)/2, (const float *)(&a[0]), 2, result, 1);
#ifdef CUDALA
        }else{
	laerror("not implemented for cuda yet");
	}
#endif
	return result;
}

template<>
const NRSMat<double> imagpart<NRSMat< std::complex<double> > >(const NRSMat< std::complex<double> > &a)
{
        NRSMat<double> result(a.nrows());

#ifdef CUDALA
        if(a.location == cpu){
#endif

	cblas_dcopy(a.nrows()*(a.ncols()+1)/2, (const double *)(&a[0])+1, 2, result, 1);
#ifdef CUDALA
        }else{
        laerror("not implemented for cuda yet");
        }
#endif
	return result;
}

template<>
const NRSMat<float> imagpart<NRSMat< std::complex<float> > >(const NRSMat< std::complex<float> > &a)
{
        NRSMat<float> result(a.nrows());

#ifdef CUDALA
        if(a.location == cpu){
#endif

	cblas_scopy(a.nrows()*(a.ncols()+1)/2, (const float *)(&a[0])+1, 2, result, 1);
#ifdef CUDALA
        }else{
        laerror("not implemented for cuda yet");
        }
#endif
	return result;
}

template<>
const NRVec<double> realpart<NRVec< std::complex<double> > >(const NRVec< std::complex<double> > &a)
{
        NRVec<double> result(a.size());

#ifdef CUDALA
        if(a.location == cpu){
#endif
	cblas_dcopy(a.size(), (const double *)&a[0], 2, result, 1);
#ifdef CUDALA
        }else{
	laerror("not implemented for cuda yet");
	}
#endif
	return result;
}

template<>
const NRVec<float> realpart<NRVec< std::complex<float> > >(const NRVec< std::complex<float> > &a)
{
        NRVec<float> result(a.size());

#ifdef CUDALA
        if(a.location == cpu){
#endif
	cblas_scopy(a.size(), (const float *)&a[0], 2, result, 1);
#ifdef CUDALA
        }else{
	laerror("not implemented for cuda yet");
	}
#endif
	return result;
}

template<>
const NRVec<double> imagpart<NRVec< std::complex<double> > >(const NRVec< std::complex<double> > &a)
{
        NRVec<double> result(a.size());

#ifdef CUDALA
        if(a.location == cpu){
#endif

	cblas_dcopy(a.size(), (const double *)&a[0]+1, 2, result, 1);
#ifdef CUDALA
        }else{
        laerror("not implemented for cuda yet");
        }
#endif
	return result;
}

template<>
const NRVec<float> imagpart<NRVec< std::complex<float> > >(const NRVec< std::complex<float> > &a)
{
        NRVec<float> result(a.size());

#ifdef CUDALA
        if(a.location == cpu){
#endif

	cblas_scopy(a.size(), (const float *)&a[0]+1, 2, result, 1);
#ifdef CUDALA
        }else{
        laerror("not implemented for cuda yet");
        }
#endif
	return result;
}



template<>
const NRMat< std::complex<double> > realmatrix<NRMat<double> > (const NRMat<double> &a)
{

        NRMat <std::complex<double> > result(a.nrows(), a.ncols());

#ifdef CUDALA
        if(a.location == cpu){
#endif


//	NRMat <std::complex<double> > result(a.nrows(), a.ncols());
	cblas_dcopy(a.nrows()*a.ncols(), a, 1, (double *)result[0], 2);
#ifdef CUDALA
        }else{
        laerror("not implemented for cuda yet");
        }
#endif

	return result;
}

template<>
const NRMat< std::complex<float> > realmatrix<NRMat<float> > (const NRMat<float> &a)
{

        NRMat <std::complex<float> > result(a.nrows(), a.ncols());

#ifdef CUDALA
        if(a.location == cpu){
#endif


//	NRMat <std::complex<float> > result(a.nrows(), a.ncols());
	cblas_scopy(a.nrows()*a.ncols(), a, 1, (float *)result[0], 2);
#ifdef CUDALA
        }else{
        laerror("not implemented for cuda yet");
        }
#endif

	return result;
}

template<>
const NRMat< std::complex<double> > imagmatrix<NRMat<double> > (const NRMat<double> &a)
{
        NRMat< std::complex<double> > result(a.nrows(), a.ncols());

#ifdef CUDALA
        if(a.location == cpu){
#endif

//	NRMat< std::complex<double> > result(a.nrows(), a.ncols());
	cblas_dcopy(a.nrows()*a.ncols(), a, 1, (double *)result[0]+1, 2);
#ifdef CUDALA
        }else{
        laerror("not implemented for cuda yet");
        }
#endif

	return result;
}

template<>
const NRMat< std::complex<float> > imagmatrix<NRMat<float> > (const NRMat<float> &a)
{
        NRMat< std::complex<float> > result(a.nrows(), a.ncols());

#ifdef CUDALA
        if(a.location == cpu){
#endif

//	NRMat< std::complex<float> > result(a.nrows(), a.ncols());
	cblas_scopy(a.nrows()*a.ncols(), a, 1, (float *)result[0]+1, 2);
#ifdef CUDALA
        }else{
        laerror("not implemented for cuda yet");
        }
#endif

	return result;
}

template<>
const NRMat< std::complex<double> > complexmatrix<NRMat<double> > (const NRMat<double> &re, const NRMat<double> &im)
{
	if(re.nrows()!=im.nrows() || re.ncols() != im.ncols()) laerror("incompatible sizes of real and imaginary parts");
        NRMat< std::complex<double> > result(re.nrows(), re.ncols());
        cblas_dcopy(re.nrows()*re.ncols(), re, 1, (double *)result[0], 2);
        cblas_dcopy(re.nrows()*re.ncols(), im, 1, (double *)result[0]+1, 2);
        return result;
}

template<>
const NRMat< std::complex<float> > complexmatrix<NRMat<float> > (const NRMat<float> &re, const NRMat<float> &im)
{
	if(re.nrows()!=im.nrows() || re.ncols() != im.ncols()) laerror("incompatible sizes of real and imaginary parts");
        NRMat< std::complex<float> > result(re.nrows(), re.ncols());
        cblas_scopy(re.nrows()*re.ncols(), re, 1, (float *)result[0], 2);
        cblas_scopy(re.nrows()*re.ncols(), im, 1, (float *)result[0]+1, 2);
        return result;
}

template<>
const SparseSMat< std::complex<double> > complexmatrix<SparseSMat<double> >(const SparseSMat<double> &re, const SparseSMat<double> &im) {
  if(re.nrows()!=im.nrows() || re.ncols() != im.ncols()) laerror("incompatible sizes of real and imaginary parts");
  SparseSMat< std::complex<double> > result(re.nrows(),re.ncols());
  std::complex<double> tmp;

  SparseSMat<double>::iterator pre(re);
  for(; pre.notend(); ++pre) {
    tmp = pre->elem;
    result.add(pre->row,pre->col,tmp,false);
  }

  SparseSMat<double>::iterator pim(im);
  for(; pim.notend(); ++pim) {
    tmp = std::complex<double>(0,1)*(pim->elem);
    result.add(pim->row,pim->col,tmp,false);
  }

  return result;
}

template<>
const SparseSMat< std::complex<double> > realmatrix<SparseSMat<double> >(const SparseSMat<double> &re) {
  SparseSMat< std::complex<double> > result(re.nrows(),re.ncols());
  std::complex<double> tmp;

  SparseSMat<double>::iterator pre(re);
  for(; pre.notend(); ++pre) {
    tmp = pre->elem;
    result.add(pre->row,pre->col,tmp,false);
  }

  return result;
}

template<>
const SparseSMat< std::complex<double> > imagmatrix<SparseSMat<double> >(const SparseSMat<double> &im) {
  SparseSMat< std::complex<double> > result(im.nrows(),im.ncols());
  std::complex<double> tmp;


  SparseSMat<double>::iterator pim(im);
  for(; pim.notend(); ++pim) {
    tmp = std::complex<double>(0,1)*(pim->elem);
    result.add(pim->row,pim->col,tmp,false);
  }

  return result;
}





NRMat<double> realmatrixfunction(NRMat<double> a, double (*f) (const double))
{
        int n = a.nrows();
        NRVec<double> w(n);
        diagonalize(a, w, true, false);

        for (int i=0; i<a.nrows(); i++) w[i] = (*f)(w[i]);
        NRMat<double> u = a;
        a.diagmultl(w);
        NRMat<double> r(n, n);
        r.gemm(0.0, u, 't', a, 'n', 1.0);
        return r;
}

NRMat<float> realmatrixfunction(NRMat<float> a, float (*f) (const float))
{
        int n = a.nrows();
        NRVec<float> w(n);
        diagonalize(a, w, true, false);

        for (int i=0; i<a.nrows(); i++) w[i] = (*f)(w[i]);
        NRMat<float> u = a;
        a.diagmultl(w);
        NRMat<float> r(n, n);
        r.gemm(0.0, u, 't', a, 'n', 1.0);
        return r;
}


NRMat<std::complex<double> > complexmatrixfunction(NRMat<double> a, double (*fre) (const double), double (*fim) (const double))
{
        int n = a.nrows();
        NRVec<double> wre(n),wim(n);
        diagonalize(a, wre, true, false);
	for (int i=0; i<a.nrows(); i++) wim[i] = (*fim)(wre[i]);
        for (int i=0; i<a.nrows(); i++) wre[i] = (*fre)(wre[i]);
        NRMat<double> u = a;
	NRMat<double> b = a;
        a.diagmultl(wre);
        b.diagmultl(wim);
	NRMat<double> t(n,n),tt(n,n);
        t.gemm(0.0, u, 't', a, 'n', 1.0);
	tt.gemm(0.0, u, 't', b, 'n', 1.0);
        NRMat<std::complex<double> > r(n, n);
	for (int i=0; i<a.nrows(); i++) for(int j=0; j<a.ncols(); ++j) r(i,j)=std::complex<double>(t(i,j),tt(i,j));
        return r;
}

NRMat<std::complex<float> > complexmatrixfunction(NRMat<float> a, float (*fre) (const float), float (*fim) (const float))
{
        int n = a.nrows();
        NRVec<float> wre(n),wim(n);
        diagonalize(a, wre, true, false);
	for (int i=0; i<a.nrows(); i++) wim[i] = (*fim)(wre[i]);
        for (int i=0; i<a.nrows(); i++) wre[i] = (*fre)(wre[i]);
        NRMat<float> u = a;
	NRMat<float> b = a;
        a.diagmultl(wre);
        b.diagmultl(wim);
	NRMat<float> t(n,n),tt(n,n);
        t.gemm(0.0, u, 't', a, 'n', 1.0);
	tt.gemm(0.0, u, 't', b, 'n', 1.0);
        NRMat<std::complex<float> > r(n, n);
	for (int i=0; i<a.nrows(); i++) for(int j=0; j<a.ncols(); ++j) r(i,j)=std::complex<float>(t(i,j),tt(i,j));
        return r;
}





// instantize template to an addresable function
std::complex<double> myccopy (const std::complex<double> &x) 
{
	return x;
}

std::complex<float> myccopy (const std::complex<float> &x) 
{
	return x;
}

double mycopy (const double x) 
{
	return x;
}

float mycopy (const float x) 
{
	return x;
}

std::complex<double> myclog (const std::complex<double> &x) 
{
	return log(x);
}

std::complex<float> myclog (const std::complex<float> &x) 
{
	return log(x);
}

std::complex<double> mycexp (const std::complex<double> &x)
{
        return std::exp(x);
}

std::complex<float> mycexp (const std::complex<float> &x)
{
        return std::exp(x);
}


std::complex<double> sqrtinv (const std::complex<double> &x)
{
        return 1./std::sqrt(x);
}

std::complex<float> sqrtinv (const std::complex<float> &x)
{
        return 1.f/std::sqrt(x);
}

double sqrtinv (const double x)
{
        return 1./std::sqrt(x);
}

float sqrtinv (const float x)
{
        return 1./std::sqrt(x);
}


NRMat<double>  log(const NRMat<double> &a)
{
	return matrixfunction(a, &myclog);
}

NRMat<float>  log(const NRMat<float> &a)
{
	return matrixfunction(a, &myclog);
}

NRMat<std::complex<double> >  log(const NRMat<std::complex<double> > &a)
{
        return matrixfunction(a, &myclog);
}

NRMat<std::complex<float> >  log(const NRMat<std::complex<float> > &a)
{
        return matrixfunction(a, &myclog);
}


NRMat<double>  exp0(const NRMat<double> &a)
{
        return matrixfunction(a, &mycexp);
}

NRMat<float>  exp0(const NRMat<float> &a)
{
        return matrixfunction(a, &mycexp);
}

NRMat<std::complex<double> >  exp0(const NRMat<std::complex<double> > &a)
{
        return matrixfunction(a, &mycexp);
}

NRMat<std::complex<float> >  exp0(const NRMat<std::complex<float> > &a)
{
        return matrixfunction(a, &mycexp);
}

NRMat<std::complex<double> >  copytest(const NRMat<std::complex<double> > &a)
{
        return matrixfunction(a, &myccopy);
}

NRMat<std::complex<float> >  copytest(const NRMat<std::complex<float> > &a)
{
        return matrixfunction(a, &myccopy);
}

NRMat<double>  copytest(const NRMat<double> &a)
{
        return matrixfunction(a, &myccopy);
}

NRMat<float>  copytest(const NRMat<float> &a)
{
        return matrixfunction(a, &myccopy);
}





const NRVec<double> diagofproduct(const NRMat<double> &a, const NRMat<double> &b,
		bool trb, bool conjb)
{
	if (trb && (a.nrows() != b.nrows() || a.ncols() != b.ncols()) ||
				!trb && (a.nrows() != b.ncols() || a.ncols() != b.nrows()))
			laerror("incompatible Mats in diagofproduct<double>()");
	NRVec<double> result(a.nrows());
	if (trb)
		for(int i=0; i<a.nrows(); i++)
			result[i] = cblas_ddot(a.ncols(), a[i], 1, b[i], 1);
	else
		for(int i=0; i<a.nrows(); i++)
			result[i] = cblas_ddot(a.ncols(), a[i], 1, b[0]+i, b.ncols());

	return result;
}

const NRVec<float> diagofproduct(const NRMat<float> &a, const NRMat<float> &b,
		bool trb, bool conjb)
{
	if (trb && (a.nrows() != b.nrows() || a.ncols() != b.ncols()) ||
				!trb && (a.nrows() != b.ncols() || a.ncols() != b.nrows()))
			laerror("incompatible Mats in diagofproduct<float>()");
	NRVec<float> result(a.nrows());
	if (trb)
		for(int i=0; i<a.nrows(); i++)
			result[i] = cblas_sdot(a.ncols(), a[i], 1, b[i], 1);
	else
		for(int i=0; i<a.nrows(); i++)
			result[i] = cblas_sdot(a.ncols(), a[i], 1, b[0]+i, b.ncols());

	return result;
}


const NRVec< std::complex<double> > diagofproduct(const NRMat< std::complex<double> > &a,
		const NRMat< std::complex<double> > &b, bool trb, bool conjb)
{
	if (trb && (a.nrows() != b.nrows() || a.ncols() != b.ncols()) ||
				!trb && (a.nrows() != b.ncols() || a.ncols() != b.nrows()))
			laerror("incompatible Mats in diagofproduct<complex>()");
	NRVec< std::complex<double> > result(a.nrows());
	if (trb) {
		if (conjb) {
			for(int i=0; i<a.nrows(); i++)
				cblas_zdotc_sub(a.ncols(), b[i], 1, a[i], 1, &result[i]);
		} else {
			for(int i=0; i<a.nrows(); i++)
				cblas_zdotu_sub(a.ncols(), b[i], 1, a[i], 1, &result[i]);
		}
	} else {
		if (conjb) {
			for(int i=0; i<a.nrows(); i++)
				cblas_zdotc_sub(a.ncols(), b[0]+i, b.ncols(), a[i], 1, &result[i]);
		} else {
			for(int i=0; i<a.nrows(); i++)
				cblas_zdotu_sub(a.ncols(), b[0]+i, b.ncols(), a[i], 1, &result[i]);
		}
	}
	return result;
}

const NRVec< std::complex<float> > diagofproduct(const NRMat< std::complex<float> > &a,
		const NRMat< std::complex<float> > &b, bool trb, bool conjb)
{
	if (trb && (a.nrows() != b.nrows() || a.ncols() != b.ncols()) ||
				!trb && (a.nrows() != b.ncols() || a.ncols() != b.nrows()))
			laerror("incompatible Mats in diagofproduct<complex>()");
	NRVec< std::complex<float> > result(a.nrows());
	if (trb) {
		if (conjb) {
			for(int i=0; i<a.nrows(); i++)
				cblas_cdotc_sub(a.ncols(), b[i], 1, a[i], 1, &result[i]);
		} else {
			for(int i=0; i<a.nrows(); i++)
				cblas_cdotu_sub(a.ncols(), b[i], 1, a[i], 1, &result[i]);
		}
	} else {
		if (conjb) {
			for(int i=0; i<a.nrows(); i++)
				cblas_cdotc_sub(a.ncols(), b[0]+i, b.ncols(), a[i], 1, &result[i]);
		} else {
			for(int i=0; i<a.nrows(); i++)
				cblas_cdotu_sub(a.ncols(), b[0]+i, b.ncols(), a[i], 1, &result[i]);
		}
	}
	return result;
}


double trace2(const NRMat<double> &a, const NRMat<double> &b, bool trb)
{
	if (trb && (a.nrows() != b.nrows() || a.ncols() != b.ncols()) ||
				!trb && (a.nrows() != b.ncols() || a.ncols() != b.nrows()))
			laerror("incompatible Mats in trace2()");
	if (trb) return cblas_ddot(a.nrows()*a.ncols(), a, 1, b, 1);

	double sum = 0.0;
	for (int i=0; i<a.nrows(); i++)
		sum += cblas_ddot(a.ncols(), a[i], 1, b[0]+i, b.ncols());

	return sum;
}

float trace2(const NRMat<float> &a, const NRMat<float> &b, bool trb)
{
	if (trb && (a.nrows() != b.nrows() || a.ncols() != b.ncols()) ||
				!trb && (a.nrows() != b.ncols() || a.ncols() != b.nrows()))
			laerror("incompatible Mats in trace2()");
	if (trb) return cblas_sdot(a.nrows()*a.ncols(), a, 1, b, 1);

	float sum = 0.0;
	for (int i=0; i<a.nrows(); i++)
		sum += cblas_sdot(a.ncols(), a[i], 1, b[0]+i, b.ncols());

	return sum;
}

// LV
std::complex<double> trace2(const NRMat<std::complex<double> > &a, const NRMat<std::complex<double> > &b, bool adjb)
{
        if (adjb && (a.nrows() != b.nrows() || a.ncols() != b.ncols()) ||
                                !adjb && (a.nrows() != b.ncols() || a.ncols() != b.nrows()))
                        laerror("incompatible Mats in trace2()");

        std::complex<double> dot;
        if (adjb) { cblas_zdotc_sub(a.nrows()*a.ncols(), b, 1, a, 1, &dot); return dot; }

        std::complex<double> sum = std::complex<double>(0.,0.);
        for (int i=0; i<a.nrows(); i++) {
                cblas_zdotu_sub(a.ncols(), a[i], 1, b[0]+i, b.ncols(), &dot);
                sum += dot;
        }

        return sum;
}

std::complex<float> trace2(const NRMat<std::complex<float> > &a, const NRMat<std::complex<float> > &b, bool adjb)
{
        if (adjb && (a.nrows() != b.nrows() || a.ncols() != b.ncols()) ||
                                !adjb && (a.nrows() != b.ncols() || a.ncols() != b.nrows()))
                        laerror("incompatible Mats in trace2()");

        std::complex<float> dot;
        if (adjb) { cblas_cdotc_sub(a.nrows()*a.ncols(), b, 1, a, 1, &dot); return dot; }

        std::complex<float> sum = std::complex<float>(0.,0.);
        for (int i=0; i<a.nrows(); i++) {
                cblas_cdotu_sub(a.ncols(), a[i], 1, b[0]+i, b.ncols(), &dot);
                sum += dot;
        }

        return sum;
}

double trace2(const NRSMat<double> &a, const NRSMat<double> &b,
		const bool diagscaled)
{
	if (a.nrows() != b.nrows()) laerror("incompatible SMats in trace2()");

	//double r = 0; for (int i=0; i<a.nrows()*(a.nrows()+1)/2; ++i) r += a[i]*b[i]; r+=r;
	double r = 2.0*cblas_ddot(a.nrows()*(a.nrows()+1)/2, a, 1, b, 1);
	if (diagscaled) return r;
	for (int i=0; i<a.nrows(); i++) r -= a(i,i)*b(i,i);
	//r -= cblas_ddot(a.nrows(),a,a.nrows()+1,b,a.nrows()+1); //@@@this was errorneous in one version of ATLAS
	return r;
}

float trace2(const NRSMat<float> &a, const NRSMat<float> &b,
		const bool diagscaled)
{
	if (a.nrows() != b.nrows()) laerror("incompatible SMats in trace2()");

	//float r = 0; for (int i=0; i<a.nrows()*(a.nrows()+1)/2; ++i) r += a[i]*b[i]; r+=r;
	float r = 2.0*cblas_sdot(a.nrows()*(a.nrows()+1)/2, a, 1, b, 1);
	if (diagscaled) return r;
	for (int i=0; i<a.nrows(); i++) r -= a(i,i)*b(i,i);
	//r -= cblas_sdot(a.nrows(),a,a.nrows()+1,b,a.nrows()+1); //@@@this was errorneous in one version of ATLAS
	return r;
}

double trace2(const NRSMat<double> &a, const NRMat<double> &b, const bool diagscaled)
{
        if (a.nrows() != b.nrows()||b.nrows()!=b.ncols()) laerror("incompatible SMats in trace2()");
double r=0;
        int i, j, k=0;
        for (i=0; i<a.nrows(); i++)
                for (j=0; j<=i;j++) r += a[k++] * (b[i][j] + (i!=j||diagscaled ? b[j][i] : 0 ));


return r;
}

float trace2(const NRSMat<float> &a, const NRMat<float> &b, const bool diagscaled)
{
        if (a.nrows() != b.nrows()||b.nrows()!=b.ncols()) laerror("incompatible SMats in trace2()");
float r=0;
        int i, j, k=0;
        for (i=0; i<a.nrows(); i++)
                for (j=0; j<=i;j++) r += a[k++] * (b[i][j] + (i!=j||diagscaled ? b[j][i] : 0 ));


return r;
}

double trace2(const NRMat<double> &a, const NRSMat<double> &b, const bool diagscaled)
{
return trace2(b,a,diagscaled);
}

float trace2(const NRMat<float> &a, const NRSMat<float> &b, const bool diagscaled)
{
return trace2(b,a,diagscaled);
}

//complex hermitian versions: lower triangle stored, upper triangle is its complex conjugate
//Tr(A B) = sum_ij A_ij B_ji = sum_ij A_ij conj(B_ij), real for hermitian A, B
std::complex<double> trace2(const NRSMat<std::complex<double> > &a, const NRSMat<std::complex<double> > &b,
		const bool diagscaled)
{
	if (a.nrows() != b.nrows()) laerror("incompatible SMats in trace2()");

	std::complex<double> dot;
	cblas_zdotc_sub(a.nrows()*(a.nrows()+1)/2, b, 1, a, 1, &dot); //sum over the lower triangle of a_ij conj(b_ij)
	double r = 2.0*dot.real();
	if (diagscaled) return r;
	for (int i=0; i<a.nrows(); i++) r -= (a(i,i)*std::conj(b(i,i))).real();
	return r;
}

std::complex<float> trace2(const NRSMat<std::complex<float> > &a, const NRSMat<std::complex<float> > &b,
		const bool diagscaled)
{
	if (a.nrows() != b.nrows()) laerror("incompatible SMats in trace2()");

	std::complex<float> dot;
	cblas_cdotc_sub(a.nrows()*(a.nrows()+1)/2, b, 1, a, 1, &dot); //sum over the lower triangle of a_ij conj(b_ij)
	float r = 2.0*dot.real();
	if (diagscaled) return r;
	for (int i=0; i<a.nrows(); i++) r -= (a(i,i)*std::conj(b(i,i))).real();
	return r;
}

std::complex<double> trace2(const NRSMat<std::complex<double> > &a, const NRMat<std::complex<double> > &b, const bool diagscaled)
{
        if (a.nrows() != b.nrows()||b.nrows()!=b.ncols()) laerror("incompatible SMats in trace2()");
std::complex<double> r=0;
        int i, j, k=0;
        for (i=0; i<a.nrows(); i++)
                for (j=0; j<=i;j++) {r += a[k] * b[j][i]; if(i!=j||diagscaled) r += std::conj(a[k]) * b[i][j]; ++k;}
return r;
}

std::complex<float> trace2(const NRSMat<std::complex<float> > &a, const NRMat<std::complex<float> > &b, const bool diagscaled)
{
        if (a.nrows() != b.nrows()||b.nrows()!=b.ncols()) laerror("incompatible SMats in trace2()");
std::complex<float> r=0;
        int i, j, k=0;
        for (i=0; i<a.nrows(); i++)
                for (j=0; j<=i;j++) {r += a[k] * b[j][i]; if(i!=j||diagscaled) r += std::conj(a[k]) * b[i][j]; ++k;}
return r;
}

std::complex<double> trace2(const NRMat<std::complex<double> > &a, const NRSMat<std::complex<double> > &b, const bool diagscaled)
{
return trace2(b,a,diagscaled);
}

std::complex<float> trace2(const NRMat<std::complex<float> > &a, const NRSMat<std::complex<float> > &b, const bool diagscaled)
{
return trace2(b,a,diagscaled);
}


//Cholesky interface
extern "C" void FORNAME(dpotrf)(const char *UPLO, const FINT *N, double *A, const FINT *LDA, FINT *INFO);
extern "C" void FORNAME(spotrf)(const char *UPLO, const FINT *N, float *A, const FINT *LDA, FINT *INFO);
extern "C" void FORNAME(zpotrf)(const char *UPLO, const FINT *N, std::complex<double> *A, const FINT *LDA, FINT *INFO);
extern "C" void FORNAME(cpotrf)(const char *UPLO, const FINT *N, std::complex<float> *A, const FINT *LDA, FINT *INFO);

void cholesky(NRMat<double> &a, bool upper)
{
if(a.nrows()!=a.ncols()) laerror("matrix must be square in Cholesky");
FINT lda=a.ncols();
FINT n=a.nrows();
char uplo= LAPACK_FORTRANCASE(upper?'u':'l');
FINT info;
a.copyonwrite();
FORNAME(dpotrf)(&uplo, &n, a, &lda, &info);
if(info) {std::cerr << "Lapack error "<<info<<std::endl; laerror("error in Cholesky");}
//zero the other triangle and switch to C array order
if(upper)
	for(int i=0; i<n; ++i) for(int j=0; j<i; ++j) {a(j,i)=a(i,j); a(i,j)=0.;}
else
	for(int i=0; i<n; ++i) for(int j=0; j<i; ++j) {a(i,j)=a(j,i); a(j,i)=0.;}

}

void cholesky(NRMat<float> &a, bool upper)
{
if(a.nrows()!=a.ncols()) laerror("matrix must be square in Cholesky");
FINT lda=a.ncols();
FINT n=a.nrows();
char uplo= LAPACK_FORTRANCASE(upper?'u':'l');
FINT info;
a.copyonwrite();
FORNAME(spotrf)(&uplo, &n, a, &lda, &info);
if(info) {std::cerr << "Lapack error "<<info<<std::endl; laerror("error in Cholesky");}
//zero the other triangle and switch to C array order
if(upper)
	for(int i=0; i<n; ++i) for(int j=0; j<i; ++j) {a(j,i)=a(i,j); a(i,j)=0.;}
else
	for(int i=0; i<n; ++i) for(int j=0; j<i; ++j) {a(i,j)=a(j,i); a(j,i)=0.;}

}


void cholesky(NRMat<std::complex<double> > &a, bool upper)
{
if(a.nrows()!=a.ncols()) laerror("matrix must be square in Cholesky");
FINT lda=a.ncols();
FINT n=a.nrows();
char uplo= LAPACK_FORTRANCASE(upper?'u':'l');
FINT info;
a.copyonwrite();
a.transposeme();//switch to Fortran order
FORNAME(zpotrf)(&uplo, &n, a, &lda, &info);
if(info) {std::cerr << "Lapack error "<<info<<std::endl; laerror("error in Cholesky");}
//zero the other triangle and switch to C array order
if(upper)
        for(int i=0; i<n; ++i) for(int j=0; j<i; ++j) {a(j,i)=a(i,j); a(i,j)=0.;}
else
        for(int i=0; i<n; ++i) for(int j=0; j<i; ++j) {a(i,j)=a(j,i); a(j,i)=0.;}
}

void cholesky(NRMat<std::complex<float> > &a, bool upper)
{
if(a.nrows()!=a.ncols()) laerror("matrix must be square in Cholesky");
FINT lda=a.ncols();
FINT n=a.nrows();
char uplo= LAPACK_FORTRANCASE(upper?'u':'l');
FINT info;
a.copyonwrite();
a.transposeme();//switch to Fortran order
FORNAME(cpotrf)(&uplo, &n, a, &lda, &info);
if(info) {std::cerr << "Lapack error "<<info<<std::endl; laerror("error in Cholesky");}
//zero the other triangle and switch to C array order
if(upper)
        for(int i=0; i<n; ++i) for(int j=0; j<i; ++j) {a(j,i)=a(i,j); a(i,j)=0.;}
else
        for(int i=0; i<n; ++i) for(int j=0; j<i; ++j) {a(i,j)=a(j,i); a(j,i)=0.;}
}


//various norms
extern "C" double FORNAME(zlange)( const char *NORM, const FINT *M, const FINT *N, const std::complex<double> *A, const FINT *LDA, double *WORK); 
extern "C" float FORNAME(clange)( const char *NORM, const FINT *M, const FINT *N, const std::complex<float> *A, const FINT *LDA, float *WORK); 
extern "C" double FORNAME(dlange)( const char *NORM, const FINT *M, const FINT *N, const double *A, const FINT *LDA, double *WORK); 
extern "C" float FORNAME(slange)( const char *NORM, const FINT *M, const FINT *N, const float *A, const FINT *LDA, float *WORK); 

//LAPACK sees the C-order matrix transposed: 1-norm and infinity norm are swapped, Frobenius and max-abs norms are invariant
static char lapack_norm_type(const char norm)
{
switch(tolower(norm))
	{
	case 'o': case '1': return 'I';
	case 'i': return 'O';
	case 'f': case 'e': return 'F';
	case 'm': return 'M';
	default: laerror("unknown norm type in MatrixNorm/CondNumber");
	}
return 0;
}

template<>
LA_traits<NRMat<std::complex<double> > >::normtype MatrixNorm(const NRMat<std::complex<double> > &A, const char norm)
{
        const char TypNorm = lapack_norm_type(norm);
        const FINT M = A.ncols(); //fortran-order view of the C-order matrix is its transpose
        const FINT N = A.nrows();
        double work[M];
        const double ret = FORNAME(zlange)(&TypNorm, &M, &N, A[0], &M, &work[0]);
        return ret;
}

template<>
LA_traits<NRMat<std::complex<float> > >::normtype MatrixNorm(const NRMat<std::complex<float> > &A, const char norm)
{
        const char TypNorm = lapack_norm_type(norm);
        const FINT M = A.ncols(); //fortran-order view of the C-order matrix is its transpose
        const FINT N = A.nrows();
        float work[M];
        const float ret = FORNAME(clange)(&TypNorm, &M, &N, A[0], &M, &work[0]);
        return ret;
}

template<>
LA_traits<NRMat<double> >::normtype MatrixNorm(const NRMat<double> &A, const char norm)
{
        const char TypNorm = lapack_norm_type(norm);
        const FINT M = A.ncols(); //fortran-order view of the C-order matrix is its transpose
        const FINT N = A.nrows();
        double work[M];
        const double ret = FORNAME(dlange)(&TypNorm, &M, &N, A[0], &M, &work[0]);
        return ret;
}

template<>
LA_traits<NRMat<float> >::normtype MatrixNorm(const NRMat<float> &A, const char norm)
{
        const char TypNorm = lapack_norm_type(norm);
        const FINT M = A.ncols(); //fortran-order view of the C-order matrix is its transpose
        const FINT N = A.nrows();
        float work[M];
        const float ret = FORNAME(slange)(&TypNorm, &M, &N, A[0], &M, &work[0]);
        return ret;
}



//condition number ||A|| ||A^{-1}|| in the 1-norm or infinity norm, from the LU factorization of a copy of A
extern "C" void FORNAME(zgetrf)(const FINT *M, const FINT *N, std::complex<double> *A, const FINT *LDA, FINT *IPIV, FINT *INFO);
extern "C" void FORNAME(cgetrf)(const FINT *M, const FINT *N, std::complex<float> *A, const FINT *LDA, FINT *IPIV, FINT *INFO);
extern "C" void FORNAME(dgetrf)(const FINT *M, const FINT *N, double *A, const FINT *LDA, FINT *IPIV, FINT *INFO);
extern "C" void FORNAME(sgetrf)(const FINT *M, const FINT *N, float *A, const FINT *LDA, FINT *IPIV, FINT *INFO);
extern "C" void FORNAME(zgecon)( const char *norm, const FINT *n, const std::complex<double> *A, const FINT *LDA, const double *anorm, double *rcond, std::complex<double> *work, double *rwork, FINT *info);
extern "C" void FORNAME(cgecon)( const char *norm, const FINT *n, const std::complex<float> *A, const FINT *LDA, const float *anorm, float *rcond, std::complex<float> *work, float *rwork, FINT *info);
extern "C" void FORNAME(dgecon)( const char *norm, const FINT *n, const double *A, const FINT *LDA, const double *anorm, double *rcond, double *work, FINT *iwork, FINT *info);
extern "C" void FORNAME(sgecon)( const char *norm, const FINT *n, const float *A, const FINT *LDA, const float *anorm, float *rcond, float *work, FINT *iwork, FINT *info);

template<>
LA_traits<NRMat<std::complex<double> > >::normtype CondNumber(const NRMat<std::complex<double> > &A, const char norm)
{
        const char TypNorm = lapack_norm_type(norm);
        if(TypNorm!='O' && TypNorm!='I') laerror("CondNumber implemented only for the 1-norm and infinity norm");
        const FINT N = A.nrows();
        if(N != A.ncols()) laerror("nonsquare matrix in CondNumber");
        double Norma(0.0), rcond(0.0);
        FINT info;

        NRMat<std::complex<double> > LU(A);
        LU.copyonwrite();
        FINT *ipiv = new FINT[N];
        FORNAME(zgetrf)(&N, &N, LU[0], &N, ipiv, &info);
        delete[] ipiv;
        if(info<0) laerror("illegal argument in zgetrf in CondNumber");
        if(info>0) return std::numeric_limits<double>::infinity(); //exactly singular

        std::complex<double> *work = new std::complex<double>[2*N];
        double *rwork = new double[2*N];
        Norma = MatrixNorm(A, norm);
        FORNAME(zgecon)(&TypNorm, &N, LU[0], &N, &Norma, &rcond, work, rwork, &info);
        delete[] work;
        delete[] rwork;
        if(info<0) laerror("illegal argument in zgecon in CondNumber");
        return rcond==0. ? std::numeric_limits<double>::infinity() : 1./rcond;
}

template<>
LA_traits<NRMat<std::complex<float> > >::normtype CondNumber(const NRMat<std::complex<float> > &A, const char norm)
{
        const char TypNorm = lapack_norm_type(norm);
        if(TypNorm!='O' && TypNorm!='I') laerror("CondNumber implemented only for the 1-norm and infinity norm");
        const FINT N = A.nrows();
        if(N != A.ncols()) laerror("nonsquare matrix in CondNumber");
        float Norma(0.0), rcond(0.0);
        FINT info;

        NRMat<std::complex<float> > LU(A);
        LU.copyonwrite();
        FINT *ipiv = new FINT[N];
        FORNAME(cgetrf)(&N, &N, LU[0], &N, ipiv, &info);
        delete[] ipiv;
        if(info<0) laerror("illegal argument in zgetrf in CondNumber");
        if(info>0) return std::numeric_limits<float>::infinity(); //exactly singular

        std::complex<float> *work = new std::complex<float>[2*N];
        float *rwork = new float[2*N];
        Norma = MatrixNorm(A, norm);
        FORNAME(cgecon)(&TypNorm, &N, LU[0], &N, &Norma, &rcond, work, rwork, &info);
        delete[] work;
        delete[] rwork;
        if(info<0) laerror("illegal argument in zgecon in CondNumber");
        return rcond==0. ? std::numeric_limits<float>::infinity() : 1./rcond;
}

template<>
LA_traits<NRMat<double> >::normtype CondNumber(const NRMat<double> &A, const char norm)
{
        const char TypNorm = lapack_norm_type(norm);
        if(TypNorm!='O' && TypNorm!='I') laerror("CondNumber implemented only for the 1-norm and infinity norm");
        const FINT N = A.nrows();
        if(N != A.ncols()) laerror("nonsquare matrix in CondNumber");
        double Norma(0.0), rcond(0.0);
        FINT info;

        NRMat<double> LU(A);
        LU.copyonwrite();
        FINT *ipiv = new FINT[N];
        FORNAME(dgetrf)(&N, &N, LU[0], &N, ipiv, &info);
        delete[] ipiv;
        if(info<0) laerror("illegal argument in dgetrf in CondNumber");
        if(info>0) return std::numeric_limits<double>::infinity(); //exactly singular

        double *work = new double[4*N];
        FINT *iwork = new FINT[N];
        Norma = MatrixNorm(A, norm);
        FORNAME(dgecon)(&TypNorm, &N, LU[0], &N, &Norma, &rcond, work, iwork, &info);
        delete[] work;
        delete[] iwork;
        if(info<0) laerror("illegal argument in dgecon in CondNumber");
        return rcond==0. ? std::numeric_limits<double>::infinity() : 1./rcond;
}

template<>
LA_traits<NRMat<float> >::normtype CondNumber(const NRMat<float> &A, const char norm)
{
        const char TypNorm = lapack_norm_type(norm);
        if(TypNorm!='O' && TypNorm!='I') laerror("CondNumber implemented only for the 1-norm and infinity norm");
        const FINT N = A.nrows();
        if(N != A.ncols()) laerror("nonsquare matrix in CondNumber");
        float Norma(0.0), rcond(0.0);
        FINT info;

        NRMat<float> LU(A);
        LU.copyonwrite();
        FINT *ipiv = new FINT[N];
        FORNAME(sgetrf)(&N, &N, LU[0], &N, ipiv, &info);
        delete[] ipiv;
        if(info<0) laerror("illegal argument in dgetrf in CondNumber");
        if(info>0) return std::numeric_limits<float>::infinity(); //exactly singular

        float *work = new float[4*N];
        FINT *iwork = new FINT[N];
        Norma = MatrixNorm(A, norm);
        FORNAME(sgecon)(&TypNorm, &N, LU[0], &N, &Norma, &rcond, work, iwork, &info);
        delete[] work;
        delete[] iwork;
        if(info<0) laerror("illegal argument in dgecon in CondNumber");
        return rcond==0. ? std::numeric_limits<float>::infinity() : 1./rcond;
}


#ifdef obsolete
void gendiagonalize(NRMat<double> &a, NRVec<double> &w, NRMat<double> b, int n)
{
if(a.nrows()!=a.ncols() || a.nrows()!=w.size() || a.nrows()!=b.nrows()  || b.nrows()!=b.ncols() ) laerror("incompatible Mats in gendiagonalize");

a.copyonwrite();
w.copyonwrite();
b.copyonwrite();
int m=w.size();
NRVec<double> dl(m);
int i,j;
double x;

if(n==0) n=m;
if(n<0 || n>m) laerror("actual dimension in gendiagonalize out of range");

//transform the problem to usual diagonalization

//cholesky decompose in b and dl
for(i=0; i<n; ++i)
{
for(j=i; j<n; ++j)
	{
        x = b(i,j) -  cblas_ddot(i,&b(i,0),1,&b(j,0),1);
	if(i==j)
		{
		if(x<=0) laerror("not positive definite metric in gendiagonalize");
       	 	dl[i] = std::sqrt(x);
		}
	else    
		b(j,i) = x / dl[i];
	}
}

// form the transpose of the upper triangle of inv(l)*a in the lower triangle of a
for(i=0; i<n; ++i)
{
for(j=i; j<n ; ++j)
	{
        x = a(i,j) - cblas_ddot(i,&b(i,0),1,&a(j,0),1);
        a(j,i) = x/dl[i];
	}
}

//pre-multiply by l^-1
for(j=0; j<n ; ++j)
{
	for(i=j;i<n;++i)
	{
        x = a(i,j) - cblas_ddot(i-j,&a(j,j),m,&b(i,j),1)
		- cblas_ddot(j,&a(j,0),1,&b(i,0),1);
        a(i,j) = x/dl[i];
	}
}

//fill in upper triangle of a for the diagonalize procedure (would not be needed with tred2,tql2)
for(i=1;i<n;++i) for(j=0; j<i; ++j) a(j,i)=a(i,j);

//diagonalize by a standard procedure
diagonalize(a,w,1,1,n);

//transform the eigenvectors back
for(j=0; j<n; ++j)//eigenvector loop
{
for(int i=n-1; i>=0; --i)//component loop
	{
        if(i<n-1) a(i,j) -= cblas_ddot(n-1-i,&b(i+1,i),m,&a(i+1,j),m);
        a(i,j) /= dl[i];
	}
}
}
#endif
//obsolete

//auxiliary routine to adjust eigenvectors to guarantee real logarithm
//at the moment not rigorous yet
void adjustphases(NRMat<double> &v)
{
int n=v.nrows();
double det=determinant(v);
int nchange=0;
for(int i=0; i<n;++i) if(v[i][i]<0.)
        {
        cblas_dscal(n,-1.,v[i],1); 
        nchange++;
        } 
if(det<0) nchange++;
if(nchange&1)//still adjust to get determinant=1
        {
        int imin=-1; double min=1e200;
        for(int i=0; i<n;++i)
                if(std::abs(v[i][i])<min)
                        {
                        imin=i;
                        min=std::abs(v[i][i]);
                        } 
        cblas_dscal(n,-1.,v[imin],1);
        }
}

void adjustphases(NRMat<float> &v)
{
int n=v.nrows();
float det=determinant(v);
int nchange=0;
for(int i=0; i<n;++i) if(v[i][i]<0.)
        {
        cblas_sscal(n,-1.,v[i],1); 
        nchange++;
        } 
if(det<0) nchange++;
if(nchange&1)//still adjust to get determinant=1
        {
        int imin=-1; float min=1e200;
        for(int i=0; i<n;++i)
                if(std::abs(v[i][i])<min)
                        {
                        imin=i;
                        min=std::abs(v[i][i]);
                        } 
        cblas_sscal(n,-1.,v[imin],1);
        }
}

}//namespace

