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


namespace LA {


#define INSTANTIZE(T) \
template void lawritemat(FILE *file,const T *a,int r,int c,const char *form0, \
		int nodim,int modulo, int issym);
INSTANTIZE(double)
INSTANTIZE(complex<double>)
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

#define EPSDET 1e-300

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
						fprintf(file, f, ((complex<double>)a[ii*(ii+1)/2+jj]).real(), ((complex<double>)a[ii*(ii+1)/2+jj]).imag());
					} else fprintf(file, f, ((complex<double>)a[(i-1)*c+j-1]).real(), ((complex<double>)a[(i-1)*c+j-1]).imag());
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
					fprintf(file, f, ((complex<double>)a[ii*(ii+1)/2+jj]).real(), ((complex<double>)a[ii*(ii+1)/2+jj]).imag());
				} else fprintf(file,f,((complex<double>)a[(i-1)*c+j-1]).real(), ((complex<double>)a[(i-1)*c+j-1]).imag());
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

	
	if (n==A.nrows() && A.nrows() != A.ncols()) laerror("linear_solve() call for non-square matrix");
	A.copyonwrite();
	ipiv = new int[A.nrows()];
	r = clapack_dgesv(CblasRowMajor, n, nrhs, A[0], A.ncols(), ipiv, B , ldb);
	if (r < 0) {
		delete[] ipiv;
		laerror("illegal argument in lapack_gesv");
	}
	if (det && r==0) {
		*det = 1.;
		//take into account some numerical instabilities in dgesv for singular matrices
		for (int i=0; i<n; ++i)  {double t=A[i][i]; if(!finite(t) || std::abs(t) < EPSDET ) {*det=0.; break;} else *det *=t;}
		//find out whether ipiv are numbered from 0 or from 1
		int shift=1;
	        for (int i=0; i<n; ++i) if(ipiv[i]==0) shift=0;
		//change sign of det by parity of ipiv permutation
		if(*det) for (int i=0; i<n; ++i) if(i+shift != ipiv[i]) {*det = -(*det); ++iswap;}
	}
/*
       std::cout <<"iswap = "<<iswap<<std::endl;
	if(det && r>0) *det = 0;

	std::cout <<"ipiv = ";
	for (int i=0; i<n; ++i) std::cout <<ipiv[i]<<" ";
	std::cout <<std::endl;
*/

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

void linear_solve(NRMat<double> &A, NRVec<double> &B, double *det, int n)
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

static void linear_solve_do(NRSMat<double> &a, double *b, const int nrhs, const int ldb, double *det, int n)
{
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
	if (r < 0) {
		delete[] ipiv;
		laerror("illegal argument in spsv() call of linear_solve()");
	}
	if (det && r == 0) {
		*det = 1.;
		for (int i=1; i<n; i++) {double t=a(i,i); if(!finite(t) || std::abs(t) < EPSDET ) {*det=0.; break;} else *det *= t;}
		//do not use ipiv, since the permutation matrix occurs twice in the decomposition and signs thus cancel (man dspsv)
	}
        if (det && r>0)  *det = 0;
	delete[] ipiv;
	if (r > 0 && b) laerror("singular matrix in linear_solve(SMat&, Mat*, double*");
}


void linear_solve(NRSMat<double> &a, NRMat<double> *B, double *det, int n)
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


// Roman, complex version of linear_solve()
extern "C" void FORNAME(zgesv)(const int *N, const int *NRHS, double *A, const int *LDA,
           int *IPIV, double *B, const int *LDB, int *INFO);

void linear_solve(NRMat< complex<double> > &A, NRMat< complex<double> > *B, complex<double> *det, int n)
{
        int r, *ipiv;

        if (A.nrows() != A.ncols()) laerror("linear_solve() call for non-square matrix");
        if (B && A.nrows() != B->ncols()) laerror("incompatible matrices in linear_solve()");
        A.copyonwrite();
        if (B) B->copyonwrite();
        ipiv = new int[A.nrows()];
  n = A.nrows();
  int nrhs = B ? B->nrows() : 0;
  int lda = A.ncols();
  int ldb = B ? B->ncols() : A.nrows();
        FORNAME(zgesv)(&n, &nrhs, (double *)A[0], &lda, ipiv,
                 B ? (double *)(*B)[0] : (double *)0, &ldb, &r);
        if (r < 0) {
                delete[] ipiv;
                laerror("illegal argument in lapack_gesv");
        }
        if (det && r>=0) {
                *det = A[0][0];
                for (int i=1; i<A.nrows(); ++i) *det *= A[i][i];
                //change sign of det by parity of ipiv permutation
                for (int i=0; i<A.nrows(); ++i) *det = -(*det);
        }
        delete [] ipiv;
        if (r>0 && B) laerror("singular matrix in zgesv");
}



//other version of linear solver based on gesvx

//------------------------------------------------------------------------------
extern "C" void FORNAME(zgesvx)(const char *fact, const char *trans, const FINT *n, const FINT *nrhs, complex<double> *A,  const FINT *lda, complex<double> *AF, const FINT *ldaf, const FINT *ipiv, char *equed, double *R,double *C, complex<double> *B, const FINT *ldb, complex<double> *X, const FINT *ldx, double *rcond, double *ferr, double *berr, complex<double> *work, double *rwork, FINT *info);
extern "C" void FORNAME(dgesvx)(const char *fact, const char *trans, const FINT *n, const FINT *nrhs, double *A,  const FINT *lda, double *AF, const FINT *ldaf, const FINT *ipiv, char *equed, double *R,double *C, double *B, const FINT *ldb, double *X, const FINT *ldx, double *rcond, double *ferr, double *berr, double *work, FINT *iwork, FINT *info);
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
int linear_solve_x(NRMat<double> &_A, double *_B, const int _rhsCount, const int _eqCount, const bool _eq, const bool _saveA, double *_rcond){
	const int A_rows = _A.nrows();
	const int A_cols = _A.ncols();

	const char fact  = LAPACK_FORTRANCASE(_eq?'E':'N');
	const char trans = LAPACK_FORTRANCASE('T');//because of c-order
	char equed = LAPACK_FORTRANCASE('B');//if fact=='N' then equed is an output argument, therefore not declared as const

	if(_eqCount < 0 || _eqCount > A_rows || _eqCount > A_cols || _rhsCount < 0){
		laerror("linear_solve_x: invalid input matrices");
	}

	double *A;
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
		}else{
			_A.copyonwrite();
		}
	}

	FORNAME(dgesvx)(&fact, &trans, &n, &nrhs, A, &lda, AF, &ldaf, &ipiv[0], &equed, &R[0], &C[0], _B, &n, X, &n, &rcond, ferr, berr, work, iwork, &info);
	
	if(_rcond)*_rcond = rcond;
	cblas_dcopy(n*nrhs, X, 1, _B, 1);//store the solution

	delete[] iwork;delete[] ipiv;
	delete[] AF;delete[] X;
	if(_saveA){
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
int linear_solve_x(NRMat<complex<double> > &_A, complex<double> *_B, const int _rhsCount, const int _eqCount, const bool _eq, const bool _saveA, double *_rcond){
	const int A_rows = _A.nrows();
	const int A_cols = _A.ncols();

	const char fact  = LAPACK_FORTRANCASE(_eq?'E':'N');
	const char trans = LAPACK_FORTRANCASE('T');//because of c-order
	char equed = LAPACK_FORTRANCASE('B');//if fact=='N' then equed is an output argument, therefore not declared as const

	if(_eqCount < 0 || _eqCount > A_rows || _eqCount > A_cols || _rhsCount < 0){
		laerror("linear_solve_x: invalid input matrices");
	}

	complex<double> *A;
	complex<double> * const _A_data = (complex<double>*)_A;

	FINT info;
	const FINT nrhs	= _rhsCount;
	const FINT n	= _eqCount;
	FINT lda	= A_cols;
	const FINT ldaf	= lda;

	double rcond;
	double ferr[nrhs], berr[nrhs];
	double R[n], C[n], rwork[2*n];
	complex<double> work[2*n];

	FINT *const ipiv = new FINT[n];

	complex<double> *X  = new complex<double>[n*nrhs];
	complex<double> *AF = new complex<double>[ldaf*n];

	A = _A_data;
	if(_eq){
		if(_saveA){//store the corresponding submatrix of _A (not needed provided fact=='N')
			A = new complex<double>[n*n];
			int offset1 = 0;int offset2 = 0;
			for(register int i=0;i<n;i++){
				cblas_zcopy(n, _A_data + offset1, 1, A + offset2, 1);
				offset1 += A_cols;
				offset2 += n;
			}
			lda = n;//!!!
		}else{
			_A.copyonwrite();
		}
	}

	FORNAME(zgesvx)(&fact, &trans, &n, &nrhs, A, &lda, AF, &ldaf, &ipiv[0], &equed, &R[0], &C[0], _B, &n, X, &n, &rcond, ferr, berr, work, rwork, &info);


	if(_rcond)*_rcond = rcond;
	cblas_zcopy(n*nrhs, X, 1, _B, 1);//store the solution

	delete[] ipiv;
	delete[] AF;delete[] X;
	if(_saveA){
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
int multiply_by_inverse(NRMat<double> &_A, NRMat<double> &_B, bool _useEq, double *_rcond){
	
	const FINT n = _A.nrows();
	const FINT m = _A.ncols();
	if(n != m || n != _B.nrows() || n != _B.ncols()){
		laerror("multiply_by_inverse: incompatible matrices");
	}
	
	const char fact  = _useEq?'E':'N';
	const char trans = 'N';//because of c-order
	char equed = 'B';//if fact=='N' then equed is an output argument, therefore not declared as const
	const int n2 = n*n;
	
	double * const A = (double*)_A;
	double * const B = (double*)_B;
	_B.copyonwrite();//even if fact='N', call copyonwrite because the solution is going to be stored in _B

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
int multiply_by_inverse(NRMat<complex<double> > &_A, NRMat<complex<double> > &_B, bool _useEq, double *_rcond){
	
	const FINT n = _A.nrows();
	const FINT m = _A.ncols();
	if(n != m || n != _B.nrows() || n != _B.ncols()){
		laerror("multiply_by_inverse: incompatible matrices");
	}
	const int n2 = n*n;
	
	const char fact  = _useEq?'E':'N';
	const char trans = 'N';//because of c-order
	char equed = 'B';//if fact=='N' then equed is an output argument, therefore not declared as const
	
	complex<double> * const A = (complex<double>*)_A;
	complex<double> * const B = (complex<double>*)_B;
	_B.copyonwrite();//even if fact='N', call copyonwrite because the solution is going to be stored in _B

	FINT info;
	double rcond;
	double ferr[n], berr[n];
	double R[n], C[n], rwork[2*n];
	complex<double> work[2*n];

	FINT *const ipiv = new FINT[n];

	complex<double> *X  = new complex<double>[n2];
	complex<double> *AF = new complex<double>[n2];

	FORNAME(zgesvx)(&fact, &trans, &n, &n, B, &n, AF, &n, &ipiv[0], &equed, &R[0], &C[0], A, &n, X, &n, &rcond, ferr, berr, work, rwork, &info);


	if(_rcond)*_rcond = rcond;
	cblas_zcopy(n2, X, 1, B, 1);//store the solution

	delete[] ipiv;
	delete[] AF;delete[] X;
	
	return (int)info;
}
//------------------------------------------------------------------------------



extern "C" void FORNAME(dsyev)(const char *JOBZ, const char *UPLO, const FINT *N,
		double *A, const FINT *LDA, double *W, double *WORK, const FINT *LWORK, FINT *INFO);

extern "C" void FORNAME(dsygv)(const FINT *ITYPE, const char *JOBZ, const char *UPLO, const FINT *N,
                double *A, const FINT *LDA, double *B, const FINT *LDB, double *W, double *WORK, const FINT *LWORK, FINT *INFO);


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
	if(b) FORNAME(dsygv)(&itypetmp,&vectors, &U, &ntmp, a, &m, *b, &ldb, w, &WORKX, &LWORK, &r );
        else FORNAME(dsyev)(&vectors, &U, &ntmp, a, &m, w, &WORKX, &LWORK, &r );
#else
        if(b) FORNAME(dsygv)(&itype,&vectors, &U, &n, a, &m, *b,&ldb, w, WORK, &LWORK, &r );
	else FORNAME(dsyev)(&vectors, &U, &n, a, &m, w, WORK, &LWORK, &r );
#endif

        delete[] WORK;
	if (LAPACK_FORTRANCASE(vectors) == LAPACK_FORTRANCASE('v') && corder) a.transposeme(n);

	if (r < 0) laerror("illegal argument in sygv/syev in diagonalize()");
	if (r > 0) laerror("convergence problem in sygv/syev in diagonalize()");
}



extern "C" void FORNAME(zheev)(const char *JOBZ, const char *UPLO, const FINT *N,
		complex<double> *A, const FINT *LDA, double *W, complex<double> *WORK, const FINT *LWORK, double *RWORK, FINT *INFO);

extern "C" void FORNAME(zhegv)(const FINT *ITYPE, const char *JOBZ, const char *UPLO, const FINT *N,
                complex<double> *A, const FINT *LDA, complex<double> *B, const FINT *LDB, double *W, complex<double> *WORK, const FINT *LWORK, double *RWORK, FINT *INFO);


// a will contain eigenvectors (columns if corder==1), w eigenvalues
void diagonalize(NRMat<complex<double> > &a, NRVec<double> &w, const bool eivec, 
		const bool corder, int n, NRMat<complex<double> > *b, const int itype)
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
	complex<double> WORKX;
	FINT ldb=0; if(b) ldb=b->ncols();
std::cout << "test vectors "<<vectors<<std::endl;

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
	complex<double> *WORK = new complex<double>[LWORK];

#ifdef FORINT
        if(b) FORNAME(zhegv)(&itypetmp,&vectors, &U, &ntmp, a, &m, *b, &ldb, w, &WORKX, &LWORK, RWORK, &r );
        else FORNAME(zheev)(&vectors, &U, &ntmp, a, &m, w, &WORKX, &LWORK, RWORK, &r );
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



extern "C" void FORNAME(dspev)(const char *JOBZ, const char *UPLO, const FINT *N,
		double *AP, double *W, double *Z, const FINT *LDZ, double *WORK, FINT *INFO);

extern "C" void FORNAME(dspgv)(const FINT *ITYPE, const char *JOBZ, const char *UPLO, const FINT *N,
                double *AP, double *BP, double *W, double *Z, const FINT *LDZ, double *WORK, FINT *INFO);


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


extern "C" void FORNAME(zhpev)(const char *JOBZ, const char *UPLO, const FINT *N,
		complex<double> *AP, double *W, complex<double> *Z, const FINT *LDZ, complex<double> *WORK, double *RWORK, FINT *INFO);

extern "C" void FORNAME(zhpgv)(const FINT *ITYPE, const char *JOBZ, const char *UPLO, const FINT *N,
                complex<double> *AP, complex<double> *BP, double *W, complex<double> *Z, const FINT *LDZ, complex<double> *WORK,  double *RWORK, FINT *INFO);


// v will contain eigenvectors, w eigenvalues
void diagonalize(NRSMat<complex<double> > &a, NRVec<double> &w, NRMat<complex<double> > *v,
		const bool corder, int n, NRSMat<complex<double> > *b, const int itype)
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

	complex<double> *WORK = new complex<double>[2*n];
	double *RWORK  = new double[3*n];
	FINT ldv=v?v->ncols():n;
#ifdef FORINT	
        const FINT itypetmp = itype;
        FINT ntmp = n;
	if(b) FORNAME(zhpgv)(&itypetmp,&job, &U, &ntmp, a, *b, w, v?(*v)[0]:(complex<double> *)0, &ldv, WORK, RWORK,  &r );
	else FORNAME(zhpev)(&job, &U, &ntmp, a, w, v?(*v)[0]:(complex<double> *)0, &ldv, WORK, RWORK,  &r );
#else
	if(b) FORNAME(zhpgv)(&itype,&job, &U, &n, a, *b, w, v?(*v)[0]:(complex<double> *)0, &ldv, WORK, RWORK,  &r );
	else FORNAME(zhpev)(&job, &U, &n, a, w, v?(*v)[0]:(complex<double> *)0, &ldv, WORK, RWORK,  &r );
#endif
	delete[] WORK;
	delete[] RWORK;
	if (v && corder) v->transposeme(n);

	if (r < 0) laerror("illegal argument in hpgv/hpev in diagonalize()");
	if (r > 0) laerror("convergence problem in hpgv/hpev in diagonalize()");
}



extern "C" void FORNAME(dgesvd)(const char *JOBU,  const char *JOBVT,  const FINT *M,
		const FINT *N,  double *A, const FINT *LDA, double *S, double *U, const FINT *LDU,
		double *VT, const FINT *LDVT, double *WORK, const FINT *LWORK, FINT *INFO );

void singular_decomposition(NRMat<double> &a, NRMat<double> *u, NRVec<double> &s,
		NRMat<double> *v, const bool vnotdagger, int m, int n)
{
	FINT m0 = a.nrows();
	FINT n0 = a.ncols();
	if(m<=0) m=(int)m0;
	if(n<=0) n=(int)n0;
	if(n>n0 || m>m0) laerror("bad dimension in singular_decomposition");
	if (u) if (m > u->nrows() || m> u->ncols())
		laerror("inconsistent dimension of U Mat in singular_decomposition()");
	if (s.size() < m && s.size() < n) 
		laerror("inconsistent dimension of S Vec in singular_decomposition()");
	if (v) if (n > v->nrows() || n > v->ncols())
		laerror("inconsistent dimension of V Mat in singular_decomposition()");

	a.copyonwrite();
	s.copyonwrite();
	if (u) u->copyonwrite();
	if (v) v->copyonwrite();
	
	// C-order (transposed) input and swap u,v matrices,
	// v should be transposed at the end
	char jobu = u ? 'A' : 'N';
	char jobv = v ? 'A' : 'N';
	double work0;
	FINT lwork = -1;
	FINT r;

#ifdef FORINT
        FINT ntmp = n;
	FINT mtmp = m;
	FORNAME(dgesvd)(&jobv, &jobu, &ntmp, &mtmp, a, &n0, s, v?(*v)[0]:0, &n0,
			u?(*u)[0]:0, &m0, &work0, &lwork, &r);
#else
	FORNAME(dgesvd)(&jobv, &jobu, &n, &m, a, &n0, s, v?(*v)[0]:0, &n0,
			u?(*u)[0]:0, &m0, &work0, &lwork, &r);
#endif

	lwork = (FINT) work0;
	double *work = new double[lwork];

#ifdef FORINT
	FORNAME(dgesvd)(&jobv, &jobu, &ntmp, &mtmp, a, &n0, s, v?(*v)[0]:0, &n0,
			u?(*u)[0]:0, &m0, work, &lwork, &r);
#else
	FORNAME(dgesvd)(&jobv, &jobu, &n, &m, a, &n0, s, v?(*v)[0]:0, &n0,
			u?(*u)[0]:0, &m0, work, &lwork, &r);
#endif

	delete[] work;
	if (v && vnotdagger) v->transposeme(n);

	if (r < 0) laerror("illegal argument in gesvd() of singular_decomposition()");
	if (r > 0) laerror("convergence problem in gesvd() of singular_decomposition()");
}






extern "C" void FORNAME(zgesvd)(const char *JOBU,  const char *JOBVT,  const FINT *M,
		const FINT *N,  complex<double> *A, const FINT *LDA, double *S, complex<double> *U, const FINT *LDU,
		complex<double> *VT, const FINT *LDVT, complex<double> *WORK, const FINT *LWORK, double *RWORK, FINT *INFO );

void singular_decomposition(NRMat<complex<double> > &a, NRMat<complex<double> > *u, NRVec<double> &s,
		NRMat<complex<double> > *v, const bool vnotdagger, int m, int n)
{
	FINT m0 = a.nrows();
	FINT n0 = a.ncols();
	if(m<=0) m=(int)m0;
	if(n<=0) n=(int)n0;
	if(n>n0 || m>m0) laerror("bad dimension in singular_decomposition");
	if (u) if (m > u->nrows() || m> u->ncols())
		laerror("inconsistent dimension of U Mat in singular_decomposition()");
	if (s.size() < m && s.size() < n) 
		laerror("inconsistent dimension of S Vec in singular_decomposition()");
	if (v) if (n > v->nrows() || n > v->ncols())
		laerror("inconsistent dimension of V Mat in singular_decomposition()");

	int nmin =  n<m?n:m;
	a.copyonwrite();
	s.copyonwrite();
	if (u) u->copyonwrite();
	if (v) v->copyonwrite();
	
	// C-order (transposed) input and swap u,v matrices,
	// v should be transposed at the end
	char jobu = u ? 'A' : 'N';
	char jobv = v ? 'A' : 'N';
	complex<double> work0;
	FINT lwork = -1;
	FINT r;
        double *rwork = new double[5*nmin];

#ifdef FORINT
        FINT ntmp = n;
	FINT mtmp = m;
	FORNAME(zgesvd)(&jobv, &jobu, &ntmp, &mtmp, a, &n0, s, v?(*v)[0]:0, &n0,
			u?(*u)[0]:0, &m0, &work0, &lwork, rwork, &r);
#else
	FORNAME(zgesvd)(&jobv, &jobu, &n, &m, a, &n0, s, v?(*v)[0]:0, &n0,
			u?(*u)[0]:0, &m0, &work0, &lwork, rwork, &r);
#endif

	lwork = (FINT) work0.real();
	complex<double> *work = new complex<double>[lwork];

#ifdef FORINT
	FORNAME(zgesvd)(&jobv, &jobu, &ntmp, &mtmp, a, &n0, s, v?(*v)[0]:0, &n0,
			u?(*u)[0]:0, &m0, work, &lwork, rwork, &r);
#else
	FORNAME(zgesvd)(&jobv, &jobu, &n, &m, a, &n0, s, v?(*v)[0]:0, &n0,
			u?(*u)[0]:0, &m0, work, &lwork, rwork, &r);
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

extern "C" void FORNAME(dggev)(const char *JOBVL, const char *JOBVR, const FINT *N,
		double *A, const FINT *LDA, double *B, const FINT *LDB, double *WR, double *WI,  double *WBETA, 
		 double *VL, const FINT *LDVL,  double *VR, const FINT *LDVR,  
		double *WORK, const FINT *LWORK, FINT *INFO );

extern "C" void FORNAME(zgeev)(const char *JOBVL, const char *JOBVR, const FINT *N,
                complex<double> *A, const FINT *LDA, complex<double> *W, complex<double> *VL, const FINT *LDVL,
                complex<double> *VR, const FINT *LDVR, complex<double> *WORK, const FINT *LWORK,
		 double *RWORK, FINT *INFO );

extern "C" void FORNAME(zggev)(const char *JOBVL, const char *JOBVR, const FINT *N,
                complex<double> *A, const FINT *LDA, complex<double> *B, const FINT *LDB, complex<double> *W, complex<double> *WBETA,
                complex<double> *VL, const FINT *LDVL,  complex<double> *VR, const FINT *LDVR,
                complex<double> *WORK, const FINT *LWORK, double *RWORK, FINT *INFO );




//statics for sorting
static int *gdperm;
static double *gdwr, *gdwi, *gdbeta;

//compare methods
static double realonly(const int i, const int j)
{
if(gdbeta)
	{
	if(gdbeta[i]==0. && gdbeta[j]!=0) return 1.;
	if(gdbeta[j]==0. && gdbeta[i]!=0) return -1.;
	if(gdbeta[i]==0. && gdbeta[j]==0) return 0.;
	double tmp = gdwr[i]/gdbeta[i]-gdwr[j]/gdbeta[j];
	if(tmp) return tmp;
	return gdwi[j]/gdbeta[j]-gdwi[i]/gdbeta[i];
	}
//else
double tmp = gdwr[i]-gdwr[j];
if(tmp) return tmp;
return gdwi[j]-gdwi[i];
}

static double realfirst(const int i, const int j)
{
if(gdwi[i] && ! gdwi[j]) return 1.;
if(!gdwi[i] && gdwi[j]) return -1.;
return realonly(i,j);
}

static double (* gdcompar[2])(const int, const int) = {&realonly, &realfirst};

//swap method
static void gdswap(const int i, const int j)
{
double tmp;
int itmp;
itmp=gdperm[i]; gdperm[i]=gdperm[j]; gdperm[j]=itmp;
tmp=gdwr[i]; gdwr[i]=gdwr[j]; gdwr[j]=tmp;
tmp=gdwi[i]; gdwi[i]=gdwi[j]; gdwi[j]=tmp;
if(gdbeta) {tmp=gdbeta[i]; gdbeta[i]=gdbeta[j]; gdbeta[j]=tmp;}
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
        if(b)  FORNAME(dggev)(&jobvr, &jobvl, &ntmp, a, &lda, *b, &ldb, wr, wi, *beta, vr?vr[0]:(double *)0,
                        &ldvr, vl?vl[0]:(double *)0, &ldvl, &work0, &lwork, &r);
	else FORNAME(dgeev)(&jobvr, &jobvl, &ntmp, a, &lda, wr, wi, vr?vr[0]:(double *)0,
			&ldvr, vl?vl[0]:(double *)0, &ldvl, &work0, &lwork, &r);
#else
        if(b)  FORNAME(dggev)(&jobvr, &jobvl, &n, a, &lda, *b, &ldb, wr, wi, *beta, vr?vr[0]:(double *)0,
                        &ldvr, vl?vl[0]:(double *)0, &ldvl, &work0, &lwork, &r);
	else FORNAME(dgeev)(&jobvr, &jobvl, &n, a, &lda, wr, wi, vr?vr[0]:(double *)0,
			&ldvr, vl?vl[0]:(double *)0, &ldvl, &work0, &lwork, &r);
#endif

        lwork = (FINT) work0;
	double *work = new double[lwork];

#ifdef FORINT	
	if(b) FORNAME(dggev)(&jobvr, &jobvl, &ntmp, a, &lda, *b, &ldb, wr, wi, *beta, vr?vr[0]:(double *)0,
                        &ldvr, vl?vl[0]:(double *)0, &ldvl, work, &lwork, &r);
	else FORNAME(dgeev)(&jobvr, &jobvl, &ntmp, a, &lda, wr, wi, vr?vr[0]:(double *)0,
			&ldvr, vl?vl[0]:(double *)0, &ldvl, work, &lwork, &r);
#else
	if(b) FORNAME(dggev)(&jobvr, &jobvl, &n, a, &lda, *b, &ldb, wr, wi, *beta, vr?vr[0]:(double *)0,
                        &ldvr, vl?vl[0]:(double *)0, &ldvl, work, &lwork, &r);
	else FORNAME(dgeev)(&jobvr, &jobvl, &n, a, &lda, wr, wi, vr?vr[0]:(double *)0,
			&ldvr, vl?vl[0]:(double *)0, &ldvl, work, &lwork, &r);
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


	if(sorttype>0)
		{
		NRVec<int> perm(n);
		for(int i=0; i<n;++i) perm[i]=i;
		gdperm= perm;
		if(beta) gdbeta= *beta; else gdbeta= NULL;
		gdwr=wr, gdwi=wi;
		genqsort(0,n-1,gdcompar[sorttype-1],gdswap);
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



//most general complex routine
template<>
void gdiagonalize(NRMat<complex<double> > &a, NRVec< complex<double> > &w,
		NRMat< complex<double> >*vl, NRMat< complex<double> > *vr,
		const bool corder, int n, const int sorttype, const int biorthonormalize,
		NRMat<complex<double> > *b, NRVec<complex<double> > *beta)
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
	complex<double> work0;
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
        if(b)  FORNAME(zggev)(&jobvr, &jobvl, &ntmp, a, &lda, *b, &ldb, w, *beta, vr?vr[0]:(complex<double> *)0,
                        &ldvr, vl?vl[0]:(complex<double> *)0, &ldvl, &work0, &lwork, rwork, &r);
	else FORNAME(zgeev)(&jobvr, &jobvl, &ntmp, a, &lda, w, vr?vr[0]:(complex<double> *)0,
			&ldvr, vl?vl[0]:(complex<double> *)0, &ldvl, &work0, &lwork, rwork, &r);
#else
        if(b)  FORNAME(zggev)(&jobvr, &jobvl, &n, a, &lda, *b, &ldb, w, *beta, vr?vr[0]:(complex<double> *)0,
                        &ldvr, vl?vl[0]:(complex<double> *)0, &ldvl, &work0, &lwork, rwork, &r);
	else FORNAME(zgeev)(&jobvr, &jobvl, &n, a, &lda, w, vr?vr[0]:(complex<double> *)0,
			&ldvr, vl?vl[0]:(complex<double> *)0, &ldvl, &work0, &lwork, rwork, &r);
#endif

        lwork = (FINT) work0.real();
	complex<double> *work = new complex<double>[lwork];

#ifdef FORINT
        if(b)  FORNAME(zggev)(&jobvr, &jobvl, &ntmp, a, &lda, *b, &ldb, w, *beta, vr?vr[0]:(complex<double> *)0,
                        &ldvr, vl?vl[0]:(complex<double> *)0, &ldvl, work, &lwork, rwork, &r);
	else FORNAME(zgeev)(&jobvr, &jobvl, &ntmp, a, &lda, w, vr?vr[0]:(complex<double> *)0,
			&ldvr, vl?vl[0]:(complex<double> *)0, &ldvl, work, &lwork, rwork, &r);
#else
        if(b)  FORNAME(zggev)(&jobvr, &jobvl, &n, a, &lda, *b, &ldb, w, *beta, vr?vr[0]:(complex<double> *)0,
                        &ldvr, vl?vl[0]:(complex<double> *)0, &ldvl, work, &lwork, rwork, &r);
	else FORNAME(zgeev)(&jobvr, &jobvl, &n, a, &lda, w, vr?vr[0]:(complex<double> *)0,
			&ldvr, vl?vl[0]:(complex<double> *)0, &ldvl, work, &lwork, rwork, &r);
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
			complex<double> tmp;
			cblas_zdotc_sub(n,(*vr)[i],1,(*vl)[i], 1, &tmp);
			tmp = 1./tmp;
			std::cout <<"scaling by "<<tmp<<"\n";
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
void gdiagonalize(NRMat<double> &a, NRVec< complex<double> > &w,
		NRMat< complex<double> >*vl, NRMat< complex<double> > *vr,
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
	for (i=0; i<n; i++) w[i] = complex<double>(wr[i], wi[i]);
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
                                                (*vl)[j][i] = complex<double>((*rvl)[i][j], (*rvl)[i+1][j]);
                                                (*vl)[j][i+1] = complex<double>((*rvl)[i][j], -(*rvl)[i+1][j]);
						}
						else
						{
						(*vl)[i][j] = complex<double>((*rvl)[i][j], (*rvl)[i+1][j]);
						(*vl)[i+1][j] = complex<double>((*rvl)[i][j], -(*rvl)[i+1][j]);
						}
					} 
				if (vr)
					for (int j=0; j<n; j++) {
					    if(corder)
                                                {
                                                (*vr)[j][i] = complex<double>((*rvr)[i][j], (*rvr)[i+1][j]);
                                                (*vr)[j][i+1] = complex<double>((*rvr)[i][j], -(*rvr)[i+1][j]);
                                                }
                                                else
                                                {
						(*vr)[i][j] = complex<double>((*rvr)[i][j], (*rvr)[i+1][j]);
						(*vr)[i+1][j] = complex<double>((*rvr)[i][j], -(*rvr)[i+1][j]);
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
const NRMat<double> realpart<NRMat< complex<double> > >(const NRMat< complex<double> > &a)
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
const NRMat<double> imagpart<NRMat< complex<double> > >(const NRMat< complex<double> > &a)
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
const NRMat< complex<double> > realmatrix<NRMat<double> > (const NRMat<double> &a)
{

        NRMat <complex<double> > result(a.nrows(), a.ncols());

#ifdef CUDALA
        if(a.location == cpu){
#endif


//	NRMat <complex<double> > result(a.nrows(), a.ncols());
	cblas_dcopy(a.nrows()*a.ncols(), a, 1, (double *)result[0], 2);
#ifdef CUDALA
        }else{
        laerror("not implemented for cuda yet");
        }
#endif

	return result;
}

template<>
const NRMat< complex<double> > imagmatrix<NRMat<double> > (const NRMat<double> &a)
{
        NRMat< complex<double> > result(a.nrows(), a.ncols());

#ifdef CUDALA
        if(a.location == cpu){
#endif

//	NRMat< complex<double> > result(a.nrows(), a.ncols());
	cblas_dcopy(a.nrows()*a.ncols(), a, 1, (double *)result[0]+1, 2);
#ifdef CUDALA
        }else{
        laerror("not implemented for cuda yet");
        }
#endif

	return result;
}

template<>
const NRMat< complex<double> > complexmatrix<NRMat<double> > (const NRMat<double> &re, const NRMat<double> &im)
{
	if(re.nrows()!=im.nrows() || re.ncols() != im.ncols()) laerror("incompatible sizes of real and imaginary parts");
        NRMat< complex<double> > result(re.nrows(), re.ncols());
        cblas_dcopy(re.nrows()*re.ncols(), re, 1, (double *)result[0], 2);
        cblas_dcopy(re.nrows()*re.ncols(), im, 1, (double *)result[0]+1, 2);
        return result;
}

template<>
const SparseSMat< complex<double> > complexmatrix<SparseSMat<double> >(const SparseSMat<double> &re, const SparseSMat<double> &im) {
  if(re.nrows()!=im.nrows() || re.ncols() != im.ncols()) laerror("incompatible sizes of real and imaginary parts");
  SparseSMat< complex<double> > result(re.nrows(),re.ncols());
  complex<double> tmp;

  SparseSMat<double>::iterator pre(re);
  for(; pre.notend(); ++pre) {
    tmp = pre->elem;
    result.add(pre->row,pre->col,tmp,false);
  }

  SparseSMat<double>::iterator pim(im);
  for(; pim.notend(); ++pim) {
    tmp = complex<double>(0,1)*(pim->elem);
    result.add(pim->row,pim->col,tmp,false);
  }

  return result;
}

template<>
const SparseSMat< complex<double> > realmatrix<SparseSMat<double> >(const SparseSMat<double> &re) {
  SparseSMat< complex<double> > result(re.nrows(),re.ncols());
  complex<double> tmp;

  SparseSMat<double>::iterator pre(re);
  for(; pre.notend(); ++pre) {
    tmp = pre->elem;
    result.add(pre->row,pre->col,tmp,false);
  }

  return result;
}

template<>
const SparseSMat< complex<double> > imagmatrix<SparseSMat<double> >(const SparseSMat<double> &im) {
  SparseSMat< complex<double> > result(im.nrows(),im.ncols());
  complex<double> tmp;


  SparseSMat<double>::iterator pim(im);
  for(; pim.notend(); ++pim) {
    tmp = complex<double>(0,1)*(pim->elem);
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


NRMat<complex<double> > complexmatrixfunction(NRMat<double> a, double (*fre) (const double), double (*fim) (const double))
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
        NRMat<complex<double> > r(n, n);
	for (int i=0; i<a.nrows(); i++) for(int j=0; j<a.ncols(); ++j) r(i,j)=complex<double>(t(i,j),tt(i,j));
        return r;
}





// instantize template to an addresable function
complex<double> myccopy (const complex<double> &x) 
{
	return x;
}

double mycopy (const double x) 
{
	return x;
}

complex<double> myclog (const complex<double> &x) 
{
	return log(x);
}

complex<double> mycexp (const complex<double> &x)
{
        return std::exp(x);
}


complex<double> sqrtinv (const complex<double> &x)
{
        return 1./std::sqrt(x);
}

double sqrtinv (const double x)
{
        return 1./std::sqrt(x);
}


NRMat<double>  log(const NRMat<double> &a)
{
	return matrixfunction(a, &myclog);
}

NRMat<complex<double> >  log(const NRMat<complex<double> > &a)
{
        return matrixfunction(a, &myclog);
}


NRMat<double>  exp0(const NRMat<double> &a)
{
        return matrixfunction(a, &mycexp);
}

NRMat<complex<double> >  exp0(const NRMat<complex<double> > &a)
{
        return matrixfunction(a, &mycexp);
}

NRMat<complex<double> >  copytest(const NRMat<complex<double> > &a)
{
        return matrixfunction(a, &myccopy);
}

NRMat<double>  copytest(const NRMat<double> &a)
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


const NRVec< complex<double> > diagofproduct(const NRMat< complex<double> > &a,
		const NRMat< complex<double> > &b, bool trb, bool conjb)
{
	if (trb && (a.nrows() != b.nrows() || a.ncols() != b.ncols()) ||
				!trb && (a.nrows() != b.ncols() || a.ncols() != b.nrows()))
			laerror("incompatible Mats in diagofproduct<complex>()");
	NRVec< complex<double> > result(a.nrows());
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

// LV
complex<double> trace2(const NRMat<complex<double> > &a, const NRMat<complex<double> > &b, bool adjb)
{
        if (adjb && (a.nrows() != b.nrows() || a.ncols() != b.ncols()) ||
                                !adjb && (a.nrows() != b.ncols() || a.ncols() != b.nrows()))
                        laerror("incompatible Mats in trace2()");

        complex<double> dot;
        if (adjb) { cblas_zdotc_sub(a.nrows()*a.ncols(), b, 1, a, 1, &dot); return dot; }

        complex<double> sum = complex<double>(0.,0.);
        for (int i=0; i<a.nrows(); i++) {
                cblas_zdotu_sub(a.ncols(), a[i], 1, b[0]+i, b.ncols(), &dot);
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

double trace2(const NRSMat<double> &a, const NRMat<double> &b, const bool diagscaled)
{
        if (a.nrows() != b.nrows()||b.nrows()!=b.ncols()) laerror("incompatible SMats in trace2()");
double r=0;
        int i, j, k=0;
        for (i=0; i<a.nrows(); i++)
                for (j=0; j<=i;j++) r += a[k++] * (b[i][j] + (i!=j||diagscaled ? b[j][i] : 0 ));


return r;
}


//Cholesky interface
extern "C" void FORNAME(dpotrf)(const char *UPLO, const FINT *N, double *A, const FINT *LDA, FINT *INFO);
extern "C" void FORNAME(zpotrf)(const char *UPLO, const FINT *N, complex<double> *A, const FINT *LDA, FINT *INFO);

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


void cholesky(NRMat<complex<double> > &a, bool upper)
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


//various norms
extern "C" double FORNAME(zlange)( const char *NORM, const FINT *M, const FINT *N, complex<double> *A, const FINT *LDA, double *WORK); 
extern "C" double FORNAME(dlange)( const char *NORM, const FINT *M, const FINT *N, double *A, const FINT *LDA, double *WORK); 

double MatrixNorm(NRMat<complex<double> > &A, const char norm)
{
        const char TypNorm = (tolower(norm) == 'o')?'I':'O'; //switch c-order/fortran-order
        const FINT M = A.nrows();
        const FINT N = A.ncols();
        double work[M];
        const double ret = FORNAME(zlange)(&TypNorm, &M, &N, A[0], &M, &work[0]);
        return ret;
}

double MatrixNorm(NRMat<double > &A, const char norm)
{
        const char TypNorm = (tolower(norm) == 'o')?'I':'O'; //switch c-order/fortran-order
        const FINT M = A.nrows();
        const FINT N = A.ncols();
        double work[M];
        const double ret = FORNAME(dlange)(&TypNorm, &M, &N, A[0], &M, &work[0]);
        return ret;
}



//condition number
extern "C" void FORNAME(zgecon)( const char *norm, const FINT *n, complex<double> *A, const FINT *LDA, const double *anorm, double *rcond, complex<double> *work, double *rwork, FINT *info);
extern "C" void FORNAME(dgecon)( const char *norm, const FINT *n, double *A, const FINT *LDA, const double *anorm, double *rcond, double *work, double *rwork, FINT *info);

double CondNumber(NRMat<complex<double> > &A, const char norm)
{
        const char TypNorm = (tolower(norm) == 'o')?'I':'O'; //switch c-order/fortran-order
        const FINT N = A.nrows();
        double Norma(0.0), ret(0.0);
        FINT info;
        complex<double> *work;
        double *rwork;

        if(N != A.ncols()){
                laerror("nonsquare matrix in zgecon");
                return 0.0;
        }
        work  = new complex<double>[2*N];
        rwork = new double[2*N];

        Norma = MatrixNorm(A, norm);
        FORNAME(zgecon)(&TypNorm, &N, A[0], &N, &Norma, &ret, &work[0], &rwork[0], &info);
        delete[] work;
        delete[] rwork;
        return ret;
}

double CondNumber(NRMat<double> &A, const char norm)
{
        const char TypNorm = (tolower(norm) == 'o')?'I':'O'; //switch c-order/fortran-order
        const FINT N = A.nrows();
        double Norma(0.0), ret(0.0);
        FINT info;
        double *work;
        double *rwork;

        if(N != A.ncols()){
                laerror("nonsquare matrix in zgecon");
                return 0.0;
        }
        work  = new double[2*N];
        rwork = new double[2*N];

        Norma = MatrixNorm(A, norm);
        FORNAME(dgecon)(&TypNorm, &N, A[0], &N, &Norma, &ret, &work[0], &rwork[0], &info);
        delete[] work;
        delete[] rwork;
        return ret;
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

}//namespace

