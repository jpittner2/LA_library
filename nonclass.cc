extern "C" {
#include "atlas_enum.h"
#include "clapack.h"
}
#include "la.h"

#ifdef FORTRAN_
#define FORNAME(x) x##_
#else
#define FORNAME(x) x
#endif

#define INSTANTIZE(T) \
template void lawritemat(FILE *file,const T *a,int r,int c,const char *form0, \
		int nodim,int modulo, int issym);
INSTANTIZE(double)
INSTANTIZE(complex<double>)
INSTANTIZE(int)
INSTANTIZE(char)

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

// LA errorr handler
void laerror(const char *s1, const char *s2, const char *s3, const char *s4)
{
  std::cerr << "LA:ERROR - ";
  if(!s1)
    std::cerr << "udefined.";
  else {
    if(s1) std::cerr << s1;
    if(s2) std::cerr << s2;
    if(s3) std::cerr << s3;
    if(s4) std::cerr << s4;
  }
  std::cerr << endl;
  exit(1);
}

//////////////////////
// LAPACK interface //
//////////////////////

// A will be overwritten, B will contain the solutions, A is nxn, B is rhs x n
void linear_solve(NRMat<double> &A, NRMat<double> *B, double *det)
{
	int r, *ipiv;
	
	if (A.nrows() != A.ncols()) laerror("linear_solve() call for non-square matrix");
	if (B && A.nrows() != B->ncols()) laerror("incompatible matrices in linear_solve()");
	A.copyonwrite();
	if (B) B->copyonwrite();
	ipiv = new int[A.nrows()];
	r = clapack_dgesv(CblasRowMajor, A.nrows(), B ? B->nrows() : 0, A[0], A.ncols(),
			ipiv, B ? B[0] : (double *)0, B ? B->ncols() : A.nrows());
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
	if (r>0 && B) laerror("singular matrix in lapack_gesv");
}


// Next routines are not available in clapack, fotran ones will b used with an
// additional swap/transpose of outputs when needed

extern "C" void FORNAME(dspsv)(const char *UPLO, const int *N, const int *NRHS,
		double *AP, int *IPIV, double *B, const int *LDB, int *INFO);

void linear_solve(NRSMat<double> &a, NRMat<double> *b, double *det)
{
	int r, *ipiv;
	if (det) cerr << "@@@ sign of the determinant not implemented correctly yet\n";
	if (b && a.nrows() != b->ncols())
		laerror("incompatible matrices in symmetric linear_solve()");
	a.copyonwrite();
	if (b) b->copyonwrite();
	ipiv = new int[a.nrows()];
	char U = 'U';
	int n = a.nrows();
	int nrhs = 0;
	if (b) nrhs = b->nrows();
	int ldb = b ? b->ncols() : a.nrows();
	FORNAME(dspsv)(&U, &n, &nrhs, a, ipiv, b?(*b)[0]:0, &ldb,&r);
	if (r < 0) {
		delete[] ipiv;
		laerror("illegal argument in spsv() call of linear_solve()");
	}
	if (det && r >= 0) {
		*det = a(0,0);
		for (int i=1; i<a.nrows(); i++) *det *= a(i,i);
		for (int i=0; i<a.nrows(); i++)
			if (ipiv[i] != i) *det = -(*det);
	}
	delete[] ipiv;
	if (r > 0 && b) laerror("singular matrix in linear_solve(SMat&, Mat*, double*");
}


extern "C" void FORNAME(dsyev)(const char *JOBZ, const char *UPLO, const int *N,
		double *A, const int *LDA, double *W, double *WORK, const int *LWORK, int *INFO);

// a will contain eigenvectors (columns if corder==1), w eigenvalues
void diagonalize(NRMat<double> &a, NRVec<double> &w, const bool eivec, 
		const bool corder, int n)
{
	int m = a.nrows();
	if (m != a.ncols()) laerror("diagonalize() call with non-square matrix");
	if (a.nrows() != w.size()) 
		laerror("inconsistent dimension of eigenvalue vector in diagonalize()");
	if(n==0) n=m;
	if(n<0||n>m) laerror("actual dimension out of range in diagonalize");

	a.copyonwrite();
	w.copyonwrite();

	int r = 0;
	char U ='U';
	char vectors = 'V';
	if (!eivec) vectors = 'N';
	int LWORK = -1;
	double WORKX;

	// First call is to determine size of workspace
	FORNAME(dsyev)(&vectors, &U, &n, a, &m, w, (double *)&WORKX, &LWORK, &r );
	LWORK = (int)WORKX;
	double *WORK = new double[LWORK];
	FORNAME(dsyev)(&vectors, &U, &n, a, &m, w, WORK, &LWORK, &r );
	delete[] WORK;
	if (vectors == 'V' && corder) a.transposeme();

	if (r < 0) laerror("illegal argument in syev() of diagonalize()");
	if (r > 0) laerror("convergence problem in syev() of diagonalize()");
}



extern "C" void FORNAME(dspev)(const char *JOBZ, const char *UPLO, const int *N,
		double *AP, double *W, double *Z, const int *LDZ, double *WORK, int *INFO);

// v will contain eigenvectors, w eigenvalues
void diagonalize(NRSMat<double> &a, NRVec<double> &w, NRMat<double> *v,
		const bool corder)
{
	int n = a.nrows();
	if (v) if (v->nrows() != v ->ncols() || n != v->nrows())
		laerror("diagonalize() call with inconsistent dimensions");
	if (n != w.size()) laerror("inconsistent dimension of eigenvalue vector");

	a.copyonwrite();
	w.copyonwrite();

	int r = 0;
	char U = 'U';
	char job = v ? 'v' : 'n';

	double *WORK = new double[3*n];
	FORNAME(dspev)(&job, &U, &n, a, w, v?(*v)[0]:(double *)0, &n, WORK,  &r );
	delete[] WORK;
	if (v && corder) v->transposeme();

	if (r < 0) laerror("illegal argument in spev() of diagonalize()");
	if (r > 0) laerror("convergence problem in spev() of diagonalize()");
}


extern "C" void FORNAME(dgesvd)(const char *JOBU,  const char *JOBVT,  const int *M,
		const int *N,  double *A, const int *LDA, double *S, double *U, const int *LDU,
		double *VT, const int *LDVT, double *WORK, const int *LWORK, int *INFO );

void singular_decomposition(NRMat<double> &a, NRMat<double> *u, NRVec<double> &s,
		NRMat<double> *v, const bool corder)
{
	int m = a.nrows();
	int n = a.ncols();
	if (u) if (m != u->nrows() || m!= u->ncols())
		laerror("inconsistent dimension of U Mat in singular_decomposition()");
	if (s.size() < m && s.size() < n) 
		laerror("inconsistent dimension of S Vec in singular_decomposition()");
	if (v) if (n != v->nrows() || n != v->ncols())
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
	int lwork = -1;
	int r;
	FORNAME(dgesvd)(&jobv, &jobu, &n, &m, a, &n, s, v?(*v)[0]:0, &n,
			u?(*u)[0]:0, &m, &work0, &lwork, &r);
	lwork = (int) work0;
	double *work = new double[lwork];
	FORNAME(dgesvd)(&jobv, &jobu, &n, &m, a, &n, s, v?(*v)[0]:0, &n,
			u?(*u)[0]:0, &m, &work0, &lwork, &r);
	delete[] work;
	if (v && corder) v->transposeme();

	if (r < 0) laerror("illegal argument in gesvd() of singular_decomposition()");
	if (r > 0) laerror("convergence problem in gesvd() of ingular_decomposition()");
}


extern "C" void FORNAME(dgeev)(const char *JOBVL, const char *JOBVR, const int *N,
		double *A, const int *LDA, double *WR, double *WI, double *VL, const int *LDVL,
		double *VR, const int *LDVR, double *WORK, const int *LWORK, int *INFO );

void gdiagonalize(NRMat<double> &a, NRVec<double> &wr, NRVec<double> &wi,
		NRMat<double> *vl, NRMat<double> *vr, const bool corder)
{
	int n = a.nrows();
	if (n != a.ncols()) laerror("gdiagonalize() call for a non-square matrix");
	if (n != wr.size()) 
		laerror("inconsistent dimension of eigen vector in gdiagonalize()");
	if (vl) if (n != vl->nrows() || n != vl->ncols())
		laerror("inconsistent dimension of vl in gdiagonalize()");
	if (vr) if (n != vr->nrows() || n != vr->ncols())
		laerror("inconsistent dimension of vr in gdiagonalize()");

	a.copyonwrite();
	wr.copyonwrite();
	wi.copyonwrite();
	if (vl) vl->copyonwrite();
	if (vr) vr->copyonwrite();
	
	char jobvl = vl ? 'V' : 'N';
	char jobvr = vr ? 'V' : 'N';
	double work0;
	int lwork = -1;
	int r;
	FORNAME(dgeev)(&jobvr, &jobvl, &n, a, &n, wr, wi, vr?vr[0]:(double *)0,
			&n, vl?vl[0]:(double *)0, &n, &work0, &lwork, &r);
	lwork = (int) work0;
	double *work = new double[lwork];
	FORNAME(dgeev)(&jobvr, &jobvl, &n, a, &n, wr, wi, vr?vr[0]:(double *)0,
			&n, vl?vl[0]:(double *)0, &n, &work0, &lwork, &r);
	delete[] work;

	if (corder) {
		if (vl) vl->transposeme();
		if (vr) vr->transposeme();
	}

	if (r < 0) laerror("illegal argument in geev() of gdiagonalize()");
	if (r > 0) laerror("convergence problem in geev() of gdiagonalize()");
}

void gdiagonalize(NRMat<double> &a, NRVec< complex<double> > &w,
		NRMat< complex<double> >*vl, NRMat< complex<double> > *vr)
{
	int n = a.nrows();
	if(n != a.ncols()) laerror("gdiagonalize() call for a non-square matrix");

	NRVec<double> wr(n), wi(n);
	NRMat<double> *rvl = 0;
	NRMat<double> *rvr = 0;
	if (vl) rvl = new NRMat<double>(n, n);
	if (vr) rvr = new NRMat<double>(n, n);
	gdiagonalize(a, wr, wi, rvl, rvr, 0);
	
	//process the results into complex matrices
	int i;
	for (i=0; i<n; i++) w[i] = complex<double>(wr[i], wi[i]);
	if (rvl || rvr) {
		i = 0;
		while (i < n) {
			if (wi[i] == 0) {
				if (vl) for (int j=0; j<n; j++) (*vl)[i][j] = (*rvl)[i][j];
				if (vr) for (int j=0; j<n; j++) (*vr)[i][j] = (*rvr)[i][j];
				i++;
			} else {
				if (vl)
					for (int j=0; j<n; j++) {
						(*vl)[i][j] = complex<double>((*rvl)[i][j], (*rvl)[i+1][j]);
						(*vl)[i+1][j] = complex<double>((*rvl)[i][j], -(*rvl)[i+1][j]);
					} 
				if (vr)
					for (int j=0; j<n; j++) {
						(*vr)[i][j] = complex<double>((*rvr)[i][j], (*rvr)[i+1][j]);
						(*vr)[i+1][j] = complex<double>((*rvr)[i][j], -(*rvr)[i+1][j]);
					}
				i += 2;
			}
		}
	}
	if (rvl) delete rvl;
	if (rvr) delete rvr;
}


const NRMat<double> realpart(const NRMat< complex<double> > &a)
{
	NRMat<double> result(a.nrows(), a.ncols());
	cblas_dcopy(a.nrows()*a.ncols(), (const double *)a[0], 2, result, 1);
	return result;
}

const NRMat<double> imagpart(const NRMat< complex<double> > &a)
{
	NRMat<double> result(a.nrows(), a.ncols());
	cblas_dcopy(a.nrows()*a.ncols(), (const double *)a[0]+1, 2, result, 1);
	return result;
}

const NRMat< complex<double> > realmatrix (const NRMat<double> &a)
{
	NRMat <complex<double> > result(a.nrows(), a.ncols());
	cblas_dcopy(a.nrows()*a.ncols(), a, 1, (double *)result[0], 2);
	return result;
}

const NRMat< complex<double> > imagmatrix (const NRMat<double> &a)
{
	NRMat< complex<double> > result(a.nrows(), a.ncols());
	cblas_dcopy(a.nrows()*a.ncols(), a, 1, (double *)result[0]+1, 2);
	return result;
}


NRMat<double> matrixfunction(NRMat<double> a, complex<double>
		(*f)(const complex<double> &), const bool adjust)
{
	int n = a.nrows();
	NRMat< complex<double> > u(n, n), v(n, n);
	NRVec< complex<double> > w(n);
	gdiagonalize(a, w, &u, &v);
	NRVec< complex<double> > z = diagofproduct(u, v, 1, 1);

	for (int i=0; i<a.nrows(); i++) w[i] = (*f)(w[i]/z[i]);
	u.diagmultl(w);

	NRMat< complex<double> > r(n, n);
	r.gemm(0.0, v, 'c', u, 'n', 1.0);
	double inorm = cblas_dnrm2(n*n, (double *)r[0]+1, 2);
	if (inorm > 1e-10) {
		cout << "norm = " << inorm << endl;
		laerror("nonzero norm of imaginary part of real matrixfunction");
	}
	return realpart(r);
}

NRMat<double> matrixfunction(NRSMat<double> a, double (*f) (double))
{
	int n = a.nrows();
	NRVec<double> w(n);
	NRMat<double> v(n, n);
	diagonalize(a, w, &v, 0);

	for (int i=0; i<a.nrows(); i++) w[i] = (*f)(w[i]);
	NRMat<double> u = v;
	v.diagmultl(w);
	NRMat<double> r(n, n);
	r.gemm(0.0, u, 't', v, 'n', 1.0);
	return r;
}

// instantize template to an addresable function
complex<double> myclog (const complex<double> &x) 
{
	return log(x);
}

NRMat<double>  log(const NRMat<double> &a)
{
	return matrixfunction(a, &myclog, 1);
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
			laerror("incompatible Mats in diagofproduct<complex>()");
	if (trb) return cblas_ddot(a.nrows()*a.ncols(), a, 1, b, 1);

	double sum = 0.0;
	for (int i=0; i<a.nrows(); i++)
		sum += cblas_ddot(a.ncols(), a[i], 1, b[0]+i, b.ncols());

	return sum;
}


double trace2(const NRSMat<double> &a, const NRSMat<double> &b,
		const bool diagscaled)
{
	if (a.nrows() != b.nrows()) laerror("incompatible SMats in trace2()");

	double r = 2.0*cblas_ddot(a.nrows()*(a.nrows()+1)/2, a, 1, b, 1);
	if (diagscaled) return r;
	for (int i=0; i<a.nrows(); i++) r -= a(i,i)*b(i,i);
	return r;
}


//generalized diagonalization, eivecs will be in columns of a
//counts with actual dimension smaller than allocated dimension

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
/*
c
c     this routine is a translation of the algol procedure reduc1,
c     num. math. 11, 99-110(1968) by martin and wilkinson.
c     handbook for auto. comp., vol.ii-linear algebra, 303-314(1971).
c
c     this routine reduces the generalized symmetric eigenproblem
c     ax=(lambda)bx, where b is positive definite, to the standard
c     symmetric eigenproblem using the cholesky factorization of b.
c
c     on input
c
c        nm must be set to the row dimension of two-dimensional
c          array parameters as declared in the calling program
c          dimension statement.
c
c        n is the order of the matrices a and b.  if the cholesky
c          factor l of b is already available, n should be prefixed
c          with a minus sign.
c
c        a and b contain the real symmetric input matrices.  only the
c          full upper triangles of the matrices need be supplied.  if
c          n is negative, the strict lower triangle of b contains,
c          instead, the strict lower triangle of its cholesky factor l.
c
c        dl contains, if n is negative, the diagonal elements of l.
c
c     on output
c
c        a contains in its full lower triangle the full lower triangle
c          of the symmetric matrix derived from the reduction to the
c          standard form.  the strict upper triangle of a is unaltered.
c
c        b contains in its strict lower triangle the strict lower
c          triangle of its cholesky factor l.  the full upper
c          triangle of b is unaltered.
c
c        dl contains the diagonal elements of l.
c
c        ierr is set to
c          zero       for normal return,
c          7*n+1      if b is not positive definite.
c
c     questions and comments should be directed to burton s. garbow,
c     mathematics and computer science div, argonne national laboratory
c
c     this version dated august 1983.
c
*/
//    .......... form l in the arrays b and dl ..........
for(i=0; i<n; ++i)
{
for(j=i; j<n; ++j)
	{
        x = b(i,j) -  cblas_ddot(i,&b(i,0),1,&b(j,0),1);
	if(i==j)
		{
		if(x<=0) laerror("not positive definite metric in gendiagonalize");
       	 	dl[i] = sqrt(x);
		}
	else    
		b(j,i) = x / dl[i];
	}
}

//     .......... form the transpose of the upper triangle of inv(l)*a
//                in the lower triangle of the array a ..........
for(i=0; i<n; ++i)
{
for(j=i; j<n ; ++j)
	{
        x = a(i,j) - cblas_ddot(i,&b(i,0),1,&a(j,0),1);
        a(j,i) = x/dl[i];
	}
}

//     .......... pre-multiply by inv(l) and overwrite ..........
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
/*
c
c     this routine is a translation of the algol procedure rebaka,
c     num. math. 11, 99-110(1968) by martin and wilkinson.
c     handbook for auto. comp., vol.ii-linear algebra, 303-314(1971).
c
c     this routine forms the eigenvectors of a generalized
c     symmetric eigensystem by back transforming those of the
c     derived symmetric matrix determined by  reduc.
c
c     on input
c
c        n is the order of the matrix system.
c
c        b contains information about the similarity transformation
c          (cholesky decomposition) used in the reduction by  reduc
c          in its strict lower triangle.
c
c        dl contains further information about the transformation.
c
c        a contains the eigenvectors to be back transformed
c          in its  columns
c
c     on output
c
c        a contains the transformed eigenvectors
c          in its columns
c
c     questions and comments should be directed to burton s. garbow,
c     mathematics and computer science div, argonne national laboratory
c
*/
for(j=0; j<n; ++j)//eigenvector loop
{
for(int i=n-1; i>=0; --i)//component loop
	{
        if(i<n-1) a(i,j) -= cblas_ddot(n-1-i,&b(i+1,i),m,&a(i+1,j),m);
        a(i,j) /= dl[i];
	}
}

}
