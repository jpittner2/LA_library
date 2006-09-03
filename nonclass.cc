#ifndef NONCBLAS
extern "C" {
#include "cblas.h"
#include "clapack.h"
}
#else
#include "noncblas.h"
#endif

#include "vec.h"
#include "smat.h"
#include "mat.h"
#include "nonclass.h"
#include "qsort.h"


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
INSTANTIZE(short)
INSTANTIZE(char)
INSTANTIZE(unsigned char)
INSTANTIZE(unsigned long)
INSTANTIZE(unsigned int)

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
		for (int i=0; i<n; ++i)  {double t=A[i][i]; if(!finite(t) || abs(t) < EPSDET ) {*det=0.; break;} else *det *=t;}
		//change sign of det by parity of ipiv permutation
		if(*det) for (int i=0; i<n; ++i) if(i!=ipiv[i]) *det = -(*det);
	}
	if(det && r>0) *det = 0;
	delete [] ipiv;
	if (r>0 && B) laerror("singular matrix in lapack_gesv");
}

void linear_solve(NRMat<double> &A, NRMat<double> *B, double *det, int n)
{
if(n<=0) n=A.nrows(); //default - whole matrix
if (n==A.nrows() && B && A.nrows() != B->ncols() || B && n>B->ncols() ||n>A.nrows()) laerror("incompatible matrices in linear_solve()");
if(B) B->copyonwrite();
linear_solve_do(A,B?(double *)B:NULL,B?B->nrows() : 0,  B?B->ncols():A.nrows(), det,n);
}

void linear_solve(NRMat<double> &A, NRVec<double> &B, double *det, int n)
{
if(n<=0) n=A.nrows(); //default - whole matrix
if(n==A.nrows() && A.nrows() != B.size() || n>B.size()||n>A.nrows() ) laerror("incompatible matrices in linear_solve()");
B.copyonwrite();
linear_solve_do(A,(double *)B,1,A.nrows(),det,n);
}


// Next routines are not available in clapack, fotran ones will be used with an
// additional swap/transpose of outputs when needed

extern "C" void FORNAME(dspsv)(const char *UPLO, const int *N, const int *NRHS,
		double *AP, int *IPIV, double *B, const int *LDB, int *INFO);

static void linear_solve_do(NRSMat<double> &a, double *b, const int nrhs, const int ldb, double *det, int n)
{
	int r, *ipiv;
	a.copyonwrite();
	ipiv = new int[n];
	char U = 'U';
	FORNAME(dspsv)(&U, &n, &nrhs, a, ipiv, b, &ldb,&r);
	if (r < 0) {
		delete[] ipiv;
		laerror("illegal argument in spsv() call of linear_solve()");
	}
	if (det && r == 0) {
		*det = 1.;
		for (int i=1; i<n; i++) {double t=a(i,i); if(!finite(t) || abs(t) < EPSDET ) {*det=0.; break;} else *det *= t;}
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


extern "C" void FORNAME(dsyev)(const char *JOBZ, const char *UPLO, const int *N,
		double *A, const int *LDA, double *W, double *WORK, const int *LWORK, int *INFO);

extern "C" void FORNAME(dsygv)(const int *ITYPE, const char *JOBZ, const char *UPLO, const int *N,
                double *A, const int *LDA, double *B, const int *LDB, double *W, double *WORK, const int *LWORK, int *INFO);


// a will contain eigenvectors (columns if corder==1), w eigenvalues
void diagonalize(NRMat<double> &a, NRVec<double> &w, const bool eivec, 
		const bool corder, int n, NRMat<double> *b, const int itype)
{
	int m = a.nrows();
	if (m != a.ncols()) laerror("diagonalize() call with non-square matrix");
	if (a.nrows() != w.size()) 
		laerror("inconsistent dimension of eigenvalue vector in diagonalize()");
	if(n==0) n=m;
	if(n<0||n>m) laerror("actual dimension out of range in diagonalize");
	if(b) if(n>b->nrows() || n> b->ncols()) laerror("wrong B matrix dimension in diagonalize");

	a.copyonwrite();
	w.copyonwrite();
	if(b) b->copyonwrite();

	int r = 0;
	char U ='U';
	char vectors = 'V';
	if (!eivec) vectors = 'N';
	int LWORK = -1;
	double WORKX;
	int ldb=0; if(b) ldb=b->ncols();

	// First call is to determine size of workspace
	if(b) FORNAME(dsygv)(&itype,&vectors, &U, &n, a, &m, *b, &ldb, w, &WORKX, &LWORK, &r );
	else FORNAME(dsyev)(&vectors, &U, &n, a, &m, w, &WORKX, &LWORK, &r );
	LWORK = (int)WORKX;
	double *WORK = new double[LWORK];
	if(b) FORNAME(dsygv)(&itype,&vectors, &U, &n, a, &m, *b,&ldb, w, WORK, &LWORK, &r );
	else FORNAME(dsyev)(&vectors, &U, &n, a, &m, w, WORK, &LWORK, &r );
	delete[] WORK;
	if (vectors == 'V' && corder) a.transposeme(n);

	if (r < 0) laerror("illegal argument in sygv/syev in diagonalize()");
	if (r > 0) laerror("convergence problem in sygv/syev in diagonalize()");
}



extern "C" void FORNAME(dspev)(const char *JOBZ, const char *UPLO, const int *N,
		double *AP, double *W, double *Z, const int *LDZ, double *WORK, int *INFO);

extern "C" void FORNAME(dspgv)(const int *ITYPE, const char *JOBZ, const char *UPLO, const int *N,
                double *AP, double *BP, double *W, double *Z, const int *LDZ, double *WORK, int *INFO);


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

	int r = 0;
	char U = 'U';
	char job = v ? 'v' : 'n';

	double *WORK = new double[3*n];
	int ldv=v?v->ncols():n;
	if(b) FORNAME(dspgv)(&itype,&job, &U, &n, a, *b, w, v?(*v)[0]:(double *)0, &ldv, WORK,  &r );
	else FORNAME(dspev)(&job, &U, &n, a, w, v?(*v)[0]:(double *)0, &ldv, WORK,  &r );
	delete[] WORK;
	if (v && corder) v->transposeme(n);

	if (r < 0) laerror("illegal argument in spgv/spev in diagonalize()");
	if (r > 0) laerror("convergence problem in spgv/spev in diagonalize()");
}


extern "C" void FORNAME(dgesvd)(const char *JOBU,  const char *JOBVT,  const int *M,
		const int *N,  double *A, const int *LDA, double *S, double *U, const int *LDU,
		double *VT, const int *LDVT, double *WORK, const int *LWORK, int *INFO );

void singular_decomposition(NRMat<double> &a, NRMat<double> *u, NRVec<double> &s,
		NRMat<double> *v, const bool corder, int m, int n)
{
	int m0 = a.nrows();
	int n0 = a.ncols();
	if(m<=0) m=m0;
	if(n<=0) n=n0;
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
	int lwork = -1;
	int r;
	FORNAME(dgesvd)(&jobv, &jobu, &n, &m, a, &n0, s, v?(*v)[0]:0, &n0,
			u?(*u)[0]:0, &m0, &work0, &lwork, &r);
	lwork = (int) work0;
	double *work = new double[lwork];
	FORNAME(dgesvd)(&jobv, &jobu, &n, &m, a, &n0, s, v?(*v)[0]:0, &n0,
			u?(*u)[0]:0, &m0, work, &lwork, &r);
	delete[] work;
	if (v && corder) v->transposeme(n);

	if (r < 0) laerror("illegal argument in gesvd() of singular_decomposition()");
	if (r > 0) laerror("convergence problem in gesvd() of ingular_decomposition()");
}


extern "C" void FORNAME(dgeev)(const char *JOBVL, const char *JOBVR, const int *N,
		double *A, const int *LDA, double *WR, double *WI, double *VL, const int *LDVL,
		double *VR, const int *LDVR, double *WORK, const int *LWORK, int *INFO );

extern "C" void FORNAME(dggev)(const char *JOBVL, const char *JOBVR, const int *N,
		double *A, const int *LDA, double *B, const int *LDB, double *WR, double *WI,  double *WBETA, 
		 double *VL, const int *LDVL,  double *VR, const int *LDVR,  
		double *WORK, const int *LWORK, int *INFO );


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
		const int sorttype, const bool biorthonormalize,
		NRMat<double> *b, NRVec<double> *beta)
{
	if(n<=0) n = a.nrows();
	if (n > a.ncols() || n>a.nrows() ) laerror("gdiagonalize() call for a non-square matrix");
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
	
	char jobvl = vl ? 'V' : 'N';
	char jobvr = vr ? 'V' : 'N';
	double work0;
	int lwork = -1;
	int r;
	int lda=a.ncols();
	int ldb=0;
	if(b) ldb=b->ncols();
	int ldvl= vl?vl->ncols():lda;
	int ldvr= vr?vr->ncols():lda;
	if(b)  FORNAME(dggev)(&jobvr, &jobvl, &n, a, &lda, *b, &ldb, wr, wi, *beta, vr?vr[0]:(double *)0,
                        &ldvr, vl?vl[0]:(double *)0, &ldvl, &work0, &lwork, &r);
	else FORNAME(dgeev)(&jobvr, &jobvl, &n, a, &lda, wr, wi, vr?vr[0]:(double *)0,
			&ldvr, vl?vl[0]:(double *)0, &ldvl, &work0, &lwork, &r);
	lwork = (int) work0;
	double *work = new double[lwork];
	if(b) FORNAME(dggev)(&jobvr, &jobvl, &n, a, &lda, *b, &ldb, wr, wi, *beta, vr?vr[0]:(double *)0,
                        &ldvr, vl?vl[0]:(double *)0, &ldvl, work, &lwork, &r);
	else FORNAME(dgeev)(&jobvr, &jobvl, &n, a, &lda, wr, wi, vr?vr[0]:(double *)0,
			&ldvr, vl?vl[0]:(double *)0, &ldvl, work, &lwork, &r);
	delete[] work;
//@@@
//cout <<"TEST dgeev\n"<<wr<<wi<<*vr<<*vl<<endl;

	if (r < 0) laerror("illegal argument in ggev/geev in gdiagonalize()");
	if (r > 0) laerror("convergence problem in ggev/geev in gdiagonalize()");

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
				tmp=1./sqrt(abs(tmp));
				cblas_dscal(n,tmp,(*vl)[i],1);
				cblas_dscal(n,tmp,(*vr)[i],1);
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
				t=.5*sqrt(t);
				alp=sqrt(.5*(t+x));
				bet=sqrt(.5*(t-x));
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


void gdiagonalize(NRMat<double> &a, NRVec< complex<double> > &w,
		NRMat< complex<double> >*vl, NRMat< complex<double> > *vr,
		const bool corder, int n, const int sorttype, const bool biorthonormalize,
		NRMat<double> *b, NRVec<double> *beta)
{
	if(!corder)  laerror("gdiagonalize() corder 0 not implemented");
	if(n<=0)  n = a.nrows();
	if(n> a.nrows() || n ==  a.nrows() && n != a.ncols()) laerror("gdiagonalize() call for a non-square matrix");

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
NRMat<complex<double> > a0=complexify(a);
	gdiagonalize(a, w, &u, &v);//a gets destroyed, eigenvectors are rows
	NRVec< complex<double> > z = diagofproduct(u, v, 1, 1);
/*
cout <<"TEST matrixfunction\n"<<w<<u<<v<<z;
cout <<"TEST matrixfunction1 "<< u*a0 - diagonalmatrix(w)*u<<endl;
cout <<"TEST matrixfunction2 "<< a0*v.transpose(1) - v.transpose(1)*diagonalmatrix(w)<<endl;
cout <<"TEST matrixfunction3 "<< u*v.transpose(1)<<diagonalmatrix(z)<<endl;
NRVec< complex<double> > wz(n);
for (int i=0; i<a.nrows(); i++) wz[i] = w[i]/z[i];
cout <<"TEST matrixfunction4 "<< a0<< v.transpose(true)*diagonalmatrix(wz)*u<<endl;
*/

	for (int i=0; i<a.nrows(); i++) w[i] = (*f)(w[i])/z[i];
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
complex<double> myclog (const complex<double> &x) 
{
	return log(x);
}

complex<double> mycexp (const complex<double> &x)
{
        return exp(x);
}


complex<double> sqrtinv (const complex<double> &x)
{
        return 1./sqrt(x);
}

double sqrtinv (const double x)
{
        return 1./sqrt(x);
}


NRMat<double>  log(const NRMat<double> &a)
{
	return matrixfunction(a, &myclog, 1);
}

NRMat<double>  exp0(const NRMat<double> &a)
{
        return matrixfunction(a, &mycexp, 1);
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
	//for (int i=0; i<a.nrows(); i++) r -= a(i,i)*b(i,i);
	r -= cblas_ddot(a.nrows(),a,a.nrows()+1,b,a.nrows()+1);
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
       	 	dl[i] = sqrt(x);
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
                if(abs(v[i][i])<min)
                        {
                        imin=i;
                        min=abs(v[i][i]);
                        } 
        cblas_dscal(n,-1.,v[imin],1);
        }
}

