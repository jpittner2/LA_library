#include "la.h"
/*Strassen algorithm*/
// called routine is fortran-compatible
extern "C" void fmm(const char c_transa,const char c_transb,const int m,const int n,const int k,const double alpha,
                const double *a,const int lda,const double *b,const int ldb,const double beta,double *c,const int ldc,
                double *d_aux,int i_naux);
extern "C" void strassen_cutoff(int c, int c1, int c2, int c3);

void NRMat<double>::s_cutoff(const int c, const int c1, const int c2, const int c3) const
{ strassen_cutoff(c,c1,c2,c3);}
void NRMat<double>::strassen(const double beta, const NRMat<double> &a, const char transa, const NRMat<double> &b, const char transb, const double alpha)
{
int l(transa=='n'?a.nn:a.mm);
int k(transa=='n'?a.mm:a.nn);
int kk(transb=='n'?b.nn:b.mm);
int ll(transb=='n'?b.mm:b.nn);

if(l!=nn|| ll!=mm||k!=kk) laerror("incompatible (or undefined size) matrices in strassen");

copyonwrite();
//swap transpositions and order of matrices
fmm(transb,transa,mm,nn,k,alpha,b,b.mm, a, a.mm, beta,*this, mm,NULL,0);
}

//stub for f77 blas called from strassen routine
extern "C" void xerbla_(const char *msg)
{
laerror(msg);
}


