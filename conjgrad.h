#include "vec.h"
#include "smat.h"
#include "mat.h"
#include "sparsemat.h"
#include "nonclass.h"
#include <iomanip>

//conjugate gradient solution of a linear system

//matrix can be any class which has nrows(), ncols(), diagonalof() and NRVec::gemv() available
//does not even have to be explicitly stored
//Conjugate gradient algorithm, cf. Bulirsch-Stoer book


template<typename T, typename Matrix>
extern void conjgrad(const Matrix &bigmat, const NRVec<T> &b, NRVec<T> &x, const bool doguess, const double tol, const int itmax, const bool verbose, bool issquare,const bool precondition)
{
int m=bigmat.nrows();
int n=bigmat.ncols();

if(x.size()!=n || b.size() != m) laerror("incompatible vectors and matrix sizes in conjgrad");
if(m!=n) issquare=0;

double t,tt,bscal,ascal;

NRVec<T> p,rr, *r;
NRVec<T> q(m),s(m);
if(issquare) r=&s; else r = new NRVec<T>(m);

if(doguess)
	{
	x.gemv(0,bigmat,'t',-1.,b);
        if(precondition) bigmat.diagonalof(x,true);
        x.normalize();
	}

s.gemv(0,bigmat,'n',-1.,x);
s+=b;
if(!issquare) (*r).gemv(0,bigmat,'t',1,s);
rr= *r;
if(precondition) bigmat.diagonalof(rr,true);
p=rr;

for(int iter=0; iter<= itmax; iter++)
	{
	double err=p.norm();
	if(verbose) cout << "conjgrad: iter= "<<iter<<" err= "<<
		setiosflags(ios::scientific)<<setprecision(8) <<err<<
		resetiosflags(ios::scientific)<<setprecision(12)<<"\n";
	if(err <= tol) break;

	q.gemv(0,bigmat,'n',1,p);
	tt= (*r) * rr;
	t=issquare?p*q:q*q;
	if(!t) {if(!issquare) delete r; laerror("conjgrad: singular matrix 1");}
	ascal=tt/t;
	x.axpy(ascal,p);
	s.axpy(-ascal,q);
	if(!issquare) (*r).gemv(0,bigmat,'t',1,s);
	rr= *r;
	if(precondition) bigmat.diagonalof(rr,true);
	if(!tt) {if(!issquare) delete r; laerror("conjgrad: singular matrix 2");}
	bscal= ((*r)*rr)/tt;
	rr.axpy(bscal,p);
	p=rr;
	}

if(!issquare) delete r;

}

