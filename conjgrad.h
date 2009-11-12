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
#ifndef _CONJGRAD_H_
#define _CONJGRAD_H_
#include "vec.h"
#include "smat.h"
#include "mat.h"
#include "sparsemat.h"
#include "nonclass.h"
#include <iomanip>

namespace LA {

//conjugate gradient solution of a linear system

//matrix can be any class which has nrows(), ncols(), diagonalof() and gemv() available
//does not even have to be explicitly stored
//Conjugate gradient algorithm, cf. Bulirsch-Stoer book


template<typename T, typename Matrix>
extern bool conjgrad(const Matrix &bigmat, const NRVec<T> &b, NRVec<T> &x, const bool doguess=true, const double tol=1e-8, const int itmax=1000, const bool verbose=true, bool issquare=1,const bool precondition=1)
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
	bigmat.gemv(0,x,'t',-1.,b); //x.gemv(0,bigmat,'t',-1.,b);
        if(precondition) bigmat.diagonalof(x,true);
        x.normalize();
	}

bigmat.gemv(0,s,'n',-1.,x); //s.gemv(0,bigmat,'n',-1.,x);
s+=b;
if(!issquare) bigmat.gemv(0,*r,'t',1,s); //(*r).gemv(0,bigmat,'t',1,s);
rr= *r;
if(precondition) bigmat.diagonalof(rr,true);
p=rr;

for(int iter=0; iter<= itmax; iter++)
	{
	double err=p.norm();
	if(verbose) 
		{
		std::cout << "conjgrad: iter= "<<iter<<" err= "<<
		std::setiosflags(std::ios::scientific)<<std::setprecision(8) <<err<<
		std::resetiosflags(std::ios::scientific)<<std::setprecision(12)<<"\n";
		std::cout.flush();
		}
	if(err <= tol)
		{
		if(!issquare) delete r;
		return true;
		} 

	bigmat.gemv(0,q,'n',1,p); //q.gemv(0,bigmat,'n',1,p);
	tt= (*r) * rr;
	t=issquare?p*q:q*q;
	if(!t) {if(!issquare) delete r; laerror("conjgrad: singular matrix 1");}
	ascal=tt/t;
	x.axpy(ascal,p);
	s.axpy(-ascal,q);
	if(!issquare) bigmat.gemv(0,*r,'t',1,s); //(*r).gemv(0,bigmat,'t',1,s);
	rr= *r;
	if(precondition) bigmat.diagonalof(rr,true);
	if(!tt) {if(!issquare) delete r; laerror("conjgrad: singular matrix 2");}
	bscal= ((*r)*rr)/tt;
	rr.axpy(bscal,p);
	p=rr;
	}

if(!issquare) delete r;
return false;
}

}//namespace
#endif
