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
#ifndef _davidson_h
#define _davidson_h
#include "vec.h"
#include "smat.h"
#include "mat.h"
#include "sparsemat.h"
#include "nonclass.h"
#include "auxstorage.h"

namespace LA {

//Davidson diagonalization of real symmetric matrix (modified Lanczos)

//matrix can be any class which has nrows(), ncols(), diagonalof(), issymmetric(), and gemv() available
//does not even have to be explicitly stored - direct CI
//therefore the whole implementation must be a template in a header
//Note that for efficiency in a direct CI case the diagonalof() should cache its result


//@@@options: left eigenvectors by matrix transpose, overridesymmetric (for nrmat)
//@@@small matrix gdiagonalize - shift complex roots up (option to gdiagonalize?)
//@@@test gdiagonalize whether it sorts the roots and what for complex ones
//@@@implement left eigenvectors for nonsymmetric case


//Davidson algorithm: J. Comp. Phys. 17:817 (1975) 


template <typename T, typename Matrix>
extern void davidson(const Matrix &bigmat, NRVec<T> &eivals, NRVec<T> *eivecs, const char *eivecsfile, 
		int nroots=1,  const bool verbose=0, const double eps=1e-6,
	 	const bool incore=1, int maxit=100, const int maxkrylov = 500,
		void (*initguess)(NRVec<T> &)=NULL)
{
bool flag=0;
int n=bigmat.nrows();
if ( n!= (int)bigmat.ncols()) laerror("non-square matrix in davidson");
if(eivals.size()<nroots) laerror("too small eivals dimension in davidson");

NRVec<T> vec1(n),vec2(n);
NRMat<T> smallH(maxkrylov,maxkrylov),smallS(maxkrylov,maxkrylov),smallV;
NRVec<T> r(maxkrylov);
NRVec<T> *v0,*v1;
AuxStorage<T> *s0,*s1;

if(incore)
	{
	v0 = new NRVec<T>[maxkrylov];
	v1 = new NRVec<T>[maxkrylov];
	}
else
	{
	s0 = new AuxStorage<T>;
	s1 = new AuxStorage<T>;
	}

int i,j;

if(maxkrylov<maxit) maxit=maxkrylov;
if(nroots>=maxkrylov) nroots =maxkrylov-1;
int  nroot=0;
int oldnroot;
smallS=0;
smallH=0;     


//default guess based on lowest diagonal element of the matrix
if(initguess) initguess(vec1);
else
	{
	const T *diagonal = bigmat.diagonalof(vec2,false,true);
	T t=1e100; int i,j;  
	vec1=0;
	for(i=0, j= -1; i<n; ++i) if(diagonal[i]<t) {t=diagonal[i]; j=i;}
	vec1[j]=1;
	}

//init Krylov matrices
bigmat.gemv(0,vec2,'n',1,vec1); //avoid bigmat.operator*(vec), since that needs to allocate another n-sized vector
smallH(0,0) = vec1*vec2;
smallS(0,0) = vec1*vec1;
int krylovsize = 0;
if(incore) v0[0]=vec1; else s0->put(vec1,0); 
if(incore) v1[0]=vec2; else s1->put(vec2,0);


//iterative Davidson loop
int it;
for(it=0; it<maxit; ++it)
{
if(it>0) //if this is the first iteration just need to diagonalise the matrix
	{
	//update reduced overlap matrix
	if(incore) v0[krylovsize]=vec1; else s0->put(vec1,krylovsize);
	for(j=0; j<krylovsize; ++j)
		{
		if(!incore) s0->get(vec2,j);
            	smallS(krylovsize,j) = smallS(j,krylovsize) = vec1*(incore?v0[j]:vec2);
		}
        smallS(krylovsize,krylovsize) = vec1*vec1;
	bigmat.gemv(0,vec2,'n',1,vec1);
	if(incore) v1[krylovsize]=vec2; else s1->put(vec2,krylovsize);

	//update reduced hamiltonian matrix
        smallH(krylovsize,krylovsize) = vec1*vec2;
	for(j=0; j<krylovsize; ++j)
		{
		if(!incore) s0->get(vec1,j);
		smallH(j,krylovsize) = (incore?v0[j]:vec1)*vec2;
            	if(bigmat.issymmetric()) smallH(krylovsize,j) = smallH(j,krylovsize);
		}
	if(!bigmat.issymmetric())
		{
		if(!incore) s0->get(vec1,krylovsize);
		for(j=0; j<krylovsize; ++j)
			{
			if(!incore) s1->get(vec2,j);
			smallH(krylovsize,j) = incore? v1[j]*v0[krylovsize] :vec1*vec2;
			}
		}
	}
smallV=smallH;
NRMat<T> smallSwork=smallS;
if(bigmat.issymmetric()) 
	diagonalize(smallV,r,1,1,krylovsize+1,&smallSwork,1); //for symmetric matrix they have already been sorted to ascending order in lapack
else 
	{
	NRVec<T> ri(krylovsize+1),beta(krylovsize+1);
	NRMat<T> scratch;
	scratch=smallV;
	gdiagonalize(scratch, r, ri,NULL, &smallV, 1, krylovsize+1, 2, 0, &smallSwork, &beta);
	for(int i=0; i<=krylovsize; ++i) {r[i]/=beta[i]; ri[i]/=beta[i];}
	}

T eival_n=r[nroot];
oldnroot=nroot;
typename LA_traits<T>::normtype test=std::abs(smallV(krylovsize,nroot));
if(test<eps) nroot++;
if(it==0) nroot = 0;
for(int iroot=0; iroot<=std::min(krylovsize,nroots-1); ++iroot)
	{
        test = std::abs(smallV(krylovsize,iroot));
        if(test>eps) nroot=std::min(nroot,iroot);
        if(verbose && iroot<=std::max(oldnroot,nroot)) 
		{
		std::cout <<"Davidson: iter="<<it <<" dim="<<krylovsize<<" root="<<iroot<<" energy="<<r[iroot]<<"\n";
		std::cout.flush();
		}
	}

if(verbose && oldnroot!=nroot) {std::cout <<"root no. "<<oldnroot<<" converged\n"; std::cout.flush();}
if (nroot>=nroots) goto converged;
if (it==maxit-1) break; //not converged

if (krylovsize==maxkrylov) //restart, krylov space exceeded
	{
	if(nroot!=0) {flag=1; goto finished;}
	smallH=0;
	smallS=0;
	vec1=0;
	for(i=0; i<=krylovsize; ++i)
		{
		if(!incore) s0->get(vec2,i);
                vec1.axpy(smallV(i,0),incore?v0[i]:vec2);
		}
	s0->put(vec1,0);
	vec1.normalize();
        krylovsize = 0;
	continue;
	}

//generate the update vector
vec1=0;
for(j=0; j<=krylovsize; ++j)
	{
	if(!incore) s0->get(vec2,j);
        vec1.axpy(-r[nroot]*smallV(j,nroot),incore?v0[j]:vec2);
	if(!incore) s1->get(vec2,j);
        vec1.axpy(smallV(j,nroot),incore?v1[j]:vec2);
	}

{
const T *diagonal = bigmat.diagonalof(vec2,false,true);
eival_n = r[nroot];
for(i=0; i<n; ++i)
	{
	T denom = diagonal[i] - eival_n;
        denom = denom<0?-std::max(0.1,std::abs(denom)):std::max(0.1,std::abs(denom));
        vec1[i] /= denom;
	}
}

//orthogonalise to previous vectors
vec1.normalize();
for(j=0; j<=krylovsize; ++j)
	{
	typename LA_traits<T>::normtype vnorm;
        if(!incore) s0->get(vec2,j);
	do	{
          	T ab = vec1*(incore?v0[j]:vec2) /smallS(j,j);
            	vec1.axpy(-ab,incore?v0[j]:vec2);
		vnorm = vec1.norm();
		vec1 *= (1./vnorm);
		} while (vnorm<0.99);
	}

//here it is possible to apply some purification procedure if the eivector has to fulfill other conditions
//vec1.normalize(); //after the purification

++krylovsize; //enlarge Krylov space
}
flag=1;
goto finished;

converged:
AuxStorage<typename LA_traits<T>::elementtype> *ev;
if(eivecsfile) ev = new AuxStorage<typename LA_traits<T>::elementtype>(eivecsfile);
if(verbose) {std::cout << "Davidson converged in "<<it<<" iterations.\n"; std::cout.flush();}
for(nroot=0; nroot<nroots; ++nroot)
	{
        eivals[nroot]=r[nroot];
	if(eivecs)
		{
		vec1=0;
		for(j=0; j<=krylovsize; ++j )
			{
            		if(!incore) s0->get(vec2,j);
			vec1.axpy(smallV(j,nroot),incore?v0[j]:vec2);
			}
		vec1.normalize();
        	if(eivecs) eivecs[nroot]|=vec1;
		if(eivecsfile)
			{
			ev->put(vec1,nroot);
			}
		}
	}

if(eivecsfile) delete ev;

finished:
if(incore) {delete[] v0; delete[] v1;}
else  {delete s0; delete s1;}

if(flag) laerror("no convergence in davidson");
}

}//namespace
#endif
