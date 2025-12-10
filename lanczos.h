/*
    LA: linear algebra C++ interface library
    Copyright (C) 2025 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>

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
#ifndef _lanczos_h
#define _lanczos_h
#include "vec.h"
#include "smat.h"
#include "mat.h"
#include "sparsemat.h"
#include "nonclass.h"
#include "auxstorage.h"

namespace LA {

//Lanczos diagonalization of hermitean matrix 
//matrix can be any class which has nrows(), ncols(), diagonalof(), issymmetric(), and gemv() available
//does not even have to be explicitly stored - direct CI
//therefore the whole implementation must be a template in a header



template <typename T, typename Matrix>
extern void lanczos(const Matrix &bigmat, NRVec<T> &eivals, NRVec<T> *eivecs, const char *eivecsfile, 
		int nroots=1,  const bool verbose=0, const double eps=1e-6,
	 	const bool incore=1, int maxit=100, const int maxkrylov = 500,
		void (*initguess)(NRVec<T> &)=NULL, const typename LA_traits<T>::normtype *target=NULL)
{
if(!bigmat.issymmetric()) laerror("lanczos only for hermitean matrices");
bool flag=0;
int n=bigmat.nrows();
if ( n!= (int)bigmat.ncols()) laerror("non-square matrix in lanczos");
if(eivals.size()<nroots) laerror("too small eivals dimension in lanczos");

NRVec<T> vec1(n),vec2(n);
NRVec<typename LA_traits<T>::normtype> r(maxkrylov);
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

if(nroots>=maxkrylov) nroots =maxkrylov-1;
int  nroot=0;
int oldnroot;

//@@@@@@@@@@
#if 0

//default guess based on lowest diagonal element of the matrix
if(initguess) initguess(vec1);
else
	{
	const T *diagonal = bigmat.diagonalof(vec2,false,true);
	typename LA_traits<T>::normtype t=1e100; int i,j;  
	vec1=0;
	for(i=0, j= -1; i<n; ++i) if(LA_traits<T>::realpart(diagonal[i])<t) {t=LA_traits<T>::realpart(diagonal[i]); j=i;}
	vec1[j]=1;
	}

//initial step
vec1.normalize();
bigmat.gemv(0,vec2,'n',1,vec1); //avoid bigmat.operator*(vec), since that needs to allocate another n-sized vector
if(incore) v0[0]=vec1; else s0->put(vec1,0); 
if(incore) v1[0]=vec2; else s1->put(vec2,0);


//iterative Lanczos
int it=0;
for(k=2; k<=maxkrylov;++k)
	{
	diagonalize3(smallV,r,1,1,krylovsize+1,&smallSwork,1); //for symmetric matrix they have already been sorted to ascending order in lapack
	if(target) //resort eigenpairs by distance from the target
		{
		NRVec<typename LA_traits<T>::normtype> key(krylovsize+1);
		for(int i=0; i<=krylovsize; ++i) key[i] = abs(r[i] - *target);
		NRPerm<int> p(krylovsize+1);
		key.sort(0,p);
	
		NRVec<typename LA_traits<T>::normtype> rp(maxkrylov);
		NRMat<T> smallVp(maxkrylov,maxkrylov);
		for(int i=0; i<=krylovsize; ++i)
			{
			rp[i]= r[p[i+1]-1];
			for(int j=0; j<=krylovsize; ++j) smallVp(j,i) = smallV(j,p[i+1]-1);
			}
		r = rp;
		smallV = smallVp;
		}

	}
flag=1;
goto finished;

converged:
AuxStorage<typename LA_traits<T>::elementtype> *ev;
if(eivecsfile) ev = new AuxStorage<typename LA_traits<T>::elementtype>(eivecsfile);
if(verbose) {std::cout << "Lanczos converged in "<<it<<" iterations.\n"; std::cout.flush();}
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

//@@@@
#endif

finished:
if(incore) {delete[] v0; delete[] v1;}
else  {delete s0; delete s1;}

if(flag) laerror("no convergence in lanczos");
}

}//namespace
#endif
