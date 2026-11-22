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

//TODO:
//@@@implement restart when Krylov space is exceeded
//@@@implement inital guess of more than 1 vector (also in davidson) and block version of the methods

namespace LA {

//Lanczos diagonalization of hermitean matrix 
//matrix can be any class which has nrows(), ncols(), diagonalof(), issymmetric(), and gemv() available
//does not even have to be explicitly stored - direct CI
//therefore the whole implementation must be a template in a header
//ghosts: treatment of the spurious copies of converged eigenvalues due to the loss of orthogonality of the Lanczos vectors:
//0 none (plain Lanczos), 1 numerically multiple eigenvalues of the tridiagonal matrix are kept once (cheap),
//2 full reorthogonalization of each new vector to all previous ones (one dot and one axpy per stored vector and iteration)



template <typename T, typename Matrix>
extern void lanczos(const Matrix &bigmat, NRVec<T> &eivals, NRVec<T> *eivecs, const char *eivecsfile, 
		int nroots=1,  const bool verbose=0, const double eps=LA_precision<typename LA_traits<T>::normtype>::eigensolver_eps(),
	 	const bool incore=1, int maxit=100, const int maxkrylov = 1000,
		void (*initguess)(NRVec<T> &)=NULL, const typename LA_traits<T>::normtype *target=NULL, const int ghosts=1)
{
if(!bigmat.issymmetric()) laerror("lanczos only for hermitean matrices");
bool flag=0;
int n=bigmat.nrows();
if ( n!= (int)bigmat.ncols()) laerror("non-square matrix in lanczos");
if(eivals.size()<nroots) laerror("too small eivals dimension in lanczos");

NRVec<T> vec1(n),vec2(n);
NRVec<T> *v0;
AuxStorage<T> *s0,*s1;

if(incore)
	{
	v0 = new NRVec<T>[maxkrylov];
	}
else
	{
	s0 = new AuxStorage<T>;
	}

int i,j;

if(nroots>=maxkrylov) nroots =maxkrylov-1;
int  nroot=0;
int oldnroot;


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

NRVec<typename LA_traits<T>::normtype> alpha(maxkrylov),beta(maxkrylov-1);

//initial step
vec1.normalize();
if(incore) v0[0]=vec1; else s0->put(vec1,0); 
bigmat.gemv(0,vec2,'n',1,vec1); //avoid bigmat.operator*(vec), since that needs to allocate another n-sized vector
{
T tmp=vec2*vec1;
alpha[0]= LA_traits<T>::realpart(tmp);
if(LA_traits<T>::imagpart(tmp)>1e-6) laerror("matrix probably not hermitian in lanczos");
}
vec2.axpy(-alpha[0],vec1); 

NRVec<typename LA_traits<T>::normtype> r(1); r[0]=alpha[0];
typename LA_traits<T>::normtype betanext=vec2.norm(); //norm of the last w vector, the next beta
NRMat<typename LA_traits<T>::normtype> smallV;
int krylovsize=1;
//iterative Lanczos
int it=0;
for(j=1; j<maxkrylov;++j)
	{
	++it;		
	if(it>maxit) {std::cout <<"too many interations in lanczos\n"; flag=1; goto finished;}
	++krylovsize;
	//extend the Krylov space (last w vector expected in vec2, last v vector in vec1)
	beta[j-1] = betanext;
	if(beta[j-1] > 0)
		{
		vec2 /= beta[j-1];
		}
	else
		{
		//generate an arbitrary vector and orthonormalize it to all previous v_j
		vec2.randomize(1.);
		for(int k=0; k<j; ++k)
			{
			T f;
			if(incore) 
				{
				f = v0[k].dot(vec2);
				vec2.axpy(-f,v0[k]);
				}
			else
				{
				NRVec<T> vec3(n);
				s0->get(vec3,k);
                                f = vec3.dot(vec2);
                                vec2.axpy(-f,vec3);
				}
			}
		vec2.normalize();
		}
	//vec2 now stores v_j
	if(incore) v0[j]=vec2; else s0->put(vec2,j);
	vec1 *= -beta[j-1];
	bigmat.gemv(1,vec1,'n',1,vec2);
	{
	T tmp=vec1*vec2;
	alpha[j]= LA_traits<T>::realpart(tmp);
	if(LA_traits<T>::imagpart(tmp)>1e-6) laerror("matrix probably not hermitian in lanczos");
	}
	vec1.axpy(-alpha[j],vec2);
	vec1.swap(vec2); //move new w vector to vec2, new v vector to vec1
	if(ghosts==2) //full reorthogonalization of the new w to all v_k, once more if its norm drops much ("twice is enough")
		{
		typename LA_traits<T>::normtype wnorm=vec2.norm();
		NRVec<T> vec3;
		if(!incore) vec3.resize(n);
		for(int pass=0; pass<2; ++pass)
			{
			for(int k=0; k<=j; ++k)
				{
				if(!incore) s0->get(vec3,k);
				const NRVec<T> &vk= incore? v0[k] : vec3;
				T f=vk.dot(vec2);
				vec2.axpy(-f,vk);
				}
			betanext=vec2.norm();
			if(betanext>0.7*wnorm) break;
			wnorm=betanext;
			}
		}
	else betanext=vec2.norm();

	//diagonalize the tridiagonal
	smallV.resize(j+1,j+1);
	r=alpha.subvector(0,j);
	NRVec<typename LA_traits<T>::normtype> rr=beta.subvector(0,j-1);
	diagonalize3(r,rr,&smallV); 
	if(ghosts==1 && j>=1) //keep one of numerically multiple eigenvalues (copies of a converged eigenvalue), that with the smallest residual
		{ //(the spurious eigenvalue test of Cullum and Willoughby is not used, it drops also true eigenvalues weakly represented in the start vector)
		const typename LA_traits<T>::normtype tol= 100*(j+1)*std::numeric_limits<typename LA_traits<T>::normtype>::epsilon()*std::max(abs(r[0]),abs(r[j]));
		NRVec<typename LA_traits<T>::normtype> rp(j+1);
		NRMat<typename LA_traits<T>::normtype> smallVp(j+1,j+1);
		int ngood=0;
		for(int i=0; i<=j; )
			{
			int k=i, best=i;
			while(k<j && r[k+1]-r[k]<=tol) {++k; if(abs(smallV(j,k))<abs(smallV(j,best))) best=k;}
			rp[ngood]=r[best];
			for(int l=0; l<=j; ++l) smallVp(l,ngood)=smallV(l,best);
			++ngood;
			i=k+1;
			}
		if(ngood<=j)
			{
			r=rp; r.resize(ngood,true);
			smallV=smallVp.submatrix(0,j,0,ngood-1);
			}
		}
	if(target) //resort eigenpairs by distance from the target
		{
		const int nr=r.size();
		NRVec<typename LA_traits<T>::normtype> key(nr);
		for(int i=0; i<nr; ++i) key[i] = abs(r[i] - *target);
		NRPerm<int> p(nr);
		key.sort(0,p);
	
		NRVec<typename LA_traits<T>::normtype> rp(nr);
		NRMat<typename LA_traits<T>::normtype> smallVp(j+1,nr);
		for(int i=0; i<nr; ++i)
			{
			rp[i]= r[p[i+1]-1];
			for(int k=0; k<=j; ++k) smallVp(k,i) = smallV(k,p[i+1]-1);
			}
		r = rp;
		smallV = smallVp;
		}

       	  if(verbose)
      	    {
	    for(int iroot=0; iroot<std::min(r.size(),nroots); ++iroot)
       	         {
       	         std::cout <<"Lanczos: iter="<<it <<" dim="<<krylovsize<<" root="<<iroot<<" eigenvalue="<<r[iroot]<<" residual="<<betanext*abs(smallV(j,iroot))<<"\n";
       	         }
       	   std::cout.flush();
       	   }
	//convergence test on the residual norms of the Ritz vectors, |H y - r y| = beta_{j+1} |smallV(j,iroot)|
	if(r.size()>=nroots)
		{
		bool conv=true;
		for(int  iroot=0; iroot<nroots; ++iroot)
			if(betanext*abs(smallV(j,iroot))>eps) conv=false;
		if(conv)
			{
			flag=0;
			goto converged;
			}
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
	if(eivecs || eivecsfile)
		{
		vec1=0;
		for(j=0; j<krylovsize; ++j )
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
if(incore) {delete[] v0;}
else  {delete s0;}

if(flag) laerror("no convergence in lanczos");
}

}//namespace
#endif
