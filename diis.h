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
//DIIS convergence acceleration according to Pulay: Chem. Phys. Lett. 73, 393 (1980); J. Comp. Chem. 3,556 (1982)
#ifndef _DIIS_H_
#define _DIIS_H_
#include "vec.h"
#include "smat.h"
#include "mat.h"
#include "sparsemat.h"
#include "nonclass.h"
#include "la_traits.h"
#include "auxstorage.h"

namespace LA {

//Pulay memorial book  remarks - for numerical stabilization small addition to diagonal (but our experience was opposite)

// Typically, T is some solution vector in form of NRVec, NRMat, or NRSMat over double or complex<double> fields
// actually it can be anything what has operator=(const T&), clear(), dot() , axpy(), norm() and copyonwrite(), and LA_traits<T>::normtype and elementtype
// and get() and put() if external storage is requested

template<typename T, typename U>
class DIIS
	{
	int dim;
	int aktdim;
	bool incore;
	int cyclicshift; //circular buffer of last dim vectors
	typedef typename LA_traits<T>::elementtype Te;
	typedef typename LA_traits<U>::elementtype Ue;
	typedef typename LA_traits<U>::normtype Un;
	NRSMat<Ue> bmat;
	AuxStorage<Te> *st;
	AuxStorage<Ue> *errst;
	T *stor;
	U *errstor;
public:
	DIIS() {dim=0; st=NULL; stor=NULL; errst=NULL; errstor=NULL;}; //for array of diis
	DIIS(const int n, const bool core=1);
	void setup(const int n, const bool core=1); //can be called repeatedly, previous storage is released
	void reset(); //forget the stored vectors, keep the storage
	~DIIS();
	DIIS(const DIIS &) = delete; //owns raw storage, not copyable
	DIIS & operator=(const DIIS &) = delete;
	typename LA_traits<U>::normtype extrapolate(T &vec, U &errvec, bool verbose=false, const Un diiseps=0, bool errvecout=false); //vec is input/output, errvec optionally too; returns square residual norm
	};

template<typename T, typename U>
DIIS<T,U>::DIIS(const int n, const bool core) : dim(n), incore(core), bmat(n+1,n+1)
{
st=incore?NULL: new AuxStorage<Te>;
errst=incore?NULL: new AuxStorage<Ue>;
stor= incore? new T[dim] : NULL;
errstor= incore? new U[dim] : NULL;
reset();
}

template<typename T, typename U>
void DIIS<T,U>::setup(const int n, const bool core)
{
if(st) delete st;
if(errst) delete errst;
if(stor) delete[] stor;
if(errstor) delete[] errstor;
dim=n;
incore=core;
bmat.resize(n+1);
st=incore?NULL: new AuxStorage<Te>;
errst=incore?NULL: new AuxStorage<Ue>;
stor= incore? new T[dim] : NULL;
errstor= incore? new U[dim] : NULL;
reset();
}

template<typename T, typename U>
void DIIS<T,U>::reset()
{
bmat= (Ue)0; for(int i=1; i<=dim; ++i) bmat(0,i) = (Ue)-1;
aktdim=cyclicshift=0;
}


template<typename T, typename U>
DIIS<T,U>::~DIIS()
{
if(st) delete st;
if(errst) delete errst;
if(stor) delete[] stor;
if(errstor) delete[] errstor;
}


template<typename T, typename U>
typename LA_traits<U>::normtype DIIS<T,U>::extrapolate(T &vec, U &errvec, bool verbose, const Un diiseps, bool errvecout)
{
if(!dim) laerror("attempt to extrapolate from uninitialized DIIS");
//if dim exceeded, shift 
if(aktdim==dim)
	{
	cyclicshift=(cyclicshift+1)%dim;
	for(int i=1; i<dim; ++i)
		for(int j=1; j<=i; ++j)
			bmat(i,j)=bmat(i+1,j+1);
	}
else
	++aktdim;

//store vector
if(incore) 
	{
	stor[(aktdim-1+cyclicshift)%dim]|=vec;
	errstor[(aktdim-1+cyclicshift)%dim]|=errvec;
	}
else 
	{
	st->put(vec,(aktdim-1+cyclicshift)%dim);
	errst->put(errvec,(aktdim-1+cyclicshift)%dim);
	}

if(aktdim==1) return (typename LA_traits<T>::normtype)1;

//calculate overlaps of the new error with old ones
typename LA_traits<T>::normtype norm=errvec.norm();
bmat(aktdim,aktdim) = norm*norm;
// LV
bmat(aktdim,aktdim) += diiseps;

if(incore)
	for(int i=1; i<aktdim; ++i) 
		bmat(i,aktdim)=errvec.dot(errstor[(i+cyclicshift-1)%dim]);
else
	{
	U tmp = errvec; tmp.copyonwrite(); //copy dimensions
	for(int i=1; i<aktdim; ++i) 
		{
		errst->get(tmp,(i-1+cyclicshift)%dim);
		bmat(i,aktdim)= errvec.dot(tmp);
		}
	}

//prepare rhs-solution vector
NRVec<Ue> rhs(dim+1);
rhs= (Ue)0; rhs[0]= (Ue)-1;

//solve for coefficients
//@@@@@@ implement checking for bad condition number and eliminating old vectors
//@@@ explicit solution - cf. remarks in Pulay memorial book
//@@@@@@or use DGGLSE, interface that in nonclass.cc, the matrix then has to be without the 0-th dimension - will be a bit confusing
//@@@@@this avoid the normal equations (which worsen the condition number), but requires work with matrices with one LARGE dimension and can be inconvenient
//for several approaches to DIIS solution cf. Shepard et al. Mol. Phys. 105, 2839 (2007)
{
NRSMat<Te> amat=bmat;
linear_solve(amat,rhs,NULL,aktdim+1);
}
if(verbose) std::cout <<"DIIS coefficients: "<<rhs<<std::endl;

//build the new linear combination
vec.clear();
if(errvecout) errvec.clear();
if(incore)
	for(int i=1; i<=aktdim; ++i) 
		{
		vec.axpy(rhs[i],stor[(i-1+cyclicshift)%dim]);
		if(errvecout) errvec.axpy(rhs[i],errstor[(i-1+cyclicshift)%dim]);
		}
else
        {
	T tmp=vec; //copy dimensions
	U errtmp; if(errvecout) errtmp=errvec;
	for(int i=1; i<=aktdim; ++i)
                {
                st->get(tmp,(i-1+cyclicshift)%dim);
		vec.axpy(rhs[i],tmp);
		if(errvecout)
			{
			errst->get(errtmp,(i-1+cyclicshift)%dim);
			errvec.axpy(rhs[i],errtmp);
			}
                }
        }

return norm;
}

}//namespace

#endif
