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

//Pulay memorial book  remarks - for numerical stabilization small addition to diagonal
#define DIISEPS 1e-9 

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
	NRSMat<Ue> bmat;
	AuxStorage<Te> *st;
	AuxStorage<Ue> *errst;
	T *stor;
	U *errstor;
public:
	DIIS() {dim=0; st=NULL; stor=NULL; errst=NULL; errstor=NULL;}; //for array of diis
	DIIS(const int n, const bool core=1);
	void setup(const int n, const bool core=1);
	~DIIS();
	typename LA_traits<U>::normtype extrapolate(T &vec, const U &errvec, bool verbose=false); //vec is input/output; returns square residual norm
	};

template<typename T, typename U>
DIIS<T,U>::DIIS(const int n, const bool core) : dim(n), incore(core), bmat(n+1,n+1)
{
st=incore?NULL: new AuxStorage<Te>;
errst=incore?NULL: new AuxStorage<Ue>;
stor= incore? new T[dim] : NULL;
errstor= incore? new U[dim] : NULL;
bmat= (Ue)0; for(int i=1; i<=n; ++i) bmat(0,i) = (Ue)-1; 
aktdim=cyclicshift=0;
}

template<typename T, typename U>
void DIIS<T,U>::setup(const int n, const bool core)
{
dim=n;
incore=core;
bmat.resize(n+1);
st=incore?NULL: new AuxStorage<Te>;
errst=incore?NULL: new AuxStorage<Ue>;
stor= incore? new T[dim] : NULL;
errstor= incore? new U[dim] : NULL;
bmat= (Ue)0; for(int i=1; i<=n; ++i) bmat(0,i) = (Ue)-1;
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
typename LA_traits<U>::normtype DIIS<T,U>::extrapolate(T &vec, const U &errvec, bool verbose)
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
bmat(aktdim,aktdim)= norm*norm + DIISEPS;
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
{
NRSMat<Te> amat=bmat;
linear_solve(amat,rhs,NULL,aktdim+1);
}
if(verbose) cout <<"DIIS coefficients: "<<rhs<<endl;

//build the new linear combination
vec.clear();
if(incore)
	for(int i=1; i<=aktdim; ++i) 
		vec.axpy(rhs[i],stor[(i-1+cyclicshift)%dim]);
else
        {
	T tmp=vec; //copy dimensions
	for(int i=1; i<=aktdim; ++i)
                {
                st->get(tmp,(i-1+cyclicshift)%dim);
		vec.axpy(rhs[i],tmp);
                }
        }

return norm;
}


#endif
