//DIIS convergence acceleration
#ifndef _DIIS_H_
#define _DIIS_H_
#include "vec.h"
#include "smat.h"
#include "mat.h"
#include "sparsemat.h"
#include "nonclass.h"
#include "la_traits.h"
#include "auxstorage.h"

// T is some solution vector in form of NRVec, NRMat, or NRSMat over double or complex<double> fields

template<typename T>
class DIIS
	{
	int dim;
	int aktdim;
	bool incore;
	int cyclicshift; //circular buffer of last dim vectors
	typedef typename LA_traits<T>::elementtype Te;
	NRSMat<Te> bmat;
	AuxStorage<Te> *st;
	T *stor;
public:
	DIIS(const int n, const bool core=1);
	~DIIS();
	typename LA_traits<T>::normtype extrapolate(T &vec); //vec is input/output; returns square residual norm
	};

template<typename T>
DIIS<T>::DIIS(const int n, const bool core) : dim(n), incore(core), bmat(n,n)
{
st=incore?NULL: new AuxStorage<Te>;
stor= incore? new T[dim] : NULL;
bmat= (Te)0; for(int i=1; i<n; ++i) bmat(0,i) = (Te)-1; 
aktdim=cyclicshift=0;

}

template<typename T>
DIIS<T>::~DIIS()
{
if(st) delete st;
if(stor) delete[] stor;
}


template<typename T>
typename LA_traits<T>::normtype DIIS<T>::extrapolate(T &vec)
{
//if dim exceeded, shift 
if(aktdim==dim)
	{
	cyclicshift=(cyclicshift+1)%dim;
	for(int i=1; i<dim-1; ++i)
		for(int j=1; j<=i; ++j)
			bmat(i,j)=bmat(i+1,j+1);
	}
else
	++aktdim;

//store vector
if(incore) stor[(aktdim-1+cyclicshift)%dim]=vec;
else st->put(vec,(aktdim-1+cyclicshift)%dim);

if(aktdim==1) return (typename LA_traits<T>::normtype)1000000000;

//calculate difference; 
vec.copyonwrite();
if(incore) vec -= stor[(aktdim-2+cyclicshift)%dim];
else
	{
	T tmp=vec;
	st->get(tmp,(aktdim-2+cyclicshift)%dim);
	vec -= tmp;
	}

//calculate overlaps of differences (if storage is cheap, they could rather be stored than recomputed)
typename LA_traits<T>::normtype norm=vec.norm();
bmat(aktdim-1,aktdim-1)= norm*norm;
if(incore)
	for(int i=1; i<aktdim-1; ++i) 
		bmat(i,aktdim-1)=vec.dot(stor[(i+cyclicshift)%dim] - stor[(i-1+cyclicshift)%dim]);
else
	{
	T tmp=vec;
	T tmp2=vec; //copy dimensions
	st->get(tmp2,(0+cyclicshift)%dim);
	for(int i=1; i<aktdim-1; ++i) 
		{
		st->get(tmp,(i+cyclicshift)%dim);
		tmp2 -= tmp;
		bmat(i,aktdim-1)= -vec.dot(tmp2);
		tmp2=tmp;
		}
	}

//prepare rhs-solution vector
NRVec<Te> rhs(dim);
rhs= (Te)0; rhs[0]= (Te)-1;

//solve for coefficients
{
NRSMat<Te> amat=bmat;
linear_solve(amat,rhs,NULL,aktdim);
}

//build the new linear combination
vec = (Te)0;
if(incore)
	for(int i=1; i<aktdim; ++i) vec.axpy(rhs[i],stor[(i+cyclicshift)%dim]);
else
        {
	T tmp=vec; //copy dimensions
	for(int i=1; i<aktdim; ++i)
                {
                st->get(tmp,(i+cyclicshift)%dim);
		vec.axpy(rhs[i],tmp);
                }
        }

return norm;
}


#endif
