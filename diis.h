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
	Te extrapolate(T &vec); //vec is input/output; returns square residual norm
	};

template<typename T>
DIIS<T>::DIIS(const int n, const bool core) : dim(n), incore(core), bmat(n+1,n+1)
{
st=incore?NULL: new AuxStorage<Te>;
stor= incore? new T[dim] : NULL;
bmat= (Te)0; for(int i=1; i<=n; ++i) bmat(0,i) = (Te)-1; 
aktdim=cyclicshift=0;

}

template<typename T>
DIIS<T>::~DIIS()
{
if(st) delete st;
if(stor) delete[] stor;
}


template<typename T>
typename DIIS<T>::Te DIIS<T>::extrapolate(T &vec)
{
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
if(incore) stor[(aktdim-1+cyclicshift)%dim]=vec;
else st.put(vec,(aktdim-1+cyclicshift)%dim);

//calculate overlaps
bmat(aktdim,aktdim)= vec.dot(vec);
if(incore)
	for(int i=1; i<aktdim; ++i) bmat(i,aktdim)=vec.dot(stor[(i-1+cyclicshift)%dim]);
else
	{
	T tmp=vec; //copy dimensions
	for(int i=1; i<aktdim; ++i) 
		{
		st.get(tmp,(i-1+cyclicshift)%dim);
		bmat(i,aktdim)=vec.dot(tmp);
		}
	}

//prepare rhs-solution vector
NRVec<Te> rhs(dim+1);
rhs= (Te)0; rhs[0]= (Te)-1;

//solve for coefficients
{
NRSMat<Te> amat=bmat;
linear_solve(amat,rhs,NULL,aktdim+1);
}

//build the new linear combination
vec.copyonwrite();
vec *= rhs[aktdim];
if(incore)
	for(int i=1; i<aktdim; ++i) vec.axpy(rhs[i],stor[(i-1+cyclicshift)%dim]);
else
        {
	T tmp=vec; //copy dimensions
	for(int i=1; i<aktdim; ++i)
                {
                st.get(tmp,(i-1+cyclicshift)%dim);
		vec.axpy(rhs[i],tmp);
                }
        }

return rhs[0];
}


#endif
