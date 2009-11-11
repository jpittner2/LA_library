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

#include <string>
#include <cmath>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include "sparsesmat.h"

template <typename T>
void SparseSMat<T>::gemm(const T beta, const SparseSMat &a, const char transa, const SparseSMat &b, const char transb, const T alpha)
{
(*this) *= beta;
if(alpha==(T)0) return;
if(a.nn!=b.nn || a.nn!=nn) laerror("incompatible sizes in SparseSMat::gemm");
copyonwrite();

for(SPMatindex k=0; k<nn; ++k) //summation loop
    if(a.v[k] && b.v[k]) //nonempty in both
	{
	NRVec<T> av(a.v[k]->size());
	NRVec<T> bv(b.v[k]->size());
	NRVec<SPMatindex> ai(a.v[k]->size());
	NRVec<SPMatindex> bi(b.v[k]->size());

	//gather the data
	typename std::map<SPMatindex,T>::iterator p;
	int i,j;
	for(p=a.v[k]->begin(), i=0; p!=a.v[k]->end(); ++p,++i) 
		{
		ai[i] = p->first;
		av[i] = p->second;
		}
        for(p=b.v[k]->begin(), i=0; p!=b.v[k]->end(); ++p,++i)
                {
                bi[i] = p->first;
                bv[i] = p->second;
                }

	//make multiply via blas
	NRMat<T> prod=av.otimes(bv,false,alpha);

	//scatter the results
	for(i=0; i<prod.nrows(); ++i) for(j=0; j<prod.ncols(); ++j)
		add(ai[i],bi[j],prod(i,j),false);

	}
simplify(); //erase elements below threshold
}


template <class T>
SparseSMat<T> & SparseSMat<T>::operator*=(const T &a)
{
if(!count) laerror("operator*= on undefined lhs");
if(a==(T)1) return *this;
if(a==(T)0) {clear(); return *this;}
copyonwrite();

for(SPMatindex i=0; i<nn; ++i) if(v[i])
	{
	typename std::map<SPMatindex,T>::iterator p;
	for(p=v[i]->begin(); p!=v[i]->end(); ++p) p->second *= a;
	}

return *this;
}



#define INSTANTIZE(T) \
template void SparseSMat<T>::gemm(const T beta, const SparseSMat &a, const char transa, const SparseSMat &b, const char transb, const T alpha); \
template SparseSMat<T> & SparseSMat<T>::operator*=(const T &a); \

INSTANTIZE(double)

INSTANTIZE(complex<double>) 

//// forced instantization of functions in the header in the corresponding object file
template class SparseSMat<double>;
template class SparseSMat<complex<double> >;


