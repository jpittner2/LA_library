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

namespace LA {


//dense times sparse (not necessarily symmetric)
template <typename T>
SparseSMat<T>  NRMat<T>::operator*(const SparseSMat<T> &rhs) const
{
SparseSMat<T> r(nn,rhs.ncols());
if(mm!=rhs.nrows())  laerror("incompatible sizes in NRMat*SparseSMat");
for(SPMatindex k=0; k<mm; ++k) //summation loop
    	{
	std::map<SPMatindex,T> * kl = rhs.line(k);
	if(kl)
		{
		//gather the data
		typename std::map<SPMatindex,T>::iterator p;
		int i,j;
		NRVec<T> kline(kl->size());
		NRVec<SPMatindex> klineind(kl->size());
		for(p=kl->begin(), i=0; p!=kl->end(); ++p,++i)
       		         {
       		         klineind[i] = p->first;
       		         kline[i] = p->second;
       		         }
		NRVec<T> kcol = column(k);

		//multiply
		NRMat<T> prod=kcol.otimes(kline,false,1.);

		//scatter the results
		for(i=0; i<prod.nrows(); ++i) for(j=0; j<prod.ncols(); ++j)
                	add(i,klineind[j],prod(i,j),false);

		}
	}
r.simplify();
return r;
}


//matrix is assummed symmetric, no transposition, but possibly make conjugation
template <typename T>
void SparseSMat<T>::gemm(const T beta, const SparseSMat &a, const char transa, const SparseSMat &b, const char transb, const T alpha)
{
(*this) *= beta;
if(alpha==(T)0) return;
if(a.nn!=a.mm || b.nn!=b.mm || nn!=mm) laerror("SparseSMat::gemm implemented only for square symmetric matrices");
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
	if(tolower(transa)=='c')
		for(p=a.v[k]->begin(), i=0; p!=a.v[k]->end(); ++p,++i) { ai[i] = p->first; av[i] = LA_traits<T>::conjugate(p->second); }
	else
		for(p=a.v[k]->begin(), i=0; p!=a.v[k]->end(); ++p,++i) { ai[i] = p->first; av[i] = p->second; }
       	for(p=b.v[k]->begin(), i=0; p!=b.v[k]->end(); ++p,++i) { bi[i] = p->first; bv[i] = p->second; }

	//make multiply via blas
	NRMat<T> prod=av.otimes(bv,tolower(transb)=='c',alpha);

	//scatter the results -- probably the computational bottleneck
	for(i=0; i<prod.nrows(); ++i) for(j=0; j<prod.ncols(); ++j)
		add(ai[i],bi[j],prod(i,j),false);

	}
simplify();
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


template <class T> 
void SparseSMat<T>::axpy(const T alpha, const SparseSMat &x, const bool transp)
{
if(nn!=x.nn || mm!=x.mm) laerror("incompatible matrix dimensions in SparseSMat::axpy");
if(alpha==(T)0) return;
copyonwrite();
for(SPMatindex i=0; i<nn; ++i) if(x.v[i])
	{
	if(!v[i]) v[i]  = new std::map<SPMatindex,T>;
	typename std::map<SPMatindex,T>::iterator p,q;
	for(p=x.v[i]->begin(); p!=x.v[i]->end(); ++p) 
		{
		q=v[i]->find(p->first);
		if(q!=v[i]->end()) q->second += p->second * alpha;
		else (*v[i])[p->first] = p->second * alpha;
		}
	}
simplify();
}


template <class T>
void SparseSMat<T>::gemv(const T beta, NRVec<T> &r, const char trans, const T alpha, const NRVec<T> &x) const
{
if(nn!=r.size() || mm!= x.size()) laerror("incompatible matrix vector dimensions in SparseSMat::gemv");
if(tolower(trans)!='n') laerror("transposition not implemented yet in SparseSMat::gemv");
r *= beta;
if(alpha == (T)0) return;
r.copyonwrite();
for(SPMatindex i=0; i<nn; ++i) if(v[i])
	{
	typename std::map<SPMatindex,T>::iterator p;
	for(p=v[i]->begin(); p!=v[i]->end(); ++p) r[i] += x[p->first] * p->second * alpha ;
	}
}


template <class T>
SparseSMat<T> & SparseSMat<T>::operator=(const T &a)
{
clear();
for(SPMatindex i=0; i<nn; ++i) 
	{
	if(!v[i]) v[i]  = new std::map<SPMatindex,T>;
	(*v[i])[i] = a;
	}
return *this;
}

template <class T>
SparseSMat<T> & SparseSMat<T>::operator+=(const T &a)
{
copyonwrite();
for(SPMatindex i=0; i<nn; ++i)
        {
        if(v[i]) 
		{
		typename std::map<SPMatindex,T>::iterator p;
		p= v[i]->find(i);
		if(p!=v[i]->end()) p->second+=a; else (*v[i])[i] = a;
		}
	else {v[i]  = new std::map<SPMatindex,T>; (*v[i])[i] = a;}
        }
return *this;
}


template <class T>
SparseSMat<T> & SparseSMat<T>::operator-=(const T &a)
{
copyonwrite();
for(SPMatindex i=0; i<nn; ++i)
        {
        if(v[i]) 
		{
		typename std::map<SPMatindex,T>::iterator p;
		p= v[i]->find(i);
		if(p!=v[i]->end()) p->second-=a; else (*v[i])[i] = -a;
		}
	else {v[i]  = new std::map<SPMatindex,T>; (*v[i])[i] = -a;}
        }
return *this;
}


template <class T>
typename LA_traits<T>::normtype SparseSMat<T>::norm(const T scalar) const
{
typename LA_traits<T>::normtype sum=0;

for(SPMatindex i=0; i<nn; ++i)
	if(v[i]) //line present
		{
		typename std::map<SPMatindex,T>::iterator p;
		bool diagonal_present=false;
		for(p=v[i]->begin(); p!=v[i]->end(); ++p) //loop over all existing elements
			{
			if(i==p->first) {diagonal_present=true; sum += LA_traits<T>::sqrabs(p->second - scalar);}
			else sum += LA_traits<T>::sqrabs(p->second);
			}
		if(!diagonal_present) sum += LA_traits<T>::sqrabs(scalar); //there was zero on the diagonal
		}
	else sum += LA_traits<T>::sqrabs(scalar); //missing whole line, subtracted diagonal element contributes

return std::sqrt(sum);
}



//get diagonal, do not construct a new object, but store in existing one
template <class T>
const T* SparseSMat<T>::diagonalof(NRVec<T> &r, const bool divide, bool cache) const
{
if(nn!=mm) laerror("non-square matrix in SparseSMat::diagonalof");
if(nn!=r.size()) laerror("incompatible vector size in diagonalof()");
NRVec<T> *rr;

r.copyonwrite();
if(divide) {rr=new NRVec<T>(nn); *rr=(T)0;}
else {r=(T)0; rr=&r;}
for(SPMatindex i=0; i<nn; ++i)
        if(v[i])
                {
		typename std::map<SPMatindex,T>::iterator p;
                p= v[i]->find(i);
                if(p!=v[i]->end())  (*rr)[i] += p->second;
		}
if(divide)
	{
	for(unsigned int i=0; i<nn; ++i) if((*rr)[i]!=0.) r[i]/=(*rr)[i];
	delete rr;
	}
return divide?NULL:&r[0];
}

template <class T>
SparseSMat<T> SparseSMat<T>::submatrix(const int fromrow, const int torow, const int fromcol, const int tocol) const
{
#ifdef DEBUG
        if(fromrow<0 || fromrow>=nn|| torow<0 || torow>=nn || fromcol<0 || fromcol>=mm || tocol<0 || tocol>=mm || fromrow>torow || fromcol>tocol){
                laerror("invalid submatrix specification");
        }
#endif
        const int m = tocol - fromcol + 1;
	const int n = torow - fromrow + 1;
        SparseSMat<T> result(n, m);
	typename SparseSMat<T>::iterator p(*this);
	for(; p.notend(); ++p) 
	    if(p->row>=fromrow && p->row<= torow && p->col >= fromcol && p->col <= tocol)
		result.add(p->row-fromrow, p->col-fromcol, p->elem, false);

return result;
}

template <class T>
void SparseSMat<T>::storesubmatrix(const int fromrow, const int fromcol, const SparseSMat<T> &rhs)
{
        const int tocol = fromcol + rhs.ncols() - 1;
        const int torow = fromrow + rhs.nrows() - 1;
#ifdef DEBUG
        if(fromrow<0 || fromrow>=nn || torow>=nn || fromcol<0 || fromcol>=mm || tocol>=mm) laerror("bad indices in storesubmatrix");
#endif
	typename SparseSMat<T>::iterator p(rhs);
	for(; p.notend(); ++p) add(p->row+fromrow, p->col+fromcol, p->elem, false);
}


template <class T>
void SparseSMat<T>::get(int fd, bool dimen, bool transp) {
  errno=0;
  SPMatindex dim[2];

if(dimen) {
    if(2*sizeof(SPMatindex)!=read(fd,&dim,2*sizeof(SPMatindex))) laerror("read() error in SparseSMat::get ");
    resize(dim[0],dim[1]);
  }
else  copyonwrite(); 

do {
    if(2*sizeof(SPMatindex)!=read(fd,&dim,2*sizeof(SPMatindex))) laerror("read() error 2 in SparseSMat::get");
    if(dim[0]==(SPMatindex) -1 || dim[1]==(SPMatindex) -1) break;
    typename LA_traits_io<T>::IOtype tmp;
    LA_traits<T>::get(fd,tmp,dimen,transp); // general way to work when elem is some complex class again
    if(transp) add(dim[0],dim[1],tmp,false);  else add(dim[1],dim[0],tmp,false); 
  } 
while(1);
}



template <class T>
void SparseSMat<T>::put(int fd, bool dimen, bool transp) const {
  errno=0;  
  if(dimen) {
    if(sizeof(SPMatindex)!=write(fd,&nn,sizeof(SPMatindex))) laerror("cannot write() 1 in SparseSMat::put");
    if(sizeof(SPMatindex)!=write(fd,&mm,sizeof(SPMatindex))) laerror("cannot write() 2 in SparseSMat::put");
  }
  
  typename SparseSMat<T>::iterator p(*this);
  for(; p.notend(); ++p) {
    if(sizeof(SPMatindex)!=write(fd,&(p->row),sizeof(SPMatindex))) laerror("cannot write() 3 in SparseSMat::put");
    if(sizeof(SPMatindex)!=write(fd,&(p->col),sizeof(SPMatindex))) laerror("cannot write() 4 in SparseSMat::put");
    typename LA_traits_io<T>::IOtype tmp = p->elem;
    LA_traits<T>::put(fd,tmp,dimen,transp); // general way to work when elem is some non-scalar class again
  }
  
  SPMatindex sentinel[2];
  sentinel[0] = sentinel[1] = (SPMatindex) -1;
  if(2*sizeof(SPMatindex) != write(fd,&sentinel,2*sizeof(SPMatindex))) laerror("cannot write() 5 in SparseSMat::put");
}



/* Commented out by Roman for ICC

#define INSTANTIZE(T) \
template void SparseSMat<T>::gemm(const T beta, const SparseSMat &a, const char transa, const SparseSMat &b, const char transb, const T alpha); \
template SparseSMat<T> & SparseSMat<T>::operator*=(const T &a); \
template void SparseSMat<T>::gemv(const T beta, NRVec<T> &r, const char trans, const T alpha, const NRVec<T> &x) const; \
template void SparseSMat<T>::axpy(const T alpha, const SparseSMat &x, const bool transp); \
template SparseSMat<T> & SparseSMat<T>::operator=(const T &a); \
template SparseSMat<T> & SparseSMat<T>::operator+=(const T &a); \
template SparseSMat<T> & SparseSMat<T>::operator-=(const T &a); \
template LA_traits<T>::normtype SparseSMat<T>::norm(const T scalar) const; \
template const T* SparseSMat<T>::diagonalof(NRVec<T> &r, const bool divide, bool cache) const; \
template void SparseSMat<T>::get(int fd, bool dimen, bool transp); \
template void SparseSMat<T>::put(int fd, bool dimen, bool transp) const; \


INSTANTIZE(double)
INSTANTIZE(complex<double>) 
*/

//// forced instantization of functions in the header in the corresponding object file
template class SparseSMat<double>;
template class SparseSMat<complex<double> >;

/*activate this if needed
template void SparseSMat<NRMat<double> >::put(int fd, bool dimen, bool transp) const;
template void SparseSMat<NRMat<double> >::get(int fd, bool dimen, bool transp);
*/


}//namespace
