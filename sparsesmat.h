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

#ifndef _SPARSESMAT_H_
#define _SPARSESMAT_H_


#include <string>
#include <cmath>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include "sparsemat.h"
#include "vec.h"
#include "mat.h"

#include <map>
#include <list>


//symmetric sparse matrix class with a representation for efficient exponentiatiation
//in particular we need a unitary symmetric complex matrix as exp(iH) with H real symmetric
//indices are counted from zero


template <typename T>
class SparseSMat
{
protected:
	SPMatindex nn;
	std::map<SPMatindex,T> **v;
	int *count;
public:
	SparseSMat() : nn(0), v(NULL), count(NULL) {};
	SparseSMat(const SPMatindex n);
	SparseSMat(const SparseSMat &rhs);
	SparseSMat(const SparseMat<T> &rhs);
	SparseSMat & operator=(const SparseSMat &rhs);
	void copyonwrite();
        void clear();
	void simplify();
	~SparseSMat() {clear();};
	//
        SparseSMat & operator=(const T a);          //assign a to diagonal
        SparseSMat & operator+=(const T a);         //assign a to diagonal
        SparseSMat & operator-=(const T a);         //assign a to diagonal
        SparseSMat & operator*=(const T a);         //multiply by a scalar
        SparseSMat & operator+=(const SparseSMat &rhs);
        SparseSMat & operator-=(const SparseSMat &rhs);
	void gemv(const T beta, NRVec<T> &r, const char trans, const T alpha, const NRVec<T> &x) const;
	void gemm(const T beta, const SparseSMat &a, const char transa, const SparseSMat &b, const char transb, const T alpha);
        const SparseSMat operator*(const SparseSMat &rhs) const;
	const typename LA_traits<T>::normtype norm(const T scalar=(T)0) const;
	void add(const SPMatindex n, const SPMatindex m, const T elem);
	unsigned int length() const;
};

template <typename T>
SparseSMat<T>::SparseSMat(const SPMatindex n)
:nn(n), 
count(new int(1))
{
v= new std::map<SPMatindex,T> * [n];
memset(v,0,nn*sizeof(std::map<SPMatindex,T> *));
}

template <typename T>
SparseSMat<T>::SparseSMat(const SparseSMat &rhs)
{
v = rhs.v;
nn = rhs.nn;
count = rhs.count;
if(count) (*count)++;
}


template <typename T>
void SparseSMat<T>::clear()
{
        if(!count) return;
        if(--(*count) <= 0) {
                if(v) 
                        {
                        for(SPMatindex i=0; i<nn; ++i) if(v[i]) delete[] v[i];
                        delete[] (v);
                        }
                delete count;
        }
}



template <typename T>
SparseSMat<T> & SparseSMat<T>::operator=(const SparseSMat &rhs)
{
  if (this != &rhs)
  {
    if(count)
      if(--(*count) == 0)
      {
	if(v) 
                        {
                        for(SPMatindex i=0; i<nn; ++i) if(v[i]) delete[] v[i];
                        delete[] (v);
                        }
        delete count;
      }
    v = rhs.v;
    nn = rhs.nn;
    count = rhs.count;
    if(count) (*count)++;
  }
return *this;
}

template <typename T>
void SparseSMat<T>::copyonwrite()
{
  if(!count) laerror("SparseSmat::copyonwrite() of an undefined object");
  if(*count > 1)
  {
    (*count)--;
    count = new int;
    *count = 1;
    std::map<SPMatindex,T> *newv= new std::map<SPMatindex,T> * [nn];
    for(SPMatindex i=0; i<nn; ++i) if(v[i])
		newv[i]= new std::map<SPMatindex,T>(*(v[i])); //deep copy of each map
	else
		newv[i]=NULL;
    v = newv;
  }
}


template <typename T>
void SparseSMat<T>::add(const SPMatindex n, const SPMatindex m, const T elem)
{
#ifdef DEBUG
if(n>=nn || m>=nn) laerror("illegal index in SparseSMat::add()");
#endif
if(!v[n]) v[n] = new std::map<SPMatindex,T>;
if(!v[m]) v[m] = new std::map<SPMatindex,T>;

typename std::map<SPMatindex,T>::iterator p;

p= v[n]->find(m);
if(p!=v[n]->end()) p->second+=elem; else (*v[n])[m] = elem;
if(n!=m)
	{
	p= v[m]->find(n);
	if(p!=v[m]->end()) p->second+=elem; else (*v[m])[n] = elem;
	}
//addition can lead to zero, but do not treat it now, make a simplify
}


template <typename T>
void SparseSMat<T>::simplify()
{
for(SPMatindex i=0; i<nn; ++i) if(v[i])
	{
	//check for zero elements and erase them from the list
	//build a list since we are not sure whether erase from within the traversal loop is safe
	std::list<SPMatindex> l;
	typename std::map<SPMatindex,T>::iterator p;
	for(p=v[i]->begin(); p!=v[i]->end(); ++p) 
		if(std::abs(p->second) < SPARSEEPSILON) l.push_front(p->first);
	typename std::list<SPMatindex>::iterator q;	
	for(q=l.begin(); q!=l.end(); ++q) v[i]->erase(*q);	
	if(v[i]->size() == 0) delete v[i];
	}
}


#endif //_SPARSESMAT_H_
