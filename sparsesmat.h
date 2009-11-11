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
#include "smat.h"

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
	SparseSMat(const NRSMat<T> &rhs);
	SparseSMat & operator=(const SparseSMat &rhs);
	void copyonwrite();
        void resize(const SPMatindex n);
        void clear() {resize(nn);}
	void simplify();
	~SparseSMat();
	//
        SparseSMat & operator=(const T a);          //assign a to diagonal
        SparseSMat & operator+=(const T a);         //assign a to diagonal
        SparseSMat & operator-=(const T a);         //assign a to diagonal
        SparseSMat & operator*=(const T a);         //multiply by a scalar
        SparseSMat & operator+=(const SparseSMat &rhs);
        SparseSMat & operator-=(const SparseSMat &rhs);
	void gemv(const T beta, NRVec<T> &r, const char trans, const T alpha, const NRVec<T> &x) const;
        const SparseSMat operator*(const SparseSMat &rhs) const; //!!!NOT A GENERAL ROUTINE, JUST FOR THE CASES WHEN THE RESULT STAYS SYMMETRIC
	const typename LA_traits<T>::normtype norm(const T scalar=(T)0) const;
	void add(const SPMatindex n, const SPMatindex m, const T elem, const bool both=true);
	unsigned int length() const;
	int nrows() const {return nn;}
	int ncols() const {return nn;}

	class iterator {//not efficient, just for output to ostreams
        private:
                matel<T> *p;
		matel<T> my;
		SPMatindex row;
		typename std::map<SPMatindex,T>::iterator *col;
		typename std::map<SPMatindex,T>::iterator mycol;
		SPMatindex mynn;
		std::map<SPMatindex,T> **myv;
		

        public:
		//compiler-generated iterator & operator=(const  iterator &rhs); 
		//compiler-generated iterator(const iterator &rhs); 
                iterator(): p(NULL),row(0),col(NULL),mynn(0),myv(NULL) {};
                iterator(const SparseSMat &rhs) : mynn(rhs.nn), myv(rhs.v), col(NULL) {row=0; p= &my; operator++();}
                iterator operator++()  {
					if(col) //finish column list
						{
						++mycol;
						if(mycol != myv[row]->end())
							{
							p->row = row;
							p->col = mycol->first;
							p->elem = mycol->second;
							return *this;
							}
						else
							{
							col=NULL;
							++row; if(row==mynn) {p=NULL; return *this;} //end()
							}	
						}
				      	nextrow: 
					while(myv[row]==NULL) {++row; if(row==mynn) {p=NULL; return *this;}} //end()

					//we are at next nonempty row
					col = &mycol;
					mycol = myv[row]->begin();
					if(mycol == myv[row]->end()) 	{col=NULL; 
									++row; 
									if(row==mynn) {p=NULL; return *this;} else goto nextrow;
									} 
					//first column of new row
					p->row = row;
                                        p->col = mycol->first;
                                        p->elem = mycol->second;
					return *this;
					};
		iterator(matel<T> *q) {p=q; col=NULL;}//just for end()
                //compiler-generated ~iterator() {};
                bool operator!=(const iterator &rhs) const {return p!=rhs.p;} //only for comparison with end()
                bool operator==(const iterator &rhs) const {return p==rhs.p;} //only for comparison with end()
                matel<T> & operator*() const {return *p;}
                matel<T> * operator->() const {return p;}
		bool notend() const {return (bool)p;};
        };
        iterator begin() const {return iterator(*this);}
        iterator end() const {return iterator(NULL);}
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
SparseSMat<T>::~SparseSMat()
{
        if(!count) return;
        if(--(*count) <= 0) {
                if(v) 
                        {
                        for(SPMatindex i=0; i<nn; ++i) if(v[i]) delete v[i];
                        delete[] (v);
                        }
                delete count;
        }
}


template <typename T>
void SparseSMat<T>::resize(const SPMatindex n)
{
if(!count) 
	{
	if(n==0) return;
	count = new int(1);
	nn=n;
	v= new std::map<SPMatindex,T> * [nn];
        for(SPMatindex i=0; i<nn; ++i) v[i]=NULL;
	return;
	}

if(*count > 1) //it was shared
  {
    (*count)--;
    if(n) 
	{
        count = new int(1);
	nn=n;
	v= new std::map<SPMatindex,T> * [nn];
        for(SPMatindex i=0; i<nn; ++i) v[i]=NULL;
	}
     else {v=NULL; nn=0; count=NULL;}
  }
else  //it was not shared
	{
	//delete all trees
	for(SPMatindex i=0; i<nn; ++i) if(v[i]) {delete[] v[i]; v[i]=NULL;}
	if(n!=nn)
		{
		nn=n;
		for(SPMatindex i=0; i<nn; ++i) v[i]=NULL;
		}
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
void SparseSMat<T>::add(const SPMatindex n, const SPMatindex m, const T elem, const bool both)
{
#ifdef DEBUG
if(n>=nn || m>=nn) laerror("illegal index in SparseSMat::add()");
#endif
if(!v[n]) v[n] = new std::map<SPMatindex,T>;
if(!v[m]) v[m] = new std::map<SPMatindex,T>;

typename std::map<SPMatindex,T>::iterator p;

p= v[n]->find(m);
if(p!=v[n]->end()) p->second+=elem; else (*v[n])[m] = elem;
if(n!=m && both) //add also transposed
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

template <typename T>
std::ostream & operator<<(std::ostream &s, const SparseSMat<T> &x)
{
  SPMatindex n;

  n = x.nrows();
  s << n << " "<< n<< std::endl;

typename SparseSMat<T>::iterator p(x);
for(; 
p.notend(); 
++p) 
s << (int)p->row << ' ' << (int)p->col  << ' ' << (typename LA_traits_io<T>::IOtype) p->elem << '\n';

  s << "-1 -1\n";
  return s;
}

template <class T>
std::istream& operator>>(std::istream  &s, SparseSMat<T> &x)
                {
                SPMatindex n,m;
		long i,j;
                s >> n >> m;
		if(n!=m) laerror("SparseSMat must be square");
                x.resize(n);
                s >> i >> j;
                typename LA_traits_io<T>::IOtype tmp;
                while(i>=0 && j>=0)
                        {
			s>>tmp;
			x.add(i,j,tmp,false); 
                        s >> i >> j;
                        }
                return s;
                }






#endif //_SPARSESMAT_H_
