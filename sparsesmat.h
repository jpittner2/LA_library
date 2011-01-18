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
#include "la_traits.h"
#include "sparsemat.h"
#include "vec.h"
#include "mat.h"
#include "smat.h"
#include "qsort.h"

#include <map>
#include <list>

#define CHOLESKYEPSILON 1e-16

namespace LA {

//symmetric sparse matrix class with a representation for efficient exponentiatiation
//in particular we need a unitary symmetric complex matrix as exp(iH) with H real symmetric
//indices are counted from zero


template <typename T>
class SparseSMat
{
protected:
	SPMatindex nn;
	SPMatindex mm;
	std::map<SPMatindex,T> **v;
	int *count;
public:
	SparseSMat() : nn(0), mm(0), v(NULL), count(NULL) {};
	explicit SparseSMat(const SPMatindex n, const SPMatindex m); //prevent double -> int -> SparseSMat
	explicit SparseSMat(const SPMatindex n);
	SparseSMat(const SparseSMat &rhs);
	explicit SparseSMat(const SparseMat<T> &rhs);
	explicit SparseSMat(const NRSMat<T> &rhs);
	explicit SparseSMat(const NRMat<T> &rhs);
	explicit SparseSMat(const CSRMat<T> &rhs);
	SparseSMat & operator=(const SparseSMat &rhs);
	void copyonwrite();
        void resize(const SPMatindex nn, const SPMatindex mm);
   	void dealloc(void) {resize(0,0);}
	inline void setcoldim(int i) {mm=(SPMatindex)i;};
	//std::map<SPMatindex,T> *line(SPMatindex n) const {return v[n];};
	typedef std::map<SPMatindex,T> *ROWTYPE;
	inline typename SparseSMat<T>::ROWTYPE & operator[](const SPMatindex i) {return v[i];}; 
        void clear() {resize(nn,mm);}
	unsigned long long simplify();
	~SparseSMat();
	inline int getcount() const {return count?*count:0;}
        SparseSMat & operator*=(const T &a);         //multiply by a scalar
        inline const SparseSMat operator*(const T &rhs) const {return SparseSMat(*this) *= rhs;}
	inline const SparseSMat operator+(const T &rhs) const {return SparseSMat(*this) += rhs;}
        inline const SparseSMat operator-(const T &rhs) const {return SparseSMat(*this) -= rhs;}
        inline const SparseSMat operator+(const SparseSMat &rhs) const {return SparseSMat(*this) += rhs;} 
        inline const SparseSMat operator-(const SparseSMat &rhs) const {return SparseSMat(*this) -= rhs;}
        SparseSMat & operator=(const T &a);          //assign a to diagonal
        SparseSMat & operator+=(const T &a);         //assign a to diagonal
        SparseSMat & operator-=(const T &a);         //assign a to diagonal
	void axpy(const T alpha, const SparseSMat &x, const bool transp=0); // this+= a*x
        inline SparseSMat & operator+=(const SparseSMat &rhs) {axpy((T)1,rhs); return *this;};
        inline SparseSMat & operator-=(const SparseSMat &rhs) {axpy((T)-1,rhs); return *this;};
	const T* diagonalof(NRVec<T> &, const bool divide=0, bool cache=false) const; //get diagonal
	void gemv(const T beta, NRVec<T> &r, const char trans, const T alpha, const NRVec<T> &x) const;
	inline const NRVec<T> operator*(const NRVec<T> &rhs) const {NRVec<T> result(nn); this->gemv((T)0,result,'n',(T)1,rhs); return result;};
	typename LA_traits<T>::normtype norm(const T scalar=(T)0) const;
        inline const SparseSMat operator*(const SparseSMat &rhs) const {SparseSMat<T> r(nn,mm); r.gemm(0,*this,'n',rhs,'n',1); return r;}; //!!!NOT A GENERAL ROUTINE, JUST FOR THE CASES WHEN THE RESULT STAYS SYMMETRIC
	void gemm(const T beta, const SparseSMat &a, const char transa, const SparseSMat &b, const char transb, const T alpha); //this := alpha*op( A )*op( B ) + beta*this !!!NOT A GENERAL ROUTINE, JUST FOR THE CASES WHEN THE RESULT STAYS SYMMETRIC
	inline void add(const SPMatindex n, const SPMatindex m, const T elem, const bool both=true);
	inline unsigned long long length() {return simplify();};
	void transposeme() const {laerror("in-place transposition not necessary/implemented for SparseSMat");};
	SparseSMat transpose(bool conj=false) const; //if we store a non-symmetric matrix there
        inline bool issymmetric() const {return true;} // LV: for davidson
	void get(int fd, bool dimen, bool transp);
        void put(int fd, bool dimen, bool transp) const;
	int nrows() const {return nn;}
	int ncols() const {return mm;}
	SparseSMat<T>  cholesky(void) const;
	SparseSMat submatrix(const int fromrow, const int torow, const int fromcol, const int tocol) const;
	void storesubmatrix(const int fromrow, const int fromcol, const SparseSMat &rhs);

	class iterator {//not efficient, just for output to ostreams
        private:
                matel<T> *p;
		matel<T> my;
		SPMatindex row;
		typename std::map<SPMatindex,T>::iterator *col;
		typename std::map<SPMatindex,T>::iterator mycol;
		SPMatindex mynn;
		SPMatindex mymm;
		std::map<SPMatindex,T> **myv;
		

        public:
		//compiler-generated iterator & operator=(const  iterator &rhs); 
		//compiler-generated iterator(const iterator &rhs); 
                iterator(): p(NULL),row(0),col(NULL),mynn(0),mymm(0),myv(NULL) {};
                iterator(const SparseSMat &rhs) : mynn(rhs.nn), mymm(rhs.mm), myv(rhs.v), col(NULL) {row=0; p= &my; operator++();}
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
:nn(n), mm(n),
count(new int(1))
{
v= new std::map<SPMatindex,T> * [n];
memset(v,0,nn*sizeof(std::map<SPMatindex,T> *));
}


template <typename T>
SparseSMat<T>::SparseSMat(const SPMatindex n, const SPMatindex m)
:nn(n), mm(m),
count(new int(1))
{
v= new std::map<SPMatindex,T> * [n];
memset(v,0,nn*sizeof(std::map<SPMatindex,T> *));
}

template <typename T>
SparseSMat<T>::SparseSMat(const NRSMat<T> &rhs)
:nn(rhs.nrows()), mm(rhs.ncols()),
count(new int(1))
{
v= new std::map<SPMatindex,T> * [nn];
memset(v,0,nn*sizeof(std::map<SPMatindex,T> *));
int i,j;
for(i=0; i<nn; ++i) for(j=0; j<=i; ++j) if(std::abs(rhs(i,j))>SPARSEEPSILON) (*this).add(i,j,rhs(i,j),true);
}

template <typename T>
SparseSMat<T>::SparseSMat(const NRMat<T> &rhs)
:nn(rhs.nrows()), mm(rhs.ncols()),
count(new int(1))
{
if(rhs.nrows()!=rhs.ncols()) laerror("non-square matrix in SparseSMat constructor from NRMat");
v= new std::map<SPMatindex,T> * [nn];
memset(v,0,nn*sizeof(std::map<SPMatindex,T> *));
int i,j;
for(i=0; i<nn; ++i) for(j=0; j<mm; ++j) if(std::abs(rhs(i,j))>SPARSEEPSILON) (*this).add(i,j,rhs(i,j),false);
}


template <typename T>
SparseSMat<T>::SparseSMat(const SparseSMat &rhs)
{
v = rhs.v;
nn = rhs.nn;
mm = rhs.mm;
count = rhs.count;
if(count) (*count)++;
}

//NRSMat from SparseSMat
#define nn2 (nn*(nn+1)/2)
template <typename T>
NRSMat<T>::NRSMat(const SparseSMat<T> &rhs)
: nn(rhs.nrows())
{
if(rhs.nrows()!=rhs.ncols()) laerror("cannot transform rectangular matrix to NRSMat");
#ifdef CUDALA
        location = cpu;
#endif
count = new int(1);
v=new T[nn2];
memset(v,0,nn2*sizeof(T));
typename SparseSMat<T>::iterator p(rhs);
for(; p.notend(); ++p) if(p->row <= p->col) (*this)(p->row,p->col)=p->elem;
}
#undef nn2


//construct dense from sparse
template <typename T>
NRMat<T>::NRMat(const SparseSMat<T> &rhs) :
nn(rhs.nrows()),
mm(rhs.ncols()),
count(new int(1))
{
#ifdef CUDALA
        location = cpu;
#endif
#ifdef MATPTR
        v = new T*[nn];
        v[0] = new T[mm*nn];
        for (int i=1; i<nn; i++) v[i] = v[i-1] + mm;
#else
        v = new T[mm*nn];
#endif
memset(&(*this)(0,0),0,mm*nn*sizeof(T));
typename SparseSMat<T>::iterator p(rhs);
for(; p.notend(); ++p) (*this)(p->row,p->col)= p->elem;
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
void SparseSMat<T>::resize(const SPMatindex n, const SPMatindex m)
{
if(!count) 
	{
	if(n==0) return;
	count = new int(1);
	nn=n;
	mm=m;
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
	mm=m;
	v= new std::map<SPMatindex,T> * [nn];
        for(SPMatindex i=0; i<nn; ++i) v[i]=NULL;
	}
     else {v=NULL; nn=0; mm=0; count=NULL;}
  }
else  //it was not shared
	{
	mm=m;
	//delete all trees
	for(SPMatindex i=0; i<nn; ++i) if(v[i]) {delete v[i]; v[i]=NULL;}
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
                        for(SPMatindex i=0; i<nn; ++i) if(v[i]) delete v[i];
                        delete[] (v);
                        }
        delete count;
      }
    v = rhs.v;
    nn = rhs.nn;
    mm = rhs.mm;
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
    typename std::map<SPMatindex,T> **newv= new std::map<SPMatindex,T> * [nn];
    for(SPMatindex i=0; i<nn; ++i) if(v[i])
		newv[i]= new typename std::map<SPMatindex,T>(*(v[i])); //deep copy of each map
	else
		newv[i]= NULL;
    v = newv;
  }
}


template <typename T>
void SparseSMat<T>::add(const SPMatindex n, const SPMatindex m, const T elem, const bool both)
{
#ifdef DEBUG
if(n>=nn || m>=mm) laerror("illegal index in SparseSMat::add()");
#endif
if(!v[n]) v[n] = new std::map<SPMatindex,T>;

typename std::map<SPMatindex,T>::iterator p;

p= v[n]->find(m);
if(p!=v[n]->end()) p->second+=elem; else (*v[n])[m] = elem;
if(n!=m && both) //add also transposed
	{
	if(!v[m]) v[m] = new std::map<SPMatindex,T>;
	p= v[m]->find(n);
	if(p!=v[m]->end()) p->second+=elem; else (*v[m])[n] = elem;
	}
//addition can lead to zero, but do not treat it now, make a simplify
}


template <typename T>
unsigned long long SparseSMat<T>::simplify()
{
unsigned long long count=0;
for(SPMatindex i=0; i<nn; ++i) if(v[i])
	{
	//check for zero elements and erase them from the list
	//build a list since we are not sure whether erase from within the traversal loop is safe
	std::list<SPMatindex> l;
	typename std::map<SPMatindex,T>::iterator p;
	for(p=v[i]->begin(); p!=v[i]->end(); ++p) 
	if(std::abs(p->second) < SPARSEEPSILON) l.push_front(p->first); else ++count;
	typename std::list<SPMatindex>::iterator q;	
	for(q=l.begin(); q!=l.end(); ++q) v[i]->erase(*q);	
	if(v[i]->size() == 0) {delete v[i]; v[i]=NULL;}
	}
return count;
}

template <typename T>
std::ostream & operator<<(std::ostream &s, const SparseSMat<T> &x)
{
SPMatindex n;

s << x.nrows() << " "<< x.ncols()<< std::endl;

typename SparseSMat<T>::iterator p(x);
for(; p.notend(); ++p) s << (int)p->row << ' ' << (int)p->col  << ' ' << (typename LA_traits_io<T>::IOtype) p->elem << '\n';
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
	x.resize(n,m);
	s >> i >> j;
	typename LA_traits_io<T>::IOtype tmp;
	while(i>=0 && j>=0)
		{
		s>>tmp;
		if(i>=n||j>=m) laerror("bad index in SparseSMat::operator>>");
		x.add(i,j,tmp,false); 
                        s >> i >> j;
                        }
                return s;
                }


template <typename T>
SparseSMat<T>  SparseSMat<T>::transpose(bool conj) const
{
SparseSMat<T> r(mm,nn);
typename SparseSMat<T>::iterator p(*this);
for(; p.notend(); ++p) r.add(p->col, p->row, (conj?LA_traits<T>::conjugate(p->elem):p->elem), false);
return r;
}



//Cholesky decomposition, pivoted, positive semidefinite, not in place
//it is NOT checked that the input matrix is symmetric/hermitean
//result.transpose(true)*result reproduces the original matrix
//Due to pivoting the result is upper triangular only before applying final permutation
//
template <typename T>
SparseSMat<T>  SparseSMat<T>::cholesky(void) const
{
if(nn!=mm) laerror("Cholesky defined only for square matrices");
//we need real values for sorting, if T is already real it makes just an unnecessary copy of one vector
NRVec<typename LA_traits<T>::normtype> diagreal(nn);
{
NRVec<T> diag(nn); diagonalof(diag);
for(SPMatindex i=0; i<nn; ++i) diagreal[i]=LA_traits<T>::realpart(diag[i]);
}

NRVec<int> pivot(nn);
for(int i=0; i<nn; ++i) pivot[i]=i;

//pivot by sorting
//!this is actually not fully correct approach, since the pivoting should be done during the Cholesky process
//Now it can happen that some elements will vanish in the process, while there will be some remaining ones later
//However, column swapping in the regular pivoting in an in-place algorithm would be rather clumsy with std::map , since simply renumbering the key is not allowed
//This works reasonably well so keep it like this at the moment
diagreal.sort(1,0,nn-1,pivot);

//prepare inverse permutation
NRVec<int> invpivot(nn);
for(int i=0; i<nn; ++i) invpivot[pivot[i]]=i;

//std::cout <<"sorted diagonal\n"<<diagreal;
//std::cout <<"pivot\n"<<pivot;

//copy-permute upper triangle
SparseSMat<T> r;
r.nn=nn;
r.mm=nn;
r.count = new int(1);
r.v = new std::map<SPMatindex,T> * [nn];
for(SPMatindex i=0; i<nn; ++i) 
       if(v[pivot[i]])
		{
		r.v[i]= new typename std::map<SPMatindex,T>; 
		typename std::map<SPMatindex,T>::iterator p;		
		for(p=v[pivot[i]]->begin(); p!=v[pivot[i]]->end(); ++p)
			{
			if(invpivot[p->first] >= i) 
				{
				(*r.v[i])[invpivot[p->first]] = p->second;
				}
			}
		}
	else
		r.v[i]= NULL;

//std::cout <<"Permuted upper triangle matrix\n"<<r;
//SparseSMat<T> r0(r);r.copyonwrite();

//perform complex, positive semidefinite Cholesky with thresholding by SPARSEEPSILON
SPMatindex i,j,k;
int rank=0;
for(k=0; k<nn; ++k)
    if(r.v[k])
	{
	typename std::map<SPMatindex,T>::iterator p;
	p= r.v[k]->find(k);
	if(p==r.v[k]->end()) continue; //must not break due to the a priori  pivoting
	if(LA_traits<T>::realpart(p->second) < CHOLESKYEPSILON) continue; //must not break due to the a priori  pivoting
	++rank;
	typename LA_traits<T>::normtype tmp = std::sqrt(LA_traits<T>::realpart(p->second));
	p->second = tmp;
	NRVec<T> linek(0.,nn);
	for(p=r.v[k]->begin(); p!=r.v[k]->end(); ++p) 
		{
		if(p->first > k) p->second /= tmp;
		linek[p->first] = p->second;
		}
	for(j=k+1; j<nn; ++j)
 	    if(r.v[j])
		{
		T akj = LA_traits<T>::conjugate(linek[j]);
		NRVec<int> linek_used(0,nn);
		if(std::abs(akj)>SPARSEEPSILON) 
			{
			for(p=r.v[j]->begin(); p!=r.v[j]->end(); ++p)
				{
					p->second -= akj * linek[p->first];
					linek_used[p->first]=1;
				}	
			//subtract also elements nonzero in line k but non-existent in line j
			for(i=j; i<nn; ++i) 
			    if(!linek_used[i] && std::abs(linek[i]) > SPARSEEPSILON)
				{
				(*r.v[j])[i] = -akj * linek[i];
				}
			}
		}
	}

/*
SparseSMat<T> br(nn);
br.gemm(0,r,'c',r,'n',1);
//cancel low triangle from br
for(k=0; k<nn; ++k)
    if(br.v[k])
	{
	 typename std::map<SPMatindex,T>::iterator p;
	for(p=br.v[k]->begin(); p!=br.v[k]->end(); ++p)
		if(p->first <k) p->second=0.;
	}
std::cout << "Error before permute = " <<(br-r0).norm()<<std::endl;
*/

//permute the result back;
for(k=0; k<nn; ++k)
    if(r.v[k])
	{
	typename std::map<SPMatindex,T>::iterator p;
	typename std::map<SPMatindex,T> *vnew = new typename std::map<SPMatindex,T>;
	for(p=r.v[k]->begin(); p!=r.v[k]->end(); ++p)
		{
        	if(std::abs(p->second) > SPARSEEPSILON) (*vnew)[pivot[p->first]] = p->second;
		}
	delete r.v[k];
	r.v[k]=vnew;
	}

return r;
}



//outer product expected to be sparse
template<typename T>
SparseSMat<T> otimes_sparse(const NRVec<T> &lhs, const NRVec<T> &rhs, const bool conjugate=false, const T &scale=1) 
{
SparseSMat<T> r(lhs.size(),rhs.size());
for(SPMatindex i=0; i<lhs.size(); ++i)
    if(lhs[i]!=(T)0)
	{
	for(SPMatindex j=0; j<rhs.size(); ++j)
	    if(rhs[j]!=(T)0)
		{
		T x=lhs[i]*(conjugate?LA_traits<T>::conjugate(rhs[j]):rhs[j])*scale;
		if(std::abs(x)>SPARSEEPSILON) r.add(i,j,x);
		}
	}
return r;
}




}//namespace
#endif //_SPARSESMAT_H_
