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
#ifndef _SPARSEMAT_H_
#define _SPARSEMAT_H_
#include "la_traits.h"

namespace LA {

//threshold for neglecting elements, if not defined, no tests are done except exact zero test in simplify - might be even faster
//seems to perform better with a threshold, in spite of abs() tests
const double SPARSEEPSILON=1e-15;

typedef unsigned int SPMatindex;
typedef int SPMatindexdiff; //more clear would be to use traits

//element of a linked list
template<typename T>
struct matel
        {
        T elem;
        SPMatindex row;
        SPMatindex col;
        matel *next;
        };


template <typename T>
class SparseMat {
protected:
	SPMatindex nn;
        SPMatindex mm;
	bool symmetric;
	unsigned int nonzero;
        int *count;
	matel<T> *list;
private:
	matel<T> **rowsorted; //NULL terminated
	matel<T> **colsorted; //NULL terminated
	void unsort();
	void deletelist();
	void copylist(const matel<T> *l);
public:
	//iterator
        class iterator {
        private:
                matel<T> *p;
        public:
                iterator() {};
                ~iterator() {};
                iterator(matel<T> *list): p(list) {};
                bool operator==(const iterator rhs) const {return p==rhs.p;}
                bool operator!=(const iterator rhs) const {return p!=rhs.p;}
                iterator operator++() {return p=p->next;}
                iterator operator++(int) {matel<T> *q=p; p=p->next; return q;}
                matel<T> & operator*() const {return *p;}
                matel<T> * operator->() const {return p;}
        };
        iterator begin() const {return list;}
        iterator end() const {return NULL;}

	//constructors etc.
	inline SparseMat() :nn(0),mm(0),symmetric(0),nonzero(0),count(NULL),list(NULL),rowsorted(NULL),colsorted(NULL) {};
	inline SparseMat(const SPMatindex n, const SPMatindex m) :nn(n),mm(m),symmetric(0),nonzero(0),count(new int(1)),list(NULL),rowsorted(NULL),colsorted(NULL) {};
	SparseMat(const SparseMat &rhs); //copy constructor
	inline int getcount() const {return count?*count:0;}
	explicit SparseMat(const NRMat<T> &rhs); //construct from a dense one
	explicit SparseMat(const NRSMat<T> &rhs); //construct from a dense symmetric one
	SparseMat & operator=(const SparseMat &rhs);
	SparseMat & operator=(const T &a);          //assign a to diagonal
	void identity() {*this = (T)1;};
    	SparseMat & operator+=(const T &a);         //assign a to diagonal
	SparseMat & operator-=(const T &a);         //assign a to diagonal
        SparseMat & operator*=(const T &a);         //multiply by a scalar
        SparseMat & operator+=(const SparseMat &rhs);
	SparseMat & addtriangle(const SparseMat &rhs, const bool lower, const char sign);
        SparseMat & join(SparseMat &rhs); //more efficient +=, rhs will be emptied
        SparseMat & operator-=(const SparseMat &rhs);
	inline const SparseMat operator+(const T &rhs) const {return SparseMat(*this) += rhs;}
        inline const SparseMat operator-(const T &rhs) const {return SparseMat(*this) -= rhs;}
        inline const SparseMat operator*(const T &rhs) const {return SparseMat(*this) *= rhs;}
        inline const SparseMat operator+(const SparseMat &rhs) const {return SparseMat(*this) += rhs;} //must not be symmetric+general
        inline const SparseMat operator-(const SparseMat &rhs) const {return SparseMat(*this) -= rhs;} //must not be symmetric+general
	inline const NRVec<T> operator*(const NRVec<T> &rhs) const; // SparseMat * Vec
	inline const NRMat<T> operator*(const NRMat<T> &rhs) const; // SparseMat * Mat
	const T* diagonalof(NRVec<T> &, const bool divide=0, bool cache=false) const; //get diagonal
	void gemv(const T beta, NRVec<T> &r, const char trans, const T alpha, const NRVec<T> &x, bool treat_as_symmetric=false) const {r.gemv(beta,*this,trans,alpha,x,treat_as_symmetric);};
	const SparseMat operator*(const SparseMat &rhs) const; 
	SparseMat & oplusequal(const SparseMat &rhs); //direct sum
	SparseMat & oplusequal(const NRMat<T> &rhs);
	SparseMat & oplusequal(const NRSMat<T> &rhs);
	const SparseMat oplus(const SparseMat &rhs) const {return SparseMat(*this).oplusequal(rhs);}; //direct sum
	const SparseMat oplus(const NRMat<T> &rhs) const {return SparseMat(*this).oplusequal(rhs);};
	const SparseMat oplus(const NRSMat<T> &rhs) const {return SparseMat(*this).oplusequal(rhs);};
	const SparseMat otimes(const SparseMat &rhs) const; //direct product
        const SparseMat otimes(const NRMat<T> &rhs) const;
        const SparseMat otimes(const NRSMat<T> &rhs) const;
        void gemm(const T beta, const SparseMat &a, const char transa, const SparseMat &b, const char transb, const T alpha);//this := alpha*op( A )*op( B ) + beta*this, if this is symemtric, only half will be added onto it
	const T dot(const SparseMat &rhs) const; //supervector dot product
	const T dot(const NRMat<T> &rhs) const; //supervector dot product
	inline ~SparseMat();
	void axpy(const T alpha, const SparseMat &x, const bool transp=0); // this+= a*x(transposed)
	inline matel<T> *getlist() const {return list;}
	void setlist(matel<T> *l) {list=l;}
	inline SPMatindex nrows() const {return nn;}
        inline SPMatindex ncols() const {return mm;}
	void get(int fd, bool dimensions=1, bool transposed=false, bool orcaformat=false); //orcaformat not supported (dense only), for LA_traits
	void put(int fd, bool dimensions=1, bool transposed=false, bool orcaformat=false) const;
	void resize(const SPMatindex n, const SPMatindex m); //destructive
        void dealloc(void) {resize(0,0);}
	void incsize(const SPMatindex n, const SPMatindex m); //increase size without destroying the data
	void transposeme();
	const SparseMat transpose() const;
	void permuteindices(const NRVec<SPMatindex> &p); //for backward compatibility, indices from 0
	void permuterows(const NRVec<SPMatindex> &p); //for backward compatibility, indices from 0
	void permutecolumns(const NRVec<SPMatindex> &p); //for backward compatibility, indices from 0
	void permuteme_rows(const NRPerm<int> &p, const bool inverse=false); //matrix indices from 0, permutation from 1
	void permuteme_cols(const NRPerm<int> &p, const bool inverse=false); //matrix indices from 0, permutation from 1
	void permuteme_both(const NRPerm<int> &p, const NRPerm<int> &q, const bool inverse=false); //matrix indices from 0, permutation from 1
	const SparseMat permuted_rows(const NRPerm<int> &p, const bool inverse=false) const {SparseMat a(*this); a.permuteme_rows(p,inverse); return a;};
	const SparseMat permuted_cols(const NRPerm<int> &p, const bool inverse=false) const {SparseMat a(*this); a.permuteme_cols(p,inverse); return a;};
	const SparseMat permuted_both(const NRPerm<int> &p, const NRPerm<int> &q, const bool inverse=false) const {SparseMat a(*this); a.permuteme_both(p,q,inverse); return a;};
	inline void setsymmetric() {if(nn!=mm) laerror("non-square cannot be symmetric"); symmetric=1;}
	inline void defineunsymmetric() {symmetric=0;} //just define and do nothing with it
	void setunsymmetric();//unwind the matrix assuming it was indeed symmetric
	inline bool issymmetric() const {return symmetric;}
	unsigned int length() const;
	void copyonwrite(bool detachonly=false, bool deep=true);
	void clear() {bool sym=symmetric; resize(nn,mm); symmetric=sym;} //zero matrix of the same dimensions and symmetry
	void simplify(const double sparseepsilon=SPARSEEPSILON);
	const T trace() const;
	const typename LA_traits<T>::normtype norm(const T scalar=(T)0) const; //is const only mathematically, not in internal implementation - we have to simplify first
	unsigned int sort(int type) const;
	inline void add(const SPMatindex n, const SPMatindex m, const T elem) {if(nonzero) unsort(); matel<T> *ltmp= new matel<T>; ltmp->next=list; list=ltmp; list->row=n; list->col=m; list->elem=elem;}
	void addsafe(const SPMatindex n, const SPMatindex m, const T elem);
	void swap(SparseMat &rhs) //more efficient swap than via tmp and constructors and operator=
		{
		SPMatindex tmpnn=nn; nn=rhs.nn; rhs.nn=tmpnn;
		SPMatindex tmpmm=mm; mm=rhs.mm; rhs.mm=tmpmm;
		int *tmpcount=count; count=rhs.count; rhs.count=tmpcount;
		bool tmpsymmetric=symmetric; symmetric=rhs.symmetric; rhs.symmetric=tmpsymmetric;
		int tmpnonzero=nonzero; nonzero=rhs.nonzero; rhs.nonzero=tmpnonzero;
		matel<T> *tmplist=list; list=rhs.list; rhs.list=tmplist;
		matel<T> **tmprowsorted=rowsorted; rowsorted=rhs.rowsorted; rhs.rowsorted=tmprowsorted;
		matel<T> **tmpcolsorted=colsorted; colsorted=rhs.colsorted; rhs.colsorted=tmpcolsorted;
		}
};

}//namespace

//due to mutual includes this has to be after full class declaration
#include "vec.h"
#include "smat.h"
#include "mat.h"

namespace LA {

template <typename T>
inline const NRVec<T> SparseMat<T>::operator*(const NRVec<T> &rhs) const
{NRVec<T> result(nn); result.gemv((T)0,*this,'n',(T)1,rhs); return result;};

template <typename T>
inline const NRMat<T> SparseMat<T>::operator*(const NRMat<T> &rhs) const
{NRMat<T> result(nn,rhs.ncols()); result.gemm((T)0,*this,'n',rhs,'n',(T)1); return result;};

template <class T>
std::ostream& operator<<(std::ostream &s, const SparseMat<T> &x)
                {
                SPMatindex n,m;
                n=x.nrows();
                m=x.ncols();
                s << (int)n << ' ' << (int)m << '\n';
		matel<T> *list=x.getlist();
		while(list)
                        {
                        s << (int)list->row << ' ' << (int)list->col  << ' ' << (typename LA_traits_io<T>::IOtype)list->elem << '\n';
			list=list->next;
                        }
		s << "-1 -1\n";
                return s;
                }

template <class T>
std::istream& operator>>(std::istream  &s, SparseMat<T> &x)
                {
                int i,j;
		int n,m;
		matel<T> *l=NULL;
                s >> n >> m;
                x.resize(n,m);
		s >> i >> j;
		while(i>=0 && j>=0)
			{
			matel<T> *ll = l;
			l= new matel<T>;
			l->next= ll;
			l->row=i;
			l->col=j;
			typename LA_traits_io<T>::IOtype tmp;
			s >> tmp;
			l->elem=tmp;	
			s >> i >> j;
			}
		x.setlist(l);
                return s;
                }


//destructor
template <typename T>
SparseMat<T>::~SparseMat()
{
	unsort();
        if(!count) return;
        if(--(*count)<=0)
                {
		deletelist();
                delete count;
                }
}

//copy constructor (sort arrays are not going to be copied)
template <typename T>
SparseMat<T>::SparseMat(const SparseMat<T> &rhs)
{
#ifdef debug
if(! &rhs) laerror("SparseMat copy constructor with NULL argument");
#endif
        nn=rhs.nn;
        mm=rhs.mm;
	symmetric=rhs.symmetric;
	if(rhs.list&&!rhs.count) laerror("some inconsistency in SparseMat contructors or assignments");
        list=rhs.list;
        if(list) {count=rhs.count; (*count)++;} else count=new int(1); //make the matrix defined, but empty and not shared
	colsorted=rowsorted=NULL;
	nonzero=0;
}

template <typename T>
const SparseMat<T> SparseMat<T>::transpose() const
{
if(list&&!count) laerror("some inconsistency in SparseMat transpose");
SparseMat<T> result;
result.nn=mm;
result.mm=nn;
result.symmetric=symmetric;
if(result.symmetric) 
	{
	result.list=list;
        if(list) {result.count=count; (*result.count)++;} else result.count=new int(1); //make the matrix defined, but empty and not shared
	}
else //really transpose it
	{
	result.count=new int(1);
	result.list=NULL;
	matel<T> *l =list;
	while(l)
		{
		result.add(l->col,l->row,l->elem);
		l=l->next;
		}
	}
result.colsorted=result.rowsorted=NULL;
result.nonzero=0;
return result;
}



template<typename T>
inline const SparseMat<T> commutator ( const SparseMat<T> &x, const SparseMat<T> &y, const bool trx=0, const bool tryy=0)
{
SparseMat<T> r;
r.gemm((T)0,x,trx?'t':'n',y,tryy?'t':'n',(T)1);
r.gemm((T)1,y,tryy?'t':'n',x,trx?'t':'n',(T)-1); //saves a temporary and simplifies the whole sum
return r;
}

template<typename T>
inline const SparseMat<T> anticommutator ( const SparseMat<T> &x, const SparseMat<T> &y, const bool trx=0, const bool tryy=0)
{
SparseMat<T> r;
r.gemm((T)0,x,trx?'t':'n',y,tryy?'t':'n',(T)1);
r.gemm((T)1,y,tryy?'t':'n',x,trx?'t':'n',(T)1); //saves a temporary and simplifies the whole sum
return r;
}


//add sparse to dense
template<typename T>
NRMat<T> & NRMat<T>::operator+=(const SparseMat<T> &rhs)
{
if((unsigned int)nn!=rhs.nrows()||(unsigned int)mm!=rhs.ncols()) laerror("incompatible matrices in +=");
copyonwrite();
matel<T> *l=rhs.getlist();
bool sym=rhs.issymmetric();
while(l)
        {
#ifdef MATPTR
        v[l->row][l->col] +=l->elem;
        if(sym && l->row!=l->col) v[l->col][l->row] +=l->elem;
#else
        v[l->row*mm+l->col] +=l->elem;
         if(sym && l->row!=l->col) v[l->col*mm+l->row] +=l->elem;
#endif
        l=l->next;
        }
return *this;
}

//subtract sparse from dense
template<typename T>
NRMat<T> & NRMat<T>::operator-=(const SparseMat<T> &rhs)
{
if((unsigned int)nn!=rhs.nrows()||(unsigned int)mm!=rhs.ncols()) laerror("incompatible matrices in -=");
copyonwrite();
matel<T> *l=rhs.getlist();
bool sym=rhs.issymmetric();
while(l)
        {
#ifdef MATPTR
        v[l->row][l->col] -=l->elem;
        if(sym && l->row!=l->col) v[l->col][l->row] -=l->elem;
#else
        v[l->row*mm+l->col] -=l->elem;
         if(sym && l->row!=l->col) v[l->col*mm+l->row] -=l->elem;
#endif
        l=l->next;
        }
return *this;
}


//a rudimentary class for sparse matrix with a dense diagonal - implemented just enough to call davidson etc. on it
template <typename T>
class SemiSparseMat {
	friend class NRMat<T>;
public:
	NRVec<T> diagonal;
	SparseMat<T> offdiagonal;
        SemiSparseMat() {};
        SemiSparseMat(const SPMatindex n, const SPMatindex m) : offdiagonal(n,m) {if(n==m) {diagonal.resize(n); diagonal.clear();}}; //zero matrix, add() accumulates into the diagonal
        SPMatindex nrows() const {return offdiagonal.nrows();}
        SPMatindex ncols() const {return offdiagonal.ncols();}
        SPMatindex issymmetric() const {return offdiagonal.issymmetric();}
	void setsymmetric() {offdiagonal.setsymmetric();}
	void gemv(const T beta, NRVec<T> &r, const char trans, const T alpha, const NRVec<T> &x, bool treat_as_symmetric=false) const
		{
		offdiagonal.gemv(beta,r,trans,alpha,x,treat_as_symmetric);
		if(diagonal.size()>0 && alpha!=(T)0)
			{
			if(diagonal.size()!=r.size()) laerror("mismatch in diagonal size");
			for(int i=0; i<diagonal.size(); ++i) r[i] += alpha * diagonal[i]*x[i];
			}
		}
	NRVec<T> operator*(const NRVec<T> &rhs) const {NRVec<T> result(nrows()); gemv((T)0,result,'n',(T)1,rhs); return result;};
	const T* diagonalof(NRVec<T> &x, const bool divide=0, bool cache=false) const
		{
		if(diagonal.size()>0) //we ASSUME the diagonal is ONLY in the vector
			{
			if(diagonal.size()!=x.size()) laerror("mismatch in diagonal size");
			if(divide) {x.copyonwrite(); for (int i=0; i<x.size(); ++i) x[i]/=diagonal[i]; return NULL;}
			else	{x|=diagonal; return &x[0];}
			}
		return offdiagonal.diagonalof(x,divide,cache);
		}
	void add(const SPMatindex n, const SPMatindex m, const T elem) {if(diagonal.size()>0 && n==m) diagonal[n]+=elem; else offdiagonal.add(n,m,elem);}
	void clear() {diagonal.clear(); offdiagonal.clear();}
	void copyonwrite() {diagonal.copyonwrite(); offdiagonal.copyonwrite();}
	void resize(const SPMatindex n, const SPMatindex m) {offdiagonal.resize(n,m); diagonal.resize(n==m?n:0); if(n==m) diagonal.clear();} //zero matrix as SparseMat::resize
};

template <class T>
std::ostream& operator<<(std::ostream &s, const SemiSparseMat<T> &x)
                {
		s << "diagonal= "<<x.diagonal;
		s << "offdiagonal= "<<x.offdiagonal;
                return s;
                }


}//namespace
#endif
