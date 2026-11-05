/*
    LA: linear algebra C++ interface library
    Copyright (C) 2010-2026 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>

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

#ifndef _CSRMAT_H_
#define _CSRMAT_H_

#include <string>
#include <cmath>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include "la_traits.h"
#include "sparsemat.h"
#include "sparsesmat.h"
#include "vec.h"
#include "mat.h"
#include "smat.h"

namespace LA {

//compressed row format read-efficient matrix storage
//row i has the elements v[k], column col[k], k=rowptr[i]..rowptr[i+1]-1, columns strictly ascending within a row
//the full matrix is always stored; symmetric only declares A = A^dagger (hermitian, real symmetric) e.g. for davidson, lanczos
//rowptr, col and v share data independently (copy-on-write), e.g. a scaled copy shares the structure with the original

template <typename T>
class CSRMat
{
protected:
	SPMatindex nn;
	SPMatindex mm;
	bool symmetric;
	NRVec<SPMatindex> rowptr;
	NRVec<SPMatindex> col;
	NRVec<T> v;
	void fromcoo(const SPMatindex n, const SPMatindex m, const int nz, const SPMatindex *ri, const SPMatindex *ci, const T *vi); //sorts, sums duplicates, drops small elements
	void permrows(const SPMatindex *to); //row r becomes row to[r]
public:
	//iterator over the stored elements in row-major order
	class iterator {
	private:
		const CSRMat *a;
		SPMatindex row;
		int k;
		matel<T> my;
		void set() {while(row<a->nn && k>=(int)a->rowptr[row+1]) ++row; if(row<a->nn) {my.row=row; my.col=a->col[k]; my.elem=a->v[k]; my.next=NULL;}}
	public:
		iterator(): a(NULL), row(0), k(0) {};
		iterator(const CSRMat &rhs): a(&rhs), row(0), k(0) {if(a->nn) set();}
		iterator & operator++() {++k; set(); return *this;}
		bool notend() const {return a && row<a->nn;}
		bool operator==(const iterator &rhs) const {return notend()==rhs.notend() && (!notend() || k==rhs.k);}
		bool operator!=(const iterator &rhs) const {return !(*this==rhs);}
		const matel<T> & operator*() const {return my;}
		const matel<T> * operator->() const {return &my;}
	};
	iterator begin() const {return iterator(*this);}
	iterator end() const {return iterator();}

	CSRMat() : nn(0), mm(0), symmetric(false) {};
	explicit CSRMat(const SPMatindex n, const SPMatindex m); //zero matrix
	explicit CSRMat(const SPMatindex n); //zero square matrix
	CSRMat(const SPMatindex n, const SPMatindex m, const NRVec<SPMatindex> &rowptr, const NRVec<SPMatindex> &col, const NRVec<T> &v); //shares the arrays
	explicit CSRMat(const SparseMat<T> &rhs);
	explicit CSRMat(const SparseSMat<T> &rhs);
	explicit CSRMat(const NRSMat<T> &rhs);
	explicit CSRMat(const NRMat<T> &rhs);
	void copyonwrite(bool detachonly=false, bool deep=true);
	void resize(const SPMatindex n, const SPMatindex m); //zero matrix
	void dealloc(void) {resize(0,0);}
	void clear() {bool sym=symmetric; resize(nn,mm); symmetric=sym;} //zero matrix of the same dimensions and symmetry
	void swap(CSRMat &rhs)
		{
		SPMatindex tmpnn=nn; nn=rhs.nn; rhs.nn=tmpnn;
		SPMatindex tmpmm=mm; mm=rhs.mm; rhs.mm=tmpmm;
		bool tmpsymmetric=symmetric; symmetric=rhs.symmetric; rhs.symmetric=tmpsymmetric;
		rowptr.swap(rhs.rowptr); col.swap(rhs.col); v.swap(rhs.v);
		}
	inline SPMatindex nrows() const {return nn;}
	inline SPMatindex ncols() const {return mm;}
	inline int length() const {return v.size();} //number of stored elements
	inline bool issymmetric() const {return symmetric;}
	inline void setsymmetric() {if(nn!=mm) laerror("non-square cannot be symmetric"); symmetric=true;} //only declares the matrix hermitian
	inline void setunsymmetric() {symmetric=false;}
	inline const NRVec<SPMatindex> & getrowptr() const {return rowptr;}
	inline const NRVec<SPMatindex> & getcol() const {return col;}
	inline const NRVec<T> & getval() const {return v;}
	const T operator()(const SPMatindex i, const SPMatindex j) const; //element (zero if not stored), binary search in the row
	void gemv(const T beta, NRVec<T> &r, const char trans, const T alpha, const NRVec<T> &x) const {r.gemv(beta,*this,trans,alpha,x);};
	inline const NRVec<T> operator*(const NRVec<T> &rhs) const {NRVec<T> result(nn); result.gemv((T)0,*this,'n',(T)1,rhs); return result;};
	const T* diagonalof(NRVec<T> &r, const bool divide=0, bool cache=false) const; //diagonal (diagonal of A^T A for non-square, as SparseMat), or divide r by it
	const CSRMat transpose(bool conj=false) const;
	CSRMat & operator*=(const T &a); //multiply by a scalar (the structure stays shared)
	CSRMat & operator=(const T &a); //a times the unit matrix
	CSRMat & operator+=(const T &a); //add a to the diagonal
	CSRMat & operator-=(const T &a) {return *this += -a;}
	void identity() {*this = (T)1;}
	void axpy(const T alpha, const CSRMat &x, const bool transp=false); //this += alpha x (alpha x^T); elements cancelled to zero stay stored, see simplify()
	CSRMat & operator+=(const CSRMat &rhs) {axpy((T)1,rhs); return *this;}
	CSRMat & operator-=(const CSRMat &rhs) {axpy((T)-1,rhs); return *this;}
	inline const CSRMat operator*(const T &rhs) const {return CSRMat(*this) *= rhs;}
	inline const CSRMat operator+(const T &rhs) const {return CSRMat(*this) += rhs;}
	inline const CSRMat operator-(const T &rhs) const {return CSRMat(*this) -= rhs;}
	inline const CSRMat operator+(const CSRMat &rhs) const {return CSRMat(*this) += rhs;}
	inline const CSRMat operator-(const CSRMat &rhs) const {return CSRMat(*this) -= rhs;}
	const typename LA_traits<T>::normtype norm(const T scalar=(T)0) const; //Frobenius norm of A - scalar I
	const T dot(const CSRMat &rhs) const; //\sum_{ij} conj(a_ij) b_ij as NRMat::dot
	const T dot(const NRMat<T> &rhs) const;
	const T trace() const;
	void simplify(const typename LA_traits<T>::normtype sparseepsilon=LA_precision<typename LA_traits<T>::normtype>::sparse_epsilon()); //removes elements with |x| <= sparseepsilon
	inline const NRMat<T> operator*(const NRMat<T> &rhs) const {NRMat<T> r(nn,rhs.ncols()); r.gemm((T)0,*this,'n',rhs,'n',(T)1); return r;};
	void gemm(const T beta, const CSRMat &a, const char transa, const CSRMat &b, const char transb, const T alpha); //this = alpha op(A) op(B) + beta this, sparse result
	inline const CSRMat operator*(const CSRMat &rhs) const {CSRMat r; r.gemm((T)0,*this,'n',rhs,'n',(T)1); return r;};
	CSRMat & oplusequal(const CSRMat &rhs); //direct sum
	CSRMat & oplusequal(const NRMat<T> &rhs) {return oplusequal(CSRMat(rhs));}
	CSRMat & oplusequal(const NRSMat<T> &rhs) {return oplusequal(CSRMat(rhs));}
	const CSRMat oplus(const CSRMat &rhs) const {return CSRMat(*this).oplusequal(rhs);}
	const CSRMat oplus(const NRMat<T> &rhs) const {return CSRMat(*this).oplusequal(rhs);}
	const CSRMat oplus(const NRSMat<T> &rhs) const {return CSRMat(*this).oplusequal(rhs);}
	const CSRMat otimes(const CSRMat &rhs) const; //direct product, (A x B)_{i nB+k, j mB+l} = a_ij b_kl as NRMat::operator|
	const CSRMat otimes(const NRMat<T> &rhs) const {return otimes(CSRMat(rhs));}
	const CSRMat otimes(const NRSMat<T> &rhs) const {return otimes(CSRMat(rhs));}
	//permutations: element of row r goes to row p(r) (with inverse p^{-1}); matrix indices from 0, permutation from 1; the full matrix is stored, so rows means only rows
	void permuteme_rows(const NRPerm<int> &p, const bool inverse=false);
	void permuteme_cols(const NRPerm<int> &p, const bool inverse=false);
	void permuteme_both(const NRPerm<int> &p, const NRPerm<int> &q, const bool inverse=false);
	const CSRMat permuted_rows(const NRPerm<int> &p, const bool inverse=false) const {CSRMat a(*this); a.permuteme_rows(p,inverse); return a;};
	const CSRMat permuted_cols(const NRPerm<int> &p, const bool inverse=false) const {CSRMat a(*this); a.permuteme_cols(p,inverse); return a;};
	const CSRMat permuted_both(const NRPerm<int> &p, const NRPerm<int> &q, const bool inverse=false) const {CSRMat a(*this); a.permuteme_both(p,q,inverse); return a;};
	void permuterows(const NRVec<SPMatindex> &p); //as SparseMat: row r goes to row p[r], indices from 0
	void permutecolumns(const NRVec<SPMatindex> &p);
	void permuteindices(const NRVec<SPMatindex> &p); //rows and columns
	void transposeme(bool conj=false) {*this=transpose(conj);}
	void get(int fd, bool dimensions=1, bool transp=0, bool orcaformat=false);
	void put(int fd, bool dimensions=1, bool transp=0, bool orcaformat=false) const;
};

template<typename T>
inline const CSRMat<T> commutator(const CSRMat<T> &x, const CSRMat<T> &y, const bool trx=0, const bool tryy=0)
{
CSRMat<T> r;
r.gemm((T)0,x,trx?'t':'n',y,tryy?'t':'n',(T)1);
r.gemm((T)1,y,tryy?'t':'n',x,trx?'t':'n',(T)-1);
return r;
}

template<typename T>
inline const CSRMat<T> anticommutator(const CSRMat<T> &x, const CSRMat<T> &y, const bool trx=0, const bool tryy=0)
{
CSRMat<T> r;
r.gemm((T)0,x,trx?'t':'n',y,tryy?'t':'n',(T)1);
r.gemm((T)1,y,tryy?'t':'n',x,trx?'t':'n',(T)1);
return r;
}

template <typename T>
inline const NRVec<T> NRVec<T>::operator*(const CSRMat<T> &mat) const
{
NOT_GPU(*this);
NRVec<T> result(mat.ncols());
result.gemv((T)0,mat,'t',(T)1,*this);
return result;
}

template <class T>
std::ostream& operator<<(std::ostream &s, const CSRMat<T> &x) //the same text format as SparseMat
		{
		s << (int)x.nrows() << ' ' << (int)x.ncols() << '\n';
		for(typename CSRMat<T>::iterator p=x.begin(); p.notend(); ++p)
			s << (int)p->row << ' ' << (int)p->col << ' ' << (typename LA_traits_io<T>::IOtype)p->elem << '\n';
		s << "-1 -1\n";
		return s;
		}

template <class T>
std::istream& operator>>(std::istream &s, CSRMat<T> &x)
		{
		SparseMat<T> a;
		s >> a;
		x=CSRMat<T>(a);
		return s;
		}

}//namespace
#endif //_CSRMAT_H_
