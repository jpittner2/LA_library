/*
    LA: linear algebra C++ interface library
    Copyright (C) 2010 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>

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
#include "qsort.h"

#ifdef CUDALA
#include "cusparse.h"
#include "cuda_runtime.h"
#endif

namespace LA {

//compressed row format read-efficient matrix storage

template <typename T>
class CSRMat
{
protected:
	SPMatindex nn;
	SPMatindex mm;
	SPMatindex nnz;
	int *count;
	SPMatindex *rowptr;
	SPMatindex *col;
	T *v;
#ifdef CUDALA
        GPUID location;
	cusparseMatDescr_t descr;
#endif

public:
	CSRMat() : nn(0), mm(0), nnz(0), count(NULL), rowptr(NULL), col(NULL), v(NULL) {};
	CSRMat(const CSRMat &rhs);
	explicit CSRMat(const SPMatindex n, const SPMatindex m); 
        explicit CSRMat(const SPMatindex n);
	explicit CSRMat(const SparseMat<T> &rhs);
	explicit CSRMat(const SparseSMat<T> &rhs);
	explicit CSRMat(const NRSMat<T> &rhs);
	explicit CSRMat(const NRMat<T> &rhs);
	CSRMat & operator=(const CSRMat &rhs);
	void copyonwrite();
        void resize(const SPMatindex nn, const SPMatindex mm);
	void dealloc(void) {resize(0,0);}
	void moveto(GPUID destination);
        void clear();
	~CSRMat();
	inline int getcount() const {return count?*count:0;}
        inline const CSRMat operator*(const T &rhs) const {return CSRMat(*this) *= rhs;}
	inline const CSRMat operator+(const T &rhs) const {return CSRMat(*this) += rhs;}
        inline const CSRMat operator-(const T &rhs) const {return CSRMat(*this) -= rhs;}
        inline const CSRMat operator+(const CSRMat &rhs) const {return CSRMat(*this) += rhs;} 
        inline const CSRMat operator-(const CSRMat &rhs) const {return CSRMat(*this) -= rhs;}
        CSRMat & operator*=(const T &a);         //multiply by a scalar
        CSRMat & operator=(const T &a);          //assign a to diagonal
        CSRMat & operator+=(const T &a);         //assign a to diagonal
        CSRMat & operator-=(const T &a);         //assign a to diagonal
	void axpy(const T alpha, const CSRMat &x, const bool transp=0); // this+= a*x
        inline CSRMat & operator+=(const CSRMat &rhs) {axpy((T)1,rhs); return *this;};
        inline CSRMat & operator-=(const CSRMat &rhs) {axpy((T)-1,rhs); return *this;};
	const T* diagonalof(NRVec<T> &, const bool divide=0, bool cache=false) const; //get diagonal
	void gemv(const T beta, NRVec<T> &r, const char trans, const T alpha, const NRVec<T> &x) const;
	inline const NRVec<T> operator*(const NRVec<T> &rhs) const {NRVec<T> result(nn); this->gemv((T)0,result,'n',(T)1,rhs); return result;};
	typename LA_traits<T>::normtype norm(const T scalar=(T)0) const;
        inline const CSRMat operator*(const CSRMat &rhs) const {CSRMat<T> r(nn,mm); r.gemm(0,*this,'n',rhs,'n',1); return r;}; 
	void gemm(const T beta, const CSRMat &a, const char transa, const CSRMat &b, const char transb, const T alpha); //this := alpha*op( A )*op( B ) + beta*this 
	inline SPMatindex length() {return nnz;};
	void transposeme() const {laerror("in-place transposition not implemented for CSRMat");};
	CSRMat transpose(bool conj=false) const {laerror("CSR transpose not implemented");}
	void get(int fd, bool dimen, bool transp);
        void put(int fd, bool dimen, bool transp) const;
	int nrows() const {return nn;}
	int ncols() const {return mm;}

	/*class iterator {//not efficient, just for output to ostreams
        private:
                matel<T> *p;
		matel<T> my;
		SPMatindex mynn;
		SPMatindex mymm;
		

        public:
		//compiler-generated iterator & operator=(const  iterator &rhs); 
		//compiler-generated iterator(const iterator &rhs); 
                iterator()
                iterator(const CSRMat &rhs)
                iterator operator++()  
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
	*/
};

template <typename T>
std::ostream & operator<<(std::ostream &s, const CSRMat<T> &x);

template <class T>
std::istream& operator>>(std::istream  &s, CSRMat<T> &x);


}//namespace
#endif //_CSRMAT_H_
