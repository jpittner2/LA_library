/*
    LA: linear algebra C++ interface library
    Copyright (C) 2008 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>
		  complex versions written by Roman Curik <roman.curik@jh-inst.cas.cz>

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
#ifndef _LA_VEC_H_
#define _LA_VEC_H_

#include "la_traits.h"

namespace LA {

//////////////////////////////////////////////////////////////////////////////
// Forward declarations
template <typename T> void lawritemat(FILE *file,const T *a,int r,int c,
		const char *form0,int nodim,int modulo, int issym);

// Memory allocated constants for cblas routines
const static complex<double> CONE = 1.0, CMONE = -1.0, CZERO = 0.0;

// Macros to construct binary operators +,-,*, from +=, -=, *=
// for 3 cases: X + a, a + X, X + Y
#define NRVECMAT_OPER(E,X) \
template<class T> \
	inline const NR##E<T> NR##E<T>::operator X(const T &a) const \
{ return NR##E(*this) X##= a; } \
	\
	template<class T> \
	inline const NR##E<T> operator X(const T &a, const NR##E<T> &rhs) \
{ return NR##E<T>(rhs) X##= a; }

#define NRVECMAT_OPER2(E,X) \
template<class T> \
	inline const NR##E<T> NR##E<T>::operator X(const NR##E<T> &a) const \
{ return NR##E(*this) X##= a; }


// NRVec class
template <typename T>
class NRVec {
protected:
	int nn;
	T *v;
	int *count;
public:
	friend class NRSMat<T>;
	friend class NRMat<T>;

	inline NRVec(): nn(0),v(0),count(0){};
	explicit inline NRVec(const int n) : nn(n), v(new T[n]), count(new int(1)) {};
	inline NRVec(const T &a, const int n);
        inline NRVec(const T *a, const int n);
	inline NRVec(T *a, const int n, bool skeleton);
	inline NRVec(const NRVec &rhs);
	NRVec(const typename LA_traits_complex<T>::NRVec_Noncomplex_type &rhs, bool imagpart=false); //construct complex from real
	inline explicit NRVec(const NRSMat<T> & S);
#ifdef MATPTR
	explicit NRVec(const NRMat<T> &rhs) : NRVec(&rhs[0][0],rhs.nrows()*rhs.ncols()) {};
#else
	explicit NRVec(const NRMat<T> &rhs);
#endif
	NRVec & operator=(const NRVec &rhs);
	NRVec & operator=(const T &a);  //assign a to every element
        void randomize(const typename LA_traits<T>::normtype &x);
	NRVec & operator|=(const NRVec &rhs);
	const bool operator!=(const NRVec &rhs) const {if(nn!=rhs.nn) return 1; return LA_traits<T>::gencmp(v,rhs.v,nn);} //memcmp for scalars else elementwise
	const bool operator==(const NRVec &rhs) const {return !(*this != rhs);};
	const bool operator>(const NRVec &rhs) const;
	const bool operator<(const NRVec &rhs) const;
	const bool operator>=(const NRVec &rhs) const {return !(*this < rhs);};
	const bool operator<=(const NRVec &rhs) const {return !(*this > rhs);};
	const NRVec operator-() const;
	inline NRVec & operator+=(const NRVec &rhs);
	inline NRVec & operator-=(const NRVec &rhs);
	inline NRVec & operator*=(const NRVec &rhs); //elementwise
	inline NRVec & operator/=(const NRVec &rhs); //elementwise
	inline NRVec & operator+=(const T &a);
	inline NRVec & operator-=(const T &a);
	inline NRVec & operator*=(const T &a);
	inline int getcount() const {return count?*count:0;}
	inline const NRVec operator+(const NRVec &rhs) const;
	inline const NRVec operator-(const NRVec &rhs) const;
	inline const NRVec operator+(const T &a) const;
	inline const NRVec operator-(const T &a) const;
	inline const NRVec operator*(const T &a) const;
	inline const T operator*(const NRVec &rhs) const; //scalar product -> dot
	inline const T dot(const NRVec &rhs) const {return *this * rhs;}; //@@@for complex do conjugate
	void gemv(const T beta, const NRMat<T> &a, const char trans, const T alpha, const NRVec &x);
	void gemv(const T beta, const NRSMat<T> &a, const char trans /*just for compatibility*/,  const T alpha, const NRVec &x);
	void gemv(const T beta, const SparseMat<T> &a, const char trans, const T alpha, const NRVec &x,const bool treat_as_symmetric=false);
	void gemv(const typename LA_traits_complex<T>::Component_type beta, const typename LA_traits_complex<T>::NRMat_Noncomplex_type &a, const char trans, const typename LA_traits_complex<T>::Component_type alpha, const NRVec &x);
	void gemv(const typename LA_traits_complex<T>::Component_type beta, const typename LA_traits_complex<T>::NRSMat_Noncomplex_type &a, const char trans, const typename LA_traits_complex<T>::Component_type alpha, const NRVec &x);
	const NRVec operator*(const NRMat<T> &mat) const {NRVec<T> result(mat.ncols()); result.gemv((T)0,mat,'t',(T)1,*this); return result;};
	const NRVec operator*(const NRSMat<T> &mat) const {NRVec<T> result(mat.ncols()); result.gemv((T)0,mat,'t',(T)1,*this); return result;};
	const NRVec operator*(const SparseMat<T> &mat) const {NRVec<T> result(mat.ncols()); result.gemv((T)0,mat,'t',(T)1,*this); return result;};
	const NRMat<T> otimes(const NRVec<T> &rhs, const bool conjugate=false, const T &scale=1) const; //outer product
	inline const NRMat<T> operator|(const NRVec<T> &rhs) const {return otimes(rhs,true);};
	inline const T sum() const {T sum=0; for(int i=0; i<nn; i++) sum += v[i]; return sum;}; //sum of its elements
	inline const T asum() const; //sum of its elements absolute values
	inline const T dot(const T *a, const int stride=1) const; // ddot with a stride-vector
	inline T & operator[](const int i);
	inline const T & operator[](const int i) const;
	inline int size() const;
	inline operator T*(); //get a pointer to the data
	inline operator const T*() const; //get a pointer to the data
	~NRVec();
	void axpy(const T alpha, const NRVec &x); // this+= a*x
	void axpy(const T alpha, const T *x, const int stride=1); // this+= a*x
	void copyonwrite();
	void clear() {copyonwrite(); LA_traits<T>::clear(v,nn);}; //zero out
	void resize(const int n);
        void get(int fd, bool dimensions=1, bool transp=0);
        void put(int fd, bool dimensions=1, bool transp=0) const;
	NRVec & normalize();
	inline const typename LA_traits<T>::normtype norm() const;
	inline const T amax() const;
	inline const NRVec unitvector() const;
	void fprintf(FILE *f, const char *format, const int modulo) const;
	void fscanf(FILE *f, const char *format);
//sparse matrix concerning members
	explicit NRVec(const SparseMat<T> &rhs);                // dense from sparse matrix with one of dimensions =1
	inline void simplify() {}; //just for compatibility with sparse ones
	bool bigger(int i, int j) const {return LA_traits<T>::bigger(v[i],v[j]);};
	bool smaller(int i, int j) const {return LA_traits<T>::smaller(v[i],v[j]);};
	void swap(int i, int j) {T tmp; tmp=v[i]; v[i]=v[j]; v[j]=tmp;};
	int sort(int direction=0, int from=0, int to= -1, int *perm=NULL); //sort, ascending by default, returns parity of permutation
};

}//namespace

//due to mutual includes this has to be after full class declaration
#include "mat.h"
#include "smat.h"
#include "sparsemat.h"

namespace LA {
// formatted I/O
template <typename T>
std::ostream & operator<<(std::ostream &s, const NRVec<T> &x)
{
  int i, n;

  n = x.size();
  s << n << std::endl;
  for(i=0; i<n; i++) s << (typename LA_traits_io<T>::IOtype)x[i] << (i == n-1 ? '\n' : ' ');
  return s;
}

template <typename T>
std::istream & operator>>(std::istream &s, NRVec<T> &x)
{
  int i,n;

  s >> n;
  x.resize(n);
  typename LA_traits_io<T>::IOtype tmp;
  for(i=0; i<n; i++) {s >> tmp; x[i]=tmp;}
  return s;
}


// INLINES

// ctors
template <typename T>
inline NRVec<T>::NRVec(const T& a, const int n) : nn(n), v(new T[n]), count(new int)
{
	*count = 1;
	if(a != (T)0)
		for(int i=0; i<n; i++)
			v[i] = a;
	else
		memset(v, 0, nn*sizeof(T));
}

template <typename T>
inline NRVec<T>::NRVec(const T *a, const int n) : nn(n), count(new int)
{
		v=new T[n];
		*count = 1;
		memcpy(v, a, n*sizeof(T));
}

template <typename T>
inline NRVec<T>::NRVec(T *a, const int n, bool skeleton) : nn(n), count(new int)
{
	if(!skeleton)
		{
		v=new T[n];
		*count = 1;
		memcpy(v, a, n*sizeof(T));
		}
	else
		{
		*count = 2;
		v=a;
		}
}

template <typename T>
inline NRVec<T>::NRVec(const NRVec<T> &rhs)
{
	v = rhs.v;
	nn = rhs.nn;
	count = rhs.count;
	if(count) (*count)++;
}

template <typename T>
inline NRVec<T>::NRVec(const NRSMat<T> &rhs)
{
	nn = rhs.nn;
	nn = NN2;
	v = rhs.v;
	count = rhs.count;
	(*count)++;
}

// x += a
template<>
inline NRVec<double> & NRVec<double>::operator+=(const double &a)
{
	copyonwrite();
	cblas_daxpy(nn, 1.0, &a, 0, v, 1);
	return *this;
}

template<>
inline NRVec< complex<double> > &
NRVec< complex<double> >::operator+=(const complex<double> &a)
{
	copyonwrite();
	cblas_zaxpy(nn, &CONE, &a, 0, v, 1);
	return *this;
}

//and for general type
template <typename T>
inline NRVec<T> & NRVec<T>::operator+=(const T &a)
{
        copyonwrite();
	int i;
	for(i=0; i<nn; ++i) v[i]+=a;
        return *this;
}


// x -= a
template<>
inline NRVec<double> & NRVec<double>::operator-=(const double &a)
{
	copyonwrite();
	cblas_daxpy(nn, -1.0, &a, 0, v, 1);
	return *this;
}

template<>
inline NRVec< complex<double> > &
NRVec< complex<double> >::operator-=(const complex<double> &a)
{
	copyonwrite();
	cblas_zaxpy(nn, &CMONE, &a, 0, v, 1);
	return *this;
}

//and for general type
template <typename T>
inline NRVec<T> & NRVec<T>::operator-=(const T &a)
{
        copyonwrite();
        int i;
        for(i=0; i<nn; ++i) v[i]-=a;
        return *this;
}


// x += x
template<>
inline NRVec<double> & NRVec<double>::operator+=(const NRVec<double> &rhs)
{
#ifdef DEBUG
	if (nn != rhs.nn) laerror("daxpy of incompatible vectors");
#endif
	copyonwrite();
	cblas_daxpy(nn, 1.0, rhs.v, 1, v, 1);
	return *this;
}

template<>
inline NRVec< complex<double> > &
NRVec< complex<double> >::operator+=(const NRVec< complex<double> > &rhs)
{
#ifdef DEBUG
	if (nn != rhs.nn) laerror("daxpy of incompatible vectors");
#endif
	copyonwrite();
	cblas_zaxpy(nn, &CONE, rhs.v, 1, v, 1);
	return *this;
}

//and for general type
template <typename T>
inline NRVec<T> & NRVec<T>::operator+=(const NRVec<T> &rhs)
{
#ifdef DEBUG
        if (nn != rhs.nn) laerror("daxpy of incompatible vectors");
#endif
        copyonwrite();
	int i;
	for(i=0; i<nn; ++i) v[i]+=rhs.v[i];
        return *this;
}

//for general type only
template <typename T>
inline NRVec<T> & NRVec<T>::operator*=(const NRVec<T> &rhs)
{
#ifdef DEBUG
        if (nn != rhs.nn) laerror("*= of incompatible vectors");
#endif
        copyonwrite();
        int i;
        for(i=0; i<nn; ++i) v[i]*=rhs.v[i];
        return *this;
}

template <typename T>
inline NRVec<T> & NRVec<T>::operator/=(const NRVec<T> &rhs)
{
#ifdef DEBUG
        if (nn != rhs.nn) laerror("/= of incompatible vectors");
#endif
        copyonwrite();
        int i;
        for(i=0; i<nn; ++i) v[i]/=rhs.v[i];
        return *this;
}



// x -= x
template<>
inline NRVec<double> & NRVec<double>::operator-=(const NRVec<double> &rhs)
{
#ifdef DEBUG
	if (nn != rhs.nn) laerror("daxpy of incompatible vectors");
#endif
	copyonwrite();
	cblas_daxpy(nn, -1.0, rhs.v, 1, v, 1);
	return *this;
}

template<>
inline NRVec< complex<double> > &
NRVec< complex<double> >::operator-=(const NRVec< complex<double> > &rhs)
{
#ifdef DEBUG
	if (nn != rhs.nn) laerror("daxpy of incompatible vectors");
#endif
	copyonwrite();
	cblas_zaxpy(nn, &CMONE, rhs.v, 1, v, 1);
	return *this;
}

//and for general type
template <typename T>
inline NRVec<T> & NRVec<T>::operator-=(const NRVec<T> &rhs)
{
#ifdef DEBUG
        if (nn != rhs.nn) laerror("daxpy of incompatible vectors");
#endif
        copyonwrite();
        int i;
        for(i=0; i<nn; ++i) v[i]-=rhs.v[i];
        return *this;
}


// x *= a
template<>
inline NRVec<double> & NRVec<double>::operator*=(const double &a)
{
	copyonwrite();
	cblas_dscal(nn, a, v, 1);
	return *this;
}

template<>
inline NRVec< complex<double> > &
NRVec< complex<double> >::operator*=(const complex<double> &a)
{
	copyonwrite();
	cblas_zscal(nn, &a, v, 1);
	return *this;
}

//and for general type
template <typename T>
inline NRVec<T> & NRVec<T>::operator*=(const T &a)
{
        copyonwrite();
        int i;
        for(i=0; i<nn; ++i) v[i]*=a;
        return *this;
}


// scalar product x.y
template<>
inline const double NRVec<double>::operator*(const NRVec<double> &rhs) const
{
#ifdef DEBUG
        if (nn != rhs.nn) laerror("dot of incompatible vectors");
#endif
        return cblas_ddot(nn, v, 1, rhs.v, 1);
}


template<>
inline const complex<double>
NRVec< complex<double> >::operator*(const NRVec< complex<double> > &rhs) const
{
#ifdef DEBUG
	if (nn != rhs.nn) laerror("dot of incompatible vectors");
#endif
	complex<double> dot;
	cblas_zdotc_sub(nn, v, 1, rhs.v, 1, &dot);
	return dot;
}

template<typename T>
inline const T NRVec<T>::operator*(const NRVec<T> &rhs) const
{
#ifdef DEBUG
        if (nn != rhs.nn) laerror("dot of incompatible vectors");
#endif
	T dot = 0;
	for(int i=0; i<nn; ++i) dot+= v[i]*rhs.v[i];
	return dot;
}



// Sum of elements
template<>
inline const double NRVec<double>::asum() const
{
	return cblas_dasum(nn, v, 1);
}


// Dot product: x * y
template<>
inline const double NRVec<double>::dot(const double *y, const int stride) const
{
	return cblas_ddot(nn, y, stride, v, 1);
}
template<>
inline const complex<double>
NRVec< complex<double> >::dot(const complex<double> *y, const int stride) const
{
	complex<double> dot;
	cblas_zdotc_sub(nn, y, stride, v, 1, &dot);
	return dot;
}

// x[i] returns i-th element
template <typename T>
inline T & NRVec<T>::operator[](const int i)
{
#ifdef DEBUG
	if(_LA_count_check && *count != 1) laerror("possible lval [] with count > 1");
	if(i < 0 || i >= nn) laerror("NRVec out of range");
	if(!v) laerror("[] on unallocated NRVec");
#endif
	return v[i];
}
template <typename T>
inline const T & NRVec<T>::operator[](const int i) const
{
#ifdef DEBUG
	if(i < 0 || i >= nn) laerror("NRVec out of range");
	if(!v) laerror("[] on unallocated NRVec");
#endif
	return v[i];
}

// length of the vector
template <typename T>
inline int NRVec<T>::size() const
{
	return nn;
}

// reference Vec to the first element
template <typename T>
inline NRVec<T>::operator T*()
{
#ifdef DEBUG
	if(!v) laerror("unallocated NRVec in operator T*");
#endif
	return v;
}
template <typename T>
inline NRVec<T>::operator const T*() const
{
#ifdef DEBUG
	if(!v) laerror("unallocated NRVec in operator T*");
#endif
	return v;
}

// return norm of the Vec
template<>
inline const double  NRVec<double>::norm() const
{
	return cblas_dnrm2(nn, v, 1);
}
template<>
inline const double NRVec< complex<double> >::norm() const
{
	return cblas_dznrm2(nn, v, 1);
}

// Max element of the array
template<>
inline const double  NRVec<double>::amax() const
{
	return v[cblas_idamax(nn, v, 1)];
}
template<>
inline const complex<double> NRVec< complex<double> >::amax() const
{
	return v[cblas_izamax(nn, v, 1)];
}


// Make Vec unitvector
template <typename T>
inline const NRVec<T> NRVec<T>::unitvector() const
{
	return NRVec<T>(*this).normalize();
}

// generate operators: Vec + a, a + Vec, Vec * a
NRVECMAT_OPER(Vec,+)
NRVECMAT_OPER(Vec,-)
NRVECMAT_OPER(Vec,*)
// generate operators: Vec + Vec, Vec - Vec
NRVECMAT_OPER2(Vec,+)
NRVECMAT_OPER2(Vec,-)

// Few forward declarations

//basic stuff which has to be in .h
// dtor
template <typename T>
NRVec<T>::~NRVec()
{
        if(!count) return;
        if(--(*count) <= 0) {
                if(v) delete[] (v);
                delete count;
        }
}

// detach from a physical vector and make own copy
template <typename T>
void NRVec<T>::copyonwrite()
{
  if(!count) laerror("Vec::copyonwrite() of an undefined vector");
  if(*count > 1)
  {
    (*count)--;
    count = new int;
    *count = 1;
    T *newv = new T[nn];
    memcpy(newv, v, nn*sizeof(T));
    v = newv;
  }
}

// Asignment
template <typename T>
NRVec<T> & NRVec<T>::operator=(const NRVec<T> &rhs)
{
  if (this != &rhs)
  {
    if(count)
      if(--(*count) == 0)
      {
        delete[] v;
        delete count;
      }
    v = rhs.v;
    nn = rhs.nn;
    count = rhs.count;
    if(count) (*count)++;
  }
  return *this;
}

// Resize
template <typename T>
void NRVec<T>::resize(const int n)
{
#ifdef DEBUG
  if(n<0) laerror("illegal vector dimension");
#endif
  if(count)
  {
    if(n==0)
	{
	if(--(*count) <= 0) {
                if(v) delete[] (v);
                delete count;
        	}
	count=0;
	nn=0;
	v=0;
	return;
	}
    if(*count > 1) {
      (*count)--;
      count = 0;
      v = 0;
      nn = 0;
    }
  }
  if(!count) {
    count = new int;
    *count = 1;
    nn = n;
    v = new T[nn];
    return;
  }
  // *count = 1 in this branch
  if (n != nn) {
    nn = n;
    delete[] v;
    v = new T[nn];
  }
}


// assignment with a physical (deep) copy
template <typename T>
NRVec<T> & NRVec<T>::operator|=(const NRVec<T> &rhs)
{
	if (this != &rhs) {
#ifdef DEBUG
		if (!rhs.v) laerror("unallocated rhs in NRVec operator |=");
#endif
		if (count)
			if (*count > 1) {
				--(*count);
				nn = 0;
				count = 0;
				v = 0;
			}
		if (nn != rhs.nn) {
			if (v) delete[] (v);
			nn = rhs.nn;
		}
		if(!v) v = new T[nn];
		if(!count) count = new int;
		*count = 1;
		memcpy(v, rhs.v, nn*sizeof(T));
	}
	return *this;
}


template<typename T>
NRVec<complex<T> > complexify(const NRVec<T> &rhs)
{
NRVec<complex<T> > r(rhs.size());
for(int i=0; i<rhs.size(); ++i)  r[i]=rhs[i];
return r;
}

}//namespace

#endif /* _LA_VEC_H_ */
