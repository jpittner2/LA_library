#ifndef _LA_VEC_H_
#define _LA_VEC_H_

#include "laerror.h"
extern "C" {
#include "cblas.h"
}
#include <stdio.h>
#include <complex>
#include <string.h>
#include <iostream>

using namespace std;

#include "la_traits.h"

template <typename T> class NRVec;
template <typename T> class NRSMat;
template <typename T> class NRMat;
template <typename T> class SparseMat;

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

#include "smat.h"
#include "mat.h"


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
	inline NRVec(const NRVec &rhs);
	inline explicit NRVec(const NRSMat<T> & S);
#ifndef MATPTR
	explicit NRVec(const NRMat<T> &rhs);
#endif
	NRVec & operator=(const NRVec &rhs);
	NRVec & operator=(const T &a);  //assign a to every element
	NRVec & operator|=(const NRVec &rhs);
	const bool operator!=(const NRVec &rhs) const {if(nn!=rhs.nn) return 1; return(memcmp(v,rhs.v,nn*sizeof(T)));}
	const bool operator==(const NRVec &rhs) const {return !(*this != rhs);};
	const NRVec operator-() const;
	inline NRVec & operator+=(const NRVec &rhs);
	inline NRVec & operator-=(const NRVec &rhs);
	inline NRVec & operator+=(const T &a);
	inline NRVec & operator-=(const T &a);
	inline NRVec & operator*=(const T &a);
	inline int getcount() const {return count?*count:0;}
	inline const NRVec operator+(const NRVec &rhs) const;
	inline const NRVec operator-(const NRVec &rhs) const;
	inline const NRVec operator+(const T &a) const;
	inline const NRVec operator-(const T &a) const;
	inline const NRVec operator*(const T &a) const;
	inline const T operator*(const NRVec &rhs) const; //scalar product -> ddot
	inline const NRVec operator*(const NRSMat<T> & S) const;
	const NRVec operator*(const NRMat<T> &mat) const;
	const NRMat<T> operator|(const NRVec<T> &rhs) const;
	inline const T sum() const; //sum of its elements
	inline const T dot(const T *a, const int stride=1) const; // ddot with a stride-vector
	inline T & operator[](const int i);
	inline const T & operator[](const int i) const;
	inline int size() const;
	inline operator T*(); //get a pointer to the data
	inline operator const T*() const; //get a pointer to the data
	~NRVec();
	void axpy(const T alpha, const NRVec &x); // this+= a*x
	void axpy(const T alpha, const T *x, const int stride=1); // this+= a*x
	void gemv(const T beta, const NRMat<T> &a, const char trans, 
			const T alpha, const NRVec &x);
	void copyonwrite();
	void resize(const int n);
        void get(int fd, bool dimensions=1);
        void put(int fd, bool dimensions=1) const;
	NRVec & normalize();
	inline const double norm() const;
	inline const T amax() const;
	inline const NRVec unitvector() const;
	void fprintf(FILE *f, const char *format, const int modulo) const;
	void fscanf(FILE *f, const char *format);
//sparse matrix concerning members
	explicit NRVec(const SparseMat<T> &rhs);                // dense from sparse matrix with one of dimensions =1
	const NRVec operator*(const SparseMat<T> &mat) const; //vector*matrix
	inline void simplify() {}; //just for compatibility with sparse ones
	void gemv(const T beta, const SparseMat<T> &a, const char trans, const T alpha, const NRVec &x);
};

template <typename T> ostream & operator<<(ostream &s, const NRVec<T> &x);
template <typename T> istream & operator>>(istream  &s, NRVec<T> &x);

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
inline NRVec<T>::NRVec(const T *a, const int n) : nn(n), v(new T[n]), count(new int)
{
	*count = 1;
	memcpy(v, a, n*sizeof(T));
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
inline NRVec<double> & NRVec<double>::operator+=(const double &a)
{
	copyonwrite();
	cblas_daxpy(nn, 1.0, &a, 0, v, 1);
	return *this;
}

inline NRVec< complex<double> > &
NRVec< complex<double> >::operator+=(const complex<double> &a)
{
	copyonwrite();
	cblas_zaxpy(nn, (void *)(&CONE), (void *)(&a), 0, (void *)v, 1);
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
inline NRVec<double> & NRVec<double>::operator-=(const double &a)
{
	copyonwrite();
	cblas_daxpy(nn, 1.0, &a, 0, v, 1);
	return *this;
}
inline NRVec< complex<double> > &
NRVec< complex<double> >::operator-=(const complex<double> &a)
{
	copyonwrite();
	cblas_zaxpy(nn, (void *)(&CMONE), (void *)(&a), 0, (void *)v, 1);
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
inline NRVec<double> & NRVec<double>::operator+=(const NRVec<double> &rhs)
{
#ifdef DEBUG
	if (nn != rhs.nn) laerror("daxpy of incompatible vectors");
#endif
	copyonwrite();
	cblas_daxpy(nn, 1.0, rhs.v, 1, v, 1);
	return *this;
}
inline NRVec< complex<double> > &
NRVec< complex<double> >::operator+=(const NRVec< complex<double> > &rhs)
{
#ifdef DEBUG
	if (nn != rhs.nn) laerror("daxpy of incompatible vectors");
#endif
	copyonwrite();
	cblas_zaxpy(nn, (void *)(&CONE), rhs.v, 1, v, 1);
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


// x -= x
inline NRVec<double> & NRVec<double>::operator-=(const NRVec<double> &rhs)
{
#ifdef DEBUG
	if (nn != rhs.nn) laerror("daxpy of incompatible vectors");
#endif
	copyonwrite();
	cblas_daxpy(nn, -1.0, rhs.v, 1, v, 1);
	return *this;
}
inline NRVec< complex<double> > &
NRVec< complex<double> >::operator-=(const NRVec< complex<double> > &rhs)
{
#ifdef DEBUG
	if (nn != rhs.nn) laerror("daxpy of incompatible vectors");
#endif
	copyonwrite();
	cblas_zaxpy(nn, (void *)(&CMONE), (void *)rhs.v, 1, (void *)v, 1);
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
inline NRVec<double> & NRVec<double>::operator*=(const double &a)
{
	copyonwrite();
	cblas_dscal(nn, a, v, 1);
	return *this;
}
inline NRVec< complex<double> > &
NRVec< complex<double> >::operator*=(const complex<double> &a)
{
	copyonwrite();
	cblas_zscal(nn, (void *)(&a), (void *)v, 1);
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
inline const double NRVec<double>::operator*(const NRVec<double> &rhs) const
{
#ifdef DEBUG
	if (nn != rhs.nn) laerror("ddot of incompatible vectors");
#endif
	return cblas_ddot(nn, v, 1, rhs.v, 1);
}
inline const complex<double>
NRVec< complex<double> >::operator*(const NRVec< complex<double> > &rhs) const
{
#ifdef DEBUG
	if (nn != rhs.nn) laerror("ddot of incompatible vectors");
#endif
	complex<double> dot;
	cblas_zdotc_sub(nn, (void *)v, 1, (void *)rhs.v, 1, (void *)(&dot));
	return dot;
}

// Vec * SMat = SMat * Vec
template <typename T>
inline const NRVec<T> NRVec<T>::operator*(const NRSMat<T> & S) const
{
	return S * (*this);
}

// Sum of elements
inline const double NRVec<double>::sum() const
{
	return cblas_dasum(nn, v, 1);
}
inline const complex<double>
NRVec< complex<double> >::sum() const
{
	complex<double> sum = CZERO;
	for (int i=0; i<nn; i++) sum += v[i];
	return sum;
}

// Dot product: x * y
inline const double NRVec<double>::dot(const double *y, const int stride) const
{
	return cblas_ddot(nn, y, stride, v, 1);
}
inline const complex<double>
NRVec< complex<double> >::dot(const complex<double> *y, const int stride) const
{
	complex<double> dot;
	cblas_zdotc_sub(nn, y, stride, v, 1, (void *)(&dot));
	return dot;
}

// x[i] returns i-th element
template <typename T>
inline T & NRVec<T>::operator[](const int i)
{
#ifdef DEBUG
	if(*count != 1) laerror("possible lval [] with count > 1");
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
inline const double  NRVec<double>::norm() const
{
	return cblas_dnrm2(nn, v, 1);
}
inline const double NRVec< complex<double> >::norm() const
{
	return cblas_dznrm2(nn, (void *)v, 1);
}

// Max element of the array
inline const double  NRVec<double>::amax() const
{
	return v[cblas_idamax(nn, v, 1)];
}
inline const complex<double> NRVec< complex<double> >::amax() const
{
	return v[cblas_izamax(nn, (void *)v, 1)];
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
#ifdef DEBUG
  if(!count) laerror("probably an assignment to undefined vector");
#endif
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


// assignmet with a physical (deep) copy
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



#endif /* _LA_VEC_H_ */
