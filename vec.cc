#include <iostream>
#include "vec.h"

//////////////////////////////////////////////////////////////////////////////
//// forced instantization in the corespoding object file
#define INSTANTIZE(T) \
template ostream & operator<<(ostream &s, const NRVec< T > &x); \
template istream & operator>>(istream  &s, NRVec< T > &x); \

INSTANTIZE(double)
INSTANTIZE(complex<double>)
INSTANTIZE(int)
INSTANTIZE(char)
template NRVec<double>;
template NRVec< complex<double> >;
template NRVec<int>;
template NRVec<char>;


/*
 * Templates first, specializations for BLAS next
 */

// conversion ctor
#ifndef MATPTR
template <typename T>
NRVec<T>::NRVec(const NRMat<T> &rhs)
{
	nn = rhs.nn*rhs.mm;
	v = rhs.v;
	count = rhs.count;
	(*count)++;
}
#endif

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
  if(n<=0) laerror("illegal vector dimension");
#endif
  if(count)
    if(*count > 1) {
      (*count)--;
      count = 0;
      v = 0;
      nn = 0;
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

// ostream << NRVec
template <typename T>
ostream & operator<<(ostream &s, const NRVec<T> &x)
{
  int i, n;

  n = x.size();
  s << n << endl;
  for(i=0; i<n; i++) s << x[i] << (i == n-1 ? '\n' : ' ');
  return s;
}

// istream >> NRVec
template <typename T>
istream & operator>>(istream &s, NRVec<T> &x)
{
  int i,n;

  s >> n;
  x.resize(n);
  for(i=0; i<n; i++) s >> x[i];
  return s;
}

// formatted print for NRVec
template<typename T>
void NRVec<T>::fprintf(FILE *file, const char *format, const int modulo) const
{
	lawritemat(file, v, 1, nn, format, 1, modulo, 0);
}

// formatted scan for NRVec
template <class T>
void NRVec<T>::fscanf(FILE *f, const char *format)
{
	int n;

	if(std::fscanf(f, "%d", &n) != 1) laerror("cannot read vector dimension");
	resize(n);
	for (int i=0; i<n; i++)
		if (std::fscanf(f, format, v+i) != 1)
			laerror("cannot read the vector eleemnt");
}

// assignmet with a physical copy
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

// unary minus
template <typename T>
const NRVec<T> NRVec<T>::operator-() const
{
	NRVec<T> result(nn);
	for (int i=0; i<nn; i++) result.v[i]= -v[i];
	return result;
}

// axpy call for T = double (not strided)
void NRVec<double>::axpy(const double alpha, const NRVec<double> &x)
{
#ifdef DEBUG
	if (nn != x.nn) laerror("axpy of incompatible vectors");
#endif
	copyonwrite();
	cblas_daxpy(nn, alpha, x.v, 1, v, 1);
}

// axpy call for T = complex<double> (not strided)
void NRVec< complex<double> >::axpy(const complex<double> alpha, 
			const NRVec< complex<double> > &x)
{
#ifdef DEBUG
	if (nn != x.nn) laerror("axpy of incompatible vectors");
#endif
	copyonwrite();
	cblas_zaxpy(nn, (void *)(&alpha), (void *)(x.v), 1, (void *)v, 1);
}

// axpy call for T = double (strided)
void NRVec<double>::axpy(const double alpha, const double *x, const int stride)
{
	copyonwrite();
	cblas_daxpy(nn, alpha, x, stride, v, 1);
}

// axpy call for T = complex<double> (strided)
void NRVec< complex<double> >::axpy(const complex<double> alpha, 
			const complex<double> *x, const int stride)
{
	copyonwrite();
	cblas_zaxpy(nn, (void *)(&alpha), (void *)x, stride, v, 1);
}

// unary minus
const NRVec<double> NRVec<double>::operator-() const
{
	NRVec<double> result(*this);
	result.copyonwrite();
	cblas_dscal(nn, -1.0, result.v, 1);
	return result;
}
const NRVec< complex<double> > 
NRVec< complex<double> >::operator-() const
{
	NRVec< complex<double> > result(*this);
	result.copyonwrite();
	cblas_zdscal(nn, -1.0, (void *)(result.v), 1);
	return result;
}

// assignment of scalar to every element
template <typename T>
NRVec<T> & NRVec<T>::operator=(const T &a)
{
	copyonwrite();
	if(a != (T)0)
		for (int i=0; i<nn; i++) v[i] = a;
	else
		memset(v, 0, nn*sizeof(T));
	return *this;
}

// Normalization of NRVec<double>
NRVec<double> & NRVec<double>::normalize()
{
	double tmp;

	tmp = cblas_dnrm2(nn, v, 1);
#ifdef DEBUG
	if(!tmp) laerror("normalization of zero vector");
#endif
	copyonwrite();
	tmp = 1.0/tmp;
	cblas_dscal(nn, tmp, v, 1);
	return *this;
}

// Normalization of NRVec< complex<double> >
NRVec< complex<double> > & NRVec< complex<double> >::normalize()
{
	complex<double> tmp;
	tmp = cblas_dznrm2(nn, (void *)v, 1);
#ifdef DEBUG
	if(!(tmp.real()) && !(tmp.imag())) laerror("normalization of zero vector");
#endif
	copyonwrite();
	tmp = 1.0/tmp;
	cblas_zscal(nn, (void *)(&tmp), (void *)v, 1);
	return *this;
}

//and for these types it does not make sense to normalize but we have them for linkage
NRVec<int> &  NRVec<int>::normalize() {laerror("normalize() impossible for integer types"); return *this;}
NRVec<char> &  NRVec<char>::normalize() {laerror("normalize() impossible for integer types"); return *this;}



// gemv call 
void NRVec<double>::gemv(const double beta, const NRMat<double> &A, 
		const char trans, const double alpha, const NRVec &x)
{
#ifdef DEBUG
	if ((trans == 'n'?A.ncols():A.nrows()) != x.size())
		laerror("incompatible sizes in gemv A*x");
#endif
	cblas_dgemv(CblasRowMajor, (trans=='n' ? CblasNoTrans:CblasTrans),
			A.nrows(), A.ncols(), alpha, A[0], A.ncols(), x.v, 1, beta, v, 1);
}
void NRVec< complex<double> >::gemv(const complex<double> beta,
		const NRMat< complex<double> > &A, const char trans, 
		const complex<double> alpha, const NRVec &x)
{
#ifdef DEBUG
	if ((trans == 'n'?A.ncols():A.nrows()) != x.size()) 
		laerror("incompatible sizes in gemv A*x");
#endif
	cblas_zgemv(CblasRowMajor, (trans=='n' ? CblasNoTrans:CblasTrans), 
			A.nrows(), A.ncols(), (void *)(&alpha), (void *)A[0], A.ncols(), 
			(void *)x.v, 1, (void *)(&beta), (void *)v, 1);
}

// Vec * Mat
const NRVec<double> NRVec<double>::operator*(const NRMat<double> &mat) const
{
#ifdef DEBUG
	if(mat.nrows() != nn) laerror("incompatible sizes in Vec*Mat");
#endif
	int n = mat.ncols();
	NRVec<double> result(n);
	cblas_dgemv(CblasRowMajor, CblasTrans, nn, n, 1.0, mat[0], n, v, 1,
			0.0, result.v, 1);
	return result;
}
const NRVec< complex<double> > 
NRVec< complex<double> >::operator*(const NRMat< complex<double> > &mat) const
{
#ifdef DEBUG
	if(mat.nrows() != nn) laerror("incompatible sizes in Vec*Mat");
#endif
	int n = mat.ncols();
	NRVec< complex<double> > result(n);
	cblas_zgemv(CblasRowMajor, CblasTrans, nn, n, &CONE, mat[0], n, v, 1,
			&CZERO, result.v, 1);
	return result;
}

// Direc product Mat = Vec | Vec
const NRMat<double> NRVec<double>::operator|(const NRVec<double> &b) const
{
	NRMat<double> result(0.,nn,b.nn);
	cblas_dger(CblasRowMajor, nn, b.nn, 1., v, 1, b.v, 1, result, b.nn);
	return result;
}
const NRMat< complex<double> > 
NRVec< complex<double> >::operator|(const NRVec< complex<double> > &b) const
{
	NRMat< complex<double> > result(0.,nn,b.nn);
	cblas_zgerc(CblasRowMajor, nn, b.nn, &CONE, v, 1, b.v, 1, result, b.nn);
	return result;
}


