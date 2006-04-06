#include <iostream>
#include "vec.h"
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include "qsort.h"
extern "C" {
extern ssize_t read(int, void *, size_t);
extern ssize_t write(int, const void *, size_t);
}


//////////////////////////////////////////////////////////////////////////////
//// forced instantization in the corespoding object file
#define INSTANTIZE(T) \
template ostream & operator<<(ostream &s, const NRVec< T > &x); \
template istream & operator>>(istream  &s, NRVec< T > &x); \
template void NRVec<T>::put(int fd, bool dim, bool transp) const; \
template void NRVec<T>::get(int fd, bool dim, bool transp); \



INSTANTIZE(double)
INSTANTIZE(complex<double>)
INSTANTIZE(int)
INSTANTIZE(unsigned int)
INSTANTIZE(short)
INSTANTIZE(unsigned short)
INSTANTIZE(char)
INSTANTIZE(unsigned char)
INSTANTIZE(unsigned long)

template class NRVec<double>;
template class NRVec<complex<double> >;
template class NRVec<char>;
template class NRVec<short>;
template class NRVec<int>;
template class NRVec<unsigned long>;




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




// formatted I/O
template <typename T>
ostream & operator<<(ostream &s, const NRVec<T> &x)
{
  int i, n;

  n = x.size();
  s << n << endl;
  for(i=0; i<n; i++) s << (typename LA_traits_io<T>::IOtype)x[i] << (i == n-1 ? '\n' : ' ');
  return s;
}

template <typename T>
istream & operator>>(istream &s, NRVec<T> &x)
{
  int i,n;

  s >> n;
  x.resize(n);
  typename LA_traits_io<T>::IOtype tmp;
  for(i=0; i<n; i++) {s >> tmp; x[i]=tmp;}
  return s;
}


//raw I/O
template <typename T>
void NRVec<T>::put(int fd, bool dim, bool transp) const
{
errno=0;
int pad=1; //align at least 8-byte
if(dim)
{
if(sizeof(int) != write(fd,&nn,sizeof(int))) laerror("cannot write");
if(sizeof(int) != write(fd,&pad,sizeof(int))) laerror("cannot write");
}
LA_traits<T>::multiput(nn,fd,v,dim);
}

template <typename T>
void NRVec<T>::get(int fd, bool dim, bool transp)
{
int nn0[2]; //align at least 8-byte
errno=0;
if(dim)
{
if(2*sizeof(int) != read(fd,&nn0,2*sizeof(int))) laerror("cannot read");
resize(nn0[0]);
}
else
copyonwrite();
LA_traits<T>::multiget(nn,fd,v,dim);
}




// formatted print for NRVec
template<typename T>
void NRVec<T>::fprintf(FILE *file, const char *format, const int modulo) const
{
	lawritemat(file, v, 1, nn, format, 1, modulo, 0);
}

// formatted scan for NRVec
template <typename T>
void NRVec<T>::fscanf(FILE *f, const char *format)
{
	int n;

	if(std::fscanf(f, "%d", &n) != 1) laerror("cannot read vector dimension");
	resize(n);
	for (int i=0; i<n; i++)
		if (std::fscanf(f, format, v+i) != 1)
			laerror("cannot read the vector eleemnt");
}

// unary minus
template <typename T>
const NRVec<T> NRVec<T>::operator-() const
{
	NRVec<T> result(nn);
	for (int i=0; i<nn; i++) result.v[i]= -v[i];
	return result;
}


//comparison operators (for lexical order)

template <typename T>
const bool NRVec<T>::operator>(const NRVec &rhs) const
{
int n=nn; if(rhs.nn<n) n=rhs.nn;
for(int i=0; i<n;++i)
	{
	if(LA_traits<T>::bigger(v[i],rhs.v[i])) return true;
	if(LA_traits<T>::smaller(v[i],rhs.v[i])) return false;
	}
return nn>rhs.nn;
}

template <typename T>
const bool NRVec<T>::operator<(const NRVec &rhs) const
{
int n=nn; if(rhs.nn<n) n=rhs.nn;
for(int i=0; i<n;++i)
        {
        if(LA_traits<T>::smaller(v[i],rhs.v[i])) return true;
        if(LA_traits<T>::bigger(v[i],rhs.v[i])) return false;
        }
return nn<rhs.nn;
}



// axpy call for T = double (not strided)
template<>
void NRVec<double>::axpy(const double alpha, const NRVec<double> &x)
{
#ifdef DEBUG
	if (nn != x.nn) laerror("axpy of incompatible vectors");
#endif
	copyonwrite();
	cblas_daxpy(nn, alpha, x.v, 1, v, 1);
}

// axpy call for T = complex<double> (not strided)
template<>
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
template<>
void NRVec<double>::axpy(const double alpha, const double *x, const int stride)
{
	copyonwrite();
	cblas_daxpy(nn, alpha, x, stride, v, 1);
}

// axpy call for T = complex<double> (strided)
template<>
void NRVec< complex<double> >::axpy(const complex<double> alpha, 
			const complex<double> *x, const int stride)
{
	copyonwrite();
	cblas_zaxpy(nn, (void *)(&alpha), (void *)x, stride, v, 1);
}

// unary minus
template<>
const NRVec<double> NRVec<double>::operator-() const
{
	NRVec<double> result(*this);
	result.copyonwrite();
	cblas_dscal(nn, -1.0, result.v, 1);
	return result;
}

template<>
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
template<>
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
template<>
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

//stubs for linkage
template<>
NRVec<int> &  NRVec<int>::normalize() {laerror("normalize() impossible for integer types"); return *this;}
template<>
NRVec<short> &  NRVec<short>::normalize() {laerror("normalize() impossible for integer types"); return *this;}
template<>
NRVec<char> &  NRVec<char>::normalize() {laerror("normalize() impossible for integer types"); return *this;}
template<>
NRVec<unsigned long> &  NRVec<unsigned long>::normalize() {laerror("normalize() impossible for integer types"); return *this;}


template<>
void NRVec<int>::gemv(const int beta,
                const NRSMat<int> &A, const char trans,
                const int alpha, const NRVec &x)
{
laerror("not yet implemented");
}

template<>
void NRVec<short>::gemv(const short beta,
                const NRSMat<short> &A, const char trans,
                const short alpha, const NRVec &x)
{
laerror("not yet implemented");
}

template<>
void NRVec<unsigned long>::gemv(const unsigned long beta,
                const NRSMat<unsigned long> &A, const char trans,
                const unsigned long alpha, const NRVec &x)
{
laerror("not yet implemented");
}

template<>
void NRVec<unsigned char>::gemv(const unsigned char beta,
                const NRSMat<unsigned char> &A, const char trans,
                const unsigned char alpha, const NRVec &x)
{
laerror("not yet implemented");
}


template<>
void NRVec<char>::gemv(const char beta,
                const NRSMat<char> &A, const char trans,
                const char alpha, const NRVec &x)
{
laerror("not yet implemented");
}

template<>
void NRVec<int>::gemv(const int beta,
                const NRMat<int> &A, const char trans,
                const int alpha, const NRVec &x)
{
laerror("not yet implemented");
}

template<>
void NRVec<short>::gemv(const short beta,
                const NRMat<short> &A, const char trans,
                const short alpha, const NRVec &x)
{
laerror("not yet implemented");
}

template<>
void NRVec<unsigned long>::gemv(const unsigned long beta,
                const NRMat<unsigned long> &A, const char trans,
                const unsigned long alpha, const NRVec &x)
{
laerror("not yet implemented");
}

template<>
void NRVec<unsigned char>::gemv(const unsigned char beta,
                const NRMat<unsigned char> &A, const char trans,
                const unsigned char alpha, const NRVec &x)
{
laerror("not yet implemented");
}




template<>
void NRVec<char>::gemv(const char beta,
                const NRMat<char> &A, const char trans,
                const char alpha, const NRVec &x)
{
laerror("not yet implemented");
}

template<>
void NRVec<int>::gemv(const int beta,
                const SparseMat<int> &A, const char trans,
                const int alpha, const NRVec &x, bool s)
{
laerror("not yet implemented");
}

template<>
void NRVec<short>::gemv(const short beta,
                const SparseMat<short> &A, const char trans,
                const short alpha, const NRVec &x, bool s)
{
laerror("not yet implemented");
}


template<>
void NRVec<char>::gemv(const char beta,
                const SparseMat<char> &A, const char trans,
                const char alpha, const NRVec &x, bool s)
{
laerror("not yet implemented");
}

template<>
void NRVec<unsigned long>::gemv(const unsigned long beta,
                const SparseMat<unsigned long> &A, const char trans,
                const unsigned long alpha, const NRVec &x, bool s)
{
laerror("not yet implemented");
}

template<>
void NRVec<unsigned char>::gemv(const unsigned char beta,
                const SparseMat<unsigned char> &A, const char trans,
                const unsigned char alpha, const NRVec &x, bool s)
{
laerror("not yet implemented");
}




// gemv calls 
template<>
void NRVec<double>::gemv(const double beta, const NRMat<double> &A, 
		const char trans, const double alpha, const NRVec &x)
{
#ifdef DEBUG
	if ((trans == 'n'?A.ncols():A.nrows()) != x.size())
		laerror("incompatible sizes in gemv A*x");
#endif
	cblas_dgemv(CblasRowMajor, (trans=='n' ? CblasNoTrans:CblasTrans),
			A.nrows(), A.ncols(), alpha, A, A.ncols(), x.v, 1, beta, v, 1);
}

template<>
void NRVec< complex<double> >::gemv(const complex<double> beta,
		const NRMat< complex<double> > &A, const char trans, 
		const complex<double> alpha, const NRVec &x)
{
#ifdef DEBUG
	if ((trans == 'n'?A.ncols():A.nrows()) != x.size()) 
		laerror("incompatible sizes in gemv A*x");
#endif
	cblas_zgemv(CblasRowMajor, (trans=='n' ? CblasNoTrans:CblasTrans), 
			A.nrows(), A.ncols(), &alpha, A, A.ncols(), 
			x.v, 1, &beta, v, 1);
}


template<>
void NRVec<double>::gemv(const double beta, const NRSMat<double> &A,
                const char trans, const double alpha, const NRVec &x)
{
#ifdef DEBUG
	if (A.ncols()!=x.size()) laerror("incompatible dimension in gemv A*x");
#endif
	NRVec<double> result(nn);
	cblas_dspmv(CblasRowMajor, CblasLower, A.ncols(), alpha, A, x.v, 1, beta, v, 1);
}


template<>
void NRVec< complex<double> >::gemv(const complex<double> beta,
                const NRSMat< complex<double> > &A, const char trans,
                const complex<double> alpha, const NRVec &x)
{
#ifdef DEBUG
	if (A.ncols()!=x.size()) laerror("incompatible dimension in gemv");
#endif
	NRVec< complex<double> > result(nn);
	cblas_zhpmv(CblasRowMajor, CblasLower, A.ncols(), &alpha, A, 
			x.v, 1, &beta, v, 1);
}





// Direc product Mat = Vec | Vec
template<>
const NRMat<double> NRVec<double>::operator|(const NRVec<double> &b) const
{
	NRMat<double> result(0.,nn,b.nn);
	cblas_dger(CblasRowMajor, nn, b.nn, 1., v, 1, b.v, 1, result, b.nn);
	return result;
}
template<>
const NRMat< complex<double> > 
NRVec< complex<double> >::operator|(const NRVec< complex<double> > &b) const
{
	NRMat< complex<double> > result(0.,nn,b.nn);
	cblas_zgerc(CblasRowMajor, nn, b.nn, &CONE, v, 1, b.v, 1, result, b.nn);
	return result;
}


template<typename T>
int NRVec<T>::sort(int direction, int from, int to, int *perm)
{
copyonwrite();
if(to == -1) to=nn-1;
if(direction) return memqsort<1,NRVec<T>,int,int>(*this,perm,from,to);
else return memqsort<0,NRVec<T>,int,int>(*this,perm,from,to);
}
