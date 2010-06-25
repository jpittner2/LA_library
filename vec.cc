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


#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include "vec.h"
#include "qsort.h"
extern "C" {
extern ssize_t read(int, void *, size_t);
extern ssize_t write(int, const void *, size_t);
}

namespace LA {


//////////////////////////////////////////////////////////////////////////////
//// forced instantization in the corespoding object file
#define INSTANTIZE(T) \
template void NRVec<T>::put(int fd, bool dim, bool transp) const; \
template void NRVec<T>::get(int fd, bool dim, bool transp); \



INSTANTIZE(double)
INSTANTIZE(complex<double>)
INSTANTIZE(char)
INSTANTIZE(short)
INSTANTIZE(int)
INSTANTIZE(long)
INSTANTIZE(long long)
INSTANTIZE(unsigned char)
INSTANTIZE(unsigned short)
INSTANTIZE(unsigned int)
INSTANTIZE(unsigned long)
INSTANTIZE(unsigned long long)




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





//raw I/O
template <typename T>
void NRVec<T>::put(int fd, bool dim, bool transp) const
{
#ifdef CUDALA
if(location!=cpu)
        {
        NRVec<T> tmp= *this;
        tmp.moveto(cpu);
        tmp.put(fd,dim,transp);
        return;
        }
#endif

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
#ifdef CUDALA
if(location!=cpu)
        {
        NRVec<T> tmp;
        tmp.moveto(cpu);
        tmp.get(fd,dim,transp);
        tmp.moveto(location);
        *this = tmp;
        return;
        }
#endif
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

	if(::fscanf(f, "%d", &n) != 1) laerror("cannot read vector dimension");
	resize(n);
	for (int i=0; i<n; i++)
		if (::fscanf(f, format, v+i) != 1)
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


template<>
void NRVec<double>::randomize(const double &x)
{
for(int i=0; i<nn; ++i) v[i] = x*(2.*random()/(1.+RAND_MAX) -1.);
}

template<>
void NRVec<complex<double> >::randomize(const double &x)
{
for(int i=0; i<nn; ++i) v[i] = complex<double> (x*(2.*random()/(1.+RAND_MAX) -1.),x*(2.*random()/(1.+RAND_MAX) -1.));
}



//complex from real constructor
template<>
NRVec<complex<double> >::NRVec(const NRVec<double> &rhs, bool imagpart)
: nn(rhs.size()),  v(new complex<double>[rhs.size()]), count(new int(1))
{
memset(v,0,nn*sizeof(complex<double>));
cblas_dcopy(nn,&rhs[0],1,((double *)v) + (imagpart?1:0),2);
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
	cblas_zaxpy(nn, &alpha, x.v, 1, v, 1);
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
	cblas_zaxpy(nn, &alpha, x, stride, v, 1);
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
	cblas_zdscal(nn, -1.0, result.v, 1);
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
	tmp = cblas_dznrm2(nn, v, 1);
#ifdef DEBUG
	if(!(tmp.real()) && !(tmp.imag())) laerror("normalization of zero vector");
#endif
	copyonwrite();
	tmp = 1.0/tmp;
	cblas_zscal(nn, &tmp, v, 1);
	return *this;
}

//stubs for linkage

#define INSTANTIZE_DUMMY(T) \
template<> void NRVec<T>::gemv(const T beta, const NRMat<T> &a, const char trans,  const T alpha, const NRVec<T> &x) { laerror("gemv on unsupported types"); } \
template<> void NRVec<T>::gemv(const T beta, const NRSMat<T> &a, const char trans,  const T alpha, const NRVec<T> &x) { laerror("gemv on unsupported types"); } \
template<> void NRVec<T>::gemv(const T beta, const SparseMat<T> &a, const char trans,  const T alpha, const NRVec<T> &x, bool s) { laerror("gemv on unsupported types"); } \
template<> void NRVec<T>::gemv(const LA_traits_complex<T>::Component_type beta, const  LA_traits_complex<T>::NRMat_Noncomplex_type  &a, const char trans,  const  LA_traits_complex<T>::Component_type alpha, const NRVec<T> &x) { laerror("gemv on unsupported types"); } \
template<> void NRVec<T>::gemv(const  LA_traits_complex<T>::Component_type beta, const  LA_traits_complex<T>::NRSMat_Noncomplex_type  &a, const char trans,  const  LA_traits_complex<T>::Component_type alpha, const NRVec<T> &x) { laerror("gemv on unsupported types"); } \
template<> NRVec<T> & NRVec<T>::normalize() {laerror("normalize() impossible for integer types"); return *this;} \
template<> const NRMat<T> NRVec<T>::otimes(const NRVec<T> &b,const bool conj, const T &scale) const {laerror("otimes presently implemented only for double and complex double"); return NRMat<T> ();}




// gemv calls 
template<>
void NRVec<double>::gemv(const double beta, const NRMat<double> &A, 
		const char trans, const double alpha, const NRVec &x)
{
#ifdef DEBUG
	if ((trans == 'n'?A.ncols():A.nrows()) != x.size())
		laerror("incompatible sizes in gemv A*x");
#endif
	SAME_LOC3(*this,x,A);
	copyonwrite();
#ifdef CUDALA
	if(location==cpu)
#endif
		cblas_dgemv(CblasRowMajor, (trans=='n' ? CblasNoTrans:CblasTrans), A.nrows(), A.ncols(), alpha, A, A.ncols(), x.v, 1, beta, v, 1);
#ifdef CUDALA
        else
		cublasDgemv((trans=='n' ?'T':'N'),A.ncols(), A.nrows(),alpha, A, A.ncols(), x.v, 1, beta, v, 1);
#endif
}


template<>
void NRVec<complex<double> >::gemv(const double beta, const NRMat<double> &A,
                const char trans, const double alpha, const NRVec<complex<double> > &x)
{
#ifdef DEBUG
        if ((trans == 'n'?A.ncols():A.nrows()) != x.size())
                laerror("incompatible sizes in gemv A*x");
#endif
        copyonwrite();
        cblas_dgemv(CblasRowMajor, (trans=='n' ? CblasNoTrans:CblasTrans), A.nrows(), A.ncols(), alpha, A, A.ncols(), (double *)x.v, 2, beta, (double *)v, 2);
        cblas_dgemv(CblasRowMajor, (trans=='n' ? CblasNoTrans:CblasTrans), A.nrows(), A.ncols(), alpha, A, A.ncols(), ((double *)x.v) + 1, 2, beta, ((double *)v)+1, 2);
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
	copyonwrite();
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
SAME_LOC3(*this,A,x);
	copyonwrite();
#ifdef CUDALA
if(location==cpu)
#endif
	cblas_dspmv(CblasRowMajor, CblasLower, A.ncols(), alpha, A, x.v, 1, beta, v, 1);
#ifdef CUDALA
else
	cublasDspmv('U',A.ncols(), alpha, A, x.v, 1, beta, v, 1);
#endif
}

template<>
void NRVec<complex<double> >::gemv(const double beta, const NRSMat<double> &A,
                const char trans, const double alpha, const NRVec<complex<double> > &x)
{
#ifdef DEBUG
	if (A.ncols()!=x.size()) laerror("incompatible dimension in gemv A*x");
#endif
	copyonwrite();
	cblas_dspmv(CblasRowMajor, CblasLower, A.ncols(), alpha, A, (double *)x.v, 2, beta, (double *)v, 2);
	cblas_dspmv(CblasRowMajor, CblasLower, A.ncols(), alpha, A, ((double *)x.v)+1, 2, beta, ((double *)v)+1, 2);
}



template<>
void NRVec< complex<double> >::gemv(const complex<double> beta,
                const NRSMat< complex<double> > &A, const char trans,
                const complex<double> alpha, const NRVec &x)
{
#ifdef DEBUG
	if (A.ncols()!=x.size()) laerror("incompatible dimension in gemv");
#endif
	copyonwrite();
	cblas_zhpmv(CblasRowMajor, CblasLower, A.ncols(), &alpha, A, 
			x.v, 1, &beta, v, 1);
}





// Direct product Mat = Vec | Vec
template<>
const NRMat<double> NRVec<double>::otimes(const NRVec<double> &b,const bool conj, const double &scale) const
{
	NRMat<double> result(0.,nn,b.nn);
	cblas_dger(CblasRowMajor, nn, b.nn, scale, v, 1, b.v, 1, result, b.nn);
	return result;
}

template<>
const NRMat< complex<double> > 
NRVec<complex<double> >::otimes(const NRVec< complex<double> > &b, const bool conj, const  complex<double> &scale) const
{
	NRMat< complex<double> > result(0.,nn,b.nn);
	if(conj) cblas_zgerc(CblasRowMajor, nn, b.nn, &scale, v, 1, b.v, 1, result, b.nn);
	else cblas_zgeru(CblasRowMajor, nn, b.nn, &scale, v, 1, b.v, 1, result, b.nn);
	return result;
}


template<typename T>
int NRVec<T>::sort(int direction, int from, int to, int *perm)
{
NOT_GPU(*this);
copyonwrite();
if(to == -1) to=nn-1;
if(direction) return memqsort<1,NRVec<T>,int,int>(*this,perm,from,to);
else return memqsort<0,NRVec<T>,int,int>(*this,perm,from,to);
}






//////////////////////////////////////////////////////////////////////////////
//// forced instantization in the corespoding object file


template class NRVec<double>;
template class NRVec<complex<double> >;
template class NRVec<char>;
template class NRVec<short>;
template class NRVec<int>;
template class NRVec<long>;
template class NRVec<long long>;
template class NRVec<unsigned char>;
template class NRVec<unsigned short>;
template class NRVec<unsigned int>;
template class NRVec<unsigned long>;
template class NRVec<unsigned long long>;


INSTANTIZE_DUMMY(char)
INSTANTIZE_DUMMY(short)
INSTANTIZE_DUMMY(int)
INSTANTIZE_DUMMY(long)
INSTANTIZE_DUMMY(long long)
INSTANTIZE_DUMMY(unsigned char)
INSTANTIZE_DUMMY(unsigned short)
INSTANTIZE_DUMMY(unsigned int)
INSTANTIZE_DUMMY(unsigned long)
INSTANTIZE_DUMMY(unsigned long long)
INSTANTIZE_DUMMY(complex<char>)
INSTANTIZE_DUMMY(complex<short>)
INSTANTIZE_DUMMY(complex<int>)
INSTANTIZE_DUMMY(complex<long>)
INSTANTIZE_DUMMY(complex<long long>)
INSTANTIZE_DUMMY(complex<unsigned char>)
INSTANTIZE_DUMMY(complex<unsigned short>)
INSTANTIZE_DUMMY(complex<unsigned int>)
INSTANTIZE_DUMMY(complex<unsigned long>)
INSTANTIZE_DUMMY(complex<unsigned long long>)

INSTANTIZE_DUMMY(complex<complex<double> >)
INSTANTIZE_DUMMY(complex<complex<float> >)

}//namespace
