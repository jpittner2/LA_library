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

#include "smat.h"
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
extern "C" {
extern ssize_t read(int, void *, size_t);
extern ssize_t write(int, const void *, size_t);
}
// TODO
// specialize unary minus




/*
 *  * Templates first, specializations for BLAS next
 *
 */

//raw I/O
template <typename T>
void NRSMat<T>::put(int fd, bool dim, bool transp) const
{
errno=0;
if(dim)
{
if(sizeof(int) != write(fd,&nn,sizeof(int))) laerror("cannot write");
if(sizeof(int) != write(fd,&nn,sizeof(int))) laerror("cannot write");
}
LA_traits<T>::multiput(NN2,fd,v,dim);
}

template <typename T>
void NRSMat<T>::get(int fd, bool dim, bool transp)
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
LA_traits<T>::multiget(NN2,fd,v,dim);
}


// conversion ctor, symmetrize general Mat into SMat
template <typename T>
NRSMat<T>::NRSMat(const NRMat<T> &rhs)
{
nn=rhs.nrows();
#ifdef DEBUG
	if (nn != rhs.ncols()) laerror("attempt to convert non-square Mat to SMat");
#endif
	count = new int;
	*count = 1;
	v = new T[NN2];
	int i, j, k=0;
	for (i=0; i<nn; i++)
		for (j=0; j<=i;j++) v[k++] = (rhs[i][j] + rhs[j][i])/((T)2);
}



// assign to diagonal
template <typename T>
NRSMat<T> & NRSMat<T>::operator=(const T &a)
{
	copyonwrite();
	memset(v,0,NN2*sizeof(T));
	for (int i=0; i<nn; i++) v[i*(i+1)/2+i] = a;
	return *this;
}

//get diagonal
template <typename T>
const T* NRSMat<T>::diagonalof(NRVec<T> &r, const bool divide, bool cache) const
{
#ifdef DEBUG
if(r.size()!=nn) laerror("incompatible vector in diagonalof()");
#endif

r.copyonwrite();

if (divide)
	for (int i=0; i<nn; i++) {T a =v[i*(i+1)/2+i]; if(a!=0.) r[i] /= a;}
else
        for (int i=0; i<nn; i++) r[i] = v[i*(i+1)/2+i];
return divide?NULL:&r[0];
}


// unary minus
template <typename T>
const NRSMat<T> NRSMat<T>::operator-() const
{
	NRSMat<T> result(nn);
	for(int i=0; i<NN2; i++) result.v[i]= -v[i];
	return result;
}

// trace of Smat
template <typename T>
const T NRSMat<T>::trace() const
{
	T tmp = 0;
	for (int i=0; i<nn; i++) tmp += v[i*(i+1)/2+i];
	return tmp;
}

template<>
void NRSMat<double>::randomize(const double &x)
{
for(int i=0; i<NN2; ++i) v[i] = x*(2.*random()/(1.+RAND_MAX) -1.);
}

template<>
void NRSMat<complex<double> >::randomize(const double &x)
{
for(int i=0; i<NN2; ++i) v[i].real() = x*(2.*random()/(1.+RAND_MAX) -1.);
for(int i=0; i<NN2; ++i) v[i].imag() = x*(2.*random()/(1.+RAND_MAX) -1.);
for(int i=0; i<nn; ++i) for(int j=0; j<=i; ++j) if(i==j) v[i*(i+1)/2+j].imag()=0; //hermitean 
}



// write matrix to the file with specific format
template <typename T>
void NRSMat<T>::fprintf(FILE *file, const char *format, const int modulo) const
{
	lawritemat(file, (const T *)(*this) ,nn, nn, format, 2, modulo, 1);
}

// read matrix from the file with specific format
template <typename T>
void NRSMat<T>::fscanf(FILE *f, const char *format)
{
	int n, m;
	if (std::fscanf(f,"%d %d",&n,&m) != 2)
		laerror("cannot read matrix dimensions in SMat::fscanf");
	if (n != m) laerror("different dimensions of SMat");
	resize(n);
	for (int i=0; i<n; i++) 
		for (int j=0; j<n; j++)
			if (std::fscanf(f,format,&((*this)(i,j))) != 1)
				laerror("Smat - cannot read matrix element");
}


/*
 * BLAS specializations for double and complex<double>
 */



// SMat * Mat
//NOTE: dsymm is not appropriate as it works on UNPACKED symmetric matrix
template<>
const NRMat<double> NRSMat<double>::operator*(const NRMat<double> &rhs) const
{
#ifdef DEBUG
	if (nn != rhs.nrows()) laerror("incompatible dimensions in SMat*Mat");
#endif
	NRMat<double> result(nn, rhs.ncols());
	for (int k=0; k<rhs.ncols(); k++)
		cblas_dspmv(CblasRowMajor, CblasLower, nn, 1.0, v, rhs[0]+k, rhs.ncols(),
				0.0, result[0]+k, rhs.ncols());
	return result;
}


template<>
const NRMat< complex<double> >
NRSMat< complex<double> >::operator*(const NRMat< complex<double> > &rhs) const
{
#ifdef DEBUG
	if (nn != rhs.nrows()) laerror("incompatible dimensions in SMat*Mat");
#endif
	NRMat< complex<double> > result(nn, rhs.ncols());
	for (int k=0; k<rhs.ncols(); k++)
		cblas_zhpmv(CblasRowMajor, CblasLower, nn, &CONE, v, rhs[0]+k, rhs.ncols(),
				&CZERO, result[0]+k, rhs.ncols());
	return result;
}



// SMat * SMat
template<>
const NRMat<double> NRSMat<double>::operator*(const NRSMat<double> &rhs) const
{
#ifdef DEBUG
	if (nn != rhs.nn) laerror("incompatible dimensions in SMat*SMat");
#endif
	NRMat<double> result(0.0, nn, nn);
	double *p, *q;

	p = v;
	for (int i=0; i<nn;i++) {
		q = rhs.v;
		for (int k=0; k<=i; k++) {
			cblas_daxpy(k+1, *p++, q, 1, result[i], 1);
			q += k+1;
		}
	}

	p = v;
	for (int i=0; i<nn;i++) {
		q = rhs.v+1;
		for (int j=1; j<nn; j++) {
			result[i][j] += cblas_ddot(i+1<j ? i+1 : j, p, 1, q, 1);
			q += j+1;
		}
		p += i+1;
	}

	p = v; 
	q = rhs.v;
	for (int i=0; i<nn; i++) {
		cblas_dger(CblasRowMajor, i, i+1, 1., p, 1, q, 1, result, nn);
		p += i+1;
		q += i+1;
	}
	
	q = rhs.v+3;
	for (int j=2; j<nn; j++) {
		p = v+1;
		for (int i=1; i<j; i++) {
			cblas_daxpy(i, *++q, p, 1, result[0]+j, nn);
			p += i+1;
		}
		q += 2;
	}

	return result;
}



template<>
const NRMat< complex<double> > 
NRSMat< complex<double> >::operator*(const NRSMat< complex<double> > &rhs) const
{
#ifdef DEBUG
	if (nn != rhs.nn) laerror("incompatible dimensions in SMat*SMat");
#endif
	NRMat< complex<double> > result(0.0, nn, nn);
	NRMat< complex<double> > rhsmat(rhs);
	result = *this * rhsmat;
	return result;
//	laerror("complex SMat*Smat not implemented");
}




// S dot S
template<>
const double NRSMat<double>::dot(const NRSMat<double> &rhs) const
{
#ifdef DEBUG
	if (nn != rhs.nn) laerror("dot of incompatible SMat's");
#endif
	return cblas_ddot(NN2, v, 1, rhs.v, 1);
}



template<>
const complex<double> 
NRSMat< complex<double> >::dot(const NRSMat< complex<double> > &rhs) const
{
#ifdef DEBUG
	if (nn != rhs.nn) laerror("dot of incompatible SMat's");
#endif
	complex<double> dot;
	cblas_zdotc_sub(NN2, (void *)v, 1, (void *)rhs.v, 1, (void *)(&dot));
	return dot;
}


template<>
const double NRSMat<double>::dot(const NRVec<double> &rhs) const
{
#ifdef DEBUG
	if (NN2 != rhs.nn) laerror("dot of incompatible SMat's");
#endif
	return cblas_ddot(NN2, v, 1, rhs.v, 1);
}



template<>
const complex<double> 
NRSMat< complex<double> >::dot(const NRVec< complex<double> > &rhs) const
{
#ifdef DEBUG
	if (NN2 != rhs.nn) laerror("dot of incompatible SMat's");
#endif
	complex<double> dot;
	cblas_zdotc_sub(NN2, (void *)v, 1, (void *)rhs.v, 1, (void *)(&dot));
	return dot;
}


// norm of the matrix
template<>
const double  NRSMat<double>::norm(const double scalar) const
{
	if (!scalar) return cblas_dnrm2(NN2, v, 1);
	double sum = 0;
	int k = 0;
	for (int i=0; i<nn; ++i)
		for (int j=0; j<=i; ++j) {
			register double tmp;
			tmp = v[k++];
			if (i == j) tmp -= scalar;
			sum += tmp*tmp;
		}
	return sqrt(sum);
}



template<>
const double NRSMat< complex<double> >::norm(const complex<double> scalar) const
{
	if (!(scalar.real()) && !(scalar.imag()))
		return cblas_dznrm2(NN2, (void *)v, 1);
	double sum = 0;
	complex<double> tmp;
	int k = 0;
	for (int i=0; i<nn; ++i)
		for (int j=0; j<=i; ++j) {
			tmp = v[k++];
			if (i == j) tmp -= scalar;
			sum += tmp.real()*tmp.real() + tmp.imag()*tmp.imag();
		}
	return sqrt(sum);
}





// axpy: S = S * a
template<>
void NRSMat<double>::axpy(const double alpha, const NRSMat<double> & x)
{
#ifdef DEBUG
	if (nn != x.nn) laerror("axpy of incompatible SMats");
#endif
	copyonwrite();
	cblas_daxpy(NN2, alpha, x.v, 1, v, 1);
}



template<>
void NRSMat< complex<double> >::axpy(const complex<double> alpha,
			const NRSMat< complex<double> > & x)
{
#ifdef DEBUG
	if (nn != x.nn) laerror("axpy of incompatible SMats");
#endif
	copyonwrite();
	cblas_zaxpy(nn, (void *)(&alpha), (void *)x.v, 1, (void *)v, 1);
}

//complex from real
template<>
NRSMat<complex<double> >::NRSMat(const NRSMat<double> &rhs, bool imagpart)
: nn(rhs.nrows()),  v(new complex<double>[rhs.nrows()*(rhs.nrows()+1)/2]), count(new int(1))
{
memset(v,0,nn*(nn+1)/2*sizeof(complex<double>));
cblas_dcopy(nn*(nn+1)/2,&rhs(0,0),1,((double *)v) + (imagpart?1:0),2);
}




//////////////////////////////////////////////////////////////////////////////
////// forced instantization in the corresponding object file
template class NRSMat<double>;
template class NRSMat< complex<double> >;
template class NRSMat<int>;
template class NRSMat<short>;
template class NRSMat<char>;
template class NRSMat<unsigned int>;
template class NRSMat<unsigned long>;

