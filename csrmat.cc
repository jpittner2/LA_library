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

#include <string>
#include <cmath>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include "csrmat.h"

namespace LA {




/*
 Commented out by Roman for ICC

#define INSTANTIZE(T) \
template void CSRMat<T>::gemm(const T beta, const CSRMat &a, const char transa, const CSRMat &b, const char transb, const T alpha); \
template CSRMat<T> & CSRMat<T>::operator*=(const T &a); \
template void CSRMat<T>::gemv(const T beta, NRVec<T> &r, const char trans, const T alpha, const NRVec<T> &x) const; \
template void CSRMat<T>::axpy(const T alpha, const CSRMat &x, const bool transp); \
template CSRMat<T> & CSRMat<T>::operator=(const T &a); \
template CSRMat<T> & CSRMat<T>::operator+=(const T &a); \
template CSRMat<T> & CSRMat<T>::operator-=(const T &a); \
template LA_traits<T>::normtype CSRMat<T>::norm(const T scalar) const; \
template const T* CSRMat<T>::diagonalof(NRVec<T> &r, const bool divide, bool cache) const; \
template void CSRMat<T>::get(int fd, bool dimen, bool transp); \
template void CSRMat<T>::put(int fd, bool dimen, bool transp) const; \


INSTANTIZE(double)
INSTANTIZE(complex<double>) 
*/

//// forced instantization of functions in the header in the corresponding object file
//@@@@@template class CSRMat<double>;
//@@@@template class CSRMat<complex<double> >;
//@@@@ unfinished class commented out



}//namespace
