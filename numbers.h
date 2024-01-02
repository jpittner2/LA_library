/*
    LA: linear algebra C++ interface library
    Copyright (C) 2008-2023 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>

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

//some number-theoretical utilities using naive algorithms suitable for small numbers

#ifndef _NUMBERS_H_
#define _NUMBERS_H_

#include <stdint.h>
#include <list>
#include <tuple>
#include <iostream>



namespace LA {

template<typename N>
N primefactor(const N &x, const N &last=0);

template<typename N>
bool isprime(const N &x) {return x>1 && x==primefactor(x);}

template<typename N>
bool isoddprime(const N &x) {return x>2 && x==primefactor(x);}

template<typename N>
class FACTORIZATION : public  std::list<std::pair<N,N> >
{
};

template<typename N>
FACTORIZATION<N> factorization(const N &x);

template<typename N>
N nextprime(N x);

template <typename N>
std::ostream & operator<<(std::ostream &s, const FACTORIZATION<N>  &x);

template <typename N>
N eulerphi(const FACTORIZATION<N> &f);

template <typename N>
N eulerphi(const N &x) {return eulerphi(factorization(x));}

template <typename N>
N pow(const N &x, N i);

template <typename N>
N powmod(const N &x, N i, const N &m);

template <typename N>
N multmod(N x, N y, const N &m);



}//namespace
#endif
