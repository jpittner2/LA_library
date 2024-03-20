/*
    LA: linear algebra C++ interface library
    Copyright (C) 2024 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>

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


#ifndef _MISCFUNC_H
#define _MISCFUNC_H



namespace LA {


const double c_euler= 0.5772156649015328606065120900824024310422;

extern unsigned long long factorial(const int n);
extern double factorialln(const int n);
extern unsigned long long binom(int n, int k);
extern unsigned long long longpow(unsigned long long x, int i);
extern double zeta(double x);
extern double gamma(double x);
extern double gammln(double x);
extern double bernoulli_number(int n);



}//namespace
#endif
