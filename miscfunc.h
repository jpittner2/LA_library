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

#include <stdint.h>
#include <stddef.h>


namespace LA {


const double c_euler= 0.5772156649015328606065120900824024310422;

extern unsigned long long factorial(const int n); //exact, n<=20
extern double double_factorial(const int n); //n! (not n!!) in double precision, n<=170
extern double factorialln(const int n);
extern unsigned long long binom(int n, int k);
extern unsigned long long longpow(unsigned long long x, int i);
extern double zeta(double x);
extern double gamma(double x);
extern double gammln(double x);
extern double bernoulli_number(int n);
extern unsigned long long simplicial(int d, int n); //simplicial numbers in a precomputed efficient way
extern int inverse_simplicial(int d,  unsigned long long s);


//auxiliary functions for hash() and checksum() of the LA classes
//hash(): 64-bit integer computed from the dimensions and from the exact binary representation of the elements;
//it tells with a negligible chance of a coincidence whether two objects (or an object and what a file was made from)
//are bit by bit the same; it depends on the byte order of the machine like the binary put() and get()
//checksum(): sum of the elements multiplied by fixed pseudorandom weights from [-1,1) which depend on the position of the
//element, on the dimensions and on the seed; it is a linear function of the elements, so that objects which
//differ little have checksums which differ little, and it is to be compared within a threshold (relative to the norm
//of the object) between objects of the same class and dimensions; objects with different elements have close
//checksums only by chance, which can be made less likely by comparing the checksums for several seeds
//NOTE: the functions below must never be changed, since hashes and checksums are meant to be stored in files
//NOTE: the weights come from the own generator hash_mix() and do not depend on RANDDOUBLE etc. (la_random.h), which
//the user can replace by his own random number generators

//starting value of hash()
#define LA_HASH_INIT 0x9E3779B97F4A7C15ULL

//pseudorandom 64-bit number as a fixed function of a counter and a seed (splitmix64); it has no state
static inline uint64_t hash_mix(const uint64_t counter, const uint64_t seed)
{
uint64_t z=seed+(counter+1)*0x9E3779B97F4A7C15ULL;
z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL;
z=(z^(z>>27))*0x94D049BB133111EBULL;
return z^(z>>31);
}

//weight from [-1,1) made of a number from hash_mix()
static inline double checksum_weight(const uint64_t z)
{
return (double)(int64_t)z*(1./9223372036854775808.);
}

//hash of plain data continuing from the value h; the data are taken by words of 8 bytes for speed
extern uint64_t hash_bytes(const void *x, const size_t nbytes, uint64_t h=LA_HASH_INIT);



}//namespace
#endif
