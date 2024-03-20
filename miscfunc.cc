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

#include "miscfunc.h"
#include "laerror.h"
#include <stdio.h>
#include <stdint.h>


namespace LA {


#define MAXFACT 20
unsigned long long factorial(const int n)
{
static int ntop=20;
static  unsigned long long a[MAXFACT+1]={1,1,2,6,24,120,720,5040,
40320,
362880,
3628800,
39916800ULL,
479001600ULL,
6227020800ULL,
87178291200ULL,
1307674368000ULL,
20922789888000ULL,
355687428096000ULL,
6402373705728000ULL,
121645100408832000ULL,
2432902008176640000ULL};

int j;
if (n < 0) laerror("negative argument of factorial");
if (n > MAXFACT) laerror("overflow in factorial");
        while (ntop<n) {
                j=ntop++;
                a[ntop]=a[j]*ntop;
                }
return a[n];
}


#define MAXBINOM 100
#define ibidxmaxeven(n) ((n-2)*(n-2)/4)
#define ibidx(n,k) (k-2+(n&1?(n-3)*(n-3)/4:(n/2-1)*(n/2-2)))
unsigned long long binom(int n, int k)
{
unsigned long long p,value;
int d;
static unsigned long long ibitab[ibidxmaxeven(MAXBINOM)]= /*only nontrivial are stored,
                                            zero initialization by compiler assumed*/
{
6,
10,
15,20,
21,35,
28,56,70
};


if(k>n||k<0) return(0);
if(k>n/2) k=n-k;
if(k==0) return(1);
if(k==1) return(n);
int ind=0;
if(n<=MAXBINOM)
        {
        ind=ibidx(n,k);
        if (ibitab[ind]) return ibitab[ind];
        }
/* nonrecurent method used anyway */
d=n-k;
p=1;
for(;n>d;n--) p *= n;
value=p/factorial(k);
if(n<=MAXBINOM) ibitab[ind]=value;
return value;
}
#undef ibidx
#undef ibidxmaxeven
#undef MAXBINOM


unsigned long long longpow(unsigned long long x, int i)
{
if(i<0) return 0;
unsigned long long y=1;
do
        {
        if(i&1) y *= x;
	i >>= 1;
        if(i) x *= x;
        }
while(i);
return y ;
}

}//namespace
