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

#include <iostream>
#include <stdint.h>
#include <limits>
#include <math.h>

#include "miscfunc.h"
#include "laerror.h"
#include "polynomial.h"

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


//n! in double precision for the cases where the exact factorial() would overflow
#define MAXDOUBLEFACT 170
double double_factorial(const int n)
{
static int ntop=MAXFACT;
static double a[MAXDOUBLEFACT+1];

int j;
if (n < 0) laerror("negative argument of double_factorial");
if (n > MAXDOUBLEFACT) laerror("overflow in double_factorial");
if (n <= MAXFACT) return (double)factorial(n);
if (ntop == MAXFACT) a[MAXFACT]=(double)factorial(MAXFACT);
        while (ntop<n) {
                j=ntop++;
                a[ntop]=a[j]*ntop;
                }
return a[n];
}


#define EPS 1e-15

static double stirfact(double n) /* n should not be less than 9 or 7 at worst */
{
double logz,cor,nn,z;

cor=0.0;
z=n+1.0;
logz= std::log(z);
if(z>=2.0) /*does not have sense for smaller anyway and could make troubles*/
        {
        int i,ii,m;
        m=(int)std::ceil(0.5-std::log(EPS/10000.)/logz/2.0);
        if(m<2)m=2; if(m>16)m=16;
        nn=z*z;
        for(i=m;i>0;i--) {ii=2*i;cor = cor/nn + bernoulli_number(ii)/((double)ii*(ii-1));}
        cor /=z;
        }

/*0.5*ln(2*M_PI)=0.91893853320467275836*/
return((z-0.5)*logz-z+0.918938533204672741780329736406+cor);
/*cor=std::log((1.0/(24.0*n)+1.0)/(12.0*n)+1.0)*/
}



#define LIMITFACT 180
double factorialln(int n)
{
static int top=MAXFACT;
static double a[LIMITFACT+1]={
0.0,
0.0,
0.69314718055994530843,
1.7917594692280549573,
3.1780538303479457518,
4.7874917427820458116,
6.579251212010101213,
8.5251613610654146669,
10.604602902745250859,
12.801827480081469091,
15.10441257307551588,
17.502307845873886549,
19.987214495661884683,
22.552163853123424531,
25.191221182738679829,
27.899271383840890337,
30.671860106080671926,
33.505073450136890756,
36.395445208033052609,
39.339884187199494647,
42.335616460753485057,
45.380138898476907627,
48.471181351835227247,
51.606675567764376922,
54.784729398112318677,
58.003605222980517908,
61.261701761002001376,
64.557538627006337606,
67.889743137181540078,
71.257038967168014665,
74.658236348830158136};
         
 
if(n < 0) laerror("negative argument of factln");
if(n < 2) return(0.0);
if(n <= LIMITFACT) /*supposed LIMITFACT > MAXFACT*/
        if(a[n]) return(a[n]); else
                {
                if(n<=MAXFACT) return(a[n]= std::log(factorial(n)));

                {
                int j;
                if(!a[top]) a[top]= std::log(factorial(top));
                while (top<n) {
                        j=top++;
                        a[top]=a[j]+std::log((double)top);
                        }
                return a[n];
                }
                }
else return(stirfact((double)n));
}



#define MAXBINOM 100
#define ibidxmaxeven(n) ((n-2)*(n-2)/4)
#define ibidx(n,k) (k-2+(n&1?(n-3)*(n-3)/4:(n/2-1)*(n/2-2)))
static unsigned long long ullgcd(unsigned long long a, unsigned long long b)
{
while(b) {unsigned long long t=a%b; a=b; b=t;}
return a;
}

unsigned long long binom(int n, int k)
{
unsigned long long value;
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
/* value = C(n-k+i,i) after step i is exact; divide by the common factor first to avoid overflow of intermediates */
value=1;
for(int i=1; i<=k; ++i)
        {
        unsigned long long m=n-k+i;
        unsigned long long g=ullgcd(m,i);
        m /= g;
        value /= (i/g); /*exact, since value*m is divisible by i and m/g is coprime to i/g*/
        if(value > std::numeric_limits<unsigned long long>::max()/m) laerror("overflow in binom");
        value *= m;
        }
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




/* Euler-Maclaurin summation: zeta(x)= sum_{k=1}^{N-1} k^-x + N^(1-x)/(x-1) + N^-x/2 + sum_{j=1}^m B_2j/(2j)! x(x+1)...(x+2j-2) N^(-x-2j+1) */
/* accurate to about 1e-15 for -1.02<=x<9.1 (the pole term 1/(x-1) is exact) */
static double zeta_euler_maclaurin(double x)
{
const int N=10, m=10;
double s,nx,t;
int j,k;

s=0.0;
for(k=1;k<N;k++) s += std::pow((double)k,-x);
nx=std::pow((double)N,-x);
s += N*nx/(x-1.0) + 0.5*nx;
t=x*nx/N;
for(j=1;j<=m;j++)
	{
	s += bernoulli_number(2*j)/double_factorial(2*j)*t;
	t *= (x+2*j-1)*(x+2*j)/((double)N*N);
	}
return s;
}


double zeta(double x)
{
if(x==std::floor(x) && std::fabs(x)<=(((unsigned int) 1)<<(4*sizeof(int)-1)) )
	{
	int n;
	n=(int)x;
	if(n==1) laerror("singular point in zeta");
	if(n==0) return(-0.5);
	if(n<0)
		{
		if(n&1) return(-bernoulli_number(1-n)/(1-n));
		else return(0.0);
		}
	if(n==2||n==4||n==6||n==8) /*precalculated few bernoulli numbers */
		return(std::fabs(bernoulli_number(n))*pow(2*M_PI,(double)n)/double_factorial(n)/2.0);
	/*otherwise continue like if general real argument*/
	}
if(x<-1.02) return (2.0*pow(2.0*M_PI,x-1.0)*sin(M_PI*x/2.0)*gamma(1.0-x)*zeta(1.0-x));

if(x>=9.1) /* in this region the summation per definition is best and quite exact*/
	{
	double s;
	int i,n;
	n=(int)std::ceil(pow(EPS/1000.0,-1.0/x));
	s=0.0;for(i=n;i>1;i--) s+= pow((double)i,-x);
	return(1.0+s);
	}

return zeta_euler_maclaurin(x);
}
//end zeta




double gamma(double x)
{
bool f;
double p;

if(x>MAXDOUBLEFACT+1) laerror("too big argument in gamma");
f= (std::fabs(x-std::floor(x+0.5))<EPS);
if(x<=0.0 && f) laerror("nonpositive integer argument in gamma");
if(f) return double_factorial(std::floor(x+0.5)-1);

if(x<8.0)
        {
        p=1.0;
        while(x<8.0) {p /= x; x += 1.0;}
        return(p* std::exp(stirfact(x-1.0)));
        }
return( std::exp(stirfact(x-1.0)));
}


static double factln(double nn)
{
if(nn<=LIMITFACT+0.5) return factorialln((int)(nn+.5));
return stirfact(nn);
}


double gammln(double x)
{
bool f;
 
f= (std::fabs(x-std::floor(x+0.5))<EPS);
if(x<=0.0 && f) laerror("nonpositive integer argument in gamma");
if(std::fabs(x-std::floor(x+0.5))<EPS) return factln(x-1.0);
if(x<8.0)
        {
        double p;
        p=0.0;  
        while(x<8.0) {p -= std::log(std::fabs(x)); x += 1.0;}
        return(p+stirfact(x-1.0));
        }
return(stirfact(x-1.0));
}





/* Bernoulli numbers */
#define MAXBERN 50
double bernoulli_number(int n)
{
static double bern[MAXBERN]={1.0,0.166666666666666666,-0.03333333333333333333,0.0238095238095238,-0.033333333333333333,0.07575757575757575757};
static int top=5;
int i; 

if(n<0) laerror("negative index of Bernoulli number");
if(n==1) return(-0.5);
if(n&1) return(0.0);
n /= 2; 
if( n >= MAXBERN) /* use zeta function for it */
        {int m;
        m=2*n;
        return(zeta((double)m)*(n&1?2.0:-2.0)*double_factorial(m)/pow(2.0*M_PI,(double)m));
        }
if(n<=top) return(bern[n]);
        
/* this is very unexact - recurrent formula with binom. coef.
for(i=top+1;i<=n;i++)
        {
        int j,d;
        double s;
        d=2*i+1;
        s=0.0;
        for(j=1;j<i;j++) s += bern[j]*ibinom(d,2*j);
        bern[i]=0.5-(1.0+s)/d;
        }
*/

/* precalculate and save for next use, because the calc. is expensive */
/* could be calculated just one at need, but let's do all of them */
for(i=top+1;i<=n;i++)
        {int m; 
        m=2*i;
        bern[i]=zeta((double)m)*(i&1?2.0:-2.0)*double_factorial(m)/pow(2.0*M_PI,(double)m);
        }
top=n;
return(bern[n]);
}



//there is a closed form s(d,n) = binom(n+d-1,d)
//we could also pre-generate these polynomials for small d
//but that would be inefficient, better is to buffer all values
//we use recurrence s(d,n) = s(d,n-1) + s(d-1,n)
//we cannot also use the above efficient binom() since here we need laerge n while d is small
//
//enlarge storage size if needed
//
#define SIMPLICIAL_MAXD 16
#define SIMPLICIAL_MAXN 1024

unsigned long long simplicial(int d, int n)
{
static unsigned long long stored[SIMPLICIAL_MAXD-2][SIMPLICIAL_MAXN]={{0,1,3}};
static int maxd=2;
static int maxn=2;

if(n<=0) return 0;
if(d==0||n==1) return 1;
if(d==1) return n;

//resize precomputed part as needed
if(n>maxn)
	{
	if(n>=SIMPLICIAL_MAXN) laerror("storage overflow in simplicial, increase MAXN");
	int newdim=2*n; 
	if(newdim<2*maxn) newdim=2*maxn;
	if(newdim>=SIMPLICIAL_MAXN) newdim=SIMPLICIAL_MAXN-1;
	for(int i=maxn+1; i<=newdim;++i) stored[0][i] = i*(i+1)/2;
	for(int dd=1; dd<=maxd-2; ++dd)
		{
		for(int i=maxn+1; i<=newdim;++i) stored[dd][i] =  stored[dd][i-1] + stored[dd-1][i];
		}
	maxn=newdim;
	}

if(d>maxd)
	{
	if(d>=SIMPLICIAL_MAXD) laerror("storage overflow in simplicial, increase MAXD");
	for(int dd=maxd+1; dd<=d; ++dd)
		{
		stored[dd-2][0]=0;
		stored[dd-2][1]=1;
		for(int i=2; i<=maxn; ++i) stored[dd-2][i] = stored[dd-3][i] + stored[dd-2][i-1];
		}
	maxd=d;
	}

return stored[d-2][n];
}


//find largest n such that simplicial(d,n)<=s
int inverse_simplicial(int d,  unsigned long long s)
{
if(s==0) return 0;
if(d==0 || s==1) return 1;
if(d==1) return (int)s;
if(d>=SIMPLICIAL_MAXD) laerror("storage overflow in inverse_simplicial - increase MAXD");

static int maxd=0;
static Polynomial<double> polynomials[SIMPLICIAL_MAXD];
static Polynomial<double> polynomials_der[SIMPLICIAL_MAXD];
for(int i=maxd+1; i<=d; ++i) //generate as needed
	{
	NRVec<double> roots(i);
	for(int j=0; j<i;++j) roots[j] = -j;
	polynomials[i-1] = polyfromroots(roots) * (1./factorial(i));
	polynomials_der[i-1] = polynomials[i-1].derivative(1);
	maxd=i;
	}

//solve polynomial equation by newton
double x=std::pow(factorial(d)*(double)s,1./d); //good init guess
double dx;
do 
	{
	double residual = value(polynomials[d-1],x)-s;
	dx = residual/value(polynomials_der[d-1],x);
	x -= dx;
	}
while(std::fabs(dx)>.5); //so usually 2 iterations are enough
int n=  std::floor(x);
while(simplicial(d,n)>s) --n;
return n;
}

#undef SIMPLICIAL_MAXD
#undef SIMPLICIAL_MAXN


}//namespace
