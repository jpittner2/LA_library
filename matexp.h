#ifndef _MATEXP_H_
#define _MATEXP_H_
//general routine for polynomial of a matrix, tuned to minimize the number
//of matrix-matrix multiplications on cost of additions and memory
// the polynom and exp routines will work on any type, for which traits class
// is defined containing definition of an element type, norm and axpy operation

#include "la_traits.h"

template<class T,class R>
const T polynom2(const T &x, const NRVec<R> &c)
{
int order=c.size()-1;
T z,y;

//trivial reference implementation by horner scheme
if(order==0) {y=x; y=c[0];} //to avoid the problem: we do not know the size of the matrix to contruct a scalar one
else
	{
	int i;
	z=x*c[order];
	for(i=order-1; i>=0; i--)
		{
		if(i<order-1) z=y*x;
		y=z+c[i];
		}
	}

return y;
}


template<class T,class R>
const T polynom(const T &x, const NRVec<R> &c)
{
int n=c.size()-1;
int i,j,k,m=0,t;

if(n<=4) return polynom2(x,c); //here the horner scheme is optimal

//first find m which minimizes the number of multiplications
j=10*n;
for(i=2;i<=n+1;i++)
    {	
    t=i-2+2*(n/i)-(n%i)?0:1;
    if(t<j)
	{
	j=t;
	m=i;
	}
    }

//allocate array for powers up to m
T *xpows = new T[m];
xpows[0]=x;
for(i=1;i<m;i++) xpows[i]=xpows[i-1]*x;


//run the summation loop
T r,s,f;
k= -1;
for(i=0; i<=n/m;i++)
	{
	for(j=0;j<m;j++)
		{
		k++;
		if(k>n) break;
		if(j==0) {if(i==0) s=x; /*just to get the dimensions of the matrix*/ s=c[k]; /*create diagonal matrix*/}
		else  
			LA_traits<T>::axpy(s,xpows[j-1],c[k]); //general  s+=xpows[j-1]*c[k]; but more efficient for matrices
		}

	if(i==0) {r=s; f=xpows[m-1];}
	else
		{
		r+= s*f;
		f=f*xpows[m-1];
		}
	}
 
delete[] xpows;
return r;
}


//for general objects
template<class T>
const T ncommutator ( const T &x, const T &y, int nest=1, const bool right=1)
{
T z;
if(right) {z=x; while(--nest>=0) z=z*y-y*z;}
else {z=y; while(--nest>=0) z=x*z-z*x;}
return z;
}

template<class T>
const T nanticommutator ( const T &x, const T &y, int nest=1, const bool right=1)
{
T z;
if(right) {z=x; while(--nest>=0) z=z*y+y*z;}
else {z=y; while(--nest>=0) z=x*z+z*x;}
return z;
}

//general BCH expansion (can be written more efficiently in a specialization for matrices)
template<class T>
const T BCHexpansion (const T &h, const T &t, const int n, const bool verbose=1)\
{
T result=h;
double factor=1.;
T z=h;
for(int i=1; i<=n; ++i)
	{
	factor/=i;
	z= z*t-t*z;
	if(verbose) cerr << "BCH contribution at order "<<i<<" : "<<z.norm()<<endl;
	result+= z*factor; 
	}
return result;
}


template<class T>
const T ipow( const T &x, int i)
{
if(i<0) laerror("negative exponent in ipow");
if(i==0) {T r=x; r=(typename LA_traits<T>::elementtype)1; return r;}//trick for matrix dimension
if(i==1) return x;
T y,z;
z=x;
while(!(i&1))
	{
	z = z*z;
	i >>= 1;
	}
y=z; 
while((i >>= 1)/*!=0*/)
                {
                z = z*z;
                if(i&1) y = y*z;
                }
return y;
}

inline int nextpow2(const double n)
{
const double log2=log(2.);
if(n<=.75) return 0; //try to keep the taylor expansion short
if(n<=1.) return 1;
return int(ceil(log(n)/log2-log(.75)));
}


template<class T>
NRVec<typename LA_traits<T>::elementtype> exp_aux(const T &x, int &power)
{
//should better be computed by mathematica to have accurate last digits, chebyshev instead, see exp in glibc
static double exptaylor[]={
1.,
1.,
0.5,
0.1666666666666666666666,
0.0416666666666666666666,
0.0083333333333333333333,
0.0013888888888888888888,
0.00019841269841269841253,
2.4801587301587301566e-05,
2.7557319223985892511e-06,
2.7557319223985888276e-07,
2.5052108385441720224e-08,
2.0876756987868100187e-09,
1.6059043836821613341e-10,
1.1470745597729724507e-11,
7.6471637318198164055e-13,
4.7794773323873852534e-14,
2.8114572543455205981e-15,
1.5619206968586225271e-16,
8.2206352466243294955e-18,
4.1103176233121648441e-19,
0.};
double mnorm= LA_traits<T>::norm(x);
power=nextpow2(mnorm);
double scale=exp(-log(2.)*power);


//find how long taylor expansion will be necessary
const double precision=1e-16;
double s,t;
s=mnorm*scale;
int n=0;
t=1.;
do	{
	n++;
	t*=s;
	}
while(t*exptaylor[n]>precision);//taylor 0 will terminate in any case


int i; //adjust the coefficients in order to avoid scaling the argument
NRVec<typename LA_traits<T>::elementtype> taylor2(n+1);
for(i=0,t=1.;i<=n;i++)
	{
	taylor2[i]=exptaylor[i]*t;
	t*=scale;
	}
return taylor2;
}



template<class T>
const T exp(const T &x)
{
int power;

//prepare the polynom of and effectively scale T
NRVec<typename LA_traits<T>::elementtype> taylor2=exp_aux(x,power);

T r=polynom(x,taylor2); //for accuracy summing from the smallest terms up would be better, but this is more efficient for matrices

//power the result back
for(int i=0; i<power; i++) r=r*r;
return r;
}




template<class M, class V>
const V exptimes(const M &mat, V vec) //uses just matrix vector multiplication
{
if(mat.nrows()!=mat.ncols()||(unsigned int) mat.nrows() != (unsigned int)vec.size()) laerror("inappropriate sizes in exptimes");
int power;
//prepare the polynom of and effectively scale the matrix
NRVec<typename LA_traits<M>::elementtype> taylor2=exp_aux(mat,power);

V result(mat.nrows());
for(int i=1; i<=(1<<power); ++i) //unfortunatelly, here we have to repeat it many times, unlike if the matrix is stored explicitly
	{
	if(i>1) vec=result; //apply again to the result of previous application
	//apply polynom of the matrix to the vector iteratively
	V y=vec;
	result=y*taylor2[0];
	for(int j=1; j<taylor2.size(); ++j)
		{
		y=mat*y;
		result.axpy(taylor2[j],y);
		}
	}

return result;
}

#endif
