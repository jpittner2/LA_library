////////////////////////////////////////////////////////////////////////////
//LA traits classes and generally needed includes

#ifndef _LA_TRAITS_INCL
#define _LA_TRAITS_INCL



using namespace std;
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <complex>
#include "laerror.h"

extern "C" {
#include "cblas.h"
}

#ifdef _GLIBCPP_NO_TEMPLATE_EXPORT
# define export 
#endif

//forward declarations
template<typename C> class NRVec;
template<typename C> class NRMat;
template<typename C> class NRSMat;
template<typename C> class SparseMat;

//let's do some simple template metaprogramming and preprocessing
//to keep the thing general and compact

typedef class scalar_false {};
typedef class scalar_true {};

//default is non-scalar
template<typename C>
class isscalar {
	typedef scalar_false scalar_type;
	};

//specializations
#define SCALAR(X) \
class isscalar<X> {typedef scalar_true scalar_type;};

//declare what is scalar
SCALAR(char)
SCALAR(short)
SCALAR(int)
SCALAR(long)
SCALAR(long long)
SCALAR(unsigned char)
SCALAR(unsigned short)
SCALAR(unsigned int)
SCALAR(unsigned long)
SCALAR(unsigned long long)
SCALAR(float)
SCALAR(double)
SCALAR(complex<float>)
SCALAR(complex<double>)
SCALAR(void *)

#undef SCALAR


//now declare the traits for scalars and for composed classes
template<typename C, typename Scalar> struct LA_traits_aux {};

//complex scalars
template<typename C>
struct LA_traits_aux<complex<C>, scalar_true> {
typedef complex<C> elementtype;
typedef complex<C> producttype;
typedef C normtype;
static normtype norm (const  complex<C> &x) {return abs(x);}
static void axpy (complex<C> &s, const complex<C> &x, const complex<C> &c) {s+=x*c;}
static void get(int fd, complex<C> &x, bool dimensions=0) {if(sizeof(complex<C>)!=read(fd,&x,sizeof(complex<C>))) laerror("read error");}
static void put(int fd, const complex<C> &x, bool dimensions=0) {if(sizeof(complex<C>)!=write(fd,&x,sizeof(complex<C>))) laerror("write error");}
static void multiget(unsigned int n,int fd, complex<C> *x, bool dimensions=0){if((ssize_t)(n*sizeof(complex<C>))!=read(fd,x,n*sizeof(complex<C>))) laerror("read error");}
static void multiput(unsigned int n, int fd, const complex<C> *x, bool dimensions=0) {if((ssize_t)(n*sizeof(complex<C>))!=write(fd,x,n*sizeof(complex<C>))) laerror("write error");}

};

//non-complex scalars
template<typename C>
struct LA_traits_aux<C, scalar_true> {
typedef C elementtype;
typedef C producttype;
typedef C normtype;
static normtype norm (const  C &x) {return abs(x);}
static void axpy (C &s, const C &x, const C &c) {s+=x*c;}
static void put(int fd, const C &x, bool dimensions=0) {if(sizeof(C)!=write(fd,&x,sizeof(C))) laerror("write error");}
static void get(int fd, C &x, bool dimensions=0) {if(sizeof(C)!=read(fd,&x,sizeof(C))) laerror("read error");}
static void multiput(unsigned int n,int fd, const C *x, bool dimensions=0){if((ssize_t)(n*sizeof(C))!=write(fd,x,n*sizeof(C))) laerror("write error");}
static void multiget(unsigned int n, int fd, C *x, bool dimensions=0) {if((ssize_t)(n*sizeof(C))!=read(fd,x,n*sizeof(C))) laerror("read error");}
};


//prepare for non-scalar classes

template<typename C>
struct LA_traits; //forward declaration needed for template recursion

#define generate_traits(X) \
template<typename C> \
struct LA_traits_aux<X<C>, scalar_false> { \
typedef C elementtype; \
typedef X<C> producttype; \
typedef typename LA_traits<C>::normtype normtype; \
static normtype norm (const X<C> &x) {return x.norm();} \
static void axpy (X<C>&s, const X<C> &x, const C c) {s.axpy(c,x);} \
static void put(int fd, const C &x, bool dimensions=1) {x.put(fd,dimensions);} \
static void get(int fd, C &x, bool dimensions=1) {x.get(fd,dimensions);} \
static void multiput(unsigned int n,int fd, const C *x, bool dimensions=1) {for(unsigned int i=0; i<n; ++i) x[i].put(fd,dimensions);} \
static void multiget(unsigned int n,int fd, C *x, bool dimensions=1) {for(unsigned int i=0; i<n; ++i) x[i].get(fd,dimensions);} \
};


//non-scalar types defined in this library
generate_traits(NRMat)
generate_traits(NRVec)
generate_traits(SparseMat)

#undef generate_traits

//non-scalar exceptions (smat product type)
template<typename C> 
struct LA_traits_aux<NRSMat<C>, scalar_false> { 
typedef C elementtype; 
typedef NRMat<C> producttype; 
typedef typename LA_traits<C>::normtype normtype; 
static normtype norm (const NRSMat<C> &x) {return x.norm();} 
static void axpy (NRSMat<C>&s, const NRSMat<C> &x, const C c) {s.axpy(c,x);} 
static void put(int fd, const C &x, bool dimensions=1) {x.put(fd,dimensions);} 
static void get(int fd, C &x, bool dimensions=1) {x.get(fd,dimensions);} 
static void multiput(unsigned int n,int fd, const C *x, bool dimensions=1) {for(unsigned int i=0; i<n; ++i) x[i].put(fd,dimensions);} \
static void multiget(unsigned int n,int fd, C *x, bool dimensions=1) {for(unsigned int i=0; i<n; ++i) x[i].get(fd,dimensions);} \
};


//the final traits class
template<typename C>
struct LA_traits : LA_traits_aux<C, typename isscalar<C>::scalar_type> {};

#endif
