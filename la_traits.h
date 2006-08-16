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

#ifdef NONCBLAS
#include "noncblas.h"
#else
extern "C" {
#include "cblas.h"
}
#endif


//forward declarations
template<typename C> class NRVec;
template<typename C> class NRMat;
template<typename C> class NRSMat;
template<typename C> class SparseMat;

//for general sortable classes
template<typename T, typename I, int type> struct LA_sort_traits;

template<typename T, typename I>
struct LA_sort_traits<T,I,0>
        {
        static inline bool compare(T object, I i, I j) {return object.bigger(i,j);};
        };

template<typename T, typename I>
struct LA_sort_traits<T,I,1>
        {
        static inline bool compare(T object, I i, I j) {return object.smaller(i,j);};
        };


//we will need to treat char and unsigned char as numbers in << and >> I/O operators
template<typename C>
struct LA_traits_io
	{
	typedef C IOtype;
	};

template<>
struct LA_traits_io<char>
	{
	typedef int IOtype;
	};

template<>
struct LA_traits_io<unsigned char>
        {
        typedef unsigned int IOtype;
        };


//let's do some simple template metaprogramming and preprocessing
//to keep the thing general and compact

typedef class scalar_false {};
typedef class scalar_true {};

//default is non-scalar
template<typename C>
class isscalar { public: typedef scalar_false scalar_type;};

//specializations
#define SCALAR(X) \
template<>\
class isscalar<X> {public: typedef scalar_true scalar_type;};

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
//NOTE! methods in traits classes have to be declared static,
//since the class itself is never instantiated.
//for performance, it can be also inlined at the same time
template<typename C, typename Scalar> struct LA_traits_aux {};


//TRAITS SPECIALIZATIONS

//complex scalars
template<typename C>
struct LA_traits_aux<complex<C>, scalar_true> {
typedef complex<C> elementtype;
typedef complex<C> producttype;
typedef C normtype;
static inline bool gencmp(const complex<C> *x, const complex<C> *y, int n) {return memcmp(x,y,n*sizeof(complex<C>));}
static bool bigger(const  complex<C> &x, const complex<C> &y) {laerror("complex comparison undefined"); return false;}
static bool smaller(const  complex<C> &x, const complex<C> &y) {laerror("complex comparison undefined"); return false;}
static inline normtype norm (const  complex<C> &x) {return abs(x);}
static inline void axpy (complex<C> &s, const complex<C> &x, const complex<C> &c) {s+=x*c;}
static inline void get(int fd, complex<C> &x, bool dimensions=0, bool transp=0) {if(sizeof(complex<C>)!=read(fd,&x,sizeof(complex<C>))) laerror("read error");}
static inline void put(int fd, const complex<C> &x, bool dimensions=0, bool transp=0) {if(sizeof(complex<C>)!=write(fd,&x,sizeof(complex<C>))) laerror("write error");}
static void multiget(unsigned int n,int fd, complex<C> *x, bool dimensions=0){if((ssize_t)(n*sizeof(complex<C>))!=read(fd,x,n*sizeof(complex<C>))) laerror("read error");}
static void multiput(unsigned int n, int fd, const complex<C> *x, bool dimensions=0) {if((ssize_t)(n*sizeof(complex<C>))!=write(fd,x,n*sizeof(complex<C>))) laerror("write error");}
static void copy(complex<C> *dest, complex<C> *src, unsigned int n) {memcpy(dest,src,n*sizeof(complex<C>));}
static void clear(complex<C> *dest, unsigned int n) {memset(dest,0,n*sizeof(complex<C>));}
};

//non-complex scalars
template<typename C>
struct LA_traits_aux<C, scalar_true> {
typedef C elementtype;
typedef C producttype;
typedef C normtype;
static inline bool gencmp(const C *x, const C *y, int n) {return memcmp(x,y,n*sizeof(C));}
static inline bool bigger(const  C &x, const C &y) {return x>y;}
static inline bool smaller(const  C &x, const C &y) {return x<y;}
static inline normtype norm (const  C &x) {return abs(x);}
static inline void axpy (C &s, const C &x, const C &c) {s+=x*c;}
static inline void put(int fd, const C &x, bool dimensions=0, bool transp=0) {if(sizeof(C)!=write(fd,&x,sizeof(C))) laerror("write error");}
static inline void get(int fd, C &x, bool dimensions=0, bool transp=0) {if(sizeof(C)!=read(fd,&x,sizeof(C))) laerror("read error");}
static void multiput(unsigned int n,int fd, const C *x, bool dimensions=0){if((ssize_t)(n*sizeof(C))!=write(fd,x,n*sizeof(C))) laerror("write error");}
static void multiget(unsigned int n, int fd, C *x, bool dimensions=0) {if((ssize_t)(n*sizeof(C))!=read(fd,x,n*sizeof(C))) laerror("read error");}
static void copy(C *dest, C *src, unsigned int n) {memcpy(dest,src,n*sizeof(C));}
static void clear(C *dest, unsigned int n) {memset(dest,0,n*sizeof(C));}
};


//non-scalars except smat

template<typename C>
struct LA_traits; //forward declaration needed for template recursion

#define generate_traits(X) \
template<typename C> \
struct LA_traits_aux<X<C>, scalar_false> { \
typedef C elementtype; \
typedef X<C> producttype; \
typedef typename LA_traits<C>::normtype normtype; \
static bool gencmp(const C *x, const C *y, int n) {for(int i=0; i<n; ++i) if(x[i]!=y[i]) return true; return false;} \
static inline bool bigger(const  C &x, const C &y) {return x>y;} \
static inline bool smaller(const  C &x, const C &y) {return x<y;} \
static inline normtype norm (const X<C> &x) {return x.norm();} \
static inline void axpy (X<C>&s, const X<C> &x, const C c) {s.axpy(c,x);} \
static void put(int fd, const C &x, bool dimensions=1, bool transp=0) {x.put(fd,dimensions,transp);} \
static void get(int fd, C &x, bool dimensions=1, bool transp=0) {x.get(fd,dimensions,transp);} \
static void multiput(unsigned int n,int fd, const C *x, bool dimensions=1) {for(unsigned int i=0; i<n; ++i) x[i].put(fd,dimensions);} \
static void multiget(unsigned int n,int fd, C *x, bool dimensions=1) {for(unsigned int i=0; i<n; ++i) x[i].get(fd,dimensions);} \
static void copy(C *dest, C *src, unsigned int n) {for(unsigned int i=0; i<n; ++i) dest[i]=src[i];} \
static void clear(C *dest, unsigned int n) {for(unsigned int i=0; i<n; ++i) dest[i].clear();}\
};


//non-scalar types defined in this library
generate_traits(NRMat)
generate_traits(NRVec)
generate_traits(SparseMat)

#undef generate_traits

//smat
template<typename C> 
struct LA_traits_aux<NRSMat<C>, scalar_false> { 
typedef C elementtype; 
typedef NRMat<C> producttype; 
typedef typename LA_traits<C>::normtype normtype; 
static bool gencmp(const C *x, const C *y, int n) {for(int i=0; i<n; ++i) if(x[i]!=y[i]) return true; return false;}
static inline bool bigger(const  C &x, const C &y) {return x>y;}
static inline bool smaller(const  C &x, const C &y) {return x<y;}
static inline normtype norm (const NRSMat<C> &x) {return x.norm();} 
static inline void axpy (NRSMat<C>&s, const NRSMat<C> &x, const C c) {s.axpy(c,x);} 
static void put(int fd, const C &x, bool dimensions=1, bool transp=0) {x.put(fd,dimensions);} 
static void get(int fd, C &x, bool dimensions=1, bool transp=0) {x.get(fd,dimensions);} 
static void multiput(unsigned int n,int fd, const C *x, bool dimensions=1) {for(unsigned int i=0; i<n; ++i) x[i].put(fd,dimensions);} 
static void multiget(unsigned int n,int fd, C *x, bool dimensions=1) {for(unsigned int i=0; i<n; ++i) x[i].get(fd,dimensions);} 
static void copy(C *dest, C *src, unsigned int n) {for(unsigned int i=0; i<n; ++i) dest[i]=src[i];} 
static void clear(C *dest, unsigned int n) {for(unsigned int i=0; i<n; ++i) dest[i].clear();}
};


//the final traits class
template<typename C>
struct LA_traits : LA_traits_aux<C, typename isscalar<C>::scalar_type> {};

#endif
