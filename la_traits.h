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
//
//for autotools
//
//#include "config.h" //this would force the user of the library to have config.h

////////////////////////////////////////////////////////////////////////////
//LA traits classes and generally needed includes

#ifndef _LA_TRAITS_INCL
#define _LA_TRAITS_INCL




#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <limits>
#include <complex>
#include <unistd.h>


//using namespace std;
#define complex std::complex

#include "laerror.h"

#include "cuda_la.h"

#ifdef NONCBLAS
#include "noncblas.h"
#else
extern "C" {
#include "cblas.h"
}
#endif

#ifdef NONCLAPACK
#include "noncblas.h"
#else
extern "C" {
#include "atlas/clapack.h"
}
#endif

namespace LA {

extern bool _LA_count_check;

//forward declarations
template<typename C> class NRVec;
template<typename C> class NRMat;
template<typename C> class NRMat_from1;
template<typename C> class NRSMat;
template<typename C> class NRSMat_from1;
template<typename C> class SparseMat;
template<typename C> class SparseSMat;
template<typename C> class CSRMat;


typedef class {} Dummy_type;
typedef class {} Dummy_type2;

//for components of complex numbers
//
template<typename C>
struct LA_traits_complex
        {
        typedef Dummy_type Component_type;
	typedef Dummy_type NRVec_Noncomplex_type;
	typedef Dummy_type NRMat_Noncomplex_type;
	typedef Dummy_type2 NRSMat_Noncomplex_type;
        };

#define SPECIALIZE_COMPLEX(T) \
template<> \
struct LA_traits_complex<complex<T> > \
        { \
        typedef T Component_type; \
	typedef NRVec<T> NRVec_Noncomplex_type; \
	typedef NRMat<T> NRMat_Noncomplex_type; \
	typedef NRSMat<T> NRSMat_Noncomplex_type; \
        };


SPECIALIZE_COMPLEX(double)
SPECIALIZE_COMPLEX(complex<double>)
SPECIALIZE_COMPLEX(float)
SPECIALIZE_COMPLEX(complex<float>)
SPECIALIZE_COMPLEX(char)
SPECIALIZE_COMPLEX(unsigned char)
SPECIALIZE_COMPLEX(short)
SPECIALIZE_COMPLEX(unsigned short)
SPECIALIZE_COMPLEX(int)
SPECIALIZE_COMPLEX(unsigned int)
SPECIALIZE_COMPLEX(long)
SPECIALIZE_COMPLEX(unsigned long)
SPECIALIZE_COMPLEX(long long)
SPECIALIZE_COMPLEX(unsigned long long)


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

class scalar_false {};
class scalar_true {};

//default is non-scalar
template<typename C>
class isscalar { public: typedef scalar_false scalar_type;};

//specializations
#define SCALAR(X) \
template<>\
class isscalar<X> {public: typedef scalar_true scalar_type;};\
template<>\
class isscalar<complex<X> > {public: typedef scalar_true scalar_type;};\
template<>\
class isscalar<complex<complex<X> > > {public: typedef scalar_true scalar_type;};\


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
SCALAR(void *)

#undef SCALAR


//declare this generically as traits for any unknown class
template<typename C, typename Scalar> struct LA_traits_aux
	{
	typedef Dummy_type normtype;
	};


//TRAITS SPECIALIZATIONS
////now declare the traits for scalars and for composed classes
////NOTE! methods in traits classes have to be declared static,
////since the class itself is never instantiated.
////for performance, it can be also inlined at the same time
//

//complex scalars
template<typename C>
struct LA_traits_aux<complex<C>, scalar_true> {
typedef complex<C> elementtype;
typedef complex<C> producttype;
typedef C normtype;
typedef C realtype;
typedef complex<C> complextype;
static inline C sqrabs(const complex<C> x) { return x.real()*x.real()+x.imag()*x.imag();}
static inline bool gencmp(const complex<C> *x, const complex<C> *y, size_t n) {return memcmp(x,y,n*sizeof(complex<C>));}
static bool bigger(const  complex<C> &x, const complex<C> &y) {laerror("complex comparison undefined"); return false;}
static bool smaller(const  complex<C> &x, const complex<C> &y) {laerror("complex comparison undefined"); return false;}
static inline normtype norm (const  complex<C> &x) {return std::abs(x);}
static inline void axpy (complex<C> &s, const complex<C> &x, const complex<C> &c) {s+=x*c;}
static inline void get(int fd, complex<C> &x, bool dimensions=0, bool transp=0) {if(sizeof(complex<C>)!=read(fd,&x,sizeof(complex<C>))) laerror("read error");}
static inline void put(int fd, const complex<C> &x, bool dimensions=0, bool transp=0) {if(sizeof(complex<C>)!=write(fd,&x,sizeof(complex<C>))) laerror("write error");}
static void multiget(size_t n,int fd, complex<C> *x, bool dimensions=0)
	{
	size_t total=0;
	size_t system_limit = (1L<<30)/sizeof(complex<C>); //do not expect too much from the system and read at most 1GB at once
	ssize_t r;
	size_t nn;
	do{
		r=read(fd,x+total,nn=(n-total > system_limit ? system_limit : n-total)*sizeof(complex<C>)); 
		if(r<0 || r==0 && nn!=0 ) {std::cout<<"read returned "<<r<<" perror "<<strerror(errno) <<std::endl; laerror("read error");}
		else total += r/sizeof(complex<C>);
		if(r%sizeof(complex<C>)) laerror("read error 2");
	  }
	while(total < n);
	}
static void multiput(size_t n, int fd, const complex<C> *x, bool dimensions=0) 
	{
	size_t total=0;
	size_t system_limit = (1L<<30)/sizeof(complex<C>); //do not expect too much from the system and write at most 1GB at once
	ssize_t r;
	size_t nn;
	do{
		r=write(fd,x+total,nn=(n-total > system_limit ? system_limit : n-total)*sizeof(complex<C>)); 
		if(r<0 || r==0 && nn!=0 ) {std::cout<<"write returned "<<r<<"  perror "<<strerror(errno) <<std::endl; laerror("write error");}
		else total += r/sizeof(complex<C>);
		if(r%sizeof(complex<C>)) laerror("write error 2");
	  }
	while(total < n);
	}
static void copy(complex<C> *dest, complex<C> *src, size_t n) {memcpy(dest,src,n*sizeof(complex<C>));}
static void clear(complex<C> *dest, size_t n) {memset(dest,0,n*sizeof(complex<C>));}
static void copyonwrite(complex<C> &x) {};
static void clearme(complex<C> &x) {x=0;};
static void deallocate(complex<C> &x) {};
static inline complex<C> conjugate(const complex<C> &x) {return complex<C>(x.real(),-x.imag());};
static inline C realpart(const complex<C> &x) {return x.real();}
static inline C imagpart(const complex<C> &x) {return x.imag();}
};


//non-complex scalars
template<typename C>
struct LA_traits_aux<C, scalar_true> {
typedef C elementtype;
typedef C producttype;
typedef C normtype;
typedef C realtype;
typedef complex<C> complextype;
static inline C sqrabs(const C x) { return x*x;}
static inline bool gencmp(const C *x, const C *y, size_t n) {return memcmp(x,y,n*sizeof(C));}
static inline bool bigger(const  C &x, const C &y) {return x>y;}
static inline bool smaller(const  C &x, const C &y) {return x<y;}
static inline normtype norm (const  C &x) {return std::abs(x);}
static inline void axpy (C &s, const C &x, const C &c) {s+=x*c;}
static inline void put(int fd, const C &x, bool dimensions=0, bool transp=0) {if(sizeof(C)!=write(fd,&x,sizeof(C))) laerror("write error");}
static inline void get(int fd, C &x, bool dimensions=0, bool transp=0) {if(sizeof(C)!=read(fd,&x,sizeof(C))) laerror("read error");}
static void multiget(size_t n,int fd, C *x, bool dimensions=0)
	{
	size_t total=0;
	size_t system_limit = (1L<<30)/sizeof(C); //do not expect too much from the system and read at most 1GB at once
	ssize_t r;
	size_t nn;
	do{
		r=read(fd,x+total,nn=(n-total > system_limit ? system_limit : n-total)*sizeof(C)); 
		if(r<0 || r==0 && nn!=0 ) {std::cout<<"read returned "<<r<<" perror "<<strerror(errno) <<std::endl; laerror("read error");}
		else total += r/sizeof(C);
		if(r%sizeof(C)) laerror("read error 2");
	  }
	while(total < n);
	}
static void multiput(size_t n, int fd, const C *x, bool dimensions=0) 
	{
	size_t total=0;
	size_t system_limit = (1L<<30)/sizeof(C); //do not expect too much from the system and write at most 1GB at once
	ssize_t r;
	size_t nn;
	do{
		r=write(fd,x+total,nn=(n-total > system_limit ? system_limit : n-total)*sizeof(C)); 
		if(r<0 || r==0 && nn!=0 ) {std::cout<<"write returned "<<r<<" perror "<<strerror(errno) <<std::endl; laerror("write error");}
		else total += r/sizeof(C);
		if(r%sizeof(C)) laerror("write error 2");
	  }
	while(total < n);
	}
static void copy(C *dest, C *src, size_t n) {memcpy(dest,src,n*sizeof(C));}
static void clear(C *dest, size_t n) {memset(dest,0,n*sizeof(C));}
static void copyonwrite(C &x) {};
static void clearme(C &x) {x=0;};
static void deallocate(C &x) {};
static inline C conjugate(const C &x) {return x;};
static inline C realpart(const C &x) {return x;}
static inline C imagpart(const C &x) {return 0;}
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
typedef X<typename LA_traits<C>::realtype> realtype; \
typedef X<typename LA_traits<C>::complextype> complextype; \
static bool gencmp(const C *x, const C *y, size_t n) {for(size_t i=0; i<n; ++i) if(x[i]!=y[i]) return true; return false;} \
static inline bool bigger(const  C &x, const C &y) {return x>y;} \
static inline bool smaller(const  C &x, const C &y) {return x<y;} \
static inline normtype norm (const X<C> &x) {return x.norm();} \
static inline void axpy (X<C>&s, const X<C> &x, const C c) {s.axpy(c,x);} \
static void put(int fd, const X<C> &x, bool dimensions=1, bool transp=0) {x.put(fd,dimensions,transp);} \
static void get(int fd, X<C> &x, bool dimensions=1, bool transp=0) {x.get(fd,dimensions,transp);} \
static void multiput(size_t n,int fd, const X<C> *x, bool dimensions=1) {for(size_t i=0; i<n; ++i) x[i].put(fd,dimensions);} \
static void multiget(size_t n,int fd, X<C> *x, bool dimensions=1) {for(size_t i=0; i<n; ++i) x[i].get(fd,dimensions);} \
static void copy(C *dest, C *src, size_t n) {for(size_t i=0; i<n; ++i) dest[i]=src[i];} \
static void clear(C *dest, size_t n) {for(size_t i=0; i<n; ++i) dest[i].clear();}\
static void copyonwrite(X<C> &x) {x.copyonwrite();}\
static void clearme(X<C> &x) {x.clear();}\
static void deallocate(X<C> &x) {x.dealloc();}\
};


//non-scalar types defined in this library
generate_traits(NRMat)
generate_traits(NRMat_from1)
generate_traits(NRVec)
generate_traits(SparseMat)
generate_traits(SparseSMat) //product leading to non-symmetric result not implemented
generate_traits(CSRMat) 

#undef generate_traits

//smat
#define generate_traits_smat(X) \
template<typename C>  \
struct LA_traits_aux<X<C>, scalar_false> {  \
typedef C elementtype;  \
typedef NRMat<C> producttype;  \
typedef typename LA_traits<C>::normtype normtype;  \
typedef X<typename LA_traits<C>::realtype> realtype; \
typedef X<typename LA_traits<C>::complextype> complextype; \
static bool gencmp(const C *x, const C *y, size_t n) {for(size_t i=0; i<n; ++i) if(x[i]!=y[i]) return true; return false;} \
static inline bool bigger(const  C &x, const C &y) {return x>y;} \
static inline bool smaller(const  C &x, const C &y) {return x<y;} \
static inline normtype norm (const X<C> &x) {return x.norm();}  \
static inline void axpy (X<C>&s, const X<C> &x, const C c) {s.axpy(c,x);}  \
static void put(int fd, const X<C> &x, bool dimensions=1, bool transp=0) {x.put(fd,dimensions);}  \
static void get(int fd, X<C> &x, bool dimensions=1, bool transp=0) {x.get(fd,dimensions);}  \
static void multiput(size_t n,int fd, const X<C> *x, bool dimensions=1) {for(size_t i=0; i<n; ++i) x[i].put(fd,dimensions);}  \
static void multiget(size_t n,int fd, X<C> *x, bool dimensions=1) {for(size_t i=0; i<n; ++i) x[i].get(fd,dimensions);}  \
static void copy(C *dest, C *src, size_t n) {for(size_t i=0; i<n; ++i) dest[i]=src[i];}  \
static void clear(C *dest, size_t n) {for(size_t i=0; i<n; ++i) dest[i].clear();} \
static void copyonwrite(X<C> &x) {x.copyonwrite();} \
static void clearme(X<C> &x) {x.clear();} \
static void deallocate(X<C> &x) {x.dealloc();} \
};

generate_traits_smat(NRSMat)
generate_traits_smat(NRSMat_from1)


//the final traits class
template<typename C>
struct LA_traits : LA_traits_aux<C, typename isscalar<C>::scalar_type> {};

}//namespace

#endif
