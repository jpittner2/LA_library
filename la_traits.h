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

//avoid unsupported register directive in new C++ versions
#if __cplusplus > 201402L
#define register
#endif


#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <limits>
#include <complex>
#include <unistd.h>


//using namespace std;

#include "laerror.h"
#include "la_random.h"

#include "cuda_la.h"

#ifdef NONCBLAS
#include "noncblas.h"
#else
extern "C" {
#ifdef HAS_MKL
#include "mkl_cblas.h"
#else
#include "cblas.h"
#endif
}
#endif

#ifdef NONCLAPACK
#include "noncblas.h"
#else
extern "C" {
#include "clapack.h"
}
#endif


//forward declarations
namespace LA_Vecmat3 {
template<typename C> class Vec3;
template<typename C> class Mat3;
}

namespace LA {



//forward declarations
template<typename C> class NRVec;
template<typename C> class NRVec_from1;
template<typename C> class NRMat;
template<typename C> class NRMat_from1;
template<typename C> class NRSMat;
template<typename C> class NRSMat_from1;
template<typename C> class SparseMat;
template<typename C> class SparseSMat;
template<typename C> class CSRMat;
template<typename C> class NRPerm;
template<typename C> class CyclePerm;
template<typename C> class Partition;
template<typename C> class CompressedPartition;

//trick to allow real and imag part of complex as l-values
template<typename T>
T &real(std::complex<T> &c) {
  return reinterpret_cast<T*>(&c)[0];
}
template<typename T>
T &imag(std::complex<T> &c) {
  return reinterpret_cast<T*>(&c)[1];
}


//

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
struct LA_traits_complex<std::complex<T> > \
        { \
        typedef T Component_type; \
	typedef NRVec<T> NRVec_Noncomplex_type; \
	typedef NRMat<T> NRMat_Noncomplex_type; \
	typedef NRSMat<T> NRSMat_Noncomplex_type; \
        };


SPECIALIZE_COMPLEX(double)
SPECIALIZE_COMPLEX(std::complex<double>)
SPECIALIZE_COMPLEX(float)
SPECIALIZE_COMPLEX(std::complex<float>)
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
        static inline bool compare(const T &object, I i, I j) {return object.bigger(i,j);};
	static inline bool comparestable(const T &object, I i, I j, I *auxkey)
                {
                bool r= object.bigger(i,j);
                if(!r && object[i]==object[j]) r= auxkey[i]>auxkey[j];
                return r;
                };

        };

template<typename T, typename I>
struct LA_sort_traits<T,I,1>
        {
        static inline bool compare(const T &object, I i, I j) {return object.smaller(i,j);};
	static inline bool comparestable(const T &object, I i, I j, I *auxkey)
                {
                bool r= object.smaller(i,j);
                if(!r && object[i]==object[j]) r= auxkey[i]<auxkey[j];
                return r;
                };
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
class isscalar<std::complex<X> > {public: typedef scalar_true scalar_type;};\
template<>\
class isscalar<std::complex<std::complex<X> > > {public: typedef scalar_true scalar_type;};\


//declare what is scalar
SCALAR(bool)
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

//#undef SCALAR


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
struct LA_traits_aux<std::complex<C>, scalar_true> {
typedef std::complex<C> elementtype;
typedef std::complex<C> producttype;
typedef C normtype;
typedef C realtype;
typedef std::complex<C> complextype;
static inline C sqrabs(const std::complex<C> x) { return x.real()*x.real()+x.imag()*x.imag();}
static inline bool gencmp(const std::complex<C> *x, const std::complex<C> *y, size_t n) {return memcmp(x,y,n*sizeof(std::complex<C>));}
static bool bigger(const  std::complex<C> &x, const std::complex<C> &y) {laerror("std::complex comparison undefined"); return false;}
static bool smaller(const  std::complex<C> &x, const std::complex<C> &y) {laerror("std::complex comparison undefined"); return false;}
static inline normtype norm (const  std::complex<C> &x) {return std::abs(x);}
static inline normtype abs2(const  std::complex<C> &x) {return std::abs(x.real())+std::abs(x.imag());} //faster without sqrt
static inline void axpy (std::complex<C> &s, const std::complex<C> &x, const std::complex<C> &c) {s+=x*c;}
static inline void get(int fd, std::complex<C> &x, bool dimensions=0, bool transp=0) {if(sizeof(std::complex<C>)!=read(fd,&x,sizeof(std::complex<C>))) laerror("read error");}
static inline void put(int fd, const std::complex<C> &x, bool dimensions=0, bool transp=0) {if(sizeof(std::complex<C>)!=write(fd,&x,sizeof(std::complex<C>))) laerror("write error");}
static void multiget(size_t n,int fd, std::complex<C> *x, bool dimensions=0)
	{
	size_t total=0;
	size_t system_limit = (1L<<30)/sizeof(std::complex<C>); //do not expect too much from the system and read at most 1GB at once
	ssize_t r;
	size_t nn;
	do{
		r=read(fd,x+total,nn=(n-total > system_limit ? system_limit : n-total)*sizeof(std::complex<C>)); 
		if(r<0 || r==0 && nn!=0 ) {std::cout<<"read returned "<<r<<" perror "<<strerror(errno) <<std::endl; laerror("read error");}
		else total += r/sizeof(std::complex<C>);
		if(r%sizeof(std::complex<C>)) laerror("read error 2");
	  }
	while(total < n);
	}
static void multiput(size_t n, int fd, const std::complex<C> *x, bool dimensions=0) 
	{
	size_t total=0;
	size_t system_limit = (1L<<30)/sizeof(std::complex<C>); //do not expect too much from the system and write at most 1GB at once
	ssize_t r;
	size_t nn;
	do{
		r=write(fd,x+total,nn=(n-total > system_limit ? system_limit : n-total)*sizeof(std::complex<C>)); 
		if(r<0 || r==0 && nn!=0 ) {std::cout<<"write returned "<<r<<"  perror "<<strerror(errno) <<std::endl; laerror("write error");}
		else total += r/sizeof(std::complex<C>);
		if(r%sizeof(std::complex<C>)) laerror("write error 2");
	  }
	while(total < n);
	}
static void copy(std::complex<C> *dest, const std::complex<C> *src, size_t n) {memcpy(dest,src,n*sizeof(std::complex<C>));}
static void clear(std::complex<C> *dest, size_t n) {memset(dest,0,n*sizeof(std::complex<C>));}
static void copyonwrite(std::complex<C> &x) {};
static bool is_plaindata() {return true;}
static bool is_complex() {return true;}
static void clearme(std::complex<C> &x) {x=0;};
static void deallocate(std::complex<C> &x) {};
static inline std::complex<C> conjugate(const std::complex<C> &x) {return std::complex<C>(x.real(),-x.imag());};
static inline C realpart(const std::complex<C> &x) {return x.real();}
static inline C imagpart(const std::complex<C> &x) {return x.imag();}
static inline void setrealpart(std::complex<C> &x, const C &y) {reinterpret_cast<C(&)[2]>(x)[0]=y;}
static inline void setimagpart(std::complex<C> &x, const C &y) {reinterpret_cast<C(&)[2]>(x)[1]=y;}
};


//non-complex scalars
template<typename C>
struct LA_traits_aux<C, scalar_true> {
typedef C elementtype;
typedef C producttype;
typedef C normtype;
typedef C realtype;
typedef std::complex<C> complextype;
static inline C sqrabs(const C x) { return x*x;}
static inline bool gencmp(const C *x, const C *y, size_t n) {return memcmp(x,y,n*sizeof(C));}
static inline bool bigger(const  C &x, const C &y) {return x>y;}
static inline bool smaller(const  C &x, const C &y) {return x<y;}
static inline normtype norm (const  C &x) {return std::abs(x);}
static inline normtype abs2 (const  C &x) {return std::abs(x);}
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
static void copy(C *dest, const C *src, size_t n) {memcpy(dest,src,n*sizeof(C));}
static void clear(C *dest, size_t n) {memset(dest,0,n*sizeof(C));}
static void copyonwrite(C &x) {};
static bool is_plaindata() {return true;}
static bool is_complex() {return false;}
static void clearme(C &x) {x=0;};
static void deallocate(C &x) {};
static inline C conjugate(const C &x) {return x;};
static inline C realpart(const C &x) {return x;}
static inline C imagpart(const C &x) {return 0;}
static inline void setrealpart(C &x, const C &y) {x=y;}
static inline void setimagpart(C &x, const C &y) {}
};


//non-scalars except smat

template<typename C>
struct LA_traits; //forward declaration needed for template recursion

#define generate_traits_nonscalar(X) \
template<typename C> \
struct LA_traits_aux<X<C>, scalar_false> { \
typedef C elementtype; \
typedef X<C> producttype; \
typedef C PAIRTYPE[2]; \
typedef C QUADRUPLETYPE[4]; \
typedef X<C> SUBMATRIXTYPE; \
typedef typename LA_traits<C>::normtype normtype; \
typedef X<typename LA_traits<C>::realtype> realtype; \
typedef X<typename LA_traits<C>::complextype> complextype; \
static bool gencmp(const X<C> *x, const X<C> *y, size_t n) {for(size_t i=0; i<n; ++i) if(x[i]!=y[i]) return true; return false;} \
static inline bool bigger(const  X<C> &x, const X<C> &y) {return x>y;} \
static inline bool smaller(const  X<C> &x, const X<C> &y) {return x<y;} \
static inline normtype norm (const X<C> &x) {return x.norm();} \
static inline void axpy (X<C>&s, const X<C> &x, const C c) {s.axpy(c,x);} \
static void put(int fd, const X<C> &x, bool dimensions=1, bool transp=0, bool orcaformat=false) {x.put(fd,dimensions,transp,orcaformat);} \
static void get(int fd, X<C> &x, bool dimensions=1, bool transp=0, bool orcaformat=false) {x.get(fd,dimensions,transp,orcaformat);} \
static void multiput(size_t n,int fd, const X<C> *x, bool dimensions=1, bool orcaformat=false) {for(size_t i=0; i<n; ++i) x[i].put(fd,dimensions,false,orcaformat);} \
static void multiget(size_t n,int fd, X<C> *x, bool dimensions=1, bool orcaformat=false) {for(size_t i=0; i<n; ++i) x[i].get(fd,dimensions,false,orcaformat);} \
static void copy(X<C> *dest, const X<C> *src, size_t n) {for(size_t i=0; i<n; ++i) dest[i]=src[i];} \
static void clear(X<C> *dest, size_t n) {for(size_t i=0; i<n; ++i) dest[i].clear();}\
static void copyonwrite(X<C> &x) {x.copyonwrite();}\
static bool is_plaindata() {return false;}\
static bool is_complex() {return LA_traits<C>::is_complex();}\
static void clearme(X<C> &x) {x.clear();}\
static void deallocate(X<C> &x) {x.dealloc();}\
};


//non-scalar types defined in this library
generate_traits_nonscalar(NRMat)
generate_traits_nonscalar(NRMat_from1)
generate_traits_nonscalar(NRVec)
generate_traits_nonscalar(NRVec_from1)
generate_traits_nonscalar(SparseMat)
generate_traits_nonscalar(SparseSMat) //product leading to non-symmetric result not implemented
generate_traits_nonscalar(CSRMat) 
generate_traits_nonscalar(NRPerm)
generate_traits_nonscalar(CyclePerm)
generate_traits_nonscalar(Partition)
generate_traits_nonscalar(CompressedPartition)



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
static void put(int fd, const X<C> &x, bool dimensions=1, bool transp=0, bool orcaformat=false) {x.put(fd,dimensions,false,orcaformat);}  \
static void get(int fd, X<C> &x, bool dimensions=1, bool transp=0, bool orcaformat=false) {x.get(fd,dimensions,false,orcaformat);}  \
static void multiput(size_t n,int fd, const X<C> *x, bool dimensions=1, bool orcaformat=false) {for(size_t i=0; i<n; ++i) x[i].put(fd,dimensions,false,orcaformat);}  \
static void multiget(size_t n,int fd, X<C> *x, bool dimensions=1, bool orcaformat=false) {for(size_t i=0; i<n; ++i) x[i].get(fd,dimensions,false,orcaformat);}  \
static void copy(C *dest, const C *src, size_t n) {for(size_t i=0; i<n; ++i) dest[i]=src[i];}  \
static void clear(C *dest, size_t n) {for(size_t i=0; i<n; ++i) dest[i].clear();} \
static void copyonwrite(X<C> &x) {x.copyonwrite();} \
static bool is_plaindata() {return false;}\
static bool is_complex() {return LA_traits<C>::is_complex();}\
static void clearme(X<C> &x) {x.clear();} \
static void deallocate(X<C> &x) {x.dealloc();} \
};

generate_traits_smat(NRSMat)
generate_traits_smat(NRSMat_from1)


//the final traits class
template<typename C>
struct LA_traits : LA_traits_aux<C, typename isscalar<C>::scalar_type> {};


//precision-dependent default thresholds; only float and double are supported (the double values are those used so far)
template<typename T> struct LA_precision;
template<> struct LA_precision<double>
	{
	static double svd_threshold() {return 1e-12;} //singular values treated as zero (svd_solve, Tucker)
	static double taylor_precision() {return 1e-14;} //truncation of the Taylor series in matexp
	static double conjgrad_tol() {return 1e-8;} //default convergence threshold of conjgrad
	static double gmres_eps() {return 1e-7;} //default relative residual of gmres
	static double eigensolver_eps() {return 1e-6;} //default convergence threshold of davidson and lanczos
	static double newton_threshold() {return 1e-14;} //convergence of Polynomial::newton
	static double det_tiny() {return 1e-300;} //determinant underflow guard in linear_solve
	};
template<> struct LA_precision<float>
	{
	static float svd_threshold() {return 1e-5f;}
	static float taylor_precision() {return 1e-7f;}
	static float conjgrad_tol() {return 1e-5f;}
	static float gmres_eps() {return 1e-5f;}
	static float eigensolver_eps() {return 1e-4f;}
	static float newton_threshold() {return 1e-6f;}
	static float det_tiny() {return 1e-37f;}
	};

}//namespace

#endif
