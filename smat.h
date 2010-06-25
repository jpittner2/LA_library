/*
    LA: linear algebra C++ interface library
    Copyright (C) 2008 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>
                  complex versions written by Roman Curik <roman.curik@jh-inst.cas.cz>


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
#ifndef _LA_SMAT_H_
#define _LA_SMAT_H_
#include "la_traits.h"

namespace LA {
#define NN2 (nn*(nn+1)/2)
template <class T>
class NRSMat { // symmetric or complex hermitean matrix in packed form
protected:
	int nn;
	T *v;
	int *count;
#ifdef CUDALA
       GPUID location;
#endif
public:
	friend class NRVec<T>;
	friend class NRMat<T>;
	
	inline NRSMat() : nn(0),v(0),count(0) 
                        {
#ifdef CUDALA
                        location = DEFAULT_LOC;
#endif
                        };
	inline explicit NRSMat(const int n, const GPUID loc= undefined);// Zero-based array
	inline NRSMat(const T &a, const int n);	//Initialize to constant
	inline NRSMat(const T *a, const int n);	// Initialize to array
	inline NRSMat(const NRSMat &rhs);		// Copy constructor
	NRSMat(const typename LA_traits_complex<T>::NRSMat_Noncomplex_type &rhs, bool imagpart=false); //construct complex from real
	explicit NRSMat(const NRMat<T> &rhs);		// symmetric part of general matrix
	explicit NRSMat(const NRVec<T> &rhs, const int n); //construct matrix from vector
	NRSMat & operator|=(const NRSMat &rhs);	//assignment to a new copy
	NRSMat & operator=(const NRSMat &rhs);	//assignment
        void randomize(const typename LA_traits<T>::normtype &x);
	NRSMat & operator=(const T &a);		//assign a to diagonal
#ifdef CUDALA
        inline GPUID getlocation() const {return location;}
        void moveto(const GPUID dest);
#else
        inline GPUID getlocation() const {return cpu;}
        void moveto(const GPUID dest) {};
#endif
        const bool operator!=(const NRSMat &rhs) const {if(nn!=rhs.nn) return 1; return LA_traits<T>::gencmp(v,rhs.v,NN2);} //memcmp for scalars else elementwise
        const bool operator==(const NRSMat &rhs) const {return !(*this != rhs);};
	inline NRSMat & operator*=(const T &a);
	inline NRSMat & operator+=(const T &a); 
	inline NRSMat & operator-=(const T &a); 
	inline NRSMat & operator+=(const NRSMat &rhs); 
	inline NRSMat & operator-=(const NRSMat &rhs); 
	const NRSMat operator-() const; //unary minus
	inline int getcount() const {return count?*count:0;}
	inline const NRSMat operator*(const T &a) const;
	inline const NRSMat operator+(const T &a) const;
	inline const NRSMat operator-(const T &a) const;
	inline const NRSMat operator+(const NRSMat &rhs) const; 
	inline const NRSMat operator-(const NRSMat &rhs) const;
	inline const NRMat<T> operator+(const NRMat<T> &rhs) const; 
	inline const NRMat<T> operator-(const NRMat<T> &rhs) const; 
	const NRMat<T> operator*(const NRSMat &rhs) const; // SMat*SMat
	const NRMat<T> operator*(const NRMat<T> &rhs) const; // SMat*Mat 
	const T dot(const NRSMat &rhs) const; // Smat.Smat//@@@for complex do conjugate
	const T dot(const NRVec<T> &rhs) const; //Smat(as vec).vec //@@@for complex do conjugate
	const NRVec<T> operator*(const NRVec<T> &rhs) const {NRVec<T> result(nn,rhs.getlocation()); result.gemv((T)0,*this,'n',(T)1,rhs); return result;}; // Mat * Vec
	const NRVec<complex<T> > operator*(const NRVec<complex<T> > &rhs) const {NRVec<complex<T> > result(nn,rhs.getlocation()); result.gemv((T)0,*this,'n',(T)1,rhs); return result;}; // Mat * Vec
	const T* diagonalof(NRVec<T> &, const bool divide=0, bool cache=false) const; //get diagonal
	void gemv(const T beta, NRVec<T> &r, const char trans, const T alpha, const NRVec<T> &x) const {r.gemv(beta,*this,trans,alpha,x);};
	void gemv(const T beta, NRVec<complex<T> > &r, const char trans, const T alpha, const NRVec<complex<T> > &x) const {r.gemv(beta,*this,trans,alpha,x);};
	inline const T& operator[](const int ij) const;
	inline T& operator[](const int ij);
	inline const T& operator()(const int i, const int j) const;
	inline T& operator()(const int i, const int j);
	inline int nrows() const;
	inline int ncols() const;
	inline int size() const;
	inline bool transp(const int i, const int j) const {return i>j;} //this can be used for compact storage of matrices, which are actually not symmetric, but one triangle of them is redundant
	const typename LA_traits<T>::normtype norm(const T scalar=(T)0) const;
	void axpy(const T alpha, const NRSMat &x); // this+= a*x
	inline const T amax() const;
	const T trace() const;
	void get(int fd, bool dimensions=1, bool transp=0);
        void put(int fd, bool dimensions=1, bool transp=0) const;
	void copyonwrite();
	void clear() {copyonwrite(); LA_traits<T>::clear(v,NN2);}; //zero out
	void resize(const int n);
	inline operator T*(); //get a pointer to the data
	inline operator const T*() const; //get a pointer to the data
	~NRSMat();
	void fprintf(FILE *f, const char *format, const int modulo) const; 
	void fscanf(FILE *f, const char *format); 
//members concerning sparse matrix
	explicit NRSMat(const SparseMat<T> &rhs);               // dense from sparse
	explicit NRSMat(const SparseSMat<T> &rhs);               // dense from sparse
	inline void simplify() {}; //just for compatibility with sparse ones
	bool issymmetric() const {return 1;}
};

}//namespace
//due to mutual includes this has to be after full class declaration
#include "vec.h"
#include "mat.h"

namespace LA {

// ctors
template <typename T>
inline NRSMat<T>::NRSMat(const int n, const GPUID loc) : nn(n), count(new int(1)) 
{
#ifdef CUDALA
        location= (loc==undefined?DEFAULT_LOC:loc);
        if(location==cpu)
#endif
	v=new T[NN2];
#ifdef CUDALA
	else v= (T*) gpualloc(NN2*sizeof(T));
#endif
}

template <typename T>
inline NRSMat<T>::NRSMat(const T& a, const int n) : nn(n), count(new int(1))
{
#ifdef CUDALA
        location=DEFAULT_LOC;
	if(location==cpu)
#endif
        {
	v=new T[NN2];
	if(a != (T)0) for(int i=0; i<NN2; i++) v[i] = a;
	else memset(v, 0, NN2*sizeof(T));
	}
#ifdef CUDALA
	else
	{
	v= (T*) gpualloc(NN2*sizeof(T));
	cublasSetVector(NN2,sizeof(T),&a,0,v,1);
	}
#endif
}

template <typename T>
inline NRSMat<T>::NRSMat(const T *a, const int n) : nn(n), count(new int(1))
{
#ifdef CUDALA
        location=DEFAULT_LOC;
        if(location==cpu)
#endif
		memcpy(v, a, NN2*sizeof(T));
#ifdef CUDALA
        else
        {
        v= (T*) gpualloc(NN2*sizeof(T));
        cublasSetVector(NN2,sizeof(T),a,1,v,1);
        }
#endif

}

template <typename T>
inline NRSMat<T>::NRSMat(const NRSMat<T> &rhs) //copy constructor
{
#ifdef CUDALA
        location=rhs.location;
#endif
	v = rhs.v;
	nn = rhs.nn;
	count = rhs.count;
	if (count) (*count)++;
}

template <typename T>
NRSMat<T>::NRSMat(const NRVec<T> &rhs, const int n) // type conversion
{
#ifdef CUDALA
        location=rhs.location;
#endif
	nn = n;
#ifdef DEBUG
	if (NN2 != rhs.size())
		laerror("matrix dimensions incompatible with vector length");
#endif
	count = rhs.count;
	v = rhs.v;
	(*count)++;
}


// S *= a
template<>
inline NRSMat<double> & NRSMat<double>::operator*=(const double & a)
{
	copyonwrite();
	cblas_dscal(NN2, a, v, 1);
	return *this;
}

template<>
inline NRSMat< complex<double> > &
NRSMat< complex<double> >::operator*=(const complex<double> & a)
{
	copyonwrite();
	cblas_zscal(NN2, &a, v, 1);
	return *this;
}
template <typename T>
inline NRSMat<T> & NRSMat<T>::operator*=(const T & a)
{
        copyonwrite();
	for(int i=0; i<NN2; ++i) v[i]*=a;
        return *this;
}



// S += D
template <typename T>
inline NRSMat<T> & NRSMat<T>::operator+=(const T &a)
{
	copyonwrite();
	for (int i=0; i<nn; i++) v[i*(i+1)/2+i] += a;
	return *this;
}

// S -= D
template <typename T>
inline NRSMat<T> & NRSMat<T>::operator-=(const T &a)
{
	copyonwrite();
	for (int i=0; i<nn; i++) v[i*(i+1)/2+i] -= a;
	return *this;
}

// S += S
template<>
inline NRSMat<double> &
NRSMat<double>::operator+=(const NRSMat<double> & rhs)
{
#ifdef DEBUG
	if (nn != rhs.nn) laerror("incompatible SMats in SMat::operator+=");
#endif
	copyonwrite();
	cblas_daxpy(NN2, 1.0, rhs.v, 1, v, 1);
	return *this;
}

template<>
inline NRSMat< complex<double> > &
NRSMat< complex<double> >::operator+=(const NRSMat< complex<double> > & rhs)
{
#ifdef DEBUG
	if (nn != rhs.nn) laerror("incompatible SMats in SMat::operator+=");
#endif
	copyonwrite();
	cblas_zaxpy(NN2, &CONE, rhs.v, 1, v, 1);
	return *this;
}

template <typename T>
inline NRSMat<T> & NRSMat<T>::operator+=(const NRSMat<T> & rhs)
{
#ifdef DEBUG
        if (nn != rhs.nn) laerror("incompatible SMats in SMat::operator+=");
#endif
        copyonwrite();
	for(int i=0; i<NN2; ++i) v[i] += rhs.v[i];
        return *this;
}


// S -= S
template<>
inline NRSMat<double> &
NRSMat<double>::operator-=(const NRSMat<double> & rhs)
{
#ifdef DEBUG
	if (nn != rhs.nn) laerror("incompatible SMats in SMat::operator-=");
#endif
	copyonwrite();
	cblas_daxpy(NN2, -1.0, rhs.v, 1, v, 1);
	return *this;
}

template<>
inline NRSMat< complex<double> > &
NRSMat< complex<double> >::operator-=(const NRSMat< complex<double> > & rhs)
{
#ifdef DEBUG
	if (nn != rhs.nn) laerror("incompatible SMats in SMat::operator-=");
#endif
	copyonwrite();
	cblas_zaxpy(NN2, &CMONE, rhs.v, 1, v, 1);
	return *this;
}

template <typename T>
inline NRSMat<T> & NRSMat<T>::operator-=(const NRSMat<T> & rhs)
{
#ifdef DEBUG
        if (nn != rhs.nn) laerror("incompatible SMats in SMat::operator-=");
#endif
        copyonwrite();
        for(int i=0; i<NN2; ++i) v[i] -= rhs.v[i];
        return *this;
}


// SMat + Mat
template <typename T>
inline const NRMat<T> NRSMat<T>::operator+(const NRMat<T> &rhs) const
{
	return NRMat<T>(rhs) += *this;
}

// SMat - Mat
template <typename T>
inline const NRMat<T> NRSMat<T>::operator-(const NRMat<T> &rhs) const
{
	return NRMat<T>(-rhs) += *this;
}

// access the element, linear array case
template <typename T>
inline T & NRSMat<T>::operator[](const int ij)
{
#ifdef DEBUG
	if (_LA_count_check && *count != 1) laerror("lval [] with count > 1 in Smat");
	if (ij<0 || ij>=NN2) laerror("SMat [] out of range");
	if (!v) laerror("[] for unallocated Smat");
#endif
	return v[ij];
}
template <typename T>
inline const T & NRSMat<T>::operator[](const int ij) const
{
#ifdef DEBUG
	if (ij<0 || ij>=NN2) laerror("SMat [] out of range");
	if (!v) laerror("[] for unallocated Smat");
#endif
	return v[ij];
}

template<typename T>
inline T SMat_index(T i, T j)
{
return (i>=j) ? i*(i+1)/2+j : j*(j+1)/2+i;
}

template<typename T>
inline T SMat_index_igej(T i, T j)
{
return i*(i+1)/2+j;
}

template<typename T>
inline T SMat_index_ilej(T i, T j)
{
return j*(j+1)/2+i;
}


template<typename T>
inline T SMat_index_1(T i, T j)
{
return (i>=j)? i*(i-1)/2+j-1 : j*(j-1)/2+i-1;
}

template<typename T>
inline T SMat_index_1igej(T i, T j)
{
return i*(i-1)/2+j-1;
}

template<typename T>
inline T SMat_index_1ilej(T i, T j)
{
return j*(j-1)/2+i-1;
}

//indexing for antisymmetric matrix (used by fourindex)

template<typename T>
inline T ASMat_index(T i, T j)
{
if(i==j) return -1;
return (i>j) ? i*(i-1)/2+j : j*(j-1)/2+i;
}

template<typename T>
inline T ASMat_index_1(T i, T j)
{
if(i==j) return -1;
return (i>j)? (i-2)*(i-1)/2+j-1 : (j-2)*(j-1)/2+i-1;
}



// access the element, 2-dim array case
template <typename T>
inline T & NRSMat<T>::operator()(const int i, const int j)
{
#ifdef DEBUG
	if (_LA_count_check && *count != 1) laerror("lval (i,j) with count > 1 in Smat");
	if (i<0 || i>=nn || j<0 || j>=nn) laerror("SMat (i,j) out of range");
	if (!v) laerror("(i,j) for unallocated Smat");
#endif
	return v[SMat_index(i,j)];
}
template <typename T>
inline const T & NRSMat<T>::operator()(const int i, const int j) const
{
#ifdef DEBUG
	if (i<0 || i>=nn || j<0 || j>=nn) laerror("SMat (i,j) out of range");
	if (!v) laerror("(i,j) for unallocated Smat");
#endif
	return v[SMat_index(i,j)];
}

// return the number of rows and columns
template <typename T>
inline int NRSMat<T>::nrows() const
{
	return nn;
}
template <typename T>
inline int NRSMat<T>::ncols() const
{
	return nn;
}

template <typename T>
inline int NRSMat<T>::size() const
{
        return NN2;
}


// max value
template<>
inline const double NRSMat<double>::amax() const
{
	return v[cblas_idamax(NN2, v, 1)];
}
template<>
inline const complex<double> NRSMat< complex<double> >::amax() const
{
	return v[cblas_izamax(NN2, v, 1)];
}

// reference pointer to Smat
template <typename T>
inline NRSMat<T>:: operator T*()
{
#ifdef DEBUG
	if (!v) laerror("unallocated SMat in operator T*");
#endif
	return v;
}
template <typename T>
inline NRSMat<T>:: operator const T*() const
{
#ifdef DEBUG
	if (!v) laerror("unallocated SMat in operator T*");
#endif
	return v;
}

//basic stuff to be available for any type ... must be in .h

// dtor
template <typename T>
NRSMat<T>::~NRSMat()
{
        if (!count) return;
        if (--(*count) <= 0) {
                if (v) 
		    {
#ifdef CUDALA
                    if(location==cpu)
#endif
			delete[] v;
#ifdef CUDALA
                        else gpufree(v);
#endif
                    }
                delete count;
        }
}


// assignment with a physical copy
template <typename T>
NRSMat<T> & NRSMat<T>::operator|=(const NRSMat<T> &rhs)
{
#ifdef DEBUG
                if (!rhs.v) laerror("unallocated rhs in NRSMat operator |=");
#endif
        if (this == &rhs) return *this;
        *this = rhs;
        this->copyonwrite();
        return *this;
}


// assignment
template <typename T>
NRSMat<T> & NRSMat<T>::operator=(const NRSMat<T> & rhs)
{
        if (this == & rhs) return *this;
        if (count)
                if(--(*count) == 0) 
			{
#ifdef CUDALA
        		if(location==cpu)
#endif
                        	delete [] v;
#ifdef CUDALA
		        else
               			 gpufree(v);
#endif
                        delete count;
                	}
        v = rhs.v;
        nn = rhs.nn;
        count = rhs.count;
#ifdef CUDALA
        location=rhs.location;
#endif
        if (count) (*count)++;
        return *this;
}


// make new instation of the Smat, deep copy
template <typename T>
void NRSMat<T>::copyonwrite()
{
        if (!count) laerror("SMat::copyonwrite() of undefined Smat");
        if (*count > 1) {
                (*count)--;
                count = new int;
                *count = 1;
		T *newv;
#ifdef CUDALA
	    if(location==cpu)
       		 {
#endif
                 newv = new T[NN2];
                 memcpy(newv, v, NN2*sizeof(T));
#ifdef CUDALA
	        }
     	else
        	{
        	 newv = (T *) gpualloc(NN2*sizeof(T));
         	if(sizeof(T)%sizeof(float)!=0) laerror("cpu memcpy alignment problem");
         	cublasScopy(NN2*sizeof(T)/sizeof(float),(const float *) v,1,(float *)newv,1);
        	}
#endif

                 v = newv;
        }
}


// resize Smat
template <typename T>
void NRSMat<T>::resize(const int n)
{
#ifdef DEBUG
        if (n < 0) laerror("illegal matrix dimension in resize of Smat");
#endif
        if (count)
	{
	    	if(n==0)
	        {
	        if(--(*count) <= 0) {
			if(v) {
#ifdef CUDALA
                        if(location==cpu)
#endif
		                 delete[] (v);
#ifdef CUDALA
                        else
                                gpufree(v);
#endif
			}
	                delete count;
	                }
	        count=0;
	        nn=0;
	        v=0;
	        return;
	        }

                if(*count > 1) {        //detach from previous
                        (*count)--;
                        count = 0;
                        v = 0;
                        nn = 0;
                }
	}
        if (!count)     {                               //new uninitialized vector or just detached
                count = new int;
                *count = 1;
                nn = n;
#ifdef CUDALA
     if(location==cpu)
#endif
                v = new T[NN2];
#ifdef CUDALA
      else
        v = (T*) gpualloc(NN2*sizeof(T));
#endif

                return;
        }
        if (n != nn) {
              nn = n;
#ifdef CUDALA
	     if(location==cpu)
#endif
			{
                	delete[] v;
                	v = new T[NN2];
			}
#ifdef CUDALA
	      else
       			 {
			 gpufree(v);
			 v = (T*) gpualloc(NN2*sizeof(T));
		         }
#endif

        }
}

#ifdef CUDALA
template<typename T>
void NRSMat<T>::moveto(const GPUID dest)
{
if(location==dest) return;
location=dest;

if(v && !count) laerror("internal inconsistency of reference counting 1");
if (!count) return;

if(v && *count==0) laerror("internal inconsistency of reference counting 2");
if(!v) return;

T *vold = v;

if(dest == cpu) //moving from GPU to CPU
        {
        v = new T[NN2];
        gpuget(NN2,sizeof(T),vold,v);
        if(*count == 1) gpufree(vold);
        else {--(*count); count = new int(1);}
        }
else    //moving from CPU to GPU
        {
        v=(T *) gpualloc(NN2*sizeof(T));
        gpuput(NN2,sizeof(T),vold,v);
        if(*count == 1) delete[] vold;
        else {--(*count); count = new int(1);}
        }
}
#endif




template<typename T>
NRSMat<complex<T> > complexify(const NRSMat<T> &rhs)
{
NRSMat<complex<T> > r(rhs.nrows());
for(int i=0; i<rhs.nrows(); ++i) 
        for(int j=0; j<=i; ++j) r(i,j)=rhs(i,j);
return r;
}


// I/O
template <typename T>
std::ostream& operator<<(std::ostream &s, const NRSMat<T> &x)
                {
#ifdef CUDALA
if(x.getlocation()==cpu)
   {
#endif
                int i,j,n;
                n=x.nrows();
                s << n << ' ' << n << '\n';
                for(i=0;i<n;i++)
                        {
                        for(j=0; j<n;j++) s << (typename LA_traits_io<T>::IOtype)x(i,j) << (j==n-1 ? '\n' : ' ');
                        }
                return s;
#ifdef CUDALA
   }
else
    {
    NRSMat<T> tmp=x;
    tmp.moveto(cpu);
    return s<<tmp;
    }
#endif
                }


template <typename T>
std::istream& operator>>(std::istream  &s, NRSMat<T> &x)
                {
#ifdef CUDALA
if(x.getlocation()==cpu)
  {
#endif
                int i,j,n,m;
                s >> n >> m;
                if(n!=m) laerror("input symmetric matrix not square");
                x.resize(n);
		typename LA_traits_io<T>::IOtype tmp;
                for(i=0;i<n;i++) for(j=0; j<m;j++) {s>>tmp; x(i,j)=tmp;}
                return s;
#ifdef CUDALA
  }
else
                {
                NRSMat<T> tmp;
                tmp.moveto(cpu);
                s >> tmp;
                tmp.moveto(x.getlocation());
                x=tmp;
                return s;
                }
#endif
                }


// generate operators: SMat + a, a + SMat, SMat * a
NRVECMAT_OPER(SMat,+)
NRVECMAT_OPER(SMat,-)
NRVECMAT_OPER(SMat,*)
// generate SMat + SMat, SMat - SMat
NRVECMAT_OPER2(SMat,+)
NRVECMAT_OPER2(SMat,-)

//optional indexing from 1
//all possible constructors have to be given explicitly, other stuff is inherited
//with exception of the operator() which differs
template<typename T>
class NRSMat_from1 : public NRSMat<T> {
public:
        NRSMat_from1(): NRSMat<T>() {};
        explicit NRSMat_from1(const int n): NRSMat<T>(n) {};
	NRSMat_from1(const NRSMat<T> &rhs): NRSMat<T>(rhs) {}; //be able to convert the parent class  transparently to this
        NRSMat_from1(const T &a, const int n): NRSMat<T>(a,n) {}; 
        NRSMat_from1(const T *a, const int n): NRSMat<T>(a,n) {}; 
        explicit NRSMat_from1(const NRMat<T> &rhs): NRSMat<T>(rhs) {};
        explicit NRSMat_from1(const NRVec<T> &rhs, const int n): NRSMat<T>(rhs,n) {};

        inline const T& operator() (const int i, const int j) const 
		{
#ifdef DEBUG
		if(i<=0||j<=0||i>NRSMat<T>::nn||j>NRSMat<T>::nn) laerror("index out of range in NRSMat_from1");
#endif
		return NRSMat<T>::v[SMat_index_1(i,j)];
		}
        inline  T& operator() (const int i, const int j) 
		{
#ifdef DEBUG
                if(i<=0||j<=0||i>NRSMat<T>::nn||j>NRSMat<T>::nn) laerror("index out of range in NRSMat_from1");
#endif
		return NRSMat<T>::v[SMat_index_1(i,j)];
		}
};

}//namespace
#endif /* _LA_SMAT_H_ */
