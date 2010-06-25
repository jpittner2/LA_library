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
#ifndef _LA_MAT_H_
#define _LA_MAT_H_
#include "la_traits.h"

namespace LA {
template <typename T>
class NRMat {
protected:
	int nn;
	int mm;
#ifdef MATPTR
	T **v;
#else
	T *v;
#endif
	int *count;
#ifdef CUDALA
        GPUID location;
#endif
public:
	friend class NRVec<T>;
	friend class NRSMat<T>;
	
	inline NRMat() : nn(0), mm(0), v(0), count(0) 
			{
#ifdef CUDALA
			location = DEFAULT_LOC;
#endif
			};
	inline NRMat(const int n, const int m ,const GPUID loc= undefined);
	inline NRMat(const T &a, const int n, const int m);
	NRMat(const T *a, const int n, const int m);
	inline NRMat(const NRMat &rhs);
	NRMat(const typename LA_traits_complex<T>::NRMat_Noncomplex_type &rhs, bool imagpart=false); //construct complex from real
	explicit NRMat(const NRSMat<T> &rhs);
#ifdef MATPTR
	explicit NRMat(const NRVec<T> &rhs, const int n, const int m, const int offset=0) :NRMat(&rhs[0][0] + offset ,n,m) {if (offset < 0 || n*m + offset > rhs.nn) laerror("matrix dimensions and offset incompatible with vector length");};
#else
	explicit NRMat(const NRVec<T> &rhs, const int n, const int m, const int offset=0);
#endif
	~NRMat();
#ifdef MATPTR
	const bool operator!=(const NRMat &rhs) const {if(nn!=rhs.nn || mm!=rhs.mm) return 1; return LA_traits<T>::gencmp(v[0],rhs.v[0],nn*mm);} //memcmp for scalars else elementwise
#else
        const bool operator!=(const NRMat &rhs) const {if(nn!=rhs.nn || mm!=rhs.mm) return 1; return LA_traits<T>::gencmp(v,rhs.v,nn*mm);} //memcmp for scalars else elementwise
#endif
	const bool operator==(const NRMat &rhs) const {return !(*this != rhs);};
	inline int getcount() const {return count?*count:0;}
#ifdef CUDALA
	inline GPUID getlocation() const {return location;}
	void moveto(const GPUID dest);
#else
	inline GPUID getlocation() const {return cpu;}
	void moveto(const GPUID dest) {};
#endif
	NRMat & operator=(const NRMat &rhs);  //assignment
	void randomize(const typename  LA_traits<T>::normtype &x); //fill with random numbers
	NRMat & operator=(const T &a);    //assign a to diagonal
	NRMat & operator|=(const NRMat &rhs); //assignment to a new copy
	NRMat & operator+=(const T &a);   //add diagonal
	NRMat & operator-=(const T &a);   //substract diagonal
	NRMat & operator*=(const T &a);   //multiply by a scalar
	NRMat & operator+=(const NRMat &rhs);
	NRMat & operator-=(const NRMat &rhs);
	NRMat & operator^=(const NRMat<T>  &rhs); //Hadamard (element-wise) product
	NRMat & operator+=(const NRSMat<T> &rhs);
	NRMat & operator-=(const NRSMat<T> &rhs);
	const NRMat operator-() const; //unary minus
	inline const NRMat operator+(const T &a) const;
	inline const NRMat operator-(const T &a) const;
	inline const NRMat operator*(const T &a) const;
	inline const NRMat operator+(const NRMat &rhs) const;
	inline const NRMat operator-(const NRMat &rhs) const;
	inline const NRMat operator+(const NRSMat<T> &rhs) const;
	inline const NRMat operator-(const NRSMat<T> &rhs) const;
	const T dot(const NRMat &rhs) const; // scalar product of Mat.Mat//@@@for complex do conjugate
	const NRMat operator*(const NRMat &rhs) const; // Mat * Mat
	const NRMat oplus(const NRMat &rhs) const; //direct sum
	const NRMat otimes(const NRMat &rhs, bool reversecolumns=false) const; //direct product
	void diagmultl(const NRVec<T> &rhs); //multiply by a diagonal matrix from L
	void diagmultr(const NRVec<T> &rhs); //multiply by a diagonal matrix from R
	const NRSMat<T> transposedtimes() const; //A^T . A
	const NRSMat<T> timestransposed() const; //A . A^T
	const NRMat operator*(const NRSMat<T> &rhs) const; // Mat * Smat
	const NRMat operator&(const NRMat &rhs) const; // direct sum
	const NRMat operator|(const NRMat<T> &rhs) const; // direct product
	const NRVec<T> operator*(const NRVec<T> &rhs) const {NRVec<T> result(nn,rhs.getlocation()); result.gemv((T)0,*this,'n',(T)1,rhs); return result;}; // Mat * Vec
	const NRVec<complex<T> > operator*(const NRVec<complex<T> > &rhs) const {NRVec<complex<T> > result(nn); result.gemv((T)0,*this,'n',(T)1,rhs); return result;}; // Mat * Vec
	const NRVec<T> rsum() const; //sum of rows
	const NRVec<T> csum() const; //sum of columns
	void orthonormalize(const bool rowcol, const NRSMat<T> *metric=NULL);//orthonormalize (true - vectors are rows)
	const NRVec<T> row(const int i, int l= -1) const; //row of, efficient
	const NRVec<T> column(const int j, int l= -1) const {if(l<0) l=nn; NRVec<T> r(l); for(int i=0; i<l; ++i) r[i]= (*this)(i,j); return r;}; //column of, general but not very efficient
	const T* diagonalof(NRVec<T> &, const bool divide=0, bool cache=false) const; //get diagonal 
	void diagonalset(const NRVec<T> &); //set diagonal elements
	void gemv(const T beta, NRVec<T> &r, const char trans, const T alpha, const NRVec<T> &x) const {r.gemv(beta,*this,trans,alpha,x);};
	void gemv(const T beta, NRVec<complex<T> > &r, const char trans, const T alpha, const NRVec<complex<T> > &x) const {r.gemv(beta,*this,trans,alpha,x);};
	inline T* operator[](const int i);  //subscripting: pointer to row i
	inline const T* operator[](const int i) const;
	inline T& operator()(const int i, const int j); // (i,j) subscripts
	inline const T& operator()(const int i, const int j) const;
	inline int nrows() const;
	inline int ncols() const;
	inline int size() const;
        void get(int fd, bool dimensions=1, bool transposed=false);
        void put(int fd, bool dimensions=1, bool transposed=false) const;
	void copyonwrite();
        void clear() {if(nn&&mm) {copyonwrite(); LA_traits<T>::clear((*this)[0],nn*mm);}}; //zero out
	void resize(int n, int m);
	inline operator T*(); //get a pointer to the data
	inline operator const T*() const;
	NRMat & transposeme(int n=0); // square matrices only 
	NRMat & conjugateme(); // square matrices only
	const NRMat transpose(bool conj=false) const;
	const NRMat conjugate() const;
        const NRMat submatrix(const int fromrow, const int torow, const int fromcol, const int tocol) const; //there is also independent less efficient routine for generally indexed submatrix
	void storesubmatrix(const int fromrow, const int fromcol, const NRMat &rhs); //overwrite a block with external matrix
	void gemm(const T &beta, const NRMat &a, const char transa, const NRMat &b,
			const char transb, const T &alpha);//this = alpha*op( A )*op( B ) + beta*this
	void fprintf(FILE *f, const char *format, const int modulo) const;
	void fscanf(FILE *f, const char *format);
	const typename LA_traits<T>::normtype norm(const T scalar=(T)0) const;
	void axpy(const T alpha, const NRMat &x); // this += a*x
	inline const T amax() const;
	const T trace() const;
	NRMat & SwapRows();
	NRMat & SwapCols();
	NRMat & SwapRowsCols();

//members concerning sparse matrix
	SparseSMat<T> operator*(const SparseSMat<T> &rhs) const;
	explicit NRMat(const SparseMat<T> &rhs);                // dense from sparse
	explicit NRMat(const SparseSMat<T> &rhs);                // dense from sparse
	NRMat & operator+=(const SparseMat<T> &rhs);
        NRMat & operator-=(const SparseMat<T> &rhs);
	void gemm(const T &beta, const SparseMat<T> &a, const char transa, const NRMat &b, const char transb, const T &alpha);//this = alpha*op( A )*op( B ) + beta*this
        inline void simplify() {}; //just for compatibility with sparse ones
	bool issymmetric() const {return 0;};
#ifndef NO_STRASSEN
//Strassen's multiplication (better than n^3, analogous syntax to gemm)
	void strassen(const T beta, const NRMat &a, const char transa, const NRMat &b, const char transb, const T alpha);//this := alpha*op( A )*op( B ) + beta*this
	void s_cutoff(const int,const int,const int,const int) const;
#endif
};

}//namespace
//due to mutual includes this has to be after full class declaration
#include "vec.h"
#include "smat.h"
#include "sparsemat.h"
#include "sparsesmat.h"

namespace LA {
// ctors
template <typename T>
NRMat<T>::NRMat(const int n, const int m, const GPUID loc) : nn(n), mm(m), count(new int)
{
	*count = 1;
#ifdef CUDALA
        location= (loc==undefined?DEFAULT_LOC:loc);
        if(location==cpu)
	{
#endif
#ifdef MATPTR
	v = new T*[n];
	v[0] = new T[m*n];
	for (int i=1; i<n; i++) v[i] = v[i-1] + m;
#else
	v = new T[m*n];
#endif
#ifdef CUDALA
	}
        else
	{
	v= (T*) gpualloc(n*m*sizeof(T));
	}
#endif
}

template <typename T>
NRMat<T>::NRMat(const T &a, const int n, const int m) : nn(n), mm(m), count(new int)
{
#ifdef CUDALA
        location=DEFAULT_LOC;
#endif

	int i;
	T *p;
	*count = 1;
#ifdef CUDALA
        if(location==cpu)
	{
#endif
#ifdef MATPTR
	v = new T*[n];
	p = v[0] = new T[m*n];
	for (int i=1; i<n; i++) v[i] = v[i-1] + m;
#else
	p = v = new T[m*n];
#endif
	if (a != (T)0)
		for (i=0; i< n*m; i++) *p++ = a;
	else
		memset(p, 0, n*m*sizeof(T));
#ifdef CUDALA
	}
	else
	{
	v= (T*) gpualloc(n*m*sizeof(T));
	cublasSetVector(n*m,sizeof(T),&a,0,v,1);
	}
#endif

}

template <typename T>
NRMat<T>::NRMat(const T *a, const int n, const int m) : nn(n), mm(m), count(new int)
{
#ifdef CUDALA
        location=DEFAULT_LOC;
#endif

	*count = 1;
#ifdef CUDALA
        if(location==cpu)
        {
#endif
#ifdef MATPTR
	v = new T*[n];
	v[0] = new T[m*n];
	for (int i=1; i<n; i++) v[i] = v[i-1] + m;
	memcpy(v[0], a, n*m*sizeof(T));
#else
	v = new T[m*n];
	memcpy(v, a, n*m*sizeof(T));
#endif
#ifdef CUDALA
        }
        else
        {
        v= (T*) gpualloc(n*m*sizeof(T));
        cublasSetVector(n*m,sizeof(T),a,1,v,1);
        }
#endif

}


//copy constructor
template <typename T>
NRMat<T>::NRMat(const NRMat &rhs)
{
#ifdef CUDALA
        location=rhs.location;
#endif
	nn = rhs.nn;
	mm = rhs.mm;
	count = rhs.count;
	v = rhs.v;
	if (count) ++(*count);
}


template <typename T>
NRMat<T>::NRMat(const NRSMat<T> &rhs)
{
NOT_GPU(rhs); 

#ifdef CUDALA
	location=rhs.location;
#endif

	int i;
	nn = mm = rhs.nrows();
	count = new int;
	*count = 1;
#ifdef MATPTR
	v = new T*[nn];
	v[0] = new T[mm*nn];
	for (int i=1; i<nn; i++) v[i] = v[i-1] + mm;
#else
	v = new T[mm*nn];
#endif

	int j, k = 0;
#ifdef MATPTR
	for (i=0; i<nn; i++)
		for (j=0; j<=i; j++) v[i][j] = v[j][i] = rhs[k++];
#else
	for (i=0; i<nn; i++)
		for (j=0; j<=i; j++) v[i*nn+j] = v[j*nn+i] = rhs[k++];
#endif
}
 
#ifndef MATPTR
template <typename T>
NRMat<T>::NRMat(const NRVec<T> &rhs, const int n, const int m, const int offset)
{
	if (offset < 0 || n*m + offset > rhs.nn) laerror("matrix dimensions and offset incompatible with vector length");

#ifdef CUDALA
location=rhs.location;
#endif

	nn = n;
	mm = m;
	count = rhs.count;
	v = rhs.v + offset;
	(*count)++;
}
#endif

// Mat + Smat
template <typename T>
inline const NRMat<T> NRMat<T>::operator+(const NRSMat<T> &rhs) const
{
	return NRMat<T>(*this) += rhs;
}

// Mat - Smat
template <typename T>
inline const NRMat<T> NRMat<T>::operator-(const NRSMat<T> &rhs) const
{
	return NRMat<T>(*this) -= rhs;
}

// Mat[i] : pointer to the first element of i-th row
template <typename T>
inline T* NRMat<T>::operator[](const int i)
{
#ifdef DEBUG
	if (_LA_count_check && *count != 1) laerror("Mat lval use of [] with count > 1");
	if (i<0 || i>=nn) laerror("Mat [] out of range");
	if (!v) laerror("[] for unallocated Mat");
#endif
#ifdef MATPTR
	return v[i];
#else
	return v+i*mm;
#endif
}
template <typename T>
inline const T* NRMat<T>::operator[](const int i) const
{
#ifdef DEBUG
	if (i<0 || i>=nn) laerror("Mat [] out of range");
	if (!v) laerror("[] for unallocated Mat");
#endif
#ifdef MATPTR
	return v[i];
#else
	return v+i*mm;
#endif
}

// Mat(i,j) reference to the matrix element M_{ij}
template <typename T>
inline T & NRMat<T>::operator()(const int i, const int j)
{
#ifdef DEBUG
	if (_LA_count_check && *count != 1) laerror("Mat lval use of (,) with count > 1");
	if (i<0 || i>=nn &&nn>0 || j<0 || j>=mm && mm>0) laerror("Mat (,) out of range");
	if (!v) laerror("(,) for unallocated Mat");
NOT_GPU(*this);
#endif
#ifdef MATPTR
	return v[i][j];
#else
	return v[i*mm+j];
#endif
}

template <typename T>
inline const T & NRMat<T>::operator()(const int i, const int j) const
{
#ifdef DEBUG
	if (i<0 || i>=nn&&nn>0 || j<0 || j>=mm&& mm>0) laerror("Mat (,) out of range");
	if (!v) laerror("(,) for unallocated Mat");
NOT_GPU(*this); //in principle we could copy the element to CPU memory, yielding, however, a highly inneficient contruct 
#endif
#ifdef MATPTR
	return v[i][j];
#else
	return v[i*mm+j];
#endif
}

// number of rows
template <typename T>
inline int NRMat<T>::nrows() const
{
	return nn;
}

// number of columns
template <typename T>
inline int NRMat<T>::ncols() const
{
	return mm;
}

template <typename T>
inline int NRMat<T>::size() const
{
	return nn*mm;
}

// reference pointer to Mat
template <typename T>
inline NRMat<T>::operator T* ()
{
#ifdef DEBUG
	if (!v) laerror("unallocated Mat in operator T*");
#endif
#ifdef MATPTR
	return v[0];
#else
	return v;
#endif
}
template <typename T>
inline NRMat<T>::operator const T* () const
{
#ifdef DEBUG
	if (!v) laerror("unallocated Mat in operator T*");
#endif
#ifdef MATPTR
	return v[0];
#else
	return v;
#endif
}

// max element of Mat
template<>
inline const double  NRMat<double>::amax() const
{
#ifdef MATPTR
	return v[0][cblas_idamax(nn*mm, v[0], 1)];
#else
	return v[cblas_idamax(nn*mm, v, 1)];
#endif
}
template<>
inline const complex<double>  NRMat< complex<double> >::amax() const
{
#ifdef MATPTR
	return v[0][cblas_izamax(nn*mm, v[0], 1)];
#else
	return v[cblas_izamax(nn*mm, v, 1)];
#endif
}


//basic stuff to be available for any type ... must be in .h
// dtor
template <typename T>
NRMat<T>::~NRMat()
{
        if (!count) return;
        if (--(*count) <= 0) {
                if (v) {
#ifdef CUDALA
		    if(location==cpu)
#endif
		    	{
#ifdef MATPTR
                        delete[] (v[0]);
#endif
                        delete[] v;
		    	}
#ifdef CUDALA
		    else 
			{
			gpufree(v);
			}
#endif
                }
                delete count;
        }
}

// assign NRMat = NRMat
template <typename T>
NRMat<T> & NRMat<T>::operator=(const NRMat<T> &rhs)
{
        if (this !=&rhs)
		{
        	if (count) 
                    if (--(*count) ==0 ) 
			{
#ifdef CUDALA
			if(location==cpu) 
			{
#endif
#ifdef MATPTR
                        delete[] (v[0]);
#endif
                        delete[] v;
#ifdef CUDALA
			}
			else gpufree(v);
#endif

                        delete count;
                	}
                v = rhs.v;
#ifdef CUDALA
                location=rhs.location;
#endif
                nn = rhs.nn;
                mm = rhs.mm;
                count = rhs.count;
                if (count) (*count)++;
		}
        return *this;
}


// Explicit deep copy of NRmat
template <typename T>
NRMat<T> & NRMat<T>::operator|=(const NRMat<T> &rhs)
{
        if (this == &rhs) return *this;
	*this = rhs;
	this->copyonwrite();
        return *this;
}

// make detach Mat and make it's own deep copy
template <typename T>
void NRMat<T>::copyonwrite()
{
        if (!count) laerror("Mat::copyonwrite of undefined matrix");
        if (*count > 1) {
             (*count)--;
             count = new int;
             *count = 1;
#ifdef CUDALA
	     if(location==cpu) //matrix is in CPU memory
	     {
#endif
#ifdef MATPTR
                T **newv = new T*[nn];
                newv[0] = new T[mm*nn];
                memcpy(newv[0], v[0], mm*nn*sizeof(T));
                v = newv;
                for (int i=1; i< nn; i++) v[i] = v[i-1] + mm;
#else
                T *newv = new T[mm*nn];
                memcpy(newv, v, mm*nn*sizeof(T));
                v = newv;
#endif
#ifdef CUDALA
	     }
          else //matrix is in GPU memory
		{
		T *newv = (T *) gpualloc(mm*nn*sizeof(T));
		if(sizeof(T)%sizeof(float)!=0) laerror("cpu memcpy alignment problem");
                cublasScopy(nn*mm*sizeof(T)/sizeof(float),(const float *) v,1,(float *)newv,1);
                v = newv;
		}
#endif
        }
}


template <typename T>
void NRMat<T>::resize(int n, int m)
{
#ifdef DEBUG
        if (n<0 || m<0) laerror("illegal dimensions in Mat::resize()");
#endif

//allow trivial dimensions
if(n==0) m=0;
if(m==0) n=0;

        if (count)
	    {
	        if(n==0 && m==0)
        		{
		        if(--(*count) <= 0) {
#ifdef CUDALA
                           if(location==cpu)
				{
#endif
#ifdef MATPTR
                		if(v) delete[] (v[0]);
#endif
                		if(v) delete[] v;
#ifdef CUDALA
				}
			    else gpufree(v);
#endif
		                delete count;
		                }
		        count=0;
		        nn=mm=0;
		        v=0;
		        return;
		        }
                if (*count > 1) {
                        (*count)--;
                        count = 0;
                        v  = 0;
                        nn = 0;
                        mm = 0;
                }
	    }
        if (!count) {
                count = new int;
                *count = 1;
                nn = n;
                mm = m;
#ifdef CUDALA
                if(location==cpu) 
		{
#endif
#ifdef MATPTR
                v = new T*[nn];
                v[0] = new T[m*n];
                for (int i=1; i< n; i++) v[i] = v[i-1] + m;
#else
                v = new T[m*n];
#endif
#ifdef CUDALA
                }
		else
		v = (T *) gpualloc(n*m*sizeof(T));
#endif

                return;
        }
        // At this point *count = 1, check if resize is necessary
        if (n!=nn || m!=mm) {
                nn = n;
                mm = m;
#ifdef CUDALA
             if(location==cpu)
	     {
#endif
#ifdef MATPTR
                delete[] (v[0]);
#endif
                delete[] v;
#ifdef MATPTR
                v = new T*[nn];
                v[0] = new T[m*n];
                for (int i=1; i< n; i++) v[i] = v[i-1] + m;
#else
                v = new T[m*n];
#endif
#ifdef CUDALA
             }
            else
	     {
             gpufree(v);
             v=(T *) gpualloc(n*m*sizeof(T));
	     }
#endif
        }
}




template<typename T>
NRMat<complex<T> > complexify(const NRMat<T> &rhs)
{
NRMat<complex<T> > r(rhs.nrows(),rhs.ncols());
for(int i=0; i<rhs.nrows(); ++i) 
	for(int j=0; j<rhs.ncols(); ++j) r(i,j)= rhs(i,j);
return r;
}



// I/O
template <typename T>
std::ostream& operator<<(std::ostream &s, const NRMat<T> &x)
{
#ifdef CUDALA
        if(x.getlocation()==cpu)
		{
#endif
                int i,j,n,m;
                n=x.nrows();
                m=x.ncols();
                s << n << ' ' << m << '\n';
                for(i=0;i<n;i++)
                        {
                        for(j=0; j<m;j++) s << (typename LA_traits_io<T>::IOtype) x[i][j] << (j==m-1 ? '\n' : ' '); // endl cannot be used in the conditional expression, since it is an overloaded function
                        }
                return s;
#ifdef CUDALA
		}
	else
		{
		NRMat<T> tmp=x;
		tmp.moveto(cpu);
		return s<<tmp;
		}
#endif
}

template <typename T>
std::istream& operator>>(std::istream  &s, NRMat<T> &x)
{
#ifdef CUDALA
        if(x.getlocation()==cpu)
                {
#endif
                int i,j,n,m;
                s >> n >> m;
                x.resize(n,m);
		typename LA_traits_io<T>::IOtype tmp;
                for(i=0;i<n;i++) for(j=0; j<m;j++) { s>>tmp; x[i][j]=tmp;}
                return s;
#ifdef CUDALA
		}
	else
		{
		NRMat<T> tmp;
		tmp.moveto(cpu);
		s >> tmp;
		tmp.moveto(x.getlocation());
		x=tmp;
		return s;
		}
#endif
}


//optional indexing from 1
//all possible constructors have to be given explicitly, other stuff is inherited
//with exception of the operator() which differs
template<typename T>
class NRMat_from1 : public NRMat<T> {
public:
        NRMat_from1(): NRMat<T>() {};
        explicit NRMat_from1(const int n): NRMat<T>(n) {};
        NRMat_from1(const NRMat<T> &rhs): NRMat<T>(rhs) {}; //be able to convert the parent class  transparently to this
        NRMat_from1(const int n, const int m): NRMat<T>(n,m) {};
        NRMat_from1(const T &a, const int n, const int m): NRMat<T>(a,n,m) {};
        NRMat_from1(const T *a, const int n, const int m): NRMat<T>(a,n,m) {};

        inline const T& operator() (const int i, const int j) const
                {
#ifdef DEBUG
        if (i<1 || i>NRMat<T>::nn || j<1 || j>NRMat<T>::mm) laerror("Mat (,) out of range");
        if (!NRMat<T>::v) laerror("(,) for unallocated Mat");
#endif
#ifdef MATPTR
        return NRMat<T>::v[i-1][j-1];
#else
        return NRMat<T>::v[(i-1)*NRMat<T>::mm+j-1];
#endif
                }
        inline  T& operator() (const int i, const int j)
                {
#ifdef DEBUG
        if (_LA_count_check && *NRMat<T>::count != 1) laerror("Mat lval use of (,) with count > 1");
        if (i<1 || i>NRMat<T>::nn || j<1 || j>NRMat<T>::mm) laerror("Mat (,) out of range");
        if (!NRMat<T>::v) laerror("(,) for unallocated Mat");
#endif
#ifdef MATPTR
        return NRMat<T>::v[i-1][j-1];
#else
        return NRMat<T>::v[(i-1)*NRMat<T>::mm+j-1];
#endif
                }
};



//Hadamard product
template<typename T>
NRMat<T> & NRMat<T>::operator^=(const NRMat<T>  &rhs){
#ifdef DEBUG
        if (nn != rhs.nn || mm!= rhs.mm)
                laerror("Mat ^= Mat of incompatible matrices");
#endif
        copyonwrite();
#ifdef MATPTR
         for (register int i=0; i< nn*mm; i++) v[0][i] *= rhs.v[0][i];
#else
	const int Dim = nn*mm;
	for(register int i=0;i<Dim;i++) v[i] *= rhs.v[i];
#endif
        return *this;
}


#ifdef CUDALA
template<typename T>
void NRMat<T>::moveto(const GPUID dest)
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
	v = new T[nn*mm];
	gpuget(nn*mm,sizeof(T),vold,v);
	if(*count == 1) gpufree(vold);
	else {--(*count); count = new int(1);}
	}
else	//moving from CPU to GPU
	{
	v=(T *) gpualloc(nn*mm*sizeof(T));
	gpuput(nn*mm,sizeof(T),vold,v);
	if(*count == 1) delete[] vold;
	else {--(*count); count = new int(1);}
	}
}
#endif
//end CUDALA



// generate operators: Mat + a, a + Mat, Mat * a
NRVECMAT_OPER(Mat,+)
NRVECMAT_OPER(Mat,-)
NRVECMAT_OPER(Mat,*)
// generate Mat + Mat, Mat - Mat
NRVECMAT_OPER2(Mat,+)
NRVECMAT_OPER2(Mat,-)

}//namespace
#endif /* _LA_MAT_H_ */
