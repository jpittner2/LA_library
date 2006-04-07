#ifndef _LA_SMAT_H_
#define _LA_SMAT_H_

#include "la_traits.h"

#define NN2 (nn*(nn+1)/2)
template <class T>
class NRSMat { // symmetric or complex hermitean matrix in packed form
protected:
	int nn;
	T *v;
	int *count;
public:
	friend class NRVec<T>;
	friend class NRMat<T>;
	
	inline NRSMat() : nn(0),v(0),count(0) {};
	inline explicit NRSMat(const int n);			// Zero-based array
	inline NRSMat(const T &a, const int n);	//Initialize to constant
	inline NRSMat(const T *a, const int n);	// Initialize to array
	inline NRSMat(const NRSMat &rhs);		// Copy constructor
	explicit NRSMat(const NRMat<T> &rhs);		// symmetric part of general matrix
	explicit NRSMat(const NRVec<T> &rhs, const int n); //construct matrix from vector
	NRSMat & operator|=(const NRSMat &rhs);	//assignment to a new copy
	NRSMat & operator=(const NRSMat &rhs);	//assignment
	NRSMat & operator=(const T &a);		//assign a to diagonal
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
	const NRVec<T> operator*(const NRVec<T> &rhs) const {NRVec<T> result(nn); result.gemv((T)0,*this,'n',(T)1,rhs); return result;}; // Mat * Vec
	const T* diagonalof(NRVec<T> &, const bool divide=0, bool cache=false) const; //get diagonal
	void gemv(const T beta, NRVec<T> &r, const char trans, const T alpha, const NRVec<T> &x) const {r.gemv(beta,*this,trans,alpha,x);};
	inline const T& operator[](const int ij) const;
	inline T& operator[](const int ij);
	inline const T& operator()(const int i, const int j) const;
	inline T& operator()(const int i, const int j);
	inline int nrows() const;
	inline int ncols() const;
	inline int size() const;
	inline bool transp(const int i, const int j) const {return i>j;} //this can be used for compact storage of matrices, which are actually not symmetric, but one triangle of them is redundant
	const double norm(const T scalar=(T)0) const;
	void axpy(const T alpha, const NRSMat &x); // this+= a*x
	inline const T amax() const;
	const T trace() const;
	void get(int fd, bool dimensions=1, bool transp=0);
        void put(int fd, bool dimensions=1, bool transp=0) const;
	void copyonwrite();
	void resize(const int n);
	inline operator T*(); //get a pointer to the data
	inline operator const T*() const; //get a pointer to the data
	~NRSMat();
	void fprintf(FILE *f, const char *format, const int modulo) const; 
	void fscanf(FILE *f, const char *format); 
//members concerning sparse matrix
	explicit NRSMat(const SparseMat<T> &rhs);               // dense from sparse
	inline void simplify() {}; //just for compatibility with sparse ones
	bool issymmetric() const {return 1;}
};

//due to mutual includes this has to be after full class declaration
#include "vec.h"
#include "mat.h"
#include "sparsemat.h"


// ctors
template <typename T>
inline NRSMat<T>::NRSMat(const int n) : nn(n), v(new T[NN2]),
				count(new int) {*count = 1;}

template <typename T>
inline NRSMat<T>::NRSMat(const T& a, const int n) : nn(n),
	        v(new T[NN2]), count(new int)
{
	*count =1;
	if(a != (T)0) for(int i=0; i<NN2; i++) v[i] = a;
}

template <typename T>
inline NRSMat<T>::NRSMat(const T *a, const int n) : nn(n),
	        v(new T[NN2]), count(new int)
{
	*count = 1;
	memcpy(v, a, NN2*sizeof(T));
}

template <typename T>
inline NRSMat<T>::NRSMat(const NRSMat<T> &rhs) //copy constructor
{
	v = rhs.v;
	nn = rhs.nn;
	count = rhs.count;
	if (count) (*count)++;
}

template <typename T>
NRSMat<T>::NRSMat(const NRVec<T> &rhs, const int n) // type conversion
{
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
	cblas_zscal(NN2, (void *)(&a), (void *)v, 1);
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
NRSMat< complex<double> > &
NRSMat< complex<double> >::operator+=(const NRSMat< complex<double> > & rhs)
{
#ifdef DEBUG
	if (nn != rhs.nn) laerror("incompatible SMats in SMat::operator+=");
#endif
	copyonwrite();
	cblas_zaxpy(NN2, (void *)(&CONE), (void *)(&rhs.v), 1, (void *)(&v), 1);
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
	cblas_zaxpy(NN2, (void *)(&CMONE), (void *)(&rhs.v), 1, (void *)(&v), 1);
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
	if (*count != 1) laerror("lval [] with count > 1 in Smat");
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
inline T SMat_index_1(T i, T j)
{
return (i>=j)? i*(i-1)/2+j-1 : j*(j-1)/2+i-1;
}

// access the element, 2-dim array case
template <typename T>
inline T & NRSMat<T>::operator()(const int i, const int j)
{
#ifdef DEBUG
	if (*count != 1) laerror("lval (i,j) with count > 1 in Smat");
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
	return v[cblas_izamax(NN2, (void *)v, 1)];
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
                if (v) delete[] (v);
                delete count;
        }
}

// assignment with a physical copy
template <typename T>
NRSMat<T> & NRSMat<T>::operator|=(const NRSMat<T> &rhs)
{
        if (this != &rhs) {
                if(!rhs.v) laerror("unallocated rhs in NRSMat operator |=");
                if(count)
                        if(*count > 1) {        // detach from the other
                                --(*count);
                                nn = 0;
                                count = 0;
                                v = 0;
                        }
                if (nn != rhs.nn) {
                        if(v) delete [] (v);
                        nn = rhs.nn;
                }
                if (!v) v = new T[NN2];
                if (!count) count = new int;
                *count = 1;
                memcpy(v, rhs.v, NN2*sizeof(T));
        }
        return *this;
}


// assignment
template <typename T>
NRSMat<T> & NRSMat<T>::operator=(const NRSMat<T> & rhs)
{
        if (this == & rhs) return *this;
        if (count)
                if(--(*count) == 0) {
                        delete [] v;
                        delete count;
                }
        v = rhs.v;
        nn = rhs.nn;
        count = rhs.count;
        if (count) (*count)++;
        return *this;
}


// make new instation of the Smat, deep copy
template <typename T>
void NRSMat<T>::copyonwrite()
{
#ifdef DEBUG
        if (!count) laerror("probably an assignment to undefined Smat");
#endif
        if (*count > 1) {
                (*count)--;
                count = new int;
                *count = 1;
                T *newv = new T[NN2];
                memcpy(newv, v, NN2*sizeof(T));
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
	                if(v) delete[] (v);
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
                v = new T[NN2];
                return;
        }
        if (n != nn) {
                nn = n;
                delete[] v;
                v = new T[NN2];
        }
}


template<typename T>
NRSMat<complex<T> > complexify(const NRSMat<T> &rhs)
{
NRSMat<complex<T> > r(rhs.nrows());
for(int i=0; i<rhs.nrows(); ++i) 
        for(int j=0; j<=i; ++j) r(i,j)=rhs(i,j);
return r;
}

// I/O
template <typename T> extern ostream& operator<<(ostream &s, const NRSMat<T> &x);
template <typename T> extern istream& operator>>(istream  &s, NRSMat<T> &x);

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
		if(i<=0||j<=0||i>nn||j>nn) laerror("index out of range in NRSMat_from1");
#endif
		return v[SMat_index_1(i,j)];
		}
        inline  T& operator() (const int i, const int j) 
		{
#ifdef DEBUG
                if(i<=0||j<=0||i>nn||j>nn) laerror("index out of range in NRSMat_from1");
#endif
		return v[SMat_index_1(i,j)];
		}
};


#endif /* _LA_SMAT_H_ */
