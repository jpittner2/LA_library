#ifndef _LA_SMAT_H_
#define _LA_SMAT_H_

#include "vec.h"
#include "mat.h"

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
	
	inline NRSMat<T>::NRSMat() : nn(0),v(0),count(0) {};
	inline explicit NRSMat(const int n);			// Zero-based array
	inline NRSMat(const T &a, const int n);	//Initialize to constant
	inline NRSMat(const T *a, const int n);	// Initialize to array
	inline NRSMat(const NRSMat &rhs);		// Copy constructor
	explicit NRSMat(const NRMat<T> &rhs);		// symmetric part of general matrix
	explicit NRSMat(const NRVec<T> &rhs, const int n); //construct matrix from vector
	NRSMat & operator|=(const NRSMat &rhs);	//assignment to a new copy
	NRSMat & operator=(const NRSMat &rhs);	//assignment
	NRSMat & operator=(const T &a);		//assign a to diagonal
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
	const T dot(const NRSMat &rhs) const; // Smat.Smat
	const NRVec<T> operator*(const NRVec<T> &rhs) const; 
	inline const T& operator[](const int ij) const;
	inline T& operator[](const int ij);
	inline const T& operator()(const int i, const int j) const;
	inline T& operator()(const int i, const int j);
	inline int nrows() const;
	inline int ncols() const;
	const double norm(const T scalar=(T)0) const;
	void axpy(const T alpha, const NRSMat &x); // this+= a*x
	inline const T amax() const;
	const T trace() const;
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
};

// INLINES
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
inline NRSMat<double> & NRSMat<double>::operator*=(const double & a)
{
	copyonwrite();
	cblas_dscal(NN2, a, v, 1);
	return *this;
}
inline NRSMat< complex<double> > &
NRSMat< complex<double> >::operator*=(const complex<double> & a)
{
	copyonwrite();
	cblas_zscal(nn, (void *)(&a), (void *)v, 1);
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

// S -= S
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

// access the element, 2-dim array case
template <typename T>
inline T & NRSMat<T>::operator()(const int i, const int j)
{
#ifdef DEBUG
	if (*count != 1) laerror("lval (i,j) with count > 1 in Smat");
	if (i<0 || i>=nn || j<0 || j>=nn) laerror("SMat (i,j) out of range");
	if (!v) laerror("(i,j) for unallocated Smat");
#endif
	return i>=j ? v[i*(i+1)/2+j] : v[j*(j+1)/2+i];
}
template <typename T>
inline const T & NRSMat<T>::operator()(const int i, const int j) const
{
#ifdef DEBUG
	if (i<0 || i>=nn || j<0 || j>=nn) laerror("SMat (i,j) out of range");
	if (!v) laerror("(i,j) for unallocated Smat");
#endif
	return i>=j ? v[i*(i+1)/2+j] : v[j*(j+1)/2+i];
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

// max value
inline const double NRSMat<double>::amax() const
{
	return v[cblas_idamax(NN2, v, 1)];
}
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

#endif /* _LA_SMAT_H_ */
