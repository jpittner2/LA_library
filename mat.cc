#include "mat.h"
// TODO :
//

//////////////////////////////////////////////////////////////////////////////
//// forced instantization in the corresponding object file
template NRMat<double>;
template NRMat< complex<double> >;
template NRMat<int>;
template NRMat<char>;


/*
 *  Templates first, specializations for BLAS next
 */



// Assign diagonal
template <typename T>
NRMat<T> & NRMat<T>::operator=(const T &a)
{
	copyonwrite();
#ifdef DEBUG
	if (nn != mm) laerror("RMat.operator=scalar on non-square matrix");
#endif
#ifdef MATPTR
	 for (int i=0; i< nn; i++) v[i][i] = a;
#else
	 for (int i=0; i< nn*nn; i+=nn+1) v[i] = a;
#endif
	 return *this;
}



//get diagonal; for compatibility with large matrices do not return newly created object
template <typename T>
void NRMat<T>::diagonalof(NRVec<T> &r) const
{
#ifdef DEBUG
        if (nn != mm) laerror("diagonalof() non-square matrix");
	if (r.size() != nn) laerror("diagonalof() incompatible vector");
#endif

#ifdef MATPTR
         for (int i=0; i< nn; i++) r[i] = v[i][i];
#else
         {int i,j; for (i=j=0; j< nn; ++j, i+=nn+1) r[j] = v[i];}
#endif

}



// M += a
template <typename T>
NRMat<T> & NRMat<T>::operator+=(const T &a)
{
	copyonwrite();
#ifdef DEBUG
	if (nn != mm) laerror("Mat.operator+=scalar on non-square matrix");
#endif
#ifdef MATPTR
	for (int i=0; i< nn; i++) v[i][i] += a;
#else
	for (int i=0; i< nn*nn; i+=nn+1) v[i] += a;
#endif
	return *this;
}

// M -= a
template <typename T>
NRMat<T> & NRMat<T>::operator-=(const T &a)
{
	copyonwrite();
#ifdef DEBUG
	if (nn != mm) laerror("Mat.operator-=scalar on non-square matrix");
#endif
#ifdef MATPTR
	for (int i=0; i< nn; i++) v[i][i] -= a;
#else
	for (int i=0; i< nn*nn; i+=nn+1) v[i] -= a;
#endif
	return *this;
}

// unary minus
template <typename T>
const NRMat<T> NRMat<T>::operator-() const
{
	NRMat<T> result(nn, mm);
#ifdef MATPTR
	for (int i=0; i<nn*mm; i++) result.v[0][i]= -v[0][i];
#else
	for (int i=0; i<nn*mm; i++) result.v[i]= -v[i];
#endif
	return result;
}

// direct sum
template <typename T>
const NRMat<T> NRMat<T>::operator&(const NRMat<T> & b) const
{
	NRMat<T> result((T)0, nn+b.nn, mm+b.mm);
	for (int i=0; i<nn; i++) memcpy(result[i], (*this)[i], sizeof(T)*mm);
	for (int i=0; i<b.nn; i++) memcpy(result[nn+i]+nn, b[i], sizeof(T)*b.mm);
	return result;
}

// direct product 
template <typename T>
const NRMat<T> NRMat<T>::operator|(const NRMat<T> &b) const
{
	NRMat<T> result(nn*b.nn, mm*b.mm);
	for (int i=0; i<nn; i++)
		for (int j=0; j<mm; j++)
			for (int k=0; k<b.nn; k++)
				for (int l=0; l<b.mm; l++)
					result[i*b.nn+k][j*b.mm+l] = (*this)[i][j]*b[k][l];
	return result;
}

// sum of columns
template <typename T>
const NRVec<T> NRMat<T>::csum() const
{
	NRVec<T> result(nn);
	T sum;
	
	for (int i=0; i<nn; i++) { 
		sum = (T)0;
		for(int j=0; j<mm; j++) sum += (*this)[i][j];
		result[i] = sum;
	}
	return result;
}

// sum of rows
template <typename T>
const NRVec<T> NRMat<T>::rsum() const
{
	NRVec<T> result(nn);
	T sum;
	
	for (int i=0; i<mm; i++) { 
		sum = (T)0;
		for(int j=0; j<nn; j++) sum += (*this)[j][i];
		result[i] = sum;
	}
	return result;
}



// transpose Mat
template <typename T>
NRMat<T> & NRMat<T>::transposeme()
{
#ifdef DEBUG
	if (nn != mm) laerror("transpose of non-square Mat");
#endif
	copyonwrite();
	for(int i=1; i<nn; i++)
		for(int j=0; j<i; j++) {
#ifdef MATPTR
			T tmp = v[i][j]; 
			v[i][j] = v[j][i]; 
			v[j][i] = tmp;
#else
			register int a; 
			register int b;
			a = i*mm+j;
			b = j*mm+i;
			T tmp = v[a];
			v[a] = v[b];
			v[b] = tmp;
#endif
		}
	return *this;
}

// Output of Mat
template <typename T>
void NRMat<T>::fprintf(FILE *file, const char *format, const int modulo) const
{
	lawritemat(file, (const T*)(*this), nn, mm, format, 2, modulo, 0);
}

// Input of Mat
template <typename T>
void NRMat<T>::fscanf(FILE *f, const char *format)
{
	int n, m;
	if (std::fscanf(f, "%d %d", &n, &m) != 2)
		laerror("cannot read matrix dimensions in Mat::fscanf()");
	resize(n,m);
	T *p = *this;
	for(int i=0; i<n; i++)
		for(int j=0; j<n; j++)
			if(std::fscanf(f,format,p++) != 1)
				laerror("cannot read matrix element in Mat::fscanf()");
}

























/*
 * BLAS specializations for double and complex<double>
 */

// Mat *= a
NRMat<double> & NRMat<double>::operator*=(const double &a)
{
	copyonwrite();
	cblas_dscal(nn*mm, a, *this, 1);
	return *this;
}
NRMat< complex<double> > &
NRMat< complex<double> >::operator*=(const complex<double> &a)
{
	copyonwrite();
	cblas_zscal(nn*mm, &a, (void *)(*this)[0], 1);
	return *this;
}

//and for general type
template <typename T>
NRMat<T> & NRMat<T>::operator*=(const T &a)
{
        copyonwrite();
#ifdef MATPTR
         for (int i=0; i< nn*nn; i++) v[0][i] *= a;
#else
         for (int i=0; i< nn*nn; i++) v[i] *= a;
#endif
        return *this;
}



// Mat += Mat
NRMat<double> & NRMat<double>::operator+=(const NRMat<double>  &rhs)
{
#ifdef DEBUG
	if (nn != rhs.nn || mm!= rhs.mm) 
		laerror("Mat += Mat of incompatible matrices");
#endif
	copyonwrite();
	cblas_daxpy(nn*mm, 1.0, rhs, 1, *this, 1);
	return *this;
}
NRMat< complex<double> > &
NRMat< complex<double> >::operator+=(const NRMat< complex<double> >  &rhs)
{
#ifdef DEBUG
	if (nn != rhs.nn || mm!= rhs.mm) 
		laerror("Mat += Mat of incompatible matrices");
#endif
	 copyonwrite();
	 cblas_zaxpy(nn*mm, &CONE, (void *)rhs[0], 1, (void *)(*this)[0], 1);
	 return *this;
}

//and for general type
template <typename T>
NRMat<T> & NRMat<T>::operator+=(const NRMat<T>  &rhs)
{
#ifdef DEBUG
        if (nn != rhs.nn || mm!= rhs.mm)
                laerror("Mat -= Mat of incompatible matrices");
#endif
        copyonwrite();
#ifdef MATPTR
         for (int i=0; i< nn*nn; i++) v[0][i] += rhs.v[0][i] ;
#else
         for (int i=0; i< nn*nn; i++) v[i] += rhs.v[i] ;
#endif
        return *this;
}


// Mat -= Mat
NRMat<double> & NRMat<double>::operator-=(const NRMat<double>  &rhs)
{
#ifdef DEBUG
	if (nn != rhs.nn || mm!= rhs.mm) 
		laerror("Mat -= Mat of incompatible matrices");
#endif
	copyonwrite();
	cblas_daxpy(nn*mm, -1.0, rhs, 1, *this, 1);
	return *this;
}
NRMat< complex<double> > &
NRMat< complex<double> >::operator-=(const NRMat< complex<double> >  &rhs)
{
#ifdef DEBUG
	if (nn != rhs.nn || mm!= rhs.mm) 
		laerror("Mat -= Mat of incompatible matrices");
#endif
	 copyonwrite();
	 cblas_zaxpy(nn*mm, &CMONE, (void *)rhs[0], 1, (void *)(*this)[0], 1);
	 return *this;
}

//and for general type
template <typename T>
NRMat<T> & NRMat<T>::operator-=(const NRMat<T>  &rhs)
{
#ifdef DEBUG
        if (nn != rhs.nn || mm!= rhs.mm)
                laerror("Mat -= Mat of incompatible matrices");
#endif
        copyonwrite();
#ifdef MATPTR
         for (int i=0; i< nn*nn; i++) v[0][i] -= rhs.v[0][i] ;
#else
         for (int i=0; i< nn*nn; i++) v[i] -= rhs.v[i] ;
#endif
        return *this;
}


// Mat += SMat
NRMat<double> & NRMat<double>::operator+=(const NRSMat<double> &rhs)
{
#ifdef DEBUG
	if (nn!=mm || nn!=rhs.nrows()) laerror("incompatible matrix size in Mat+=SMat");
#endif
	const double *p = rhs;
	copyonwrite();
	for (int i=0; i<nn; i++) {
		cblas_daxpy(i+1, 1.0, p, 1, (*this)[i], 1); 
		p += i+1;
	}
	p = rhs; p++;
	for (int i=1; i<nn; i++) {
		cblas_daxpy(i, 1.0, p, 1, (*this)[0]+i, nn);
		p += i+1;
	}
	return *this;
}
NRMat< complex<double> > & 
NRMat< complex<double> >::operator+=(const NRSMat< complex<double> > &rhs)
{
#ifdef DEBUG
	if (nn!=mm || nn!=rhs.nrows()) laerror("incompatible matrix size in Mat+=SMat");
#endif
	const complex<double> *p = rhs;
	copyonwrite();
	for (int i=0; i<nn; i++) {
		cblas_zaxpy(i+1, (void *)&CONE, (void *)p, 1, (void *)(*this)[i], 1); 
		p += i+1;
	}
	p = rhs; p++;
	for (int i=1; i<nn; i++) {
		cblas_zaxpy(i, (void *)&CONE, (void *)p, 1, (void *)((*this)[i]+i), nn);
		p += i+1;
	}
	return *this;
}

//and for general type
template <typename T>
NRMat<T> & NRMat<T>::operator+=(const NRSMat<T> &rhs)
{
#ifdef DEBUG
        if (nn!=mm || nn!=rhs.nrows()) laerror("incompatible matrix size in Mat+=SMat");
#endif
        const T *p = rhs;
        copyonwrite();
        for (int i=0; i<nn; i++) {
		for(int j=0; j<i+1; ++j) *((*this)[i]+j) += p[j];
                p += i+1;
        }
        p = rhs; p++;
        for (int i=1; i<nn; i++) {
		for(int j=0; j<i; ++j) *((*this)[i]+i+nn*j) += p[j];
                p += i+1;
        }
        return *this;
}


// Mat -= SMat
NRMat<double> & NRMat<double>::operator-=(const NRSMat<double> &rhs)
{
#ifdef DEBUG
	if (nn!=mm || nn!=rhs.nrows()) laerror("incompatible matrix size in Mat-=SMat");
#endif
	const double *p = rhs;
	copyonwrite();
	for (int i=0; i<nn; i++) {
		cblas_daxpy(i+1, -1.0, p, 1, (*this)[i], 1); 
		p += i+1;
	}
	p = rhs; p++;
	for (int i=1; i<nn; i++) {
		cblas_daxpy(i, -1.0, p, 1, (*this)[0]+i, nn);
		p += i+1;
	}
	return *this;
}
NRMat< complex<double> > & 
NRMat< complex<double> >::operator-=(const NRSMat< complex<double> > &rhs)
{
#ifdef DEBUG
	if (nn!=mm || nn!=rhs.nrows()) laerror("incompatible matrix size in Mat-=SMat");
#endif
	const complex<double> *p = rhs;
	copyonwrite();
	for (int i=0; i<nn; i++) {
		cblas_zaxpy(i+1, (void *)&CMONE, (void *)p, 1, (void *)(*this)[i], 1); 
		p += i+1;
	}
	p = rhs; p++;
	for (int i=1; i<nn; i++) {
		cblas_zaxpy(i, (void *)&CMONE, (void *)p, 1, (void *)((*this)[i]+i), nn);
		p += i+1;
	}
	return *this;
}


//and for general type
template <typename T>
NRMat<T> & NRMat<T>::operator-=(const NRSMat<T> &rhs)
{
#ifdef DEBUG
        if (nn!=mm || nn!=rhs.nrows()) laerror("incompatible matrix size in Mat+=SMat");
#endif
        const T *p = rhs;
        copyonwrite();
        for (int i=0; i<nn; i++) {
		for(int j=0; j<i+1; ++j) *((*this)[i]+j) -= p[j];
                p += i+1;
        }
        p = rhs; p++;
        for (int i=1; i<nn; i++) {
		for(int j=0; j<i; ++j) *((*this)[i]+i+nn*j) -= p[j];
                p += i+1;
        }
        return *this;
}

// Mat.Mat - scalar product
const double NRMat<double>::dot(const NRMat<double> &rhs) const
{
#ifdef DEBUG
	if(nn!=rhs.nn || mm!= rhs.mm) laerror("Mat.Mat incompatible matrices");
#endif
	return cblas_ddot(nn*mm, (*this)[0], 1, rhs[0], 1);
}
const complex<double>
NRMat< complex<double> >::dot(const NRMat< complex<double> > &rhs) const
{
#ifdef DEBUG
	if(nn!=rhs.nn || mm!= rhs.mm) laerror("Mat.Mat incompatible matrices");
#endif
	complex<double> dot;
	cblas_zdotc_sub(nn*mm, (void *)(*this)[0], 1, (void *)rhs[0], 1, 
			(void *)(&dot));
	return dot;
}

// Mat * Mat
const NRMat<double> NRMat<double>::operator*(const NRMat<double> &rhs) const
{
#ifdef DEBUG
	if (mm != rhs.nn) laerror("product of incompatible matrices");
#endif
	NRMat<double> result(nn, rhs.mm);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nn, rhs.mm, mm, 1.0,
			*this, mm, rhs, rhs.mm, 0.0, result, rhs.mm);
	return result;
}
const NRMat< complex<double> > 
NRMat< complex<double> >::operator*(const NRMat< complex<double> > &rhs) const
{
#ifdef DEBUG
	if (mm != rhs.nn) laerror("product of incompatible matrices");
#endif
	NRMat< complex<double> > result(nn, rhs.mm);
	cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nn, rhs.mm, mm,
			(const void *)(&CONE),(const void *)(*this)[0], mm, (const void *)rhs[0],
			rhs.mm, (const void *)(&CZERO), (void *)result[0], rhs.mm);
	return result;
}

// Multiply by diagonal from L
void NRMat<double>::diagmultl(const NRVec<double> &rhs)
{
#ifdef DEBUG
	if (nn != rhs.size()) laerror("incompatible matrix dimension in diagmultl");
#endif
	copyonwrite();
	for(int i=0; i<nn; i++) cblas_dscal(mm, rhs[i], (*this)[i], 1);
}
void NRMat< complex<double> >::diagmultl(const NRVec< complex<double> > &rhs)
{
#ifdef DEBUG
	if (nn != rhs.size()) laerror("incompatible matrix dimension in diagmultl");
#endif
	copyonwrite();
	for (int i=0; i<nn; i++) cblas_zscal(mm, &rhs[i], (*this)[i], 1);
}

// Multiply by diagonal from R
void NRMat<double>::diagmultr(const NRVec<double> &rhs)
{
#ifdef DEBUG
	if (mm != rhs.size()) laerror("incompatible matrix dimension in diagmultr");
#endif
	copyonwrite();
	for (int i=0; i<mm; i++) cblas_dscal(nn, rhs[i], (*this)[i], mm);
}
void NRMat< complex<double> >::diagmultr(const NRVec< complex<double> > &rhs)
{
#ifdef DEBUG
	if (mm != rhs.size()) laerror("incompatible matrix dimension in diagmultl");
#endif
	copyonwrite();
	for (int i=0; i<mm; i++) cblas_zscal(nn, &rhs[i], (*this)[i], mm);
}

// Mat * Smat, decomposed to nn x Vec * Smat
const NRMat<double> 
NRMat<double>::operator*(const NRSMat<double> &rhs) const
{
#ifdef DEBUG
	if (mm != rhs.nrows()) laerror("incompatible dimension in Mat*SMat");
#endif
	NRMat<double> result(nn, rhs.ncols());
	for (int i=0; i<nn; i++)
		cblas_dspmv(CblasRowMajor, CblasLower, mm, 1.0, &rhs[0], 
				(*this)[i], 1, 0.0, result[i], 1);
	return result;
}
const NRMat< complex<double> >
NRMat< complex<double> >::operator*(const NRSMat< complex<double> > &rhs) const
{
#ifdef DEBUG
	if (mm != rhs.nrows()) laerror("incompatible dimension in Mat*SMat");
#endif
	NRMat< complex<double> > result(nn, rhs.ncols());
	for (int i=0; i<nn; i++)
		cblas_zhpmv(CblasRowMajor, CblasLower, mm, (void *)&CONE, (void *)&rhs[0],
				(void *)(*this)[i], 1, (void *)&CZERO, (void *)result[i], 1);
	return result;
}

// Mat * Vec
const NRVec<double> 
NRMat<double>::operator*(const NRVec<double> &vec) const
{
#ifdef DEBUG
	if(mm != vec.size()) laerror("incompatible sizes in Mat*Vec");
#endif
	NRVec<double> result(nn);
	cblas_dgemv(CblasRowMajor, CblasNoTrans, nn, mm, 1.0, (*this)[0],
			mm, &vec[0], 1, 0.0, &result[0], 1);
	return result;
}
const NRVec< complex<double> >
NRMat< complex<double> >::operator*(const NRVec< complex<double> > &vec) const
{
#ifdef DEBUG
	if(mm != vec.size()) laerror("incompatible sizes in Mat*Vec");
#endif
	NRVec< complex<double> > result(nn);
	cblas_zgemv(CblasRowMajor, CblasNoTrans, nn, mm, (void *)&CONE, (void *)(*this)[0],
			mm, (void *)&vec[0], 1, (void *)&CZERO, (void *)&result[0], 1);
	return result;
}

// sum of rows
const NRVec<double> NRMat<double>::rsum() const
{
	NRVec<double> result(mm);
	for (int i=0; i<mm; i++) result[i] = cblas_dasum(nn,(*this)[0]+i,mm);
	return result;
}

// sum of columns
const NRVec<double> NRMat<double>::csum() const
{
	NRVec<double> result(nn);
	for (int i=0; i<nn; i++) result[i] = cblas_dasum(mm, (*this)[i], 1);
	return result;
}

// complex conjugate of Mat
NRMat<double> &NRMat<double>::conjugateme() {return *this;}

NRMat< complex<double> > & NRMat< complex<double> >::conjugateme()
{
	copyonwrite();
	cblas_dscal(mm*nn, -1.0, (double *)((*this)[0])+1, 2);
	return *this;
}

// transpose and optionally conjugate
const NRMat<double> NRMat<double>::transpose(bool conj) const
{
	NRMat<double> result(mm,nn);
	for(int i=0; i<nn; i++) cblas_dcopy(mm, (*this)[i], 1, result[0]+i, nn);
	return result;
}
const NRMat< complex<double> >
NRMat< complex<double> >::transpose(bool conj) const
{
	NRMat< complex<double> > result(mm,nn);
	for (int i=0; i<nn; i++) 
		cblas_zcopy(mm, (void *)(*this)[i], 1, (void *)(result[0]+i), nn);
	if (conj) cblas_dscal(mm*nn, -1.0, (double *)(result[0])+1, 2);
	return result;
}

// gemm : this = alpha*op( A )*op( B ) + beta*this
void NRMat<double>::gemm(const double &beta, const NRMat<double> &a,
		const char transa, const NRMat<double> &b, const char transb, 
		const double &alpha)
{
	int l(transa=='n'?a.nn:a.mm);
	int k(transa=='n'?a.mm:a.nn);
	int kk(transb=='n'?b.nn:b.mm);
	int ll(transb=='n'?b.mm:b.nn);

#ifdef DEBUG
	if (l!=nn || ll!=mm || k!=kk) laerror("incompatible matrices in Mat:gemm()");
#endif
	if (alpha==0.0 && beta==1.0) return;

	copyonwrite();
	cblas_dgemm(CblasRowMajor, (transa=='n' ? CblasNoTrans : CblasTrans),
			(transb=='n' ? CblasNoTrans : CblasTrans), nn, mm, k, alpha, a,
			a.mm, b , b.mm, beta, *this , mm);
}
void NRMat< complex<double> >::gemm(const complex<double> & beta,
		const NRMat< complex<double> > & a, const char transa, 
		const NRMat< complex<double> > & b, const char transb, 
		const complex<double> & alpha)
{
	int l(transa=='n'?a.nn:a.mm);
	int k(transa=='n'?a.mm:a.nn);
	int kk(transb=='n'?b.nn:b.mm);
	int ll(transb=='n'?b.mm:b.nn);

#ifdef DEBUG
	if (l!=nn || ll!=mm || k!=kk) laerror("incompatible matrices in Mat:gemm()");
#endif
	if (alpha==CZERO && beta==CONE) return;

	copyonwrite();
	cblas_zgemm(CblasRowMajor,
			(transa=='n' ? CblasNoTrans : (transa=='c'?CblasConjTrans:CblasTrans)), 
			(transb=='n' ? CblasNoTrans : (transa=='c'?CblasConjTrans:CblasTrans)),
			nn, mm, k, &alpha, a , a.mm, b , b.mm, &beta, *this , mm);
}

// norm of Mat
const double  NRMat<double>::norm(const double scalar) const
{
	if (!scalar) return cblas_dnrm2(nn*mm, (*this)[0], 1);
	double sum = 0;
	for (int i=0; i<nn; i++)
		for (int j=0; j<mm; j++) {
			register double tmp;
#ifdef MATPTR
			tmp = v[i][j];
#else
			tmp = v[i*mm+j];
#endif
			if (i==j) tmp -= scalar;
			sum += tmp*tmp;
		}
	return sqrt(sum);
}
const double  NRMat< complex<double> >::norm(const complex<double> scalar) const
{
	if (scalar == CZERO) return cblas_dznrm2(nn*mm, (*this)[0], 1);
	double sum = 0;
	for (int i=0; i<nn; i++)
		for (int j=0; j<mm; j++) {
			register complex<double> tmp;
#ifdef MATPTR
			tmp = v[i][j];
#else
			tmp = v[i*mm+j];
#endif
			if (i==j) tmp -= scalar;
			sum += tmp.real()*tmp.real()+tmp.imag()*tmp.imag();
		}
	return sqrt(sum);
}

// axpy: this = a * Mat
void NRMat<double>::axpy(const double alpha, const NRMat<double> &mat)
{
#ifdef DEBUG
	if (nn!=mat.nn || mm!=mat.mm) laerror("daxpy of incompatible matrices");
#endif
	copyonwrite();
	cblas_daxpy(nn*mm, alpha, mat, 1, *this, 1);
}
void NRMat< complex<double> >::axpy(const complex<double> alpha, 
		const NRMat< complex<double> > & mat)
{
#ifdef DEBUG
	if (nn!=mat.nn || mm!=mat.mm) laerror("zaxpy of incompatible matrices");
#endif
	copyonwrite();
	cblas_zaxpy(nn*mm, (void *)&alpha, mat, 1, (void *)(*this)[0], 1);
}

// trace of Mat
const double NRMat<double>::trace() const
{
#ifdef DEBUG
	if (nn != mm) laerror("no-square matrix in Mat::trace()");
#endif
	return cblas_dasum(nn, (*this)[0], nn+1);
}
const complex<double> NRMat< complex<double> >::trace() const
{
#ifdef DEBUG
	if (nn != mm) laerror("no-square matrix in Mat::trace()");
#endif
	register complex<double> sum = CZERO;
	for (int i=0; i<nn*nn; i+=(nn+1))
#ifdef MATPTR
		sum += v[0][i];
#else
		sum += v[i];
#endif
	return sum;
}

//////////////////////////////////////////////////////////////////////////////
//// forced instantization in the corespoding object file
#define INSTANTIZE(T) \
template ostream & operator<<(ostream &s, const NRMat< T > &x); \
template istream & operator>>(istream  &s, NRMat< T > &x); \

INSTANTIZE(double)
INSTANTIZE(complex<double>)
INSTANTIZE(int)
INSTANTIZE(char)


export template <typename T>
ostream& operator<<(ostream &s, const NRMat<T> &x)
                {
                int i,j,n,m;
                n=x.nrows();
                m=x.ncols();
                s << n << ' ' << m << '\n';
                for(i=0;i<n;i++)
                        {
                        for(j=0; j<m;j++) s << x[i][j] << (j==m-1 ? '\n' : ' '); // endl cannot be used in the conditional expression, since it is an overloaded function
                        }
                return s;
                }

export template <typename T>
istream& operator>>(istream  &s, NRMat<T> &x)
                {
                int i,j,n,m;
                s >> n >> m;
                x.resize(n,m);
                for(i=0;i<n;i++) for(j=0; j<m;j++) s>>x[i][j] ;
                return s;
                }
















