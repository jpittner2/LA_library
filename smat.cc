#include "smat.h"
// TODO
// specialize unary minus


//////////////////////////////////////////////////////////////////////////////
////// forced instantization in the corresponding object file
template NRSMat<double>;
template NRSMat< complex<double> >;
template NRSMat<int>;
template NRSMat<char>;



/*
 *  * Templates first, specializations for BLAS next
 *
 */

// conversion ctor, symmetrize general Mat into SMat
template <typename T>
NRSMat<T>::NRSMat(const NRMat<T> &rhs)
{
#ifdef DEBUG
	if (nn != rhs.ncols()) laerror("attempt to convert non-square Mat to SMat");
#endif
	count = new int;
	*count = 1;
	v = new T[NN2];
	int i, j, k=0;
	for (i=0; i<nn; i++)
		for (j=0; j<=i;j++) v[k++] = (rhs[i][j] + rhs[j][i])/((T)2);
}



// assign to diagonal
template <typename T>
NRSMat<T> & NRSMat<T>::operator=(const T &a)
{
	copyonwrite();
	for (int i=0; i<nn; i++) v[i*(i+1)/2+i] = a;
	return *this;
}

//get diagonal
template <typename T>
void NRSMat<T>::diagonalof(NRVec<T> &r, const bool divide) const
{
#ifdef DEBUG
if(r.size()!=nn) laerror("incompatible vector in diagonalof()");
#endif

r.copyonwrite();

if (divide)
	for (int i=0; i<nn; i++) {T a =v[i*(i+1)/2+i]; if(a!=0.) r[i] /= a;}
else
        for (int i=0; i<nn; i++) r[i] = v[i*(i+1)/2+i];
}


// unary minus
template <typename T>
const NRSMat<T> NRSMat<T>::operator-() const
{
	NRSMat<T> result(nn);
	for(int i=0; i<NN2; i++) result.v[i]= -v[i];
	return result;
}

// trace of Smat
template <typename T>
const T NRSMat<T>::trace() const
{
	T tmp = 0;
	for (int i=0; i<nn; i++) tmp += v[i*(i+1)/2+i];
	return tmp;
}



// write matrix to the file with specific format
template <typename T>
void NRSMat<T>::fprintf(FILE *file, const char *format, const int modulo) const
{
	lawritemat(file, (const T *)(*this) ,nn, nn, format, 2, modulo, 1);
}

// read matrix from the file with specific format
template <typename T>
void NRSMat<T>::fscanf(FILE *f, const char *format)
{
	int n, m;
	if (std::fscanf(f,"%d %d",&n,&m) != 2)
		laerror("cannot read matrix dimensions in SMat::fscanf");
	if (n != m) laerror("different dimensions of SMat");
	resize(n);
	for (int i=0; i<n; i++) 
		for (int j=0; j<n; j++)
			if (std::fscanf(f,format,&((*this)(i,j))) != 1)
				laerror("Smat - cannot read matrix element");
}


/*
 * BLAS specializations for double and complex<double>
 */

// SMat * Mat
const NRMat<double> NRSMat<double>::operator*(const NRMat<double> &rhs) const
{
#ifdef DEBUG
	if (nn != rhs.nrows()) laerror("incompatible dimensions in SMat*Mat");
#endif
	NRMat<double> result(nn, rhs.ncols());
	for (int k=0; k<rhs.ncols(); k++)
		cblas_dspmv(CblasRowMajor, CblasLower, nn, 1.0, v, rhs[0]+k, rhs.ncols(),
				0.0, result[0]+k, rhs.ncols());
	return result;
}
const NRMat< complex<double> >
NRSMat< complex<double> >::operator*(const NRMat< complex<double> > &rhs) const
{
#ifdef DEBUG
	if (nn != rhs.nrows()) laerror("incompatible dimensions in SMat*Mat");
#endif
	NRMat< complex<double> > result(nn, rhs.ncols());
	for (int k=0; k<rhs.ncols(); k++)
		cblas_zhpmv(CblasRowMajor, CblasLower, nn, &CONE, v, rhs[0]+k, rhs.ncols(),
				&CZERO, result[0]+k, rhs.ncols());
	return result;
}

// SMat * SMat
const NRMat<double> NRSMat<double>::operator*(const NRSMat<double> &rhs) const
{
#ifdef DEBUG
	if (nn != rhs.nn) laerror("incompatible dimensions in SMat*SMat");
#endif
	NRMat<double> result(0.0, nn, nn);
	double *p, *q;

	p = v;
	for (int i=0; i<nn;i++) {
		q = rhs.v;
		for (int k=0; k<=i; k++) {
			cblas_daxpy(k+1, *p++, q, 1, result[i], 1);
			q += k+1;
		}
	}

	p = v;
	for (int i=0; i<nn;i++) {
		q = rhs.v+1;
		for (int j=1; j<nn; j++) {
			result[i][j] += cblas_ddot(i+1<j ? i+1 : j, p, 1, q, 1);
			q += j+1;
		}
		p += i+1;
	}

	p = v; 
	q = rhs.v;
	for (int i=0; i<nn; i++) {
		cblas_dger(CblasRowMajor, i, i+1, 1., p, 1, q, 1, result, nn);
		p += i+1;
		q += i+1;
	}
	
	q = rhs.v+3;
	for (int j=2; j<nn; j++) {
		p = v+1;
		for (int i=1; i<j; i++) {
			cblas_daxpy(i, *++q, p, 1, result[0]+j, nn);
			p += i+1;
		}
		q += 2;
	}

	return result;
}
const NRMat< complex<double> > 
NRSMat< complex<double> >::operator*(const NRSMat< complex<double> > &rhs) const
{
#ifdef DEBUG
	if (nn != rhs.nn) laerror("incompatible dimensions in SMat*SMat");
#endif
	NRMat< complex<double> > result(0.0, nn, nn);
	NRMat< complex<double> > rhsmat(rhs);
	result = *this * rhsmat;
	return result;
//	laerror("complex SMat*Smat not implemented");
}
// S dot S
const double NRSMat<double>::dot(const NRSMat<double> &rhs) const
{
#ifdef DEBUG
	if (nn != rhs.nn) laerror("dot of incompatible SMat's");
#endif
	return cblas_ddot(NN2, v, 1, rhs.v, 1);
}
const complex<double> 
NRSMat< complex<double> >::dot(const NRSMat< complex<double> > &rhs) const
{
#ifdef DEBUG
	if (nn != rhs.nn) laerror("dot of incompatible SMat's");
#endif
	complex<double> dot;
	cblas_zdotc_sub(nn, (void *)v, 1, (void *)rhs.v, 1, (void *)(&dot));
	return dot;
}

// x = S * x
const NRVec<double> NRSMat<double>::operator*(const NRVec<double> &rhs) const
{
#ifdef DEBUG
	if (nn!=rhs.size()) laerror("incompatible dimension in Smat*Vec");
#endif
	NRVec<double> result(nn);
	cblas_dspmv(CblasRowMajor, CblasLower, nn, 1.0, v, rhs, 1, 0.0, result, 1);
	return result;
}
const NRVec< complex<double> >
NRSMat< complex<double> >::operator*(const NRVec< complex<double> > &rhs) const
{
#ifdef DEBUG
	if (nn!=rhs.size()) laerror("incompatible dimension in Smat*Vec");
#endif
	NRVec< complex<double> > result(nn);
	cblas_zhpmv(CblasRowMajor, CblasLower, nn, (void *)(&CONE), (void *)v, 
			(const void *)rhs, 1, (void *)(&CZERO), (void *)result, 1);
	return result;
}

// norm of the matrix
const double  NRSMat<double>::norm(const double scalar) const
{
	if (!scalar) return cblas_dnrm2(NN2, v, 1);
	double sum = 0;
	int k = 0;
	for (int i=0; i<nn; ++i)
		for (int j=0; j<=i; ++j) {
			register double tmp;
			tmp = v[k++];
			if (i == j) tmp -= scalar;
			sum += tmp*tmp;
		}
	return sqrt(sum);
}
const double
NRSMat< complex<double> >::norm(const complex<double> scalar) const
{
	if (!(scalar.real()) && !(scalar.imag()))
		return cblas_dznrm2(NN2, (void *)v, 1);
	double sum = 0;
	complex<double> tmp;
	int k = 0;
	for (int i=0; i<nn; ++i)
		for (int j=0; j<=i; ++j) {
			tmp = v[k++];
			if (i == j) tmp -= scalar;
			sum += tmp.real()*tmp.real() + tmp.imag()*tmp.imag();
		}
	return sqrt(sum);
}

// axpy: S = S * a
void NRSMat<double>::axpy(const double alpha, const NRSMat<double> & x)
{
#ifdef DEBUG
	if (nn != x.nn) laerror("axpy of incompatible SMats");
#endif
	copyonwrite();
	cblas_daxpy(NN2, alpha, x.v, 1, v, 1);
}
void NRSMat< complex<double> >::axpy(const complex<double> alpha,
			const NRSMat< complex<double> > & x)
{
#ifdef DEBUG
	if (nn != x.nn) laerror("axpy of incompatible SMats");
#endif
	copyonwrite();
	cblas_zaxpy(nn, (void *)(&alpha), (void *)x.v, 1, (void *)v, 1);
}


export template <typename T>
ostream& operator<<(ostream &s, const NRSMat<T> &x)
                {
                int i,j,n;
                n=x.nrows();
                s << n << ' ' << n << '\n';
                for(i=0;i<n;i++)
                        {
                        for(j=0; j<n;j++) s << x(i,j) << (j==n-1 ? '\n' : ' ');
                        }
                return s;
                }


export template <typename T>
istream& operator>>(istream  &s, NRSMat<T> &x)
                {
                int i,j,n,m;
                s >> n >> m;
                if(n!=m) laerror("input symmetric matrix not square");
                x.resize(n);
                for(i=0;i<n;i++) for(j=0; j<m;j++) s>>x(i,j);
                return s;
                }

//not implemented yet
const NRVec<int> NRSMat<int>::operator*(NRVec<int> const&rhs) const
{
laerror("NRSMat<int>::operator*(NRVec<int> const&) not implemented yet");
return rhs;
}

const NRVec<char> NRSMat<char>::operator*(NRVec<char> const&rhs) const
{
laerror("NRSMat<char>::operator*(NRVec<char> const&) not implemented yet");
return rhs;
}



//////////////////////////////////////////////////////////////////////////////
//// forced instantization in the corespoding object file
#define INSTANTIZE(T) \
template ostream & operator<<(ostream &s, const NRSMat< T > &x); \
template istream & operator>>(istream  &s, NRSMat< T > &x); \

INSTANTIZE(double)
INSTANTIZE(complex<double>)
INSTANTIZE(int)
INSTANTIZE(char)

