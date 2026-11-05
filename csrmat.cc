/*
    LA: linear algebra C++ interface library
    Copyright (C) 2008-2026 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>

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

#include <string>
#include <cmath>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include "csrmat.h"
#include "nonclass.h"
#include <algorithm>

namespace LA {

//raw pointers of possibly empty arrays (NULL when empty)
template<typename U> static inline const U *s_ptr(const NRVec<U> &x) {return x.size()? (const U *)x : NULL;}
template<typename U> static inline U *s_wptr(NRVec<U> &x) {return x.size()? (U *)x : NULL;}

template <typename T>
CSRMat<T>::CSRMat(const SPMatindex n, const SPMatindex m)
: nn(n), mm(m), symmetric(false), rowptr(n+1)
{
rowptr.clear();
}

template <typename T>
CSRMat<T>::CSRMat(const SPMatindex n)
: nn(n), mm(n), symmetric(false), rowptr(n+1)
{
rowptr.clear();
}

template <typename T>
CSRMat<T>::CSRMat(const SPMatindex n, const SPMatindex m, const NRVec<SPMatindex> &rp, const NRVec<SPMatindex> &c, const NRVec<T> &x)
: nn(n), mm(m), symmetric(false), rowptr(rp), col(c), v(x)
{
#ifdef DEBUG
if(rp.size()!=(int)n+1 || rp[0]!=0 || (int)rp[n]!=c.size() || c.size()!=x.size()) laerror("inconsistent arrays in CSRMat constructor");
for(SPMatindex i=0; i<n; ++i)
	{
	if(rp[i+1]<rp[i]) laerror("decreasing rowptr in CSRMat constructor");
	for(SPMatindex k=rp[i]; k<rp[i+1]; ++k) if(c[k]>=m || (k>rp[i] && c[k]<=c[k-1])) laerror("column index out of range or columns not ascending in CSRMat constructor");
	}
#endif
}

template <typename T>
void CSRMat<T>::copyonwrite(bool detachonly, bool deep)
{
if(rowptr.size()) rowptr.copyonwrite(detachonly);
if(col.size()) col.copyonwrite(detachonly);
if(v.size()) v.copyonwrite(detachonly,deep);
}

template <typename T>
void CSRMat<T>::resize(const SPMatindex n, const SPMatindex m)
{
nn=n; mm=m; symmetric=false;
col.resize(0); v.resize(0);
if(n==0 && m==0) {rowptr.resize(0); return;}
rowptr.resize(n+1);
rowptr.clear();
}

//counting sort, O(nnz+n+m); the columns of the result are ascending in each row also if they were not in the input
template <typename T>
const CSRMat<T> CSRMat<T>::transpose(bool conj) const
{
CSRMat<T> r(mm,nn);
r.symmetric=symmetric;
const int nz=v.size();
if(nz==0) return r;
const SPMatindex *rp=s_ptr(rowptr), *c=s_ptr(col);
const T *x=s_ptr(v);
r.col.resize(nz); r.v.resize(nz);
SPMatindex *rrp=r.rowptr, *rc=r.col;
T *rv=r.v;
for(int k=0; k<nz; ++k) ++rrp[c[k]+1];
for(SPMatindex j=0; j<mm; ++j) rrp[j+1]+=rrp[j];
for(SPMatindex i=0; i<nn; ++i) for(SPMatindex k=rp[i]; k<rp[i+1]; ++k)
	{
	SPMatindex p=rrp[c[k]]++;
	rc[p]=i;
	rv[p]= conj? LA_traits<T>::conjugate(x[k]) : x[k];
	}
for(SPMatindex j=mm; j>0; --j) rrp[j]=rrp[j-1]; //rrp[j] was advanced to the start of row j+1
rrp[0]=0;
return r;
}

//removes small elements; also sums duplicates (adjacent in sorted rows, only during the construction from coordinates)
template <typename T>
void CSRMat<T>::simplify(const typename LA_traits<T>::normtype eps)
{
const int nz=v.size();
if(nz==0) return;
copyonwrite();
SPMatindex *rp=rowptr, *c=col;
T *x=v;
int p=0, k=0;
for(SPMatindex i=0; i<nn; ++i)
	{
	const int end=rp[i+1];
	while(k<end)
		{
		const SPMatindex cc=c[k];
		T s=x[k++];
		while(k<end && c[k]==cc) s+=x[k++];
		if(std::abs(s)>eps) {c[p]=cc; x[p]=s; ++p;}
		}
	rp[i+1]=p;
	}
if(p<nz) {col.resize(p,true); v.resize(p,true);}
}

//from coordinate format: grouped by columns (i.e. the transposed matrix in CSR, rows unsorted), transposed back (rows sorted), duplicates summed
template <typename T>
void CSRMat<T>::fromcoo(const SPMatindex n, const SPMatindex m, const int nz, const SPMatindex *ri, const SPMatindex *ci, const T *vi)
{
CSRMat<T> t(m,n);
if(nz)
	{
	t.col.resize(nz); t.v.resize(nz);
	SPMatindex *trp=t.rowptr, *tc=t.col;
	T *tv=t.v;
	for(int k=0; k<nz; ++k)
		{
#ifdef DEBUG
		if(ri[k]>=n || ci[k]>=m) laerror("index out of range in CSRMat construction");
#endif
		++trp[ci[k]+1];
		}
	for(SPMatindex j=0; j<m; ++j) trp[j+1]+=trp[j];
	for(int k=0; k<nz; ++k) {SPMatindex p=trp[ci[k]]++; tc[p]=ri[k]; tv[p]=vi[k];}
	for(SPMatindex j=m; j>0; --j) trp[j]=trp[j-1];
	trp[0]=0;
	}
*this=t.transpose();
simplify();
}

//duplicates summed; a symmetric SparseMat (one triangle, complex symmetric) is unfolded without conjugation
template <typename T>
CSRMat<T>::CSRMat(const SparseMat<T> &rhs)
: nn(0), mm(0), symmetric(false)
{
const bool sym=rhs.issymmetric();
int nz=0;
for(const matel<T> *l=rhs.getlist(); l; l=l->next) nz+= (sym && l->row!=l->col)? 2:1;
NRVec<SPMatindex> ri(nz), ci(nz);
NRVec<T> vi(nz);
SPMatindex *r=s_wptr(ri), *c=s_wptr(ci);
T *x=s_wptr(vi);
int p=0;
for(const matel<T> *l=rhs.getlist(); l; l=l->next)
	{
	r[p]=l->row; c[p]=l->col; x[p++]=l->elem;
	if(sym && l->row!=l->col) {r[p]=l->col; c[p]=l->row; x[p++]=l->elem;}
	}
fromcoo(rhs.nrows(),rhs.ncols(),nz,r,c,x);
symmetric= sym && !LA_traits<T>::is_complex();
}

template <typename T>
CSRMat<T>::CSRMat(const SparseSMat<T> &rhs)
: nn(rhs.nrows()), mm(rhs.ncols()), symmetric(false), rowptr(nn+1)
{
const typename LA_traits<T>::normtype eps=LA_precision<typename LA_traits<T>::normtype>::sparse_epsilon();
typedef typename std::map<SPMatindex,T>::const_iterator IT;
int nz=0;
for(SPMatindex i=0; i<nn; ++i) {const std::map<SPMatindex,T> *l=rhs.line(i); if(l) for(IT q=l->begin(); q!=l->end(); ++q) if(std::abs(q->second)>eps) ++nz;}
col.resize(nz); v.resize(nz);
SPMatindex *rp=rowptr, *c=s_wptr(col);
T *x=s_wptr(v);
int p=0;
rp[0]=0;
for(SPMatindex i=0; i<nn; ++i)
	{
	const std::map<SPMatindex,T> *l=rhs.line(i);
	if(l) for(IT q=l->begin(); q!=l->end(); ++q) if(std::abs(q->second)>eps) {c[p]=q->first; x[p++]=q->second;}
	rp[i+1]=p;
	}
}

template <typename T>
CSRMat<T>::CSRMat(const NRMat<T> &rhs)
: nn(rhs.nrows()), mm(rhs.ncols()), symmetric(false), rowptr(nn+1)
{
const typename LA_traits<T>::normtype eps=LA_precision<typename LA_traits<T>::normtype>::sparse_epsilon();
int nz=0;
for(SPMatindex i=0; i<nn; ++i) {const T *a=rhs[i]; for(SPMatindex j=0; j<mm; ++j) if(std::abs(a[j])>eps) ++nz;}
col.resize(nz); v.resize(nz);
SPMatindex *rp=rowptr, *c=s_wptr(col);
T *x=s_wptr(v);
int p=0;
rp[0]=0;
for(SPMatindex i=0; i<nn; ++i)
	{
	const T *a=rhs[i];
	for(SPMatindex j=0; j<mm; ++j) if(std::abs(a[j])>eps) {c[p]=j; x[p++]=a[j];}
	rp[i+1]=p;
	}
}

//hermitian as NRSMat: the upper triangle is the conjugate of the stored lower one
template <typename T>
CSRMat<T>::CSRMat(const NRSMat<T> &rhs)
: nn(rhs.nrows()), mm(rhs.ncols()), symmetric(true), rowptr(nn+1)
{
const typename LA_traits<T>::normtype eps=LA_precision<typename LA_traits<T>::normtype>::sparse_epsilon();
int nz=0;
for(SPMatindex i=0; i<nn; ++i) for(SPMatindex j=0; j<nn; ++j) if(std::abs(rhs(i,j))>eps) ++nz;
col.resize(nz); v.resize(nz);
SPMatindex *rp=rowptr, *c=s_wptr(col);
T *x=s_wptr(v);
int p=0;
rp[0]=0;
for(SPMatindex i=0; i<nn; ++i)
	{
	for(SPMatindex j=0; j<nn; ++j)
		{
		const T a=rhs(i,j);
		if(std::abs(a)>eps) {c[p]=j; x[p++]= j>i? LA_traits<T>::conjugate(a) : a;}
		}
	rp[i+1]=p;
	}
}

template <typename T>
const T CSRMat<T>::operator()(const SPMatindex i, const SPMatindex j) const
{
#ifdef DEBUG
if(i>=nn || j>=mm) laerror("index out of range in CSRMat::operator()");
#endif
if(v.size()==0) return (T)0;
const SPMatindex *rp=rowptr, *c=col;
const T *x=v;
const SPMatindex *q=std::lower_bound(c+rp[i],c+rp[i+1],j);
if(q!=c+rp[i+1] && *q==j) return x[q-c];
return (T)0;
}

template <typename T>
void CSRMat<T>::put(int fd, bool dimensions, bool transp, bool orcaformat) const
{
if(orcaformat) laerror("ORCA format not supported for CSRMat");
if(transp) {transpose().put(fd,dimensions,false); return;}
errno=0;
if(dimensions)
	{
	if(sizeof(SPMatindex)!=write(fd,&nn,sizeof(SPMatindex))) laerror("cannot write() 1 in CSRMat::put");
	if(sizeof(SPMatindex)!=write(fd,&mm,sizeof(SPMatindex))) laerror("cannot write() 2 in CSRMat::put");
	}
int sym=symmetric;
if(sizeof(int)!=write(fd,&sym,sizeof(int))) laerror("cannot write() 3 in CSRMat::put");
rowptr.put(fd,true);
col.put(fd,true);
v.put(fd,true);
}

template <typename T>
void CSRMat<T>::get(int fd, bool dimensions, bool transp, bool orcaformat)
{
if(orcaformat) laerror("ORCA format not supported for CSRMat");
errno=0;
SPMatindex dim[2];
if(dimensions) {if(2*sizeof(SPMatindex)!=read(fd,&dim,2*sizeof(SPMatindex))) laerror("read() error 1 in CSRMat::get");}
else {dim[0]= transp?mm:nn; dim[1]= transp?nn:mm;}
int sym;
if(sizeof(int)!=read(fd,&sym,sizeof(int))) laerror("read() error 2 in CSRMat::get");
rowptr.get(fd,true);
col.get(fd,true);
v.get(fd,true);
nn=dim[0]; mm=dim[1]; symmetric=sym;
if(rowptr.size()!=(nn||mm ? (int)nn+1 : 0) || col.size()!=v.size()) laerror("inconsistent data in CSRMat::get");
if(transp) transposeme();
}

template <typename T>
CSRMat<T> & CSRMat<T>::operator*=(const T &a)
{
if(a==(T)0) {clear(); return *this;}
if(LA_traits<T>::imagpart(a)!=0) symmetric=false;
if(v.size()) v*=a;
return *this;
}

template <typename T>
CSRMat<T> & CSRMat<T>::operator=(const T &a)
{
if(nn!=mm) laerror("assignment of scalar to non-square CSRMat");
resize(nn,mm);
symmetric= LA_traits<T>::imagpart(a)==0;
if(nn==0 || std::abs(a)<=LA_precision<typename LA_traits<T>::normtype>::sparse_epsilon()) return *this;
col.resize(nn); v.resize(nn);
SPMatindex *rp=rowptr, *c=col;
T *x=v;
for(SPMatindex i=0; i<nn; ++i) {rp[i]=i; c[i]=i; x[i]=a;}
rp[nn]=nn;
return *this;
}

//in place if all diagonal elements are stored (only the values are copied if shared), otherwise via axpy
template <typename T>
CSRMat<T> & CSRMat<T>::operator+=(const T &a)
{
if(nn!=mm) laerror("addition of scalar to non-square CSRMat");
if(a==(T)0) return *this;
const SPMatindex *rp=s_ptr(rowptr), *c=s_ptr(col);
bool all= v.size()>0;
if(all) for(SPMatindex i=0; i<nn; ++i)
	{
	const SPMatindex *q=std::lower_bound(c+rp[i],c+rp[i+1],i);
	if(q==c+rp[i+1] || *q!=i) {all=false; break;}
	}
if(!all)
	{
	CSRMat<T> d(nn);
	d=a;
	const bool sym=symmetric;
	axpy((T)1,d);
	symmetric= sym && LA_traits<T>::imagpart(a)==0;
	return *this;
	}
if(LA_traits<T>::imagpart(a)!=0) symmetric=false;
v.copyonwrite();
T *x=v;
for(SPMatindex i=0; i<nn; ++i) x[std::lower_bound(c+rp[i],c+rp[i+1],i)-c]+=a;
return *this;
}

//merge of the sorted rows; for the same structure (shared or equal arrays) only the values by BLAS axpy
//the result is declared hermitian if both are (a matrix without elements counts as hermitian) and alpha is real
template <typename T>
void CSRMat<T>::axpy(const T alpha, const CSRMat<T> &x0, const bool transp)
{
if((!transp && (nn!=x0.nn || mm!=x0.mm)) || (transp && (nn!=x0.mm || mm!=x0.nn))) laerror("incompatible dimensions for axpy");
if(alpha==(T)0 || x0.v.size()==0) return;
CSRMat<T> xt;
if(transp) xt=x0.transpose();
const CSRMat<T> &x= transp? xt : x0;
const bool sym= (v.size()==0? nn==mm : symmetric) && x.symmetric && LA_traits<T>::imagpart(alpha)==0;
if(v.size()==0)
	{
	rowptr=x.rowptr; col=x.col; v=x.v;
	if(alpha!=(T)1) v*=alpha;
	symmetric=sym;
	return;
	}
const SPMatindex *rp=rowptr, *c=col, *xrp=x.rowptr, *xc=x.col;
const int nz=v.size(), xnz=x.v.size();
if(nz==xnz && ((rp==xrp && c==xc) || (!memcmp(rp,xrp,(nn+1)*sizeof(SPMatindex)) && !memcmp(c,xc,nz*sizeof(SPMatindex)))))
	{
	v.axpy(alpha,x.v);
	symmetric=sym;
	return;
	}
const T *a=v, *b=x.v;
NRVec<SPMatindex> nrp(nn+1);
SPMatindex *np=nrp;
np[0]=0;
for(SPMatindex i=0; i<nn; ++i)
	{
	SPMatindex p=rp[i], q=xrp[i], cnt=0;
	while(p<rp[i+1] && q<xrp[i+1]) {if(c[p]<xc[q]) ++p; else if(c[p]>xc[q]) ++q; else {++p; ++q;} ++cnt;}
	np[i+1]=np[i]+cnt+(rp[i+1]-p)+(xrp[i+1]-q);
	}
NRVec<SPMatindex> ncol(np[nn]);
NRVec<T> nv(np[nn]);
SPMatindex *nc=ncol;
T *y=nv;
int k=0;
for(SPMatindex i=0; i<nn; ++i)
	{
	SPMatindex p=rp[i], q=xrp[i];
	while(p<rp[i+1] || q<xrp[i+1])
		{
		if(q==xrp[i+1] || (p<rp[i+1] && c[p]<xc[q])) {nc[k]=c[p]; y[k++]=a[p++];}
		else if(p==rp[i+1] || c[p]>xc[q]) {nc[k]=xc[q]; y[k++]=alpha*b[q++];}
		else {nc[k]=c[p]; y[k++]=a[p++]+alpha*b[q++];}
		}
	}
rowptr=nrp; col=ncol; v=nv;
symmetric=sym;
}

template <typename T>
const typename LA_traits<T>::normtype CSRMat<T>::norm(const T scalar) const
{
if(scalar==(T)0) return v.size()? v.norm() : (typename LA_traits<T>::normtype)0;
if(nn!=mm) laerror("subtraction of scalar from non-square CSRMat in norm()");
const SPMatindex *rp=s_ptr(rowptr), *c=s_ptr(col);
const T *x=s_ptr(v);
typename LA_traits<T>::normtype sum=0;
for(SPMatindex i=0; i<nn; ++i)
	{
	bool d=false;
	if(x) for(SPMatindex k=rp[i]; k<rp[i+1]; ++k)
		{
		T e=x[k];
		if(c[k]==i) {e-=scalar; d=true;}
		sum+=LA_traits<T>::sqrabs(e);
		}
	if(!d) sum+=LA_traits<T>::sqrabs(scalar);
	}
return std::sqrt(sum);
}

template <typename T>
const T CSRMat<T>::dot(const CSRMat<T> &rhs) const
{
if(nn!=rhs.nn || mm!=rhs.mm) laerror("dot of incompatible CSRMat matrices");
if(v.size()==0 || rhs.v.size()==0) return (T)0;
const SPMatindex *rp=rowptr, *c=col, *xrp=rhs.rowptr, *xc=rhs.col;
if(rp==xrp && c==xc) return v.dot(rhs.v);
const T *a=v, *b=rhs.v;
T s=0;
for(SPMatindex i=0; i<nn; ++i)
	{
	SPMatindex p=rp[i], q=xrp[i];
	while(p<rp[i+1] && q<xrp[i+1])
		{
		if(c[p]<xc[q]) ++p;
		else if(c[p]>xc[q]) ++q;
		else s+=LA_traits<T>::conjugate(a[p++])*b[q++];
		}
	}
return s;
}

template <typename T>
const T CSRMat<T>::dot(const NRMat<T> &rhs) const
{
if((int)nn!=rhs.nrows() || (int)mm!=rhs.ncols()) laerror("dot of incompatible matrices");
if(v.size()==0) return (T)0;
const SPMatindex *rp=rowptr, *c=col;
const T *a=v;
T s=0;
for(SPMatindex i=0; i<nn; ++i)
	{
	const T *b=rhs[i];
	for(SPMatindex k=rp[i]; k<rp[i+1]; ++k) s+=LA_traits<T>::conjugate(a[k])*b[c[k]];
	}
return s;
}

template <typename T>
const T CSRMat<T>::trace() const
{
if(v.size()==0) return (T)0;
const SPMatindex *rp=rowptr, *c=col;
const T *x=v;
T s=0;
for(SPMatindex i=0; i<nn && i<mm; ++i)
	{
	const SPMatindex *q=std::lower_bound(c+rp[i],c+rp[i+1],i);
	if(q!=c+rp[i+1] && *q==i) s+=x[q-c];
	}
return s;
}

//sparse times sparse, row-wise with a dense accumulator: F. G. Gustavson, ACM Trans. Math. Softw. 4, 250 (1978)
//two passes (count, then compute); the columns of each result row are sorted; the result is not declared hermitian
template <typename T>
void CSRMat<T>::gemm(const T beta, const CSRMat<T> &a0, const char transa, const CSRMat<T> &b0, const char transb, const T alpha)
{
const char ta=tolower(transa), tb=tolower(transb);
CSRMat<T> at, bt;
if(ta!='n') at=a0.transpose(ta=='c');
if(tb!='n') bt=b0.transpose(tb=='c');
const CSRMat<T> &a= ta!='n'? at : a0;
const CSRMat<T> &b= tb!='n'? bt : b0;
if(a.mm!=b.nn) laerror("incompatible matrices in CSRMat::gemm");
if(beta!=(T)0 && (nn!=a.nn || mm!=b.mm)) laerror("incompatible result matrix in CSRMat::gemm");
const SPMatindex n=a.nn, m=b.mm;
CSRMat<T> r(n,m);
if(alpha!=(T)0 && a.v.size() && b.v.size())
	{
	const SPMatindex *arp=a.rowptr, *ac=a.col, *brp=b.rowptr, *bc=b.col;
	const T *av=a.v, *bv=b.v;
	NRVec<SPMatindex> marker(m);
	SPMatindex *mk=marker;
	for(SPMatindex j=0; j<m; ++j) mk[j]=n;
	SPMatindex *rp=r.rowptr;
	for(SPMatindex i=0; i<n; ++i)
		{
		SPMatindex cnt=0;
		for(SPMatindex k=arp[i]; k<arp[i+1]; ++k) for(SPMatindex q=brp[ac[k]]; q<brp[ac[k]+1]; ++q) if(mk[bc[q]]!=i) {mk[bc[q]]=i; ++cnt;}
		rp[i+1]=rp[i]+cnt;
		}
	r.col.resize(rp[n]); r.v.resize(rp[n]);
	if(rp[n])
		{
		SPMatindex *c=r.col;
		T *y=r.v;
		NRVec<T> acc(m);
		T *ac2=acc;
		for(SPMatindex j=0; j<m; ++j) mk[j]=n;
		for(SPMatindex i=0; i<n; ++i)
			{
			SPMatindex len=0;
			for(SPMatindex k=arp[i]; k<arp[i+1]; ++k)
				{
				const T x=av[k];
				for(SPMatindex q=brp[ac[k]]; q<brp[ac[k]+1]; ++q)
					{
					const SPMatindex j=bc[q];
					if(mk[j]!=i) {mk[j]=i; c[rp[i]+len++]=j; ac2[j]=x*bv[q];}
					else ac2[j]+=x*bv[q];
					}
				}
			std::sort(c+rp[i],c+rp[i]+len);
			for(SPMatindex t=rp[i]; t<rp[i]+len; ++t) y[t]=alpha*ac2[c[t]];
			}
		}
	}
if(beta==(T)0) *this=r;
else
	{
	*this*=beta;
	axpy((T)1,r);
	}
symmetric=false;
}

template <typename T>
CSRMat<T> & CSRMat<T>::oplusequal(const CSRMat<T> &rhs)
{
const bool sym= (v.size()==0? nn==mm : symmetric) && rhs.symmetric;
const SPMatindex n=nn+rhs.nn, m=mm+rhs.mm;
const int nz=v.size(), rnz=rhs.v.size();
NRVec<SPMatindex> nrp(n+1), ncol(nz+rnz);
NRVec<T> nv(nz+rnz);
SPMatindex *np=nrp, *nc=s_wptr(ncol);
T *y=s_wptr(nv);
const SPMatindex *rp=s_ptr(rowptr), *c=s_ptr(col), *rrp=s_ptr(rhs.rowptr), *rc=s_ptr(rhs.col);
const T *x=s_ptr(v), *rx=s_ptr(rhs.v);
for(SPMatindex i=0; i<=nn; ++i) np[i]= rp? rp[i] : 0;
for(SPMatindex i=1; i<=rhs.nn; ++i) np[nn+i]=nz+rrp[i];
for(int k=0; k<nz; ++k) {nc[k]=c[k]; y[k]=x[k];}
for(int k=0; k<rnz; ++k) {nc[nz+k]=mm+rc[k]; y[nz+k]=rx[k];}
nn=n; mm=m;
rowptr=nrp; col=ncol; v=nv;
symmetric=sym;
return *this;
}

template <typename T>
const CSRMat<T> CSRMat<T>::otimes(const CSRMat<T> &rhs) const
{
CSRMat<T> r(nn*rhs.nn,mm*rhs.mm);
r.symmetric= symmetric && rhs.symmetric;
const int nz=v.size(), rnz=rhs.v.size();
if(nz==0 || rnz==0) return r;
const SPMatindex *rp=rowptr, *c=col, *brp=rhs.rowptr, *bc=rhs.col;
const T *x=v, *bx=rhs.v;
r.col.resize(nz*rnz); r.v.resize(nz*rnz);
SPMatindex *np=r.rowptr, *nc=r.col;
T *y=r.v;
int p=0;
for(SPMatindex i=0; i<nn; ++i) for(SPMatindex k=0; k<rhs.nn; ++k)
	{
	for(SPMatindex a=rp[i]; a<rp[i+1]; ++a) for(SPMatindex b=brp[k]; b<brp[k+1]; ++b) {nc[p]=c[a]*rhs.mm+bc[b]; y[p++]=x[a]*bx[b];}
	np[i*rhs.nn+k+1]=p;
	}
return r;
}

//row r becomes row to[r]: the rows are moved as blocks, their columns stay sorted
template <typename T>
void CSRMat<T>::permrows(const SPMatindex *to)
{
const int nz=v.size();
NRVec<SPMatindex> nrp(nn+1);
SPMatindex *np=nrp;
const SPMatindex *rp=rowptr;
np[0]=0;
for(SPMatindex r=0; r<nn; ++r) np[to[r]+1]=rp[r+1]-rp[r];
for(SPMatindex r=0; r<nn; ++r) np[r+1]+=np[r];
if(nz)
	{
	NRVec<SPMatindex> ncol(nz);
	NRVec<T> nv(nz);
	SPMatindex *nc=ncol;
	T *y=nv;
	const SPMatindex *c=col;
	const T *x=v;
	for(SPMatindex r=0; r<nn; ++r) for(SPMatindex k=rp[r], t=np[to[r]]; k<rp[r+1]; ++k, ++t) {nc[t]=c[k]; y[t]=x[k];}
	col=ncol; v=nv;
	}
rowptr=nrp;
}

template <typename T>
void CSRMat<T>::permuteme_rows(const NRPerm<int> &p, const bool inverse)
{
if((SPMatindex)p.size()!=nn) laerror("inconsistent dimension in permuteme_rows");
NRPerm<int> pinv;
if(inverse) pinv=p.inverse();
const NRPerm<int> &pp= inverse?pinv:p;
NRVec<SPMatindex> to(nn);
for(SPMatindex r=0; r<nn; ++r) to[r]=pp[r+1]-1;
if(nn) permrows(to);
symmetric=false;
}

//columns via the transposed matrix (counting sorts keep the rows sorted)
template <typename T>
void CSRMat<T>::permuteme_cols(const NRPerm<int> &p, const bool inverse)
{
if((SPMatindex)p.size()!=mm) laerror("inconsistent dimension in permuteme_cols");
*this=transpose();
permuteme_rows(p,inverse);
*this=transpose();
}

template <typename T>
void CSRMat<T>::permuteme_both(const NRPerm<int> &p, const NRPerm<int> &q, const bool inverse)
{
const bool sym= symmetric && p==q;
permuteme_rows(p,inverse);
permuteme_cols(q,inverse);
symmetric=sym;
}

template <typename T>
void CSRMat<T>::permuterows(const NRVec<SPMatindex> &p)
{
if((SPMatindex)p.size()!=nn) laerror("inconsistent dimension in permuterows");
if(nn) permrows(p);
symmetric=false;
}

template <typename T>
void CSRMat<T>::permutecolumns(const NRVec<SPMatindex> &p)
{
if((SPMatindex)p.size()!=mm) laerror("inconsistent dimension in permutecolumns");
*this=transpose();
permuterows(p);
*this=transpose();
}

template <typename T>
void CSRMat<T>::permuteindices(const NRVec<SPMatindex> &p)
{
if((SPMatindex)p.size()!=nn || (SPMatindex)p.size()!=mm) laerror("inconsistent dimension in permuteindices");
const bool sym=symmetric;
permuterows(p);
permutecolumns(p);
symmetric=sym;
}

//square: diagonal (binary search in each row); non-square: diagonal of A^T A (for preconditioning as SparseMat::diagonalof); cache is ignored
template <typename T>
const T* CSRMat<T>::diagonalof(NRVec<T> &r, const bool divide, bool cache) const
{
#ifdef DEBUG
if((int)mm!=r.size()) laerror("incompatible vector size in diagonalof()");
#endif
NOT_GPU(r);
r.copyonwrite();
T *rr=r;
const SPMatindex *rp=s_ptr(rowptr), *c=s_ptr(col);
const T *x=s_ptr(v);
if(nn==mm)
	{
	for(SPMatindex i=0; i<nn; ++i)
		{
		T d=0;
		if(x)
			{
			const SPMatindex *q=std::lower_bound(c+rp[i],c+rp[i+1],i);
			if(q!=c+rp[i+1] && *q==i) d=x[q-c];
			}
		if(divide) {if(d!=(T)0) rr[i]/=d;}
		else rr[i]=d;
		}
	}
else
	{
	NRVec<T> d(mm);
	d.clear();
	T *dd=d;
	for(int k=0; k<v.size(); ++k) dd[c[k]]+=x[k]*x[k];
	if(divide) {for(SPMatindex j=0; j<mm; ++j) if(dd[j]!=(T)0) rr[j]/=dd[j];}
	else for(SPMatindex j=0; j<mm; ++j) rr[j]=dd[j];
	}
return divide?NULL:rr;
}

//r = beta r + alpha op(A) x, op = 'n', 't' or 'c' (conjugate transpose, the same as 't' for real types)
template <typename T>
void NRVec<T>::gemv(const T beta, const CSRMat<T> &a, const char trans, const T alpha, const NRVec<T> &x)
{
const char tr=tolower(trans);
const bool tp= tr!='n';
const bool cj= tr=='c' && LA_traits<T>::is_complex();
if((tp?a.nrows():a.ncols())!=(SPMatindex)x.size()) laerror("incompatible sizes in gemv");
#ifdef DEBUG
if((tp?a.ncols():a.nrows())!=(SPMatindex)nn) laerror("incompatible result vector in gemv");
#endif
NOT_GPU(*this);
NOT_GPU(x);
copyonwrite();
if(beta==(T)0) clear();
else if(beta!=(T)1) (*this)*=beta;
if(alpha==(T)0 || a.length()==0) return;
const SPMatindex *rp=a.getrowptr(), *c=a.getcol();
const T *av=a.getval();
const T *xv=x.v;
const SPMatindex n=a.nrows();
if(!tp)
	for(SPMatindex i=0; i<n; ++i)
		{
		T s=0;
		for(SPMatindex k=rp[i]; k<rp[i+1]; ++k) s+=av[k]*xv[c[k]];
		v[i]+=alpha*s;
		}
else
	for(SPMatindex i=0; i<n; ++i)
		{
		const T ax=alpha*xv[i];
		if(cj) for(SPMatindex k=rp[i]; k<rp[i+1]; ++k) v[c[k]]+=LA_traits<T>::conjugate(av[k])*ax;
		else for(SPMatindex k=rp[i]; k<rp[i+1]; ++k) v[c[k]]+=av[k]*ax;
		}
}

//vector from a matrix with one row or one column
template <typename T>
NRVec<T>::NRVec(const CSRMat<T> &rhs)
: NRVec(rhs.nrows()>1?rhs.nrows():rhs.ncols())
{
if(rhs.nrows()>1 && rhs.ncols()>1) laerror("cannot construct a vector from a sparse matrix with more than one row/column");
clear();
const SPMatindex *rp=s_ptr(rhs.getrowptr()), *c=s_ptr(rhs.getcol());
const T *x=s_ptr(rhs.getval());
if(!x) return;
if(rhs.nrows()>1) {for(SPMatindex i=0; i<rhs.nrows(); ++i) if(rp[i+1]>rp[i]) v[i]=x[rp[i]];}
else for(SPMatindex k=0; k<rp[1]; ++k) v[c[k]]=x[k];
}

//conversions to the other classes

template <typename T>
NRMat<T>::NRMat(const CSRMat<T> &rhs)
{
nn=rhs.nrows();
mm=rhs.ncols();
count=new int(1);
#ifdef CUDALA
location=cpu;
#endif
T *p;
#ifdef MATPTR
	v= new T*[nn];
	p=v[0] = new T[mm*nn];
	for (int i=1; i< nn; i++) v[i] = v[i-1] + mm;
#else
	p= v = new T[mm*nn];
#endif
memset(p,0,nn*mm*sizeof(T));
const SPMatindex *rp=s_ptr(rhs.getrowptr()), *c=s_ptr(rhs.getcol());
const T *x=s_ptr(rhs.getval());
if(x) for(int i=0; i<nn; ++i) for(SPMatindex k=rp[i]; k<rp[i+1]; ++k) p[(size_t)i*mm+c[k]]=x[k];
}

template <typename T>
NRMat<T> & NRMat<T>::operator+=(const CSRMat<T> &rhs)
{
if(nn!=(int)rhs.nrows() || mm!=(int)rhs.ncols()) laerror("incompatible matrices in +=");
NOT_GPU(*this);
copyonwrite();
const SPMatindex *rp=s_ptr(rhs.getrowptr()), *c=s_ptr(rhs.getcol());
const T *x=s_ptr(rhs.getval());
if(x) for(int i=0; i<nn; ++i) {T *a=(*this)[i]; for(SPMatindex k=rp[i]; k<rp[i+1]; ++k) a[c[k]]+=x[k];}
return *this;
}

template <typename T>
NRMat<T> & NRMat<T>::operator-=(const CSRMat<T> &rhs)
{
if(nn!=(int)rhs.nrows() || mm!=(int)rhs.ncols()) laerror("incompatible matrices in -=");
NOT_GPU(*this);
copyonwrite();
const SPMatindex *rp=s_ptr(rhs.getrowptr()), *c=s_ptr(rhs.getcol());
const T *x=s_ptr(rhs.getval());
if(x) for(int i=0; i<nn; ++i) {T *a=(*this)[i]; for(SPMatindex k=rp[i]; k<rp[i+1]; ++k) a[c[k]]-=x[k];}
return *this;
}

//C = beta C + alpha op(A) op(B), A sparse: rows of op(B) accumulated (op(B) transposed first if needed)
template <typename T>
void NRMat<T>::gemm(const T &beta, const CSRMat<T> &a, const char transa, const NRMat<T> &b, const char transb, const T &alpha)
{
const char ta=tolower(transa), tb=tolower(transb);
const bool tpa= ta!='n', cja= ta=='c' && LA_traits<T>::is_complex();
const int an= tpa? a.ncols() : a.nrows(), ak= tpa? a.nrows() : a.ncols();
const int bk= tb!='n'? b.ncols() : b.nrows(), bm= tb!='n'? b.nrows() : b.ncols();
if(ak!=bk) laerror("incompatible matrices in NRMat::gemm with CSRMat");
#ifdef DEBUG
if(nn!=an || mm!=bm) laerror("incompatible result matrix in NRMat::gemm with CSRMat");
#endif
NOT_GPU(*this);
copyonwrite();
if(beta==(T)0) clear();
else if(beta!=(T)1) (*this)*=beta;
if(alpha==(T)0 || a.length()==0) return;
NRMat<T> bt;
if(tb!='n') bt=b.transpose(tb=='c');
const NRMat<T> &bb= tb!='n'? bt : b;
const SPMatindex *rp=a.getrowptr(), *c=a.getcol();
const T *x=a.getval();
for(SPMatindex i=0; i<a.nrows(); ++i) for(SPMatindex k=rp[i]; k<rp[i+1]; ++k)
	{
	const T s= alpha*(cja? LA_traits<T>::conjugate(x[k]) : x[k]);
	T *d= (*this)[tpa? c[k] : i];
	const T *e= bb[tpa? i : c[k]];
	for(int j=0; j<mm; ++j) d[j]+=s*e[j];
	}
}

template <typename T>
const NRMat<T> NRMat<T>::operator*(const CSRMat<T> &rhs) const
{
if((SPMatindex)mm!=rhs.nrows()) laerror("incompatible matrices in NRMat*CSRMat");
NOT_GPU(*this);
NRMat<T> r(nn,rhs.ncols());
r.clear();
if(rhs.length()==0) return r;
const SPMatindex *rp=rhs.getrowptr(), *c=rhs.getcol();
const T *x=rhs.getval();
for(int i=0; i<nn; ++i)
	{
	const T *a=(*this)[i];
	T *d=r[i];
	for(int k=0; k<mm; ++k) if(a[k]!=(T)0) for(SPMatindex q=rp[k]; q<rp[k+1]; ++q) d[c[q]]+=a[k]*x[q];
	}
return r;
}

//lower triangle, as from SparseSMat
template <typename T>
NRSMat<T>::NRSMat(const CSRMat<T> &rhs)
: nn(rhs.nrows())
{
if(rhs.nrows()!=rhs.ncols()) laerror("cannot transform rectangular matrix to NRSMat");
#ifdef CUDALA
location=cpu;
#endif
count=new int(1);
v=new T[NN2];
memset(v,0,NN2*sizeof(T));
const SPMatindex *rp=s_ptr(rhs.getrowptr()), *c=s_ptr(rhs.getcol());
const T *x=s_ptr(rhs.getval());
if(x) for(int i=0; i<nn; ++i) for(SPMatindex k=rp[i]; k<rp[i+1] && (int)c[k]<=i; ++k) v[(size_t)i*(i+1)/2+c[k]]=x[k];
}

template <typename T>
SparseMat<T>::SparseMat(const CSRMat<T> &rhs)
: nn(rhs.nrows()), mm(rhs.ncols()), symmetric(0), nonzero(0), count(new int(1)), list(NULL), rowsorted(NULL), colsorted(NULL)
{
for(typename CSRMat<T>::iterator p=rhs.begin(); p.notend(); ++p) add(p->row,p->col,p->elem);
}

template <typename T>
SparseSMat<T>::SparseSMat(const CSRMat<T> &rhs)
: nn(rhs.nrows()), mm(rhs.ncols()), count(new int(1))
{
v= new std::map<SPMatindex,T> * [nn];
memset(v,0,nn*sizeof(std::map<SPMatindex,T> *));
for(typename CSRMat<T>::iterator p=rhs.begin(); p.notend(); ++p) add(p->row,p->col,p->elem,false);
}


//real and imaginary parts, complex matrices from real ones (nonclass.h): the index arrays stay shared,
//unless elements which became zero have to be removed
template<typename R>
static const CSRMat<R> s_part(const CSRMat<std::complex<R> > &a, const NRVec<R> &x, const bool sym)
{
CSRMat<R> r(a.nrows(),a.ncols());
if(a.length())
	{
	r=CSRMat<R>(a.nrows(),a.ncols(),a.getrowptr(),a.getcol(),x);
	const R eps=LA_precision<R>::sparse_epsilon();
	const R *y=x;
	for(int k=0; k<x.size(); ++k) if(std::abs(y[k])<=eps) {r.simplify(); break;}
	}
if(sym && a.nrows()==a.ncols()) r.setsymmetric();
return r;
}

template<typename R>
static const CSRMat<std::complex<R> > s_cmplx(const CSRMat<R> &a, const bool imag)
{
CSRMat<std::complex<R> > r(a.nrows(),a.ncols());
if(a.length()) r=CSRMat<std::complex<R> >(a.nrows(),a.ncols(),a.getrowptr(),a.getcol(),NRVec<std::complex<R> >(a.getval(),imag));
if(!imag && a.issymmetric()) r.setsymmetric();
return r;
}

template<>
const CSRMat<double> realpart<CSRMat<std::complex<double> > >(const CSRMat<std::complex<double> > &a)
{return s_part(a,a.length()? realpart(a.getval()) : NRVec<double>(),a.issymmetric());}

template<>
const CSRMat<float> realpart<CSRMat<std::complex<float> > >(const CSRMat<std::complex<float> > &a)
{return s_part(a,a.length()? realpart(a.getval()) : NRVec<float>(),a.issymmetric());}

template<>
const CSRMat<double> imagpart<CSRMat<std::complex<double> > >(const CSRMat<std::complex<double> > &a)
{return s_part(a,a.length()? imagpart(a.getval()) : NRVec<double>(),false);}

template<>
const CSRMat<float> imagpart<CSRMat<std::complex<float> > >(const CSRMat<std::complex<float> > &a)
{return s_part(a,a.length()? imagpart(a.getval()) : NRVec<float>(),false);}

template<>
const CSRMat<std::complex<double> > realmatrix<CSRMat<double> >(const CSRMat<double> &a)
{return s_cmplx(a,false);}

template<>
const CSRMat<std::complex<float> > realmatrix<CSRMat<float> >(const CSRMat<float> &a)
{return s_cmplx(a,false);}

template<>
const CSRMat<std::complex<double> > imagmatrix<CSRMat<double> >(const CSRMat<double> &a)
{return s_cmplx(a,true);}

template<>
const CSRMat<std::complex<float> > imagmatrix<CSRMat<float> >(const CSRMat<float> &a)
{return s_cmplx(a,true);}

//re + i im: a BLAS axpy of the values if both have the same structure, otherwise a merge
template<>
const CSRMat<std::complex<double> > complexmatrix<CSRMat<double> >(const CSRMat<double> &re, const CSRMat<double> &im)
{CSRMat<std::complex<double> > r=s_cmplx(re,false); r.axpy(std::complex<double>(0,1),s_cmplx(im,false)); return r;}

template<>
const CSRMat<std::complex<float> > complexmatrix<CSRMat<float> >(const CSRMat<float> &re, const CSRMat<float> &im)
{CSRMat<std::complex<float> > r=s_cmplx(re,false); r.axpy(std::complex<float>(0,1),s_cmplx(im,false)); return r;}


#define INSTANTIZE(T) \
template class CSRMat<T>; \
template NRMat<T>::NRMat(const CSRMat<T> &rhs); \
template NRSMat<T>::NRSMat(const CSRMat<T> &rhs); \
template SparseMat<T>::SparseMat(const CSRMat<T> &rhs); \
template SparseSMat<T>::SparseSMat(const CSRMat<T> &rhs); \
template void NRVec<T>::gemv(const T beta, const CSRMat<T> &a, const char trans, const T alpha, const NRVec<T> &x); \
template NRVec<T>::NRVec(const CSRMat<T> &rhs); \
template NRMat<T> & NRMat<T>::operator+=(const CSRMat<T> &rhs); \
template NRMat<T> & NRMat<T>::operator-=(const CSRMat<T> &rhs); \
template void NRMat<T>::gemm(const T &beta, const CSRMat<T> &a, const char transa, const NRMat<T> &b, const char transb, const T &alpha); \
template const NRMat<T> NRMat<T>::operator*(const CSRMat<T> &rhs) const; \

INSTANTIZE(double)
INSTANTIZE(float)
INSTANTIZE(std::complex<double>)
INSTANTIZE(std::complex<float>)

}//namespace
