/*
    LA: linear algebra C++ interface library
    Copyright (C) 2021 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>

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

#include "vec.h"
#include "permutation.h"
#include "miscfunc.h"
#include <stdio.h>
#include <string.h>
#include <list>
#include "qsort.h"
#include "bitvector.h"


namespace LA {

template <typename T>
void NRPerm<T>::identity()
{
T n=this->size();
#ifdef DEBUG
        if(n<0) laerror("invalid permutation size");
#endif
if(n==0) return;
this->copyonwrite();
for(T i=1; i<=n; ++i) (*this)[i]=i;
}


template <typename T>
bool NRPerm<T>::is_identity() const
{
T n=this->size();
#ifdef DEBUG
        if(n<0) laerror("invalid permutation size");
#endif
if(n==0) return 1;
for(T i=1; i<=n; ++i) if((*this)[i]!=i) return 0;
return 1;
}



template <typename T>
bool NRPerm<T>::is_valid() const
{
T n = this->size();
if(n<0) return 0;
NRVec_from1<T> used(n);
used.clear();
for(T i=1; i<=n; ++i)
	{
	T x= (*this)[i];
	if(x<1||x>n) return 0;
	used[x] += 1;
	}
for(T i=1; i<=n; ++i) if(used[i]!=1) return 0;
return 1;
}

template <typename T>
NRPerm<T> NRPerm<T>::inverse() const
{
#ifdef DEBUG
	if(!this->is_valid()) laerror("inverse of invalid permutation");
#endif

NRPerm<T> q(this->size());
for(T i=1; i<=this->size(); ++i) q[(*this)[i]]=i;
return q;
}

template <typename T>
NRPerm<T> NRPerm<T>::reverse() const
{
#ifdef DEBUG
        if(!this->is_valid()) laerror("reverse of invalid permutation");
#endif

NRPerm<T> q(this->size());
for(T i=1; i<=this->size(); ++i) q[i]=(*this)[this->size()-i+1];
return q;
}


template <typename T>
NRPerm<T> NRPerm<T>::cutleft(int n) const
{
if(n<0||n>size()) laerror("illegal n in ::cutleft");
if(n==0) return *this;
for(int i=1; i<=n; ++i) if((*this)[i]!=i) laerror("left subset is not identity permutation");
if(n==size()) return NRPerm<T>();
NRPerm<T> res(size()-n);
for(int i=n+1; i<=size(); ++i) res[i-n] = (*this)[i]-n;
#ifdef DEBUG
if(!res.is_valid()) laerror("inetrnal error in cutleft");
#endif
return res;
}

template <typename T>
NRPerm<T> NRPerm<T>::cutright(int n) const
{
if(n<0||n>size()) laerror("illegal n in ::cutright");
if(n==0) return *this;
for(int i=size(); i>size()-n; --i) if((*this)[i]!=i) laerror("right subset is not identity permutation");
if(n==size()) return NRPerm<T>();
NRPerm<T> res(size()-n);
for(int i=1; i<=size()-n; ++i) res[i] = (*this)[i];
#ifdef DEBUG
if(!res.is_valid()) laerror("inetrnal error in cutleft");
#endif
return res;
}



template <typename T>
NRPerm<T> NRPerm<T>::operator&(const NRPerm<T> &q) const
{
#ifdef DEBUG
        if(!this->is_valid() || !q.is_valid()) laerror("concatenation of invalid permutation");
#endif

NRPerm<T> r(size()+q.size());
for(int i=1; i<=size(); ++i) r[i]=(*this)[i];
for(int i=1; i<=q.size(); ++i) r[size()+i]=size()+q[i];
return r;
}

template <typename T>
NRPerm<T> NRPerm<T>::operator|(const NRPerm<T> &q) const
{
#ifdef DEBUG
        if(!this->is_valid() || !q.is_valid()) laerror("concatenation of invalid permutation");
#endif

NRPerm<T> r(size()+q.size());
for(int i=1; i<=q.size(); ++i) r[i]=size()+q[i];
for(int i=1; i<=size(); ++i) r[q.size()+i]=(*this)[i];
return r;
}



template <typename T>
NRPerm<T> NRPerm<T>::operator*(const NRPerm<T> &q) const
{
#ifdef DEBUG
        if(!this->is_valid() || !q.is_valid()) laerror("multiplication of invalid permutation");
#endif

T n=this->size();
if(n!=q.size()) laerror("product of incompatible permutations");
NRPerm<T> r(n);
for(T i=1; i<=n; ++i) r[i] = (*this)[q[i]];
return r;
}

template <typename T>
NRPerm<T> NRPerm<T>::multiply(const NRPerm<T> &q, bool inverse) const
{
#ifdef DEBUG
        if(!this->is_valid() || !q.is_valid()) laerror("multiplication of invalid permutation");
#endif

T n=this->size();
if(n!=q.size()) laerror("product of incompatible permutations");
NRPerm<T> r(n);
if(inverse) for(T i=1; i<=n; ++i) r[q[i]] = (*this)[i];
else for(T i=1; i<=n; ++i) r[i] = (*this)[q[i]];
return r;
}



template <typename T>
NRPerm<T> NRPerm<T>::conjugate_by(const NRPerm<T> &q, bool inverse) const
{
#ifdef DEBUG
        if(!this->is_valid() || !q.is_valid()) laerror("multiplication of invalid permutation");
#endif

T n=this->size();
if(n!=q.size()) laerror("product of incompatible permutations");
NRPerm<T> qi=q.inverse();
NRPerm<T> r(n);
if(inverse)  for(T i=1; i<=n; ++i) r[i] = q[(*this)[qi[i]]];
else for(T i=1; i<=n; ++i) r[i] = qi[(*this)[q[i]]];
return r;
}

template <typename T>
NRPerm<T> NRPerm<T>::commutator(const NRPerm<T> &q, bool inverse) const
{
#ifdef DEBUG
        if(!this->is_valid() || !q.is_valid()) laerror("multiplication of invalid permutation");
#endif

T n=this->size();
if(n!=q.size()) laerror("product of incompatible permutations");
NRPerm<T> qi=q.inverse();
NRPerm<T> pi=this->inverse();
NRPerm<T> r(n);
if(inverse)  for(T i=1; i<=n; ++i) r[i] = qi[pi[q[(*this)[i]]]];
else for(T i=1; i<=n; ++i) r[i] = pi[qi[(*this)[q[i]]]];
return r;
}



template <typename T>
CyclePerm<T> CyclePerm<T>::conjugate_by(const CyclePerm<T> &q) const
{
#ifdef DEBUG
        if(!this->is_valid() || !q.is_valid()) laerror("multiplication of invalid permutation");
#endif 
return q.inverse()*((*this)*q);
}


//NOTE: for larger permutations it might be more efficient to use cycle decomposition but where exactly the breakeven is?
//at the moment we do it the trivial way for small permutations and using sort for larger ones
template <typename T>
int NRPerm<T>::parity() const
{
if(!this->is_valid()) return 0;
T n=this->size();
if(n==1) return 1;
if(n>=10) //???
	{
	NRPerm<T> tmp(*this);
	return (tmp.sort()&1) ? -1:1;
	}
T count=0;
for(T i=2;i<=n;i++) for(T j=1;j<i;j++) if((*this)[j]>(*this)[i]) count++;
return (count&1)? -1:1;
}

template <typename T>
NRPerm<T>::NRPerm(const CyclePerm<T> &rhs, const int n)
{
#ifdef DEBUG
        if(!rhs.is_valid()) laerror("invalid cycle permutation");
#endif
int m;
if(n) m=n; else m=rhs.max();
this->resize(m);

identity();
T ncycles=rhs.size();
for(T j=1; j<=ncycles; ++j)
        {
	T length= rhs[j].size();
        for(T i=1; i<=length; ++i) (*this)[rhs[j][i]] = rhs[j][i%length+1];
        }
#ifdef DEBUG
if(!is_valid()) laerror("internal error in NRPerm constructor from CyclePerm");
#endif
}

template <typename T>
void NRPerm<T>::randomize(void)
{
int n=this->size();
if(n<=0) laerror("cannot randomize empty permutation");
this->copyonwrite();
this->identity();
for(int i=n-1; i>=1; --i)
	{
	int j= RANDINT32()%(i+1);
	T tmp = (*this)[i+1];
	(*this)[i+1]=(*this)[j+1];
	(*this)[j+1]=tmp;
	}
}


template <typename T>
bool NRPerm<T>::next(void)
{
this->copyonwrite();
int n=this->size();
// Find longest non-increasing suffix and the pivot
int i = n;
while (i > 1 && (*this)[i-1] >= (*this)[i]) --i;
if (i<=1) return false;
T piv=(*this)[i-1];

// Find rightmost element greater than the pivot
int j = n;
while ((*this)[j] <= piv) --j;
// Now the value array[j] will become the new pivot, Assertion: j >= i
    
// Swap the pivot with j
(*this)[i - 1] = (*this)[j];
(*this)[j] = piv;
    
// Reverse the suffix
j = n;
while (i < j) {
        T temp = (*this)[i];
        (*this)[i] = (*this)[j];
        (*this)[j] = temp;
        i++;
        j--;
    }

return true;
}

//Algorithm L from Knuth's volume 4
template <typename T>
PERM_RANK_TYPE NRPerm<T>::generate_all(void (*callback)(const NRPerm<T>&), int select)
{
int n=this->size();
NRVec_from1<T> c((T)0,n);
NRVec_from1<T> d((T)1,n);
int j,k,s;
T q;
T t;

this->identity();
PERM_RANK_TYPE sumperm=0;

p2:
        sumperm++;
        if(callback) {if(!select || (select&1) == (sumperm&1)) {(*callback)(*this); this->copyonwrite();}}
        j=n; s=0;
p4:
        q=c[j]+d[j];
        if(q<0) goto p7;
        if(q==j) goto p6;

        t=(*this)[j-c[j]+s]; (*this)[j-c[j]+s]=(*this)[j-q+s]; (*this)[j-q+s]=t;
        c[j]=q;
        goto p2;
p6:
        if(j==1) goto end;
        s++;
p7:
        d[j]= -d[j];
        j--;
        goto p4;

end:
return select? sumperm/2:sumperm;
}



template <typename T> 
static PermutationAlgebra<T,T>  *list_all_return;

static PERM_RANK_TYPE list_all_index;

template <typename T> 
static void list_all_callback(const NRPerm<T> &p)
{
(*list_all_return<T>)[list_all_index].weight= (list_all_index&1)?-1:1;
(*list_all_return<T>)[list_all_index].perm=p;
(*list_all_return<T>)[list_all_index].perm.copyonwrite();
++list_all_index;
}

template <typename T>
PermutationAlgebra<T,T>  NRPerm<T>::list_all(int parity_sel)
{
PERM_RANK_TYPE number = factorial(this->size());
if(parity_sel) number/=2;
PermutationAlgebra<T,T>  ret(number);
list_all_return<T> = &ret;
list_all_index=0;
this->generate_all(list_all_callback,parity_sel);
return ret;
}




template <typename T, typename R>
bool PermutationAlgebra<T,R>::operator==(PermutationAlgebra<T,R> &rhs)
{
simplify();
rhs.simplify();
if(size()!=rhs.size()) return false;
for(int i=0; i<size(); ++i) if((*this)[i].weight!=rhs[i].weight || (*this)[i].perm!=rhs[i].perm) return false;
return true;
}



//Algorithm L2 from Knuth's volume 4 for multiset permutations
//input must be initialized with (repeated) numbers in any order
template <typename T>
PERM_RANK_TYPE NRPerm<T>::generate_all_multi(void (*callback)(const NRPerm<T>&))
{
PERM_RANK_TYPE sumperm=0;
this->copyonwrite();
myheapsort(this->size(),&(*this)[1],1);
while(1)
        {
        int j,k,l;
        T t;

        sumperm++;
        if(callback) (*callback)(*this);
        j=this->size()-1;
        while(j>0 && (*this)[j]>=(*this)[j+1]) j--;
        if(j==0) break; /*finished*/
        l=this->size();
        while((*this)[j]>=(*this)[l]) l--;
        t=(*this)[j];(*this)[j]=(*this)[l];(*this)[l]=t;
        k=j+1;
        l=this->size();
        while(k<l)
                {
                t=(*this)[k];(*this)[k]=(*this)[l];(*this)[l]=t;
                k++;l--;
                }
        }
return sumperm;
}



template <typename T>
static T _n;

template <typename T>
static void (*_callback)(const NRPerm<T>& ); 

static PERM_RANK_TYPE _sumperm;

template <typename T>
static NRPerm<T> *_perm;

template <typename T>
static void permg(T n)
{
if(n<= _n<T>)
        {
        for(T i=1;i<= _n<T>;i++)
                {
                if(!(*_perm<T>)[i]) //place not occupied
			{
                	(*_perm<T>)[i]=n;
                	permg(n+1);
                	(*_perm<T>)[i]=0;
			}
                }
        }
else
        {
        _sumperm++;
	if(_callback<T>) (*_callback<T>)(*_perm<T>);
        }
}


template <typename T>
PERM_RANK_TYPE NRPerm<T>::generate_all2(void (*callback)(const NRPerm<T>&))
{
this->copyonwrite();
this->clear();
_n<T> = this->size();
_callback<T> =callback;
_sumperm=0;
_perm<T> = this;
permg(1);
return _sumperm;
}


template <typename T>
PERM_RANK_TYPE NRPerm<T>::generate_all_lex(void (*callback)(const NRPerm<T>&))
{
PERM_RANK_TYPE np=0;
this->identity();
do{
++np;
if(callback) (*callback)(*this);
}while(this->next());
return np;
}

template <typename T>
PermutationAlgebra<T,T>  NRPerm<T>::list_all_lex()
{       
PERM_RANK_TYPE number = factorial(this->size());
PermutationAlgebra<T,T>  ret(number);
PERM_RANK_TYPE np=0;
this->identity();
do{
ret[np].perm = *this; ret[np].perm.copyonwrite();
ret[np].weight=0;
++np;
}while(this->next());
return ret;
}



template <typename T>
static T _n2;

template <typename T>
static void (*_callback2)(const NRPerm<T>& );

static PERM_RANK_TYPE _sumperm2;

template <typename T>
static NRPerm<T> *_perm2;

template <typename T>
static const T *pclasses2;


static int sameclass;

template <typename T>
static void permg2(T n) //place number n in a free box in all possible ways and according to restrictions
{
T i;
if(n<=_n2<T>)
        {
        T c=pclasses2<T>[n];
        for(i=1;i<=_n2<T>;i++)
                {
                if((*_perm2<T>)[i]>=1) goto skipme; /*place already occupied*/
                if (sameclass)
                        {
                        if(sameclass==1 && c!=pclasses2<T>[i]) goto skipme;
                        if(sameclass== -1 && c==pclasses2<T>[i] ) goto skipme;
                        if(sameclass== -2) /*allow only permutations which leave elements of each class sorted*/
                                {
                                T j,k;
                                for(j=i+1; j<=_n2<T>; j++)
                                        if((k=(*_perm2<T>)[j])>=1)
                                                 if(/* automatically fulfilled k<n &&*/ c==pclasses2<T>[k]) goto skipme;
                                }
                        }
                /*put it there and next number*/
                (*_perm2<T>)[i]=n;
                permg2(n+1);
                (*_perm2<T>)[i]=0;
                skipme:;
                }
        }
else
        {
        _sumperm2++;
	if(_callback2<T>) {(*_callback2<T>)(*_perm2<T>); _perm2<T> -> copyonwrite();}
        }
}




template <typename T>
PERM_RANK_TYPE NRPerm<T>::generate_restricted(void (*callback)(const NRPerm<T>&), const NRVec_from1<T> &classes, int restriction_type)
{
this->copyonwrite();
this->clear();
_n2<T> = this->size();
_callback2<T> =callback;
_sumperm2=0;
_perm2<T> = this;
pclasses2<T> = &classes[1]-1;
sameclass=restriction_type;
permg2<T>(1);
return _sumperm2;
}

template <typename T> 
static PermutationAlgebra<T,T>  *list_restricted_return;

static PERM_RANK_TYPE list_restricted_index;

static bool list_restricted_invert;

template <typename T> 
static void list_restricted_callback(const NRPerm<T> &p)
{
(*list_restricted_return<T>)[list_restricted_index].weight=p.parity();
(*list_restricted_return<T>)[list_restricted_index].perm= list_restricted_invert?p.inverse():p;
(*list_restricted_return<T>)[list_restricted_index].perm.copyonwrite();
++list_restricted_index;
}

template <typename T>
PermutationAlgebra<T,T>  NRPerm<T>::list_restricted(const NRVec_from1<T> &classes, int restriction_type, bool invert)
{
PERM_RANK_TYPE number = this->generate_restricted(NULL,classes,restriction_type);
PermutationAlgebra<T,T>  ret(number);
list_restricted_return<T> = &ret;
list_restricted_index=0;
list_restricted_invert=invert;
generate_restricted(list_restricted_callback,classes,restriction_type);
return ret;
}


template <typename T>
PERM_RANK_TYPE NRPerm<T>::rank() const
{
int c,i,k;
PERM_RANK_TYPE r;
int n= this->size();

r=0;
for (k=1; k<=n; ++k)
        {
        T l=(*this)[k];
        c=0;
        for(i=k+1;i<=n;++i) if((*this)[i]<l) ++c;
        r+= c;
	i=n-k;
        if(i) r*= i;
        }
return r;
}


template <typename T>
NRVec_from1<T> NRPerm<T>::inversions(const int type, PERM_RANK_TYPE *prank) const
{
PERM_RANK_TYPE s=0;
int n=this->size();
int j,k;
T l;

NRVec_from1<T> i(n);
i.clear();

switch(type) {
case 3:  /*number of elements right from p[j] smaller < p[j]*/
for(j=1;j<n;++j) /*number of elements right from j smaller <j*/
        {
        l=(*this)[j];
        for(k=n;k>j;--k)
                {
                if((*this)[k]<l) ++i[j];
                }
        }
        break;
case 2: /*number of elements left from p[j] bigger >p[j]*/
for(j=2;j<=n;++j) /*number of elements left from j bigger >j*/
        {
        l=(*this)[j];
        for(k=1;k<j;++k)
                {
                if((*this)[k]>l) ++i[j];
                }
        }
        break;
case 1:
for(j=1;j<=n;++j) /*number of elements right from j smaller <j*/
        {
        for(k=n;k>=1;--k)
                {
                if((*this)[k]==j) break;
                if((*this)[k]<j) ++i[j];
                }
        }
        break;
case 0:
for(j=1;j<=n;++j) /*number of elements left from j bigger >j*/
        {
        for(k=1;k<=n;++k)
                {
                if((*this)[k]==j) break;
                if((*this)[k]>j) ++i[j];
                }
        }
        break;
default: laerror("illegal type in inversions");

if(prank)
	{
	if(type!=3) laerror("rank can be computed from inversions only for type 3");
	l=1;
	for(j=1;j<n;++j)
                {
                l*= j;
                *prank += l*i[n-j];
                }
	}
}

return i;
}

template <typename T>
NRPerm<T>::NRPerm(const int type, const NRVec_from1<T> &i)
{
int n=i.size();
this->resize(n);

int k,l;
T j;

switch(type) {
case 2:
case 1:
        for(l=0,j=1; j<=n; ++j,++l)
                {
                /*shift left and place*/
                for(k=n-l+1; k<=n-i[j]; ++k) (*this)[k-1]=(*this)[k];
                (*this)[n-i[j]]=j;
                }
        break;
case 3:
case 0:
        for(l=0,j=n; j>0; --j,++l)
                {
                /*shift right and place*/
                for(k=l; k>=i[j]+1; --k) (*this)[k+1]=(*this)[k];
                (*this)[i[j]+1]=j;
                }
        break;
default: laerror("illegal type in nrperm from inversions");
}

if(type>=2) (*this) = this->inverse();
}



template <typename T>
NRPerm<T>::NRPerm(const int n, const PERM_RANK_TYPE rank)
{
this->resize(n);
NRVec_from1<T> inv(n) ;
#ifdef DEBUG
if(rank>=factorial(n)) laerror("illegal rank for this n");
#endif
inv[n]=0;
int k;
PERM_RANK_TYPE r=rank;
for(k=n-1; k>=0; --k)
        {
        PERM_RANK_TYPE t;
        t=factorial(k);
        inv[n-k]=r/t;
        r=r%t;
        }

*this =  NRPerm(3,inv);
}

template <typename T, typename R>
PermutationAlgebra<T,R> PermutationAlgebra<T,R>::cutleft(int n) const
{
PermutationAlgebra<T,R> res(this->size());
for(int i=0; i<this->size(); ++i) {res[i].perm = (*this)[i].perm.cutleft(n) ; res[i].weight=(*this)[i].weight;}
return res;
}

template <typename T, typename R>
PermutationAlgebra<T,R> PermutationAlgebra<T,R>::cutright(int n) const
{
PermutationAlgebra<T,R> res(this->size());
for(int i=0; i<this->size(); ++i) {res[i].perm = (*this)[i].perm.cutright(n) ; res[i].weight=(*this)[i].weight;}
return res;
}


template <typename T, typename R>
PermutationAlgebra<T,R> PermutationAlgebra<T,R>::operator&(const NRPerm<T> &rhs) const
{
PermutationAlgebra<T,R> res(this->size());
for(int i=0; i<this->size(); ++i) {res[i].perm = (*this)[i].perm & rhs; res[i].weight=(*this)[i].weight;}
return res;
}

template <typename T, typename R>
PermutationAlgebra<T,R> PermutationAlgebra<T,R>::operator|(const NRPerm<T> &rhs) const
{
PermutationAlgebra<T,R> res(this->size());
for(int i=0; i<this->size(); ++i)  {res[i].perm = (*this)[i].perm | rhs; res[i].weight=(*this)[i].weight;}
return res;
}

template <typename T, typename R>
PermutationAlgebra<T,R> PermutationAlgebra<T,R>::operator*(const NRPerm<T> &rhs) const
{
PermutationAlgebra<T,R> res(this->size());
for(int i=0; i<this->size(); ++i)  {res[i].perm = (*this)[i].perm * rhs; res[i].weight=(*this)[i].weight;}
return res;
}

template <typename T> template<typename R>
PermutationAlgebra<T,R> NRPerm<T>::operator*(const  PermutationAlgebra<T,R> &pa) const
{
PermutationAlgebra<T,R> res(pa.size());
for(int i=0; i<pa.size(); ++i) {res[i].perm = *this * pa[i]; res[i].weight= pa[i].weight;}
return res;
}




template <typename T, typename R>
PermutationAlgebra<T,R> PermutationAlgebra<T,R>::operator&(const PermutationAlgebra<T,R> &rhs) const
{
PermutationAlgebra<T,R> res(this->size()*rhs.size());
for(int i=0; i<this->size(); ++i)
	for(int j=0; j<rhs.size(); ++j)
		res[i*rhs.size()+j] = (*this)[i]&rhs[j];
return res;
}

template <typename T, typename R>
PermutationAlgebra<T,R> PermutationAlgebra<T,R>::operator|(const PermutationAlgebra<T,R> &rhs) const
{
PermutationAlgebra<T,R> res(this->size()*rhs.size());
for(int i=0; i<this->size(); ++i)
        for(int j=0; j<rhs.size(); ++j)
                res[i*rhs.size()+j] = (*this)[i]|rhs[j];
return res;
}


template <typename T, typename R>
PermutationAlgebra<T,R> PermutationAlgebra<T,R>::operator*(const PermutationAlgebra<T,R> &rhs) const
{
if(this->size()>0 && rhs.size()>0 && (*this)[0].perm.size()!=rhs[0].perm.size()) laerror("permutation sizes do not match");
PermutationAlgebra<T,R> res(this->size()*rhs.size());
for(int i=0; i<this->size(); ++i)
        for(int j=0; j<rhs.size(); ++j)
                res[i*rhs.size()+j] = (*this)[i]*rhs[j];
res.simplify();
return res;
}


template <typename T, typename R>
PermutationAlgebra<T,R> PermutationAlgebra<T,R>::operator+(const PermutationAlgebra<T,R> &rhs) const
{
if(this->size()>0 && rhs.size()>0 && (*this)[0].perm.size()!=rhs[0].perm.size()) laerror("permutation sizes do not match");
PermutationAlgebra<T,R> res=this->concat(rhs);
res.simplify();
return res;
}






////////////////////////////////////////////////////////

template <typename T>
CyclePerm<T>:: CyclePerm(const NRPerm<T> &p)
{
#ifdef DEBUG
        if(!p.is_valid()) laerror("invalid permutation");
#endif
T n=p.size();
NRVec_from1<T> used((T)0,n),tmp(n);
T firstunused=1;
T currentcycle=0;
std::list<NRVec_from1<T> > cyclelist;
do
        {
        //find a cycle starting with first unused element
        T cyclelength=0;
        T next = firstunused;
        do
                {
                ++cyclelength;
                used[next]=1;
                tmp[cyclelength] = next;
                next = p[next];
                }
        while(used[next]==0);
        if(cyclelength>1) //nontrivial cycle
                {
		NRVec_from1<T> cycle(&tmp[1],cyclelength);
		cyclelist.push_front(cycle);
                ++currentcycle;
                }
        while(used[firstunused]) {++firstunused; if(firstunused>n) break;} //find next unused element
        }
while(firstunused<=n);
//convert list to NRVec
this->resize(currentcycle);
T i=1;
for(typename std::list<NRVec_from1<T> >::iterator l=cyclelist.begin(); l!=cyclelist.end(); ++l) (*this)[i++] = *l;
}



template <typename T>
bool CyclePerm<T>::is_valid() const
{
for(T i=1; i<=this->size(); ++i)
	{
	T n=(*this)[i].size();
	if(n<=0) return false;
	for(T j=1; j<=n; ++j)
		{
		T x=(*this)[i][j];
		if(x<=0) return false;

		//now check for illegal duplicity of numbers withis a cycle or across cycles
		for(T ii=i; ii<=this->size(); ++ii)
			{
			T nn=(*this)[ii].size();
			for(T jj=(ii==i?j+1:1); jj<=nn; ++jj)
				{
				T xx=(*this)[ii][jj];
				if(x==xx) return false;
				}
			}
		}
	}
return true;
}


template <typename T>
bool CyclePerm<T>::is_identity() const
{
#ifdef DEBUG
if(!this->is_valid()) laerror("operation with an invalid cycleperm");
#endif
for(T i=1; i<=this->size(); ++i) if((*this)[i].size()>1) return false; //at least one nontrivial cycle
return true;
}


template <typename T>
CyclePerm<T> CyclePerm<T>::inverse() const
{
#ifdef DEBUG
if(!this->is_valid()) laerror("operation with an invalid cycleperm");
#endif

CyclePerm<T> r;
T ncycles=this->size();
r.resize(ncycles);
for(T i=1; i<=ncycles; ++i)
	{
	T length=(*this)[i].size();
	r[i].resize(length);
	//reverse order in cycles (does not matter in cycle lengths 1 and 2 anyway)
	for(T j=1; j<=length; ++j) r[i][j] = (*this)[i][length-j+1];
	}
return r;
}

//multiplication via NRPerm - could there be a more efficient direct algorithm?
template <typename T>
CyclePerm<T> CyclePerm<T>::operator*(const CyclePerm &q) const
{
int m=this->max();
int mm=q.max();
if(mm>m) mm=m;
NRPerm<T> qq(q,m);
NRPerm<T> pp(*this,m);
NRPerm<T> rr=pp*qq;
return CyclePerm<T>(rr);
}


//mixed type multiplications
template <typename T>
NRPerm<T> NRPerm<T>::operator*(const CyclePerm<T> &r) const
{
NRPerm<T> rr(r,this->size());
return *this * rr;
}

template <typename T>
NRPerm<T> CyclePerm<T>::operator*(const NRPerm<T> &r) const
{
NRPerm<T> tt(*this,r.size());
return tt*r;
}

template <typename T>
void CyclePerm<T>::simplify(bool keep1)
{
int j=0;
for(int i=1; i<=this->size(); ++i)
	{
	int il= (*this)[i].size();
	if(keep1 && il>0 || il>1 )
		{
		++j;
		if(j!=i) (*this)[j] = (*this)[i]; //keep this
		}
	}
this->resize(j,true);
}


template <typename T>
int CyclePerm<T>::parity() const
{
#ifdef DEBUG
if(!this->is_valid()) laerror("operation with an invalid cycleperm");
#endif
int n_even_cycles=0;
T ncycles=this->size();
for(T i=1; i<=ncycles; ++i)
        {
        T length=(*this)[i].size();
	if((length&1)==0) ++n_even_cycles;
	}
return (n_even_cycles&1)?-1:1;
}


template <typename T>
PERM_RANK_TYPE  CyclePerm<T>::order() const
{
#ifdef DEBUG
if(!this->is_valid()) laerror("operation with an invalid cycleperm");
#endif
PERM_RANK_TYPE r=1;
T ncycles=this->size();
for(T i=1; i<=ncycles; ++i)
        {
        T length=(*this)[i].size();
	r=lcm(r,(PERM_RANK_TYPE)length);
        }
return r;
}


template <typename T>
CompressedPartition<T> CyclePerm<T>::cycles(T n) const
{
#ifdef DEBUG
if(!this->is_valid()) laerror("operation with an invalid cycleperm");
#endif
if(n==0) n=max();
CompressedPartition<T> r(n); r.clear();
T ncycles=this->size();
for(T i=1; i<=ncycles; ++i)
        {
	T length=(*this)[i].size();
	if(length<=0||length>n) laerror("unexpected cycle length in permutation");
	r[length]++;
	}
//fill in trivial cycles of length one
r[1] += n - r.sum();
if(r[1]<0) laerror("inconsistent cycle lengths in CyclePerm::cycles");
return r;
}


//auxiliary function for input of a permutation in cycle format
//returns pointer after closing bracket or NULL if no cycle found
//or input error
template <typename T>
const char *read1cycle(NRVec_from1<T> &c, const char *p)
{
if(*p==0) return NULL;
const char *openbracket = strchr(p,'(');
if(!openbracket) return NULL;
const char *closebracket = strchr(openbracket+1,')');
if(!closebracket) return NULL;
const char *s = openbracket+1;
int r;
int length=0;
std::list<T> cycle;
do      {
	long int tmp;
        int nchar;
        if(*s==',') ++s;
        r = sscanf(s,"%ld%n",&tmp,&nchar);
        if(r==1)
                {
                ++length;
                s += nchar;
		cycle.push_back((T)tmp);
                }
        }
        while(r==1 && s<closebracket);

//make vector from list
c.resize(length);
int i=0;
for(typename std::list<T>::iterator l=cycle.begin(); l!=cycle.end(); ++l) c[++i] = *l;

return closebracket+1;
}

template <typename T>
void CyclePerm<T>::readfrom(const std::string &line)
{
const char *p=line.c_str();
std::list<NRVec<T> > cyclelist;
int ncycles=0;
int count=0;
NRVec_from1<T> c;
while(p=read1cycle(c,p))
        {
        //printf("cycle %d of length %d read\n",count,c.size());
        if(c.size()!=0) //store a nonempty cycle
                {
                ++count;
		cyclelist.push_back(c);
                }
        }


//convert list to vector
this->resize(count);
T i=0;
for(typename std::list<NRVec<T> >::iterator l=cyclelist.begin(); l!=cyclelist.end(); ++l) (*this)[++i] = *l;
#ifdef DEBUG
if(!this->is_valid()) laerror("readfrom received input of invalid CyclePerm");
#endif
}


template <typename T>
std::istream & operator>>(std::istream &s, CyclePerm<T> &x)
{
std::string l;
getline(s,l);
x.readfrom(l);
return s;
}

template <typename T>
std::ostream & operator<<(std::ostream &s, const CyclePerm<T> &x)
{
for(int i=1; i<=x.size(); ++i)
	{
	s<<"(";
	for(int j=1; j<=x[i].size(); ++j)
		{
		s<<x[i][j];
		if(j<x[i].size()) s<<" ";
		}
	s<<")";
	}
return s;
}


template <typename T>
CyclePerm<T> CyclePerm<T>::pow(const int n, const bool keep1) const
{
#ifdef DEBUG
        if(!this->is_valid()) laerror("power of invalid permutation");
#endif
CyclePerm<T> r;
if(n==0) {r.identity(); return r;}

//probably negative n will work automatically using the modulo approach
std::list<NRVec_from1<T> > cyclelist;
T currentcycle=0;

//go through all our cycles and compute power of each cycle separately
for(int i=1; i<=this->size(); ++i)
	{
	int cyclesize = (*this)[i].size();
	if(cyclesize>0 && keep1 || cyclesize>1)
		{
		int modulo = n%cyclesize;
		if(modulo<0) modulo += cyclesize;
		if(modulo==0) 
			{
			if(keep1) //keep cycles of length 1 to keep info about the size of the permutation
				{
				for(int j=1; j<=cyclesize; ++j) {NRVec_from1<T> c(1); c[1] = (*this)[i][j]; ++currentcycle; cyclelist.push_back(c);}
				}
			}
		else //the nontrivial case
			{
			int nsplit=gcd(modulo,cyclesize);
			int splitsize=cyclesize/nsplit;
			for(int j=0; j<nsplit; ++j) //loop over split cycles
				{
				NRVec_from1<T> c(splitsize);
				for(int k=1; k<=splitsize; ++k) //fill in the cycle
					{
					c[k] = (*this)[i][((k-1)*modulo)%cyclesize + 1 + j];
					}
				++currentcycle; cyclelist.push_back(c);	
				}
			}
		}
	}

//convert list to NRVec
r.resize(currentcycle);
int i=1;
for(typename std::list<NRVec_from1<T> >::iterator l=cyclelist.begin(); l!=cyclelist.end(); ++l) r[i++] = *l;
return r;
}

///////////////////////////////////////////////////////

template <typename T>
PERM_RANK_TYPE CompressedPartition<T>::Sn_class_size() const
{
#ifdef DEBUG
if(!this->is_valid()) laerror("operation with an invalid partition");
#endif
int n=this->size();
PERM_RANK_TYPE r=factorial(n);
for(int i=1; i<=n; ++i) 
	{
	T m=(*this)[i];
	if(i>1 && m>0) r/=longpow(i,m);
	if(m>1) r/=factorial(m);
	}
return r;
}

template <typename T>
PERM_RANK_TYPE Partition<T>::Sn_irrep_dim() const
{
#ifdef DEBUG
if(!this->is_valid()) laerror("operation with an invalid partition");
#endif
int n=this->size();
PERM_RANK_TYPE prod=1;
Partition<T> adj=this->adjoint();
//hook length formula
for(int i=1; i<=adj[1]; ++i) //rows
	for(int j=1; j<= (*this)[i]; ++j) //cols
		prod *= (*this)[i]-j+adj[j]-i+1;

return factorial(n)/prod;
}


/*hammermesh eq 10-25, highest weight == generalized partition of r into n parts*/
template <typename T>
PERM_RANK_TYPE Partition<T>::Un_irrep_dim(const int n) const
{
#ifdef DEBUG
if(!this->is_valid()) laerror("operation with an invalid partition");
#endif
if(nparts()>n) return 0; //too antisymmetric partition
int i,j;
double prod;
int r=this->size();
NRVec_from1<int> p(n);
for(i=1;i<=n;i++) p[i]= (i<=r?(*this)[i]:0)+n-i;
prod=1;
for(j=n;j>=2;j--)
        {
        for(i=1;i<j;i++) prod*= (p[i]-p[j]);
	prod /= factorial(j-1); //can be fractional in the intermediate steps - needs double
	}

return (PERM_RANK_TYPE) (prod+0.2);
}



template <typename T>
Partition<T> Partition<T>::adjoint() const
{
#ifdef DEBUG
if(!this->is_valid()) laerror("operation with an invalid partition");
#endif
int n=this->size();
Partition<T> r(n);
r.clear();
for(int i=1;i<=n;++i)
                {
		int j;
                for(j=1; j<=n&&(*this)[j]>=i; ++j);
                r[i]=j-1;
                }
return r;
}

template <typename T>
Partition<T>::Partition(const YoungTableaux<T> &x)
{
#ifdef DEBUG
if(!x.is_valid()) laerror("operation with an invalid tableaux");
#endif
int nparts=x.size();
int n=0;
for(int i=1; i<=nparts; ++i) n+= x[i].size();
this->resize(n);
this->clear();
for(int i=1; i<=nparts; ++i) (*this)[i]=x[i].size();
}


//aux for the recursive procedure
static PERM_RANK_TYPE partitioncount;
template <typename T> static void (*_pcallback)(const Partition<T>&);
static int partitiontyp;
static int partitiontypabs;
template <typename T> static Partition<T> *partition;

template <typename T>
void partgen(int remain, int pos)
{
int hi,lo;
if(remain==0) {++partitioncount; if(_pcallback<T>) (*_pcallback<T>)(*partition<T>); return;}
if(partitiontyp) lo=(remain+partitiontypabs-pos)/(partitiontypabs-pos+1); else lo=1;
hi=remain;
if(partitiontyp>0) hi -= partitiontypabs-pos;
if(pos>1 && (*partition<T>)[pos-1] < hi) hi= (*partition<T>)[pos-1];
for((*partition<T>)[pos]=hi; (*partition<T>)[pos]>=lo; --(*partition<T>)[pos]) partgen<T>(remain-(*partition<T>)[pos],pos+1);
(*partition<T>)[pos]=0;
}



template <typename T>
PERM_RANK_TYPE Partition<T>::generate_all(void (*callback)(const Partition<T>&), int nparts) 
{
int n=this->size();
if(n==0) return 0;
if(nparts>0 && n<nparts) return 0;

this->copyonwrite();
this->clear();
partitioncount=0;
_pcallback<T> =callback;
partitiontyp=nparts;
partition<T> = this;
partitiontypabs= nparts>=0?nparts:-nparts;
partgen<T>(n,1);
return partitioncount;
}


template <typename T>
std::ostream & operator<<(std::ostream &s, const CompressedPartition<T> &x)
{
int n=x.size();
T sum=0;
for(int i=n; i>0;--i)
	if(x[i])
		{
		s<<i;
		if(x[i]>1) s<<'^'<<x[i];
		sum+= i*x[i];
		if(sum<n) s<<',';
		}
return s;
}



template <typename T>
int Partition<T>::parity() const
{
#ifdef DEBUG
if(!this->is_valid()) laerror("operation with an invalid partition");
#endif
T n_even_cycles=0;
T n=this->size();
for(T i=1; i<=n; ++i)
        {
	if((*this)[i]!=0 && ((*this)[i] &1 ) ==0) ++n_even_cycles;
	}
return (n_even_cycles&1)?-1:1;
}

template <typename T>
int CompressedPartition<T>::parity() const
{
#ifdef DEBUG
if(!this->is_valid()) laerror("operation with an invalid partition");
#endif
T n_even_cycles=0;
T n=this->size();
for(T i=2; i<=n; i+=2) n_even_cycles += (*this)[i];
return (n_even_cycles&1)?-1:1;
}





/*see M. Aigner - Combinatorial Theory - number of partitions of number n*/
#define MAXPART 260
#define MIN(x,y) ((x)<(y)?(x):(y))
/* here even few more columns more could be saved, and also the property part(n,k>=n/2)=part(n-k) could be used */
#define partind(n,k) (((n)-3)*((n)-2)/2+(k)-2)
#define getpart(n,k) (((k)>(n) || (k)<=0 )?0:(((k)==(n) || (k)==1)?1:(part[partind((n),(k))])))

PERM_RANK_TYPE partitions(int n, int k)
{
static PERM_RANK_TYPE part[partind(MAXPART+1,1)]={1};
static PERM_RANK_TYPE npart[MAXPART+1]={0,1,2,3};
static int ntop=3;
int i,j,l;
PERM_RANK_TYPE s;

if( n<=0 || k<-1 ) laerror("Partitions are available only for positive numbers");
if(k== -1) /* total partitions */
	{
	if(n>ntop)(void)partitions(n,3); /* make sure that table is calculated */
	return npart[n];
	}

if(n==k||k==1) return 1;
if(k==0||k>n) return 0;
if(k==2) return n/2;
if(n>MAXPART && k>=n/2) return(partitions(n-k,-1)); 
if( n>MAXPART) laerror("sorry, too big argument to partition enumerator");
if(n<=ntop) return(getpart(n,k));

/*calculate, if necessary, few next rows*/
for(i=4;i<=n;i++)
    {
    npart[i]=2;
    for(j=2;j<=i-1;j++)
	{
	s=0;
	for(l=1; l<=MIN(i-j,j);l++) s+=getpart(i-j,l);
	npart[i]+=(part[partind(i,j)]=s);
	}
    }

ntop=n;
return(part[partind(n,k)]);

}

#undef getpart
#undef partind
#undef MIN



//////////////////////////////////////////////////////////////////////////////////////

template <typename T>
YoungTableaux<T>::YoungTableaux(const Partition<T> &frame)
: NRVec_from1<NRVec_from1<T> >()
{
#ifdef DEBUG
if(!frame.is_valid()) laerror("invalid partition used as young frame");
#endif
int nlines=frame.nparts();
this->resize(nlines);
for(int i=1; i<=nlines; ++i) {(*this)[i].resize(frame[i]); (*this)[i].clear();}
}

template <typename T>
bool YoungTableaux<T>::is_valid() const
{
int nrows=this->size();
for(int i=1; i<=nrows; ++i)
	{
	if((*this)[i].size()<=0) return false;
	if(i>1 && (*this)[i].size()> (*this)[i-1].size()) return false;
	}
return true; 
}


template <typename T>
T YoungTableaux<T>::sum() const
{
#ifdef DEBUG
if(!is_valid()) laerror("invalid young frame");
#endif
int sum=0;
int nrows=this->size();
for(int i=1; i<=nrows; ++i) sum += (*this)[i].size();
return sum;
}

template <typename T>
T YoungTableaux<T>::max() const
{
#ifdef DEBUG
if(!is_valid()) laerror("invalid young frame");
#endif
int m= -1;
int nrows=this->size();
for(int i=1; i<=nrows; ++i) for(int j=1; j<= (*this)[i].size(); ++j) if((*this)[i][j]>m) m=(*this)[i][j];
return m;
}



template <typename T>
bool YoungTableaux<T>::is_standard() const
{
#ifdef DEBUG
if(!is_valid()) laerror("invalid young frame");
#endif
int nrows=this->size();
//check numbers monotonous in rows and columns
for(int i=1; i<=nrows; ++i)
	{
	for(int j=1; j<=(*this)[i].size(); ++j) 
		{
		if(j>1 && (*this)[i][j] < (*this)[i][j-1]) return false;
		if(i>1 && (*this)[i][j] < (*this)[i-1][j]) return false;
		}
	}
return true;
}

template <typename T>
std::ostream & operator<<(std::ostream &s, const YoungTableaux<T> &x)
{
int nrows=x.size();
for(int i=1; i<=nrows; ++i)
        {
	for(int j=1; j<=x[i].size(); ++j)
		{
		s.width(4);
		s<<x[i][j]<<' ';
		}
	s<<std::endl;
	}
return s;
}


template <typename T>
NRVec_from1<T>  YoungTableaux<T>::yamanouchi() const
{
#ifdef DEBUG
if(!is_valid()) laerror("invalid young frame");
#endif
int n=sum();
NRVec_from1<T> yama(n);
int i,j;
for (i=1;i<=this->size();i++) for (j=1;j<=(*this)[i].size();j++) yama[n-(*this)[i][j]+1]=i;
return yama;
}


template <typename T>
T  YoungTableaux<T>::character_contribution(int ncyc) const
{
#ifdef DEBUG
if(!is_valid()) laerror("invalid young frame");
if(!is_standard()) laerror("nonstandardly filled young frame");
#endif


if(!ncyc) ncyc=max(); //number of types of applied points
NRVec_from1<T> onxlines((T)0,ncyc);

for(int i=1;i<=(*this).size();i++) //rows
        {
	NRVec_from1<T> wasfound((T)0,ncyc);
        for(int j=1;j<=(*this)[i].size();j++) //columns
                {
                T k = (*this)[i][j];
                onxlines[k] += (1-wasfound[k]);
                wasfound[k]=1;
                }
        }

/*now sum the number of odd applications for all cycles*/
T contrib=0;
for(int i=1;i<=ncyc;i++) contrib += ((onxlines[i]&1)^1);

return (1-2*(contrib&1)); //add it to the character +1 for even no. of odd apl., -1 for odd no. of odd apl.
}

static bool _callyoung;

template <typename T>
static void (*_young_callback)(const YoungTableaux<T>&);

template <typename T>
static T _character;

template <typename T>
static YoungTableaux<T> *_tchi;

template <typename T>
static T _nowapplying;

template <typename T>
static T _ncycles;

template <typename T>
static NRVec_from1<T> *_aplnumbers;

template <typename T>
static NRVec_from1<T> *_oclin;

template <typename T>
static NRVec_from1<T> *_occol;

template <typename T>
static T _nn;

template <typename T>
static inline T mymin(T i,T j)
{
  return(i<j?i:j);
}


template <typename T>
void placedot(T l, T c)
{
(*_tchi<T>)[l][c]= _nowapplying<T>;
(*_oclin<T>)[l]++;
(*_occol<T>)[c]++;
}

//forward declaration for recursion
template <typename T>
extern void nextapply(T ncyc);

template <typename T>
void regulapply(T ncyc, T napl) //ncyc is number of cycles; napl is the size of now processed cycle 
{

/*the main idea is that only the first point of each set can be placed
in several independent ways; the other points (if present) must be placed in
a given fixed way depending on the first one; than it is also necessary to test
if the application was successfull at all and return back or call nextapply, respectively*/

Partition<T> usedcols(_nn<T>),usedlines(_nn<T>); //stores the positions where points were placed, is necessary for later cleanup

/*try to place first point*/
for(usedlines[1]=mymin((T)_tchi<T>->nrows(),(*_occol<T>)[1]+napl); usedlines[1]>0; usedlines[1]--)
   {
   if((*_oclin<T>)[usedlines[1]]<=(*_tchi<T>)[usedlines[1]].size()) /*line is not fully occupied*/
	{
	int i;

	usedcols[1]= (*_oclin<T>)[usedlines[1]]; /*in the line there is only one possible column*/
	/* place first point */
	placedot(usedlines[1],usedcols[1]);

	for(i=2;i<=napl;i++) usedlines[i]=usedcols[i]= 0; /* flag for cleaning that they were not placed*/
	/*now place other ones, if not possible go to irregular*/
	for(i=2;i<=napl;i++)
		{
		T thisrow;
		thisrow=usedlines[i-1];
		if (thisrow==1)
			{
			if((*_oclin<T>)[1]> (*_tchi<T>)[1].size()) goto irregular; /*no place remained*/ 
			placedot((usedlines[i]=1),(usedcols[i]= (*_oclin<T>)[1]));
			}
		else
			{
			T thiscol;
			thiscol=usedcols[i-1];
			if(!(*_tchi<T>)[thisrow-1][thiscol]) /*the box at the top of last placed is free*/
				{
				placedot((usedlines[i]=thisrow-1),(usedcols[i]=thiscol));
				}
			else if((*_oclin<T>)[thisrow] <= (*_tchi<T>)[thisrow].size()) /*the position at the right exists*/
				{
#ifdef DEBUG
				if((*_tchi<T>)[thisrow][thiscol+1]) laerror("error in regulapply!!!");
#endif
				placedot((usedlines[i]=thisrow),(usedcols[i]=thiscol+1));
				}
			else goto irregular;
			}
		}

	/*test if it is regular*/
	for(i=1;i<=napl;i++)
		{
		/*test if the box left and up from actual position is full (provided it exists)*/
		if(usedlines[i]>1) if(!(*_tchi<T>)[usedlines[i]-1][usedcols[i]]) goto irregular;
		if(usedcols[i]>1)  if(!(*_tchi<T>)[usedlines[i]][usedcols[i]-1]) goto irregular;
		}

	nextapply(ncyc+1);
	
	irregular:

	/* clear all points */
	for(i=napl;i>0;i--)
	    if(usedlines[i]>0 && usedcols[i]>0)
		{
		(*_tchi<T>)[usedlines[i]][usedcols[i]]=0;
		(*_oclin<T>)[usedlines[i]]--;
		(*_occol<T>)[usedcols[i]]--;
		}

	}
   /*end of loop for nonfull lines of first point*/
   }
}


template <typename T>
void nextapply(T ncyc)
{
_nowapplying<T>++;
if(ncyc> _ncycles<T>) 
	{
	if(_callyoung)
		{
		if(_young_callback<T>) _young_callback<T>(*_tchi<T>);
		_character<T> ++;
		}
	else _character<T> += _tchi<T>->character_contribution(_ncycles<T>);
	}
else regulapply(ncyc,(*_aplnumbers<T>)[ncyc]);
_nowapplying<T>--;
}



template <typename T>
T Sn_character(const Partition<T> &irrep, const  Partition<T> &cclass)
{
#ifdef DEBUG
if(!irrep.is_valid()||!cclass.is_valid()) laerror("invalid input to Sn_character");
#endif

//prepare numbers to apply to the tableaux
int n=irrep.sum();
if(cclass.sum()!=n) laerror("irrep and class partitions do not match");
T ncycles=cclass.nparts();

NRVec_from1<T> aplnumbers(ncycles);
CompressedPartition<T> ccclass(cclass);
T k=0;
for(int j=n;j>0;j--) for(T l=1;l<=ccclass[j];l++) aplnumbers[++k]=j;
if(aplnumbers[k]==1)
        {
        /*rotate to the right*/
        for(T l=k;l>1;l--) aplnumbers[l]=aplnumbers[l-1];
        aplnumbers[1]=1;
        }

//applying aplnumbers[i] pieces of "i"
//std::cout<<"TEST aplnumbers "<<aplnumbers<<std::endl;

//prepare static variables for the recursive procedure and generate all regular applications
YoungTableaux<T> y(irrep);
//y.clear(); //already done in the constructor
_callyoung=false;
_ncycles<T> =ncycles;
_tchi<T> = &y;
_nn<T> = n;
_nowapplying<T> =0;
_aplnumbers<T> = &aplnumbers;
_character<T> =0;
Partition<T> oclin(n);
_oclin<T> = &oclin;
Partition<T> occol(n);
_occol<T>  = &occol;
for(int i=1; i<=n; ++i) (*_oclin<T>)[i]= (*_occol<T>)[i]= 1;
nextapply<T>(1);

return _character<T>;
}



template <typename T>
PERM_RANK_TYPE YoungTableaux<T>::generate_all_standard(void (*callback)(const YoungTableaux<T>&))
{
#ifdef DEBUG
if(!this->is_valid()) laerror("invalid young frame");
#endif

//prepare numbers to apply to the tableaux
T n=this->sum();
T ncycles=n;
NRVec_from1<T> aplnumbers(n);
for(T i=1; i<=n; ++i) aplnumbers[i]=1;

//prepare static variables for the recursive procedure and generate all regular applications
this->clear(); 
_callyoung=true;
_young_callback<T> =callback;
_ncycles<T> =ncycles;
_tchi<T> = this;
_nn<T> = n;
_nowapplying<T> =0;
_aplnumbers<T> = &aplnumbers;
_character<T> =0;
Partition<T> oclin(n);
_oclin<T> = &oclin;
Partition<T> occol(n);
_occol<T>  = &occol;
for(int i=1; i<=n; ++i) (*_oclin<T>)[i]= (*_occol<T>)[i]= 1;
nextapply<T>(1);
return _character<T>;
}


////generation of the young operator

template <typename T>
static NRPerm<T> _aperm;

template <typename T>
static NRPerm<T> _sperm;

template <typename T>
static const YoungTableaux<T> *_tyou;

template <typename T>
static const Partition<T> *_tyou_cols;

template <typename T>
static const Partition<T> *_tyou_rows;

static PERM_RANK_TYPE _nyoungterms, _expectterms;

template <typename T>
static T _antparity;

template <typename T, typename R>
PermutationAlgebra<T,R> *_young_r;



template <typename T>
static void symetr(T ilin, T iel)
{

if(ilin > (*_tyou_cols<T>)[1])
	{
	(*_young_r<T,T>)[_nyoungterms].weight = _antparity<T>;
	(*_young_r<T,T>)[_nyoungterms].perm = _aperm<T>*_sperm<T>;
	++_nyoungterms;
	} 
else if(iel > (*_tyou_rows<T>)[ilin]) symetr(ilin+1,(T)1);
else
	{
	int i;

	for(i=1;i<=(*_tyou_rows<T>)[ilin];i++)
	    if(!_sperm<T>[(*_tyou<T>)[ilin][i]])
		{
		_sperm<T>[(*_tyou<T>)[ilin][i]]= (*_tyou<T>)[ilin][iel];
		symetr(ilin,iel+1);
		_sperm<T>[(*_tyou<T>)[ilin][i]]=0;
		}
	}
}




template <typename T>
static void antisym(T icol,T iel)
{

if(icol > (*_tyou_rows<T>)[1])
	{
	_antparity<T> = _aperm<T>.parity();
	symetr<T>(1,1);
	}
else
if(iel > (*_tyou_cols<T>)[icol]) antisym(icol+1,(T)1);
else
	{
	int i;

	for(i=1;i<=(*_tyou_cols<T>)[icol];i++)
	    if(!_aperm<T>[(*_tyou<T>)[i][icol]])
		{
		_aperm<T>[(*_tyou<T>)[i][icol]]= (*_tyou<T>)[iel][icol];
		antisym(icol,iel+1);
		_aperm<T>[(*_tyou<T>)[i][icol]]=0;
		}
	}
}







template <typename T>
PermutationAlgebra<T,T> YoungTableaux<T>::young_operator() const
{
#ifdef DEBUG
if(!this->is_standard()) laerror("young_operator called for non-standard tableaux");
#endif
_nyoungterms =0;
_tyou<T> = this;
Partition<T> rows=Partition<T>(*this);
Partition<T> cols=rows.adjoint();
_tyou_rows<T> = &rows;
_tyou_cols<T> = &cols;

T n=rows.sum();
_aperm<T>.resize(n); _aperm<T>.clear();
_sperm<T>.resize(n); _sperm<T>.clear();


_expectterms=1;
for(int i=1;i<=cols[1];i++) _expectterms *= factorial(rows[i]);
for(int i=1;i<=rows[1];i++) _expectterms *= factorial(cols[i]);

PermutationAlgebra<T,T> r(_expectterms);
_young_r<T,T> = &r;

antisym<T>(1,1);

if(_nyoungterms!=_expectterms) laerror("youngconstruct: inconsistent number of terms");

return r;
}



template <typename T>
static NRVec_from1<CompressedPartition<T> > *_irreps;

template <typename T>
static NRVec_from1<CompressedPartition<T> > *_classes;

template <typename T>
static PERM_RANK_TYPE _ithrough;


template <typename T>
static void _process_partition(const Partition<T> &p)
{
++_ithrough<T>;
(*_irreps<T>)[_ithrough<T>] = CompressedPartition<T>(p);
Partition<T> q=p.adjoint();
(*_classes<T>)[_ithrough<T>] = CompressedPartition<T>(q);
}

template <typename T>
Sn_characters<T>::Sn_characters(const int n0)
:n(n0)
{
Partition<T> p(n);
PERM_RANK_TYPE np = partitions(n);
irreps.resize(np);
classes.resize(np);
classsizes.resize(np);
chi.resize(np,np);
_irreps<T> = &irreps;
_classes<T> = &classes;
_ithrough<T> = 0;
//generate partitions for classes and irreps
PERM_RANK_TYPE tot=p.generate_all(_process_partition<T>);
//compute class sizes
for(int i=1; i<=np; ++i) classsizes[i]= classes[i].Sn_class_size();
//compute characters
for(int i=1; i<=np; ++i) 
	{
	for(int j=1; j<=np; ++j)
		{
		chi(i,j) = Sn_character<T>(irreps[i],classes[j]);
		}
	}
}


template <typename T>
bool Sn_characters<T>::is_valid() const
{
//consistency of n and partition number
PERM_RANK_TYPE np = partitions(n);
if(np!=classes.size() || np!=irreps.size() || np!=classsizes.size() ||np!=chi.nrows() ||np!=chi.ncols()) return false;

//consistency of sum = n in all partitions
for(int i=1; i<=np; ++i) 
	{
	if(irreps[i].sum()!=n) return false;
	if(classes[i].sum()!=n) return false;
	}

//consistency of irrep dim squared to n!
//consistency of irrep dimensions by hook length formula to character of identity (asserted as class no. 1)
if(classes[1][1]!=n) laerror("assetion failed on class no. 1 being identity");
PERM_RANK_TYPE nf = factorial(n);
PERM_RANK_TYPE s=0;
for(int i=1; i<=np; ++i)
        {
	T d=chi(i,1);
	s += d*d;
	T dd=Partition<T>(irreps[i]).Sn_irrep_dim();
	if(d!=dd) return false;
	}
if(s!=nf) return false;

//consistency of class sizes sum to n! 
s=0; for(int i=1; i<=np; ++i) s+= classsizes[i];
if(s!=nf) return false;

//check that irrep [n] is totally symmetric
if(irreps[1][n]!=1)  laerror("assetion failed on first irrep being [n]");
for(int i=1; i<=np; ++i) if(chi(1,i)!=1) return false;

//check that irrep[1^n] is totally antisymmetric
if(irreps[np][1]!=n) laerror("assetion failed on last irrep being [1^n]");
for(int i=1; i<=np; ++i) if(chi(np,i)!=classes[i].parity()) return false;

//check character orthonormality between irreps
for(int i=1; i<=np; ++i) for(int j=1; j<=i; ++j)
	{
	s=0;
	for(int k=1; k<=np; ++k) s+= classsizes[k]*chi(i,k)*chi(j,k);
	if(i==j)
		{
		if(s!=nf) return false;
		}
	else
		{
		if(s!=0) return false;
		}
	}

return true;
}

template <typename T>
std::ostream & operator<<(std::ostream &s, const Sn_characters<T> &c)
{
const int w=4+c.n; //some reasonable width
s<<"S"<<c.n<<" "<<c.classes.size()<<std::endl;
s.width(w); s<<" ";
for(int i=1; i<=c.classes.size(); ++i)
	{
	s.width(w); 
        s<<c.classsizes[i]<<" ";
	}
s<<std::endl;
s.width(w); s<<" ";
for(int i=1; i<=c.classes.size(); ++i) 
	{
	std::ostringstream  classlabel;
	classlabel<<"("<<c.classes[i]<<")";
	s.width(w); 
	s<<classlabel.str()<<" ";
	} 
s<<std::endl;
for(int i=1; i<=c.chi.nrows(); ++i)
	{
	std::ostringstream irrep;
	irrep<<"["<<c.irreps[i]<<"] ";
	s.width(w); s<<irrep.str();
	for(int j=1; j<=c.chi.ncols(); ++j) {s.width(w); s<<c.chi(i,j)<<" ";}
	s<<std::endl;
	}

return s;
}


template <typename T>
bool CycleIndex<T>::is_valid() const
{
if(classes.size()!=classsizes.size()) return false;
T n=classes[1].sum();
for(int i=2; i<=classes.size(); ++i) 
	{
	if(classes[i].sum()!=n) return false;
	}
return true;
}

template <typename T>
Polynomial<T> CycleIndex<T>::substitute(const Polynomial<T> &p, PERM_RANK_TYPE *denom) const
{
Polynomial<T> r(0);
r[0]=(T)0;
*denom =0;

for(int i=1; i<=classes.size(); ++i)
	{
	Polynomial<T> term(0);
	term[0]=(T)1;
	for(int j=1; j<=classes[i].size(); ++j) if(classes[i][j]) term *= (p.powx(j)).pow((int)classes[i][j]);
	r += term*classsizes[i];
	*denom += classsizes[i];
	}

return r;
}


template<typename T>
NRMat<PERM_RANK_TYPE> Multable(T n)
{
NRPerm<T> p(n);
PermutationAlgebra<T,T> all = p.list_all_lex();
NRMat<PERM_RANK_TYPE> r(all.size(),all.size());
for(PERM_RANK_TYPE i=0; i<all.size(); ++i) r[0][i] = r[i][0]=i; //identity
for(PERM_RANK_TYPE i=1; i<all.size(); ++i)
	for(PERM_RANK_TYPE j=1; j<all.size(); ++j)
		{
		NRPerm<T> tmp = all[i].perm * all[j].perm;
		r(i,j) = tmp.rank();
		}
//consistency checks
#ifdef DEBUG
bitvector occ(all.size());
for(PERM_RANK_TYPE i=0; i<all.size(); ++i) 
	{
	occ.clear();
	for(PERM_RANK_TYPE j=0; j<all.size(); ++j) {if(occ[r(i,j)]) laerror("inconsistency in Multable"); occ.set(r(i,j));}
	occ.clear();
	for(PERM_RANK_TYPE j=0; j<all.size(); ++j) {if(occ[r(j,i)]) laerror("inconsistency in Multable"); occ.set(r(j,i));}
	}

#endif
return r;
}

template<typename T, typename R>
NRMat<R> RegularRepresentation(const PermutationAlgebra<T,R> &a, const NRMat<PERM_RANK_TYPE> &mtable)
{
NRMat<R> r(mtable.nrows(),mtable.ncols());
r.clear();
for(int i=0; i<a.size(); ++i)
	{
	PERM_RANK_TYPE rx=a[i].perm.rank();	
	for(PERM_RANK_TYPE j=0; j<mtable.nrows();++j) r(mtable(rx,j),j) += a[i].weight;
	}
return r;
}


template<typename T>
PermutationAlgebra<T,T> general_antisymmetrizer(const NRVec<NRVec_from1<T> > &groups, int restriction_type, bool inverted)
{
PermutationAlgebra<T,T> r;
int ngroups=groups.size();
if(ngroups==0) return r;
NRVec<PermutationAlgebra<T,T> > lists(ngroups);
for(int i=0; i<ngroups; ++i)
	{
	int ni = groups[i].size();
	NRPerm<T> tmp(ni);
	lists[i] = tmp.list_restricted(groups[i],restriction_type,inverted);
	}
//cross-product the lists
r=lists[0];
for(int i=1; i<ngroups; ++i) r= r&lists[i];
return r;
}

//DOES NOT WORK template<typename T,typename U>
template<typename T> template <typename U> //this works
void NRPerm<T>::testik(U u)
{
}

/***************************************************************************//**
 * forced instantization in the corresponding object file
 ******************************************************************************/

#define INSTANTIZE(T) \
template class NRPerm<T>; \
template class CyclePerm<T>; \
template class CompressedPartition<T>; \
template class Partition<T>; \
template class YoungTableaux<T>; \
template class Sn_characters<T>; \
template class CycleIndex<T>; \
template NRMat<PERM_RANK_TYPE> Multable(T n); \
template PermutationAlgebra<T,T> general_antisymmetrizer(const NRVec<NRVec_from1<T> > &groups, int, bool); \
template std::istream & operator>>(std::istream &s, CyclePerm<T> &x); \
template std::ostream & operator<<(std::ostream &s, const CyclePerm<T> &x); \
template std::ostream & operator<<(std::ostream &s, const CompressedPartition<T> &x); \
template std::ostream & operator<<(std::ostream &s, const YoungTableaux<T> &x); \
template T Sn_character(const Partition<T> &irrep, const  Partition<T> &cclass); \
template std::ostream & operator<<(std::ostream &s, const Sn_characters<T> &x); \


#define INSTANTIZE2(T,R) \
template class WeightPermutation<T,R>; \
template class PermutationAlgebra<T,R>; \
template std::istream & operator>>(std::istream &s, WeightPermutation<T,R> &x); \
template std::ostream & operator<<(std::ostream &s, const WeightPermutation<T,R> &x); \
template NRMat<R> RegularRepresentation(const PermutationAlgebra<T,R> &a, const NRMat<PERM_RANK_TYPE> &mtable); \


INSTANTIZE(int)
INSTANTIZE(unsigned int)


INSTANTIZE2(int,int)


INSTANTIZE2(int,double)

}//namespace
