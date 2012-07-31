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

#include <string>
#include <cmath>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include "bitvector.h"
#include "sparsemat.h"

namespace LA {

template<typename T>
static inline const T MAX(const T &a, const T &b)
        {return b > a ? (b) : (a);}

template<typename T>
static inline void SWAP(T &a, T &b)
        {T dum=a; a=b; b=dum;}







template <class T>
void SparseMat<T>::get(int fd, bool dimen, bool transp)
{
errno=0;
SPMatindex dim[2];
if(dimen)
{
if(2*sizeof(SPMatindex) != read(fd,&dim,2*sizeof(SPMatindex))) laerror("cannot read");
if(transp) resize(dim[1],dim[0]); else resize(dim[0],dim[1]);
int symnon[2];
if(2*sizeof(int) != read(fd,&symnon,2*sizeof(int))) laerror("cannot read");
symmetric=symnon[0];
nonzero=symnon[1];
}
else
copyonwrite();

matel<T> *l=NULL;
do
          {
	  if(2*sizeof(SPMatindex) != read(fd,&dim,2*sizeof(SPMatindex))) laerror("cannot read");
	  if(dim[0]+1==0 && dim[1]+1==0) break;
          matel<T> *ll = l;
          l= new matel<T>;
          l->next= ll;
          if(transp) {l->row=dim[1]; l->col=dim[0];} else {l->row=dim[0]; l->col=dim[1];}
	  LA_traits<T>::get(fd,l->elem,dimen,transp); //general way to work when elem is some complex class again
          } while(1);
list=l;
}


template <class T>
void SparseMat<T>::put(int fd,bool dimen, bool transp) const
{
errno=0;
if(dimen)
{
if(sizeof(SPMatindex) != write(fd,&(transp ? mm : nn),sizeof(SPMatindex))) laerror("cannot write");
if(sizeof(SPMatindex) != write(fd,&(transp ? nn : mm),sizeof(SPMatindex))) laerror("cannot write");
int symnon[2];
symnon[0]=symmetric;
symnon[1]=nonzero;
if(2*sizeof(int) != write(fd,symnon,2*sizeof(int))) laerror("cannot write");
}
matel<T> *l=list;
while(l)
	{
	if(sizeof(SPMatindex) != write(fd,&(transp ? l->col : l->row),sizeof(SPMatindex))) laerror("cannot write");
	if(sizeof(SPMatindex) != write(fd,&(transp ? l->row : l->col),sizeof(SPMatindex))) laerror("cannot write");
	LA_traits<T>::put(fd,l->elem,dimen,transp);//general way to work when elem is some non-scalar class again
	l=l->next;
	}
SPMatindex sentinel[2];
sentinel[0]=sentinel[1]=(SPMatindex) -1;
if(2*sizeof(SPMatindex) != write(fd,&sentinel,2*sizeof(SPMatindex))) laerror("cannot write");
}


//helpers to be used from different functions
template <class T>
void SparseMat<T>::unsort()
{
if(symmetric) colsorted=NULL;
if(colsorted) delete[] colsorted;
if(rowsorted) delete[] rowsorted;
colsorted=rowsorted=NULL;
nonzero=0;
}

template <class T>
void SparseMat<T>::deletelist()
{
if(colsorted||rowsorted) unsort();//prevent obsolete pointers
if(*count >1) laerror("trying to delete shared list");
matel<T> *l=list;
while(l)
	{
	matel<T> *ltmp=l;
	l=l->next;
	delete ltmp;
	}
list=NULL;
delete count;
count=NULL;
}

//no checks, not to be public
template <class T>
void SparseMat<T>::copylist(const matel<T> *l)
{
list=NULL;
while(l)
	{
	add(l->row,l->col,l->elem);
	l=l->next;
	}
}

template <class T>
void SparseMat<T>::copyonwrite(bool detachonly)
{
        if(!count) laerror("probably an assignment to undefined sparse matrix");
        if(*count > 1)
                {
                (*count)--;
                count = new int; *count=1;
		if(!list) laerror("empty list with count>1");
		unsort();
		if(!detachonly) copylist(list);
                }
}


//global for sort !!! is not thread-safe
static void *globsorted;

//global functions cannot be partially specialized in templates, we have to make it a member function

//!!! gencmp's and genswap are critical for performance, make sure that compiler really inlines them
template<class T, int type>
struct gencmp {
inline static SPMatindexdiff EXEC(register const SPMatindex i, register const SPMatindex j) 
	{
	register SPMatindexdiff k;
	register matel<T> *ii,*jj;
	ii=((matel<T> **)globsorted)[i];
	jj=((matel<T> **)globsorted)[j];
	if (k=ii->col-jj->col) return k; else return ii->row-jj->row;
	}
};


template<class T>
struct gencmp<T,0> {
inline static SPMatindexdiff EXEC(register const SPMatindex i, register const SPMatindex j)
	{
        register SPMatindexdiff k;
        register matel<T> *ii,*jj;
        ii=((matel<T> **)globsorted)[i];
        jj=((matel<T> **)globsorted)[j];
	if (k=ii->row-jj->row) return k; else return ii->col-jj->col;
	}
};




template<class T>
inline void genswap(const SPMatindex i,const SPMatindex j)
{
SWAP(((matel<T> **)globsorted)[i],((matel<T> **)globsorted)[j]);
}



template<class T, int type>
void myqsort(SPMatindex l,SPMatindex r) /*safer version for worst case*/
{
register SPMatindex i,j,piv;

/* other method for small arrays recommended in NUMREC is not used here
does not give so large gain for moderate arrays and complicates the
things, but would be worth trying (cf. profile) */

if(r<=l) return; /*1 element*/
if(gencmp<T,type>::EXEC(r,l)<0) genswap<T>(l,r);
if(r-l==1) return; /*2 elements and preparation for median*/
piv= (l+r)/2; /*pivoting by median of 3 - safer */
if(gencmp<T,type>::EXEC(piv,l)<0) genswap<T>(l,piv); /*and change the pivot element implicitly*/
if(gencmp<T,type>::EXEC(r,piv)<0) genswap<T>(r,piv); /*and change the pivot element implicitly*/
if(r-l==2) return; /*in the case of 3 elements we are finished too */

/*general case , l-th r-th already processed*/
i=l+1; j=r-1;
do{
  /*important sharp inequality - stops at sentinel element for efficiency*/
  /* this is inefficient if all keys are equal - unnecessary n log n swaps are done, but we assume that it is atypical input*/
  while(gencmp<T,type>::EXEC(i++,piv)<0);
  i--;
  while(gencmp<T,type>::EXEC(j--,piv)>0);
  j++;
  if(i<j)
        {
        /* swap and keep track of position of pivoting element */
        genswap<T>(i,j);
        if(i==piv) piv=j; else if(j==piv) piv=i;
        }
  if(i<=j) {i++; j--;}
  }while(i<=j);

if(j-l < r-i)   /*because of the stack in bad case process first the shorter subarray*/
        {if(l<j) myqsort<T,type>(l,j); if(i<r) myqsort<T,type>(i,r);}
else
        {if(i<r) myqsort<T,type>(i,r); if(l<j) myqsort<T,type>(l,j);}
}


template <class T>
unsigned int SparseMat<T>::length() const
{
if(nonzero) return nonzero;
unsigned int n=0;
matel<T> *l=list;
while(l)
	{
	++n;
	l=l->next;
	}

const_cast<SparseMat<T> *>(this)->nonzero=n;
return n;
}

 
template <class T>
unsigned int SparseMat<T>::sort(int type) const //must be const since used from operator* which must be const to be compatible with other stuff, dirty casts here
{
if(type==0&&rowsorted || type==1&&colsorted) return nonzero;
if(!list) return ((SparseMat<T> *)this)->nonzero=0;

if(type!=2) const_cast<SparseMat<T> *>(this) ->setunsymmetric(); else type=0;//symmetric and sorted not supported simultaneously, type 2 is special just for simplify

//create array from list, reallocate as necessary
unsigned int size=3*MAX(nn,mm); //initial guess for a number of nonzero elements
matel<T> **sorted= new matel<T>* [size];
((SparseMat<T> *)this)->nonzero=0;
matel<T> *l = list;
while(l)
        {
        sorted[(((SparseMat<T> *)this)->nonzero)++]=l;
        if(nonzero >= size ) //reallocate
		{
		size*=2;
		matel<T> **newsorted= new matel<T>* [size];
		memcpy(newsorted,sorted,size/2*sizeof(matel<T>*));
		delete[] sorted;
		sorted=newsorted;
		}
        l= l->next;
        }

//now sort the array of pointers according to type
globsorted =sorted; 
if(type==0) {myqsort<T,0>(0,nonzero-1); ((SparseMat<T> *)this)->rowsorted=sorted;} //type handled at compile time for more efficiency
else {myqsort<T,1>(0,nonzero-1); ((SparseMat<T> *)this)->colsorted=sorted;} //should better be const cast

//cout <<"sort: nonzero ="<<nonzero<<"\n";
return nonzero; //number of (in principle) nonzero elements
}


template <class T>
void SparseMat<T>::simplify()
{
unsigned int n;
if(!list) return;
copyonwrite();
if(symmetric)
	{
	unsort(); 
	matel<T> *p;
	p=list;
	while(p)
		{
		if(p->row>p->col) SWAP(p->row,p->col); //get into one triangle, not OK for complex hermitean 
		p=p->next;
		}
	n=sort(2); //sort and further handle like a triangle matrix
	}
else n=sort(0); //sorts according to row,column

unsigned int i,j;
SPMatindex r,c;
j=0;
r=rowsorted[j]->row;
c=rowsorted[j]->col;
for(i=1; i<n;i++)
        {
        if(r==rowsorted[i]->row && c==rowsorted[i]->col) {rowsorted[j]->elem +=rowsorted[i]->elem; delete rowsorted[i]; rowsorted[i]=NULL;}
                else
                        {
                        j=i;
                        r=rowsorted[j]->row;
                        c=rowsorted[j]->col;
                        }
        }

//check if  summed to zero
for(i=0; i<n;i++) if(rowsorted[i] &&
	std::abs(rowsorted[i]->elem)<=SPARSEEPSILON
	)	 {delete rowsorted[i]; rowsorted[i]=NULL;}

//restore connectivity
int nonz=0;
matel<T> *p,*first,*prev;
first=NULL;
prev=NULL;
for(i=0; i<n;i++) if(p=rowsorted[i])
        {
	++nonz;
        if(!first) first=p;
        if(prev) prev->next=p;
        p->next=NULL;
        prev=p;
        }
list=first;
nonzero=nonz;
unsort(); //since there were NULLs introduced, rowsorted is not dense
}


template <class T>
void SparseMat<T>::resize(const SPMatindex n, const SPMatindex m)
{
	unsort(); 
        if(count) 
		{
		if(*count > 1) {(*count)--; count=NULL; list=NULL;} //detach from previous
		else if(*count==1) deletelist();
		if(count) delete count;
		}
        nn=n;
        mm=m;
        if(nn||mm) count=new int(1); //empty but defined matrix
	list=NULL;
	symmetric=0;
	nonzero=0;
	colsorted=rowsorted=NULL;
}

template <class T>
void SparseMat<T>::incsize(const SPMatindex n, const SPMatindex m)
{
	if(symmetric && n!=m) laerror("unsymmetric size increment of a symmetric sparsemat");
	if(!count ) count=new int(1);
	copyonwrite();//this errors if !count 
	unsort();
        nn+=n;
        mm+=m;
}



template <class T>
void SparseMat<T>::addsafe(const SPMatindex n, const SPMatindex m, const T elem)
{
#ifdef debug
if(n<0||n>=nn||m<0||m>=mm) laerror("SparseMat out of range");
#endif
if(std::abs(elem)<=SPARSEEPSILON) return;
if(!count) {count=new int;  *count=1; list=NULL;} //blank new matrix
else copyonwrite(); //makes also unsort
add(n,m,elem);
}


//assignment operator
template <class T>
SparseMat<T> & SparseMat<T>::operator=(const SparseMat<T> &rhs)
{
        if (this != &rhs)
                {
		unsort();
                if(count)
                    if(--(*count) ==0) {deletelist(); delete count;} // old stuff obsolete
                list=rhs.list;
                nn=rhs.nn;
                mm=rhs.mm;
                if(list) count=rhs.count; else count= new int(0); //make the matrix defined, but empty and not shared, count will be incremented below
		symmetric=rhs.symmetric;
                if(count) (*count)++;
                }
        return *this;
}

template <class T>
SparseMat<T> & SparseMat<T>::join(SparseMat<T> &rhs)
{
if(symmetric!=rhs.symmetric||nn!=rhs.nn||mm!=rhs.mm) laerror("incompatible matrices in join()");
if(*rhs.count!=1) laerror("shared rhs in join()");
if(!count) {count=new int;  *count=1; list=NULL;}
else copyonwrite();
matel<T> **last=&list;
while(*last) last= &((*last)->next);
*last=rhs.list;
rhs.list=NULL;
return *this;
}


template <class T>
SparseMat<T> & SparseMat<T>::addtriangle(const SparseMat &rhs, const bool lower, const char sign)
{
if(nn!=rhs.nn||mm!=rhs.mm) laerror("incompatible dimensions for +=");
if(!count) {count=new int;  *count=1; list=NULL;}
else copyonwrite();
register matel<T> *l=rhs.list;
while(l)
	{
	if(rhs.symmetric || lower && l->row <=l->col || !lower && l->row >=l->col)
	    if(std::abs(l->elem)>SPARSEEPSILON) 
		add( l->row,l->col,sign=='+'?l->elem:- l->elem);
	l=l->next;
	}
return *this;
}

template <class T>
SparseMat<T> & SparseMat<T>::operator+=(const SparseMat<T> &rhs)
{
if(symmetric&&!rhs.symmetric) laerror("cannot add general to symmetric sparse");
if(nn!=rhs.nn||mm!=rhs.mm) laerror("incompatible dimensions for +=");
if(!count) {count=new int;  *count=1; list=NULL;}
else copyonwrite();
bool symmetrize= !symmetric && rhs.symmetric;
register matel<T> *l=rhs.list;
if(symmetrize)
while(l)
        {
        if(std::abs(l->elem)>SPARSEEPSILON) 
		{add( l->row,l->col,l->elem); if( l->row!=l->col) add( l->col,l->row,l->elem);}
        l=l->next;
        }
else
while(l)
	{
	if(std::abs(l->elem)>SPARSEEPSILON) 
		add( l->row,l->col,l->elem);
	l=l->next;
	}
simplify(); //maybe leave up to the user
return *this;
}

template <class T>
SparseMat<T> & SparseMat<T>::operator-=(const SparseMat<T> &rhs)
{
if(symmetric&&!rhs.symmetric) laerror("cannot add general to symmetric sparse");
if(nn!=rhs.nn||mm!=rhs.mm) laerror("incompatible dimensions for -=");
if(!count) {count=new int;  *count=1; list=NULL;}
else copyonwrite();
bool symmetrize= !symmetric && rhs.symmetric;
register matel<T> *l=rhs.list;
if(symmetrize)
while(l)
        {
        if(std::abs(l->elem)>SPARSEEPSILON) 
		{add( l->row,l->col,- l->elem); if( l->row!=l->col) add( l->col,l->row,- l->elem);}
        l=l->next;
        }
else
while(l)
        {
        if(std::abs(l->elem)>SPARSEEPSILON) 
		add( l->row,l->col,- l->elem);
        l=l->next;
        }
simplify(); //maybe leave up to the user
return *this;
}


//constructor from a dense matrix
template <class T>
SparseMat<T>::SparseMat(const NRMat<T> &rhs)
{
nn=rhs.nrows();
mm=rhs.ncols();
count=new int;
*count=1; 
list=NULL;
symmetric=0;
colsorted=rowsorted=NULL;
SPMatindex i,j;
for(i=0;i<nn;++i)
	for(j=0; j<mm;++j)
		{register T t(rhs(i,j));
		if( std::abs(t)>SPARSEEPSILON)
		 add(i,j,t);
		}
}


//get diagonal, do not construct a new object, but store in existing one, important for huge CI matrices
// with the divide option is used as a preconditioner, another choice of preconditioner is possible
template <class T>
const T* SparseMat<T>::diagonalof(NRVec<T> &r, const bool divide, bool cache) const
{
#ifdef DEBUG
if((int)mm!=r.size()) laerror("incompatible vector size in diagonalof()");
#endif
matel<T> *l=list;
NRVec<T> *rr;

r.copyonwrite();
if(divide) {rr=new NRVec<T>(mm); *rr=(T)0;}
else {r=(T)0; rr=&r;}
if(nn==mm) //square
	while(l)
		{
		if(l->row == l->col) (*rr)[l->row]+= l->elem;
		l= l->next;
		}
else //diagonal of A^TA, assuming it has been simplified (only one entry per position), will be used as preconditioner only anyway
	while(l)
		{
		(*rr)[l->col] += l->elem*l->elem *(l->col!=l->row && symmetric?2.:1.);
		l= l->next;
		}
if(divide)
	{
	for(unsigned int i=0; i<mm; ++i) if((*rr)[i]!=0.) r[i]/=(*rr)[i];
	delete rr;
	}
return divide?NULL:&r[0];
}


//constructor dense matrix from sparse
template <class T>
NRMat<T>::NRMat(const SparseMat<T> &rhs)
{
nn=rhs.nrows();
mm=rhs.ncols();
count=new int(1);
T *p;
#ifdef MATPTR
        v= new T*[nn];
        p=v[0] = new T[mm*nn];
        for (int i=1; i< nn; i++) v[i] = v[i-1] + mm;
#else
        p= v = new T[mm*nn];
#endif
memset(p,0,nn*mm*sizeof(T));
matel<T> *l=rhs.getlist();
bool sym=rhs.issymmetric();
while(l)
	{
#ifdef MATPTR
	v[l->row][l->col] +=l->elem;
	if(sym && l->row!=l->col) v[l->col][l->row] +=l->elem;
#else
	v[l->row*mm+l->col] +=l->elem;
	 if(sym && l->row!=l->col) v[l->col*mm+l->row] +=l->elem;
#endif
	l=l->next;
	}
}




//constructor dense symmetric packed matrix from sparse
#define nn2 (nn*(nn+1)/2)
template <class T>
NRSMat<T>::NRSMat(const SparseMat<T> &rhs)
{
if(!rhs.issymmetric()||rhs.nrows()!=rhs.ncols()) laerror("sparse matrix is not symmetric");
nn=rhs.nrows();
count=new int(1);
v=new T[nn2];
memset(v,0,nn2*sizeof(T));
matel<T> *l=rhs.getlist();
while(l)
	{
	(*this)(l->row,l->col)=l->elem;
	l=l->next;
	}
}
#undef nn2

//constructor dense vector from sparse
template <class T>
NRVec<T>::NRVec(const SparseMat<T> &rhs)
{
if(rhs.nrows()>1 && rhs.ncols()>1) laerror("cannot construct a vector from a sparse matrix with more than one row/column");
nn=rhs.nrows()>1?rhs.nrows():rhs.ncols();
v=new T[nn]; 
memset(v,0,nn*sizeof(T));
count=new int(1);
matel<T> *l=rhs.getlist();

if(rhs.nrows()>1) while(l)
        {
	v[l->row]+=l->elem;
        l=l->next;
        }
else while(l)
        {
        v[l->col]+=l->elem;
        l=l->next;
        }
}

//assignment of a scalar matrix
template <class T>
SparseMat<T> & SparseMat<T>::operator=(const T &a)
{
if(!count ||nn<=0||mm<=0) laerror("assignment of scalar to undefined sparse matrix");
if(nn!=mm) laerror("assignment of scalar to non-square sparse matrix");
resize(nn,mm);//clear 
if(std::abs(a)<=SPARSEEPSILON) return *this;
SPMatindex i;
for(i=0;i<nn;++i) add(i,i,a);
return *this;
}

template <class T>
SparseMat<T> & SparseMat<T>::operator+=(const T &a)
{
if(!count ||nn<=0||mm<=0) laerror("assignment of scalar to undefined sparse matrix");
if(nn!=mm) laerror("assignment of scalar to non-square sparse matrix");
if(a==(T)0) return *this;
SPMatindex i;
for(i=0;i<nn;++i) add(i,i,a);
simplify(); //maybe leave up to the user
return *this;
}

template <class T>
SparseMat<T> & SparseMat<T>::operator-=(const T &a)
{
if(!count ||nn<=0||mm<=0) laerror("assignment of scalar to undefined sparse matrix");
if(nn!=mm) laerror("assignment of scalar to non-square sparse matrix");
if(a==(T)0) return *this;
SPMatindex i;
for(i=0;i<nn;++i) add(i,i,-a);
simplify(); //maybe leave up to the user
return *this;
}



//constructor from a dense symmetric matrix
template <class T>
SparseMat<T>::SparseMat(const NRSMat<T> &rhs)
{
nn=rhs.nrows();
mm=rhs.ncols();
count=new int;
*count=1;
list=NULL;
symmetric=1;
colsorted=rowsorted=NULL;
SPMatindex i,j;
for(i=0;i<nn;++i)
        for(j=0; j<=i;++j)
		{register T t;
                if(
			std::abs(t=rhs(i,j))>SPARSEEPSILON
			) add(i,j,t);
		}
}

template <class T>
void SparseMat<T>::transposeme()
{
if(!count) laerror("transposeme on undefined lhs");
if(symmetric||!list) return;
copyonwrite();//also unsort
register matel<T> *l=list;
while(l)
	{
	SWAP(l->row,l->col);
	l=l->next;
	}
SWAP(nn,mm);
}

template <class T>
void SparseMat<T>::setunsymmetric()
{
if(!symmetric) return;
unsort();
symmetric=0;
if(!count) return;
copyonwrite();
matel<T> *l=list;
while(l) //include the mirror picture of elements into the list
	{
	if(
		std::abs(l->elem)>SPARSEEPSILON && 
	   l->row!=l->col) add(l->col,l->row,l->elem); //not OK for complex-hermitean
	l=l->next;
	}
}


template <class T>
SparseMat<T> & SparseMat<T>::operator*=(const T &a)
{
if(!count) laerror("operator*= on undefined lhs");
if(!list||a==(T)1) return *this;
if(a==(T)0) {clear(); return *this;}
copyonwrite();

register matel<T> *l=list;
while(l) 
        {
	l->elem*=a;
        l=l->next;
        }
return *this;
}

template<>
const double SparseMat<double>::dot(const NRMat<double> &rhs) const
{
double r=0;
matel<double> *l=list;
while(l)
	{
	r+= l->elem*rhs[l->row][l->col];
	if(symmetric&&l->row!=l->col) r+=l->elem*rhs[l->col][l->row];
	l=l->next;
	}
return r;
}

template<>
void NRMat<complex<double> >::gemm(const complex<double>  &beta, const SparseMat<complex<double> > &a, const char transa, const NRMat<complex<double> > &b, const char transb, const complex<double>  &alpha)
{
laerror("not implemented yet");
}


template<>
void NRMat<double>::gemm(const double &beta, const SparseMat<double> &a, const char transa, const NRMat<double> &b, const char transb, const double &alpha)
{
bool transpa = tolower(transa)!='n'; //not OK for complex
bool transpb = tolower(transb)!='n'; //not OK for complex
if(nn!=(int)(transpa?a.ncols():a.nrows()) | mm!= (transpb?b.nrows():b.ncols()) || (int)(transpa?a.nrows():a.ncols()) != (transpb?b.ncols():b.nrows())) laerror("incompatible sizes in gemm");
copyonwrite();
if(beta!=(double)0) (*this) *= beta; 
else memset(v,0,nn*mm*sizeof(double));

matel<double> *l=a.getlist();
if(alpha==(double)0 || !l) return;

//raw pointers to the full matrices
const double *bp= b;
double *p= *this;

int ldb=(transpb?b.ncols():1);
int bstep=(transpb?1:b.ncols());
int len=(transpb?b.nrows():b.ncols());

if(a.issymmetric())
	{
	while(l)
	        {
		cblas_daxpy(len, alpha*l->elem , bp+l->row*bstep, ldb, p+l->col*mm, 1);
	        if(l->row!=l->col) cblas_daxpy(len, alpha*l->elem , bp+l->col*bstep, ldb, p+l->row*mm, 1);
	        l=l->next;
	        }
	}
else
	{
	if(transpa)
	while(l)
	        {
		cblas_daxpy(len, alpha*l->elem , bp+l->row*bstep, ldb, p+l->col*mm, 1);
	        l=l->next;
	        }
	else
	while(l)
	        {
		cblas_daxpy(len, alpha*l->elem , bp+l->col*bstep, ldb, p+l->row*mm, 1);
	        l=l->next;
	        }
	}

}



template<class T>
void NRVec<T>::gemv(const T beta, const SparseMat<T> &a, const char trans, const T alpha, const NRVec<T> &x, const bool treat_as_symmetric)
{
if((trans=='n'?a.ncols():a.nrows())!= (SPMatindex)x.size()) laerror("incompatible sizes in gemv");
copyonwrite();
if(beta!=(T)0) {if(beta!=(T)1) (*this) *= beta;}
else memset(v,0,nn*sizeof(T));

bool transp = tolower(trans)!='n'; //not OK for complex

matel<T> *l=a.getlist();

if(alpha==(T)0 || !l) return;
T *vec=x.v;

if(alpha==(T)1)
{
        if(a.issymmetric()||treat_as_symmetric)
        {
        while(l)
                {
                v[l->row]+= l->elem*vec[l->col];
                if(l->row!=l->col) v[l->col]+= l->elem*vec[l->row];
                l=l->next;
                }
        }
        else
        {
        if(transp)
        while(l)
                {
                v[l->col]+= l->elem*vec[l->row];
                l=l->next;
                }
        else
        while(l)
                {
                v[l->row]+= l->elem*vec[l->col];
                l=l->next;
                }
        }
}
else
{
	if(a.issymmetric()||treat_as_symmetric)
	{
	while(l)
	        {
	        v[l->row]+= alpha*l->elem*vec[l->col];
	        if(l->row!=l->col) v[l->col]+= alpha*l->elem*vec[l->row];
	        l=l->next;
	        }
	}
	else
	{
	if(transp)
	while(l)
	        {
	        v[l->col]+= alpha*l->elem*vec[l->row];
	        l=l->next;
	        }
	else
	while(l)
	        {
	        v[l->row]+= alpha*l->elem*vec[l->col];
	        l=l->next;
	        }
	}
}
}




template<class T>
const T SparseMat<T>::trace() const
{
matel<T> *l=list;
T sum(0);
while(l)
	{
	if(l->row==l->col) sum+= l->elem;
	l=l->next;
	}
return sum;
}


template<class T>
const typename LA_traits<T>::normtype SparseMat<T>::norm(const T scalar) const
{
if(!list) return typename LA_traits<T>::normtype(0);
const_cast<SparseMat<T> *>(this)->simplify(); 

matel<T> *l=list;
typename LA_traits<T>::normtype sum(0);

if(scalar!=(T)0)
	{
	if(nn!=mm) laerror("subtraction of scalar from non-square sparse matrix in norm()");
	bitvector has_diagonal_element(nn); has_diagonal_element.clear();
	if(symmetric) 
	while(l)
	        {
		T hlp=l->elem;
		bool b= l->row==l->col;
		if(b) {hlp-=scalar; has_diagonal_element.set(l->row);}
		typename LA_traits<T>::normtype tmp=LA_traits<T>::sqrabs(hlp);
       	 sum+= tmp;
		if(!b) sum+=tmp;
       	 l=l->next;
       	 }
	else
	while(l)
	       	 {
		T hlp=l->elem;
		if(l->row==l->col) {hlp-=scalar; has_diagonal_element.set(l->row);}
	        sum+= LA_traits<T>::sqrabs(hlp);
	        l=l->next;
	        }
	sum += (nn-has_diagonal_element.population()) * LA_traits<T>::sqrabs(scalar); //add contribution of the subtracted scalar from zero non-stored diagonal elements
	}
else
	{
        if(symmetric)
        while(l)
                {
                typename LA_traits<T>::normtype tmp=LA_traits<T>::sqrabs(l->elem);
         sum+= tmp;
                if(l->row!=l->col) sum+=tmp;
         l=l->next;
         }
        else
        while(l)
                 {
                sum+= LA_traits<T>::sqrabs(l->elem);
                l=l->next;
                }
	}
return (typename LA_traits<T>::normtype) std::sqrt(sum); //not OK for int, would need traits technique
}


template<class T>
void SparseMat<T>::axpy(const T alpha, const SparseMat<T> &x, const bool transp)
{
if(!transp && (nn!=x.nn||mm!=x.mm) || transp && (mm!=x.nn||nn!=x.mm) ) laerror("incompatible dimensions for axpy");
if(!count) {count=new int;  *count=1; list=NULL;}
else copyonwrite();

if(alpha==(T)0||x.list==NULL) return;
if(!transp||x.symmetric)
	{
	if(alpha==(T)1) {*this +=x; return;}
	if(alpha==(T)-1) {*this -=x; return;}
	}
if(symmetric!=x.symmetric) laerror("general axpy not supported for different symmetry types");
//now does not matter if both are general or both symmetric (transposition will not matter)

register matel<T> *l=x.list;
if(transp)
while(l)
        {
	register T t=alpha*l->elem;
	if(std::abs(t)>SPARSEEPSILON) 
				add( l->col,l->row,t);
        l=l->next;
        }
else
while(l)
	{
	register T t=alpha*l->elem;
	if(std::abs(t)>SPARSEEPSILON) 
				add( l->row,l->col,t);
	l=l->next;
	}
}


template<class T>
const T SparseMat<T>::dot(const SparseMat<T> &rhs) const //complex conj. not implemented yet
{
if(nn!=rhs.nn || mm!=rhs.mm) laerror("dot of incompatible sparse matrices");
if(symmetric||rhs.symmetric) laerror("dot of symmetric sparse matrices not implemented");

T result=0;
if(list && rhs.list) //both nonzero
    {
    unsigned int na=sort(0);
    unsigned int nb=rhs.sort(0);

    //now merge the sorted lists
    register unsigned int i,j;
    register SPMatindex ra,ca;
    j=0;
    for(i=0; i<na;i++)
        {
	register SPMatindex rb=0,cb=0;
        ra=rowsorted[i]->row;
        ca=rowsorted[i]->col;
        while(j<nb && (rb=rhs.rowsorted[j]->row) <ra) j++; /*skip in rhs*/
        while(j<nb && (cb=rhs.rowsorted[j]->col) <ca) j++; /*skip in rhs*/
	
	if(j==nb) break; //we can exit the i-loop, no suitable element in b any more
        if(ra==rb&&ca==cb)
                {
        	T tmp=rowsorted[i]->elem;
                register unsigned int k;
                /*j remembers the position, k forwards in the rhs.rowsorted to find all combinations*/
                k=j;
                do {
                        result += tmp*rhs.rowsorted[k]->elem;
                        k++;
                        } while(k<nb && (rhs.rowsorted[k]->row == ra) && (rhs.rowsorted[k]->col == ca));
                }
        /*else skip in left operand*/
        }
    }
return result;
}


template<class T>
void SparseMat<T>::permuterows(const NRVec<SPMatindex> &p)
{
if((SPMatindex)p.size()!=nn) laerror("inconsistent dimension in permuterows");

matel<T> *l=list;

while(l)
	{
	l->row = p[l->row];
	if(symmetric) l->col= p[l->col];
	l=l->next;
	}
}


template<class T>
void SparseMat<T>::permutecolumns(const NRVec<SPMatindex> &p)
{
if((SPMatindex)p.size()!=nn) laerror("inconsistent dimension in permuterows");

matel<T> *l=list;

while(l)
        {
        if(symmetric) l->row = p[l->row];
        l->col= p[l->col];
        l=l->next;
        }
}

template<class T>
void SparseMat<T>::permuteindices(const NRVec<SPMatindex> &p)
{
if((SPMatindex)p.size()!=nn) laerror("inconsistent dimension in permuterows");

matel<T> *l=list;

while(l)
        {
        l->row = p[l->row];
        l->col= p[l->col];
        l=l->next;
        }
}




template<class T>
const SparseMat<T> SparseMat<T>::operator*(const SparseMat<T> &rhs) const
{
if(mm!=rhs.nn) laerror("product of incompatible sparse matrices");
if(symmetric||rhs.symmetric) laerror("product of symmetric sparse matrices not implemented");

SparseMat<T> result(nn,rhs.mm);
if(list && rhs.list) //both nonzero
    {
    unsigned int na=sort(1);
    unsigned int nb=rhs.sort(0);

    //now merge the sorted lists
    register unsigned int i,j;
    register SPMatindex rb=0,ca;
    j=0;
    for(i=0; i<na;i++)
        {
        ca=colsorted[i]->col;
        while(j<nb && (rb=rhs.rowsorted[j]->row) <ca) j++; /*skip in rhs.rowsorted*/
	if(j==nb) break; //we can exit the i-loop, no suitable element in mb any more
        if(rb==ca)
                {
        	T tmp=colsorted[i]->elem;
                register unsigned int k;
                /*j remembers the position, k forwards in the rhs.rowsorted to find all combinations*/
                k=j;
                do {
                        result.add(colsorted[i]->row,rhs.rowsorted[k]->col,tmp*rhs.rowsorted[k]->elem);
                        k++;
                        } while(k<nb && ((rhs.rowsorted[k]->row) == ca));
                }
        /*else skip in left operand*/
        }
    result.simplify();//otherwise number of terms tends to grow exponentially
    }
return result;
}



template <class T>
void SparseMat<T>::gemm(const T beta, const SparseMat<T> &a, const char transa, const SparseMat<T> &b, const char transb, const T alpha)
{
SPMatindex l(transa=='n'?a.nn:a.mm);
SPMatindex k(transa=='n'?a.mm:a.nn);
SPMatindex kk(transb=='n'?b.nn:b.mm);
SPMatindex ll(transb=='n'?b.mm:b.nn);
if(a.symmetric||b.symmetric) laerror("symmetric sparse matrices not supported in gemm");

if(beta==(T)0) resize(l,ll); //empty matrix
else	*this *= beta; //takes care about beta=1
if(l!=nn|| ll!=mm||k!=kk) laerror("incompatible sparse matrices in gemm");

if(alpha==(T)0 || !a.list ||!b.list) return;
copyonwrite();

//regular case, specialize for transpositions
matel<T> **ma,**mb;
unsigned int na,nb;
bool tra= transa!='n';
bool trb= transb!='n';
if(!tra) {na=a.sort(1); ma=a.colsorted;} else {na=a.sort(0); ma=a.rowsorted;}
if(!trb) {nb=b.sort(0); mb=b.rowsorted;} else {nb=b.sort(1); mb=b.colsorted;}

//now merge the sorted lists
register unsigned int i,j;
register SPMatindex rb=0,ca,row;
j=0;
for(i=0; i<na;i++)
        {
        ca=tra?ma[i]->row:ma[i]->col;
	row=tra?ma[i]->col:ma[i]->row;
        while(j<nb && (rb=trb?mb[j]->col:mb[j]->row) <ca) j++; /*skip in mb*/ 
	if(j==nb) break; //we can exit the i-loop, no suitable element in mb any more
        if(rb==ca)
                {
		T tmp=alpha * ma[i]->elem;
                register unsigned int k;
                /*j remembers the position, k forwards in the mb to find all combinations*/
                k=j;
                do {
			register SPMatindex col; 
		     col=trb?mb[k]->row:mb[k]->col;
		     if(!symmetric||row<=col) add(row,col,tmp*mb[k]->elem);
                        k++;
                        } while(k<nb && ((trb?mb[k]->col:mb[k]->row) == ca));
                }
        /*else skip in ma*/
        }

simplify();
}



//direct sum and product -- only partly implemented at the moment
template<typename T>
SparseMat<T> & SparseMat<T>::oplusequal(const NRMat<T> &rhs)
{
if(issymmetric()) laerror("oplusequal symmetric-unsymmetric");
SPMatindex n0=nn;
SPMatindex m0=mm;
incsize(rhs.nrows(), rhs.ncols());
T tmp;
for(SPMatindex i=0; i<(SPMatindex)rhs.nrows(); ++i)
	for(SPMatindex j=0; j<(SPMatindex)rhs.ncols(); ++j)
		if(std::abs(tmp=rhs(i,j))>SPARSEEPSILON) add(n0+i,m0+j,tmp);
return *this;
}



template<typename T>
SparseMat<T> & SparseMat<T>::oplusequal(const NRSMat<T> &rhs)
{
if(!issymmetric()) laerror("oplusequal symmetric-unsymmetric");
SPMatindex n0=nn;
SPMatindex m0=mm;
T tmp;
incsize(rhs.nrows(), rhs.ncols());
for(SPMatindex i=0; i<(SPMatindex)rhs.nrows(); ++i)
        for(SPMatindex j=0; j<(SPMatindex)rhs.ncols(); ++j)
                if(std::abs(tmp=rhs(i,j))>SPARSEEPSILON) add(n0+i,m0+j,tmp);
return *this;
}



template <class T>
SparseMat<T> & SparseMat<T>::oplusequal(const SparseMat<T> &rhs)
{
if(symmetric != rhs.symmetric) laerror("incompatible symmetry of sparsemats in oplusequal");
if(!count) {count=new int;  *count=1; list=NULL;}

SPMatindex n0=nn;
SPMatindex m0=mm; 
incsize(rhs.nrows(), rhs.ncols());

register matel<T> *l=rhs.list;
while(l)
	{
	if(std::abs(l->elem)>SPARSEEPSILON) 
		add(n0+l->row,m0+l->col,l->elem);
	l=l->next;
	}
return *this;
}



/*
 Commented out by Roman for ICC

#define INSTANTIZE(T) \
template SparseMat<T> & SparseMat<T>::oplusequal(const SparseMat<T> &rhs);\
template SparseMat<T> & SparseMat<T>::oplusequal(const NRMat<T> &rhs);\
template SparseMat<T> & SparseMat<T>::oplusequal(const NRSMat<T> &rhs);\
template void SparseMat<T>::get(int fd, bool dimen, bool transp); \
template void SparseMat<T>::put(int fd, bool dimen, bool transp) const; \
template void SparseMat<T>::copyonwrite(); \
template void SparseMat<T>::unsort(); \
template void SparseMat<T>::resize(const SPMatindex n, const SPMatindex m); \
template void SparseMat<T>::incsize(const SPMatindex n, const SPMatindex m); \
template unsigned int SparseMat<T>::sort(int type) const; \
template unsigned int SparseMat<T>::length() const; \
template void SparseMat<T>::deletelist(); \
template void SparseMat<T>::simplify(); \
template void SparseMat<T>::copylist(const matel<T> *l); \
template SparseMat<T> & SparseMat<T>::operator=(const SparseMat<T> &rhs); \
template SparseMat<T> & SparseMat<T>::operator+=(const SparseMat<T> &rhs); \
template SparseMat<T> & SparseMat<T>::operator-=(const SparseMat<T> &rhs); \
template SparseMat<T>::SparseMat(const NRMat<T> &rhs); \
template SparseMat<T>::SparseMat(const NRSMat<T> &rhs); \
template void SparseMat<T>::transposeme(); \
template const T* SparseMat<T>::diagonalof(NRVec<T> &r, const bool divide, bool cache) const; \
template SparseMat<T> & SparseMat<T>::operator*=(const T &a); \
template void SparseMat<T>::setunsymmetric(); \
template SparseMat<T> & SparseMat<T>::operator=(const T &a); \
template SparseMat<T> & SparseMat<T>::operator+=(const T &a); \
template SparseMat<T> & SparseMat<T>::operator-=(const T &a); \
template NRMat<T>::NRMat(const SparseMat<T> &rhs); \
template NRSMat<T>::NRSMat(const SparseMat<T> &rhs); \
template NRVec<T>::NRVec(const SparseMat<T> &rhs); \
template const NRVec<T> NRVec<T>::operator*(const SparseMat<T> &mat) const; \
template SparseMat<T> & SparseMat<T>::join(SparseMat<T> &rhs); \
template const T SparseMat<T>::trace() const; \
template const LA_traits<T>::normtype SparseMat<T>::norm(const T scalar) const; \
template void SparseMat<T>::axpy(const T alpha, const SparseMat<T> &x, const bool transp); \
template const SparseMat<T> SparseMat<T>::operator*(const SparseMat<T> &rhs) const; \
template const T SparseMat<T>::dot(const SparseMat<T> &rhs) const; \
template void SparseMat<T>::gemm(const T beta, const SparseMat<T> &a, const char transa, const SparseMat<T> &b, const char transb, const T alpha); \
template void NRVec<T>::gemv(const T beta, const SparseMat<T> &a, const char trans, const T alpha, const NRVec<T> &x, const bool treat_as_symmetric);\
template void SparseMat<T>::permuterows(const NRVec<SPMatindex> &p);\
template void SparseMat<T>::permutecolumns(const NRVec<SPMatindex> &p);\
template void SparseMat<T>::permuteindices(const NRVec<SPMatindex> &p);\



INSTANTIZE(double)
INSTANTIZE(complex<double>) //some functions are not OK for hermitean matrices, needs a revision!!!
*/

//////////////////////////////////////////////////////////////////////////////
//// forced instantization in the corresponding object file
template class SparseMat<double>;
template class SparseMat<complex<double> >;

#define INSTANTIZE(T) \
template NRMat<T>::NRMat(const SparseMat<T> &rhs); \
template NRSMat<T>::NRSMat(const SparseMat<T> &rhs); \
template NRVec<T>::NRVec(const SparseMat<T> &rhs);

INSTANTIZE(double)
INSTANTIZE(complex<double>)


}//namespace
