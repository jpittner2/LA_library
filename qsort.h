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
#ifndef _QSORT_H
#define _QSORT_H
//quicksort, returns parity of the permutation
//
namespace LA {

template<typename INDEX, typename COMPAR>
int genqsort(INDEX l, INDEX r,COMPAR (*cmp)(const INDEX, const INDEX), void (*swap)(const INDEX,const INDEX)) 
{
INDEX i,j,piv;
int parity=0;

if(r<=l) return parity; //1 element
if(cmp(r,l)<0) {parity^=1; swap(l,r);}
if(r-l==1) return parity; //2 elements and preparation for median
piv= l+(r-l)/2; //pivoting by median of 3 - safer 
if(cmp(piv,l)<0) {parity^=1; swap(l,piv);} //and change the pivot element implicitly
if(cmp(r,piv)<0) {parity^=1; swap(r,piv);} //and change the pivot element implicitly
if(r-l==2) return parity; //in the case of 3 elements we are finished too

//general case , l-th r-th already processed
i=l+1; j=r-1;
do{
  //important sharp inequality - stops at sentinel element for efficiency
  // this is inefficient if all keys are equal - unnecessary n log n swaps are done, but we assume that it is atypical input
  while(cmp(i++,piv)<0);
  i--;
  while(cmp(j--,piv)>0);
  j++;
  if(i<j)
	{
	// swap and keep track of position of pivoting element 
	parity^=1; swap(i,j); 
	if(i==piv) piv=j; else if(j==piv) piv=i;
	}
  if(i<=j) {i++; j--;}
  }while(i<=j);

if(j-l < r-i)   //because of the stack in bad case process first the shorter subarray
	{if(l<j) parity ^=genqsort(l,j,cmp,swap); if(i<r) parity ^=genqsort(i,r,cmp,swap);}
else
	{if(i<r) parity ^=genqsort(i,r,cmp,swap); if(l<j) parity ^=genqsort(l,j,cmp,swap);}
return parity;
}



//for SORTABLE classes which provide LA_sort_traits<SORTABLE,INDEX,type>::compare and swap member functions
//this allows to use it in general templates also for complex elements, for which comparison falls back to error
template<int type, typename SORTABLE, typename INDEX, typename PERMINDEX>
int memqsort(SORTABLE &object, PERMINDEX *perm, INDEX l, INDEX r)
{
INDEX i,j,piv;
int parity=0;

if(r<=l) return parity; //1 element
if(LA_sort_traits<SORTABLE,INDEX,type>::compare(object,l,r)) {parity^=1; object.swap(l,r); if(perm) {PERMINDEX tmp=perm[l]; perm[l]=perm[r]; perm[r]=tmp;}}
if(r-l==1) return parity; //2 elements and preparation for median
piv= l+(r-l)/2; //pivoting by median of 3 - safer 
if(LA_sort_traits<SORTABLE,INDEX,type>::compare(object,l,piv)) {parity^=1; object.swap(l,piv); if(perm) {PERMINDEX tmp=perm[l]; perm[l]=perm[piv]; perm[piv]=tmp;}} //and change the pivot element implicitly
if(LA_sort_traits<SORTABLE,INDEX,type>::compare(object,piv,r)) {parity^=1; object.swap(r,piv); if(perm) {PERMINDEX tmp=perm[r]; perm[r]=perm[piv]; perm[piv]=tmp;}} //and change the pivot element implicitly
if(r-l==2) return parity; //in the case of 3 elements we are finished too

//general case , l-th r-th already processed
i=l+1; j=r-1;
do{
  //important sharp inequality - stops at sentinel element for efficiency
  // this is inefficient if all keys are equal - unnecessary n log n swaps are done, but we assume that it is atypical input
  while(LA_sort_traits<SORTABLE,INDEX,type>::compare(object,piv,i++));
  i--;
  while(LA_sort_traits<SORTABLE,INDEX,type>::compare(object,j--,piv));
  j++;
  if(i<j)
	{
	// swap and keep track of position of pivoting element 
	parity^=1; object.swap(i,j);  if(perm) {PERMINDEX tmp=perm[i]; perm[i]=perm[j]; perm[j]=tmp;}
	if(i==piv) piv=j; else if(j==piv) piv=i;
	}
  if(i<=j) {i++; j--;}
  }while(i<=j);

if(j-l < r-i)   //because of the stack in bad case process first the shorter subarray
	{if(l<j) parity ^=memqsort<type,SORTABLE,INDEX,PERMINDEX>(object,perm,l,j); if(i<r) parity ^=memqsort<type,SORTABLE,INDEX,PERMINDEX>(object,perm,i,r);}
else
	{if(i<r) parity ^=memqsort<type,SORTABLE,INDEX,PERMINDEX>(object,perm,i,r); if(l<j) parity ^=memqsort<type,SORTABLE,INDEX,PERMINDEX>(object,perm,l,j);}
return parity;
}


template<typename S, typename PERMINDEX>
int ptrqsortup(S *l, S *r, PERMINDEX *perm=NULL)
{
S *i,*j,*piv;
int parity=0;

if(r-l<=0) return parity; //1 element
if(*l > *r) {parity^=1; {S tmp; tmp=*l; *l= *r; *r=tmp;} if(perm) {PERMINDEX tmp=*perm; *perm=perm[r-l]; perm[r-l]=tmp;}}
if(r-l==1) return parity; //2 elements and preparation for median
piv= l+(r-l)/2; //pivoting by median of 3 - safer 
if(*l>*piv) {parity^=1; {S tmp; tmp=*l; *l=*piv; *piv=tmp;} if(perm) {PERMINDEX tmp= *perm; *perm=perm[piv-l]; perm[piv-l]=tmp;}} //and change the pivot element implicitly
if(*piv>*r) {parity^=1; {S tmp; tmp=*r; *r=*piv; *piv=tmp;} if(perm) {PERMINDEX tmp=perm[r-l]; perm[r-l]=perm[piv-l]; perm[piv-l]=tmp;}} //and change the pivot element implicitly
if(r-l==2) return parity; //in the case of 3 elements we are finished too

//general case , l-th r-th already processed
i=l+1; j=r-1;
do{
  //important sharp inequality - stops at sentinel element for efficiency
  // this is inefficient if all keys are equal - unnecessary n log n swaps are done, but we assume that it is atypical input
  while(*piv > *i++);
  i--;
  while(*j-- > *piv);
  j++;
  if(i<j)
	{
	// swap and keep track of position of pivoting element 
	parity^=1; {S tmp; tmp=*i; *i=*j; *j=tmp;} if(perm) {PERMINDEX tmp=perm[i-l]; perm[i-l]=perm[j-l]; perm[j-l]=tmp;}
	if(i==piv) piv=j; else if(j==piv) piv=i;
	}
  if(i<=j) {i++; j--;}
  }while(i<=j);

if(j-l < r-i)   //because of the stack in bad case process first the shorter subarray
	{if(l<j) parity ^=ptrqsortup(l,j,perm); if(i<r) parity ^=ptrqsortup(i,r,perm+(i-l));}
else
	{if(i<r) parity ^=ptrqsortup(i,r,perm+(i-l)); if(l<j) parity ^=ptrqsortup(l,j,perm);}
return parity;
}


template<typename S, typename PERMINDEX>
int ptrqsortdown(S *l, S *r, PERMINDEX *perm=NULL)
{
S *i,*j,*piv;
int parity=0;

if(r-l<=0) return parity; //1 element
if(*l < *r) {parity^=1; {S tmp; tmp=*l; *l= *r; *r=tmp;} if(perm) {PERMINDEX tmp=*perm; *perm=perm[r-l]; perm[r-l]=tmp;}}
if(r-l==1) return parity; //2 elements and preparation for median
piv= l+(r-l)/2; //pivoting by median of 3 - safer 
if(*l<*piv) {parity^=1; {S tmp; tmp=*l; *l=*piv; *piv=tmp;} if(perm) {PERMINDEX tmp= *perm; *perm=perm[piv-l]; perm[piv-l]=tmp;}} //and change the pivot element implicitly
if(*piv<*r) {parity^=1; {S tmp; tmp=*r; *r=*piv; *piv=tmp;} if(perm) {PERMINDEX tmp=perm[r-l]; perm[r-l]=perm[piv-l]; perm[piv-l]=tmp;}} //and change the pivot element implicitly
if(r-l==2) return parity; //in the case of 3 elements we are finished too

//general case , l-th r-th already processed
i=l+1; j=r-1;
do{
  //important sharp inequality - stops at sentinel element for efficiency
  // this is inefficient if all keys are equal - unnecessary n log n swaps are done, but we assume that it is atypical input
  while(*piv < *i++);
  i--;
  while(*j-- < *piv);
  j++;
  if(i<j)
	{
	// swap and keep track of position of pivoting element 
	parity^=1; {S tmp; tmp=*i; *i=*j; *j=tmp;} if(perm) {PERMINDEX tmp=perm[i-l]; perm[i-l]=perm[j-l]; perm[j-l]=tmp;}
	if(i==piv) piv=j; else if(j==piv) piv=i;
	}
  if(i<=j) {i++; j--;}
  }while(i<=j);

if(j-l < r-i)   //because of the stack in bad case process first the shorter subarray
	{if(l<j) parity ^=ptrqsortdown(l,j,perm); if(i<r) parity ^=ptrqsortdown(i,r,perm+(i-l));}
else
	{if(i<r) parity ^=ptrqsortdown(i,r,perm+(i-l)); if(l<j) parity ^=ptrqsortdown(l,j,perm);}
return parity;
}

}//namespace
#endif
