#ifndef _QSORT_H
#define _QSORT_H
//general quicksort suitable if we do not want to create extra class for the member type but prefer to encapsulate it into compare and swap soutines
//returns parity of the permutation

template<typename INDEX, typename COMPAR>
int genqsort(INDEX l, INDEX r,COMPAR (*cmp)(const INDEX, const INDEX), void (*swap)(const INDEX,const INDEX)) 
{
INDEX i,j,piv;
int parity=0;

if(r<=l) return parity; //1 element
if(cmp(r,l)<0) {parity^=1; swap(l,r);}
if(r-l==1) return parity; //2 elements and preparation for median
piv= (l+r)/2; //pivoting by median of 3 - safer 
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
template<int type, typename SORTABLE, typename INDEX>
int memqsort(SORTABLE &object, INDEX l, INDEX r)
{
INDEX i,j,piv;
int parity=0;

if(r<=l) return parity; //1 element
if(LA_sort_traits<SORTABLE,INDEX,type>::compare(object,l,r)) {parity^=1; object.swap(l,r);}
if(r-l==1) return parity; //2 elements and preparation for median
piv= (l+r)/2; //pivoting by median of 3 - safer 
if(LA_sort_traits<SORTABLE,INDEX,type>::compare(object,l,piv)) {parity^=1; object.swap(l,piv);} //and change the pivot element implicitly
if(LA_sort_traits<SORTABLE,INDEX,type>::compare(object,piv,r)) {parity^=1; object.swap(r,piv);} //and change the pivot element implicitly
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
	parity^=1; object.swap(i,j); 
	if(i==piv) piv=j; else if(j==piv) piv=i;
	}
  if(i<=j) {i++; j--;}
  }while(i<=j);

if(j-l < r-i)   //because of the stack in bad case process first the shorter subarray
	{if(l<j) parity ^=memqsort<type,SORTABLE,INDEX>(object,l,j); if(i<r) parity ^=memqsort<type,SORTABLE,INDEX>(object,i,r);}
else
	{if(i<r) parity ^=memqsort<type,SORTABLE,INDEX>(object,i,r); if(l<j) parity ^=memqsort<type,SORTABLE,INDEX>(object,l,j);}
return parity;
}

#endif
