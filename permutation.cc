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

#include "permutation.h"
#include <stdio.h>
#include <string.h>


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
NRPerm<T> NRPerm<T>::operator*(const NRPerm<T> q) const
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
NRPerm<T> NRPerm<T>::conjugate_by(const NRPerm<T> q) const
{
#ifdef DEBUG
        if(!this->is_valid() || !q.is_valid()) laerror("multiplication of invalid permutation");
#endif

T n=this->size();
if(n!=q.size()) laerror("product of incompatible permutations");
NRPerm<T> qi=q.inverse();
NRPerm<T> r(n);
for(T i=1; i<=n; ++i) r[i] = qi[(*this)[q[i]]];
return r;

}

template <typename T>
int NRPerm<T>::parity() const
{
if(!this->is_valid()) return 0;
T n=this->size();
if(n==1) return 1;
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
	int j= random()%(i+1);
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
NRVec_from1<T> c(0,n);
NRVec_from1<T> d(1,n);
int j,k,s;
T q;
T t;

this->identity();
PERM_RANK_TYPE sumperm=0;

p2:
        sumperm++;
        if(!select || (select&1) == (sumperm&1)) (*callback)(*this);
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
static T _n;

template <typename T>
static void (*_callback)(const NRPerm<T>& ); 

PERM_RANK_TYPE _sumperm;

template <typename T>
NRPerm<T> *_perm;

template <typename T>
void permg(T n)
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
	(*_callback<T>)(*_perm<T>);
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
(*callback)(*this);
}while(this->next());
return np;
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

#define MAXFACT 20
PERM_RANK_TYPE factorial(const int n)
{
static int ntop=20;
static  PERM_RANK_TYPE a[MAXFACT+1]={1,1,2,6,24,120,720,5040,
40320,
362880,
3628800,
39916800ULL,
479001600ULL,
6227020800ULL,
87178291200ULL,
1307674368000ULL,
20922789888000ULL,
355687428096000ULL,
6402373705728000ULL,
121645100408832000ULL,
2432902008176640000ULL};

int j;
if (n < 0) laerror("negative argument of factorial");
if (n > MAXFACT) laerror("overflow in factorial");
        while (ntop<n) {
                j=ntop++;
                a[ntop]=a[j]*ntop;
                }
return a[n];
}


////////////////////////////////////////////////////////

template <typename T>
CyclePerm<T>:: CyclePerm(const NRPerm<T> &p)
{
#ifdef DEBUG
        if(!p.is_valid()) laerror("invalid permutation");
#endif
T n=p.size();
NRVec_from1<T> used(0,n),tmp(n);
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
CyclePerm<T> CyclePerm<T>::operator*(const CyclePerm q) const
{
int m=this->max();
int mm=q.max();
if(mm>m) mm=m;
NRPerm<T> qq(q,m);
NRPerm<T> pp(*this,m);
NRPerm<T> rr=pp*qq;
return CyclePerm<T>(rr);
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
CompressedPartition<T> CyclePerm<T>::cycles(const T n) const
{
#ifdef DEBUG
if(!this->is_valid()) laerror("operation with an invalid cycleperm");
#endif
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
std::cout<<"TEST "<<nparts()<<" "<<n<<std::endl;
if(nparts()>n) return 0; //too antisymmetric partition
int i,j;
double prod;
int r=this->size();
NRVec_from1<int> p(n);
for(i=1;i<=n;i++) p[i]= (i<=r?(*this)[i]:0)+n-i;
std::cout<<"TEST "<<p<<std::endl;
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
if(x.is_valid()) laerror("operation with an invalid tableaux");
#endif
int nparts=x.size();
int n=0;
for(int i=1; i<=nparts; ++i) n+= x[i].size();
this->resize(n);
this->clear();
for(int i=1; i<=nparts; ++i) (*this)[i]=x[i].size();
}

PERM_RANK_TYPE longpow(PERM_RANK_TYPE x, int i)
{
if(i<0) return 0;
PERM_RANK_TYPE y=1;
do
        {
        if(i&1) y *= x;
	i >>= 1;
        if(i) x *= x;
        }
while(i);
return y ;
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
if(remain==0) {++partitioncount; (*_pcallback<T>)(*partition<T>); return;}
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
for(int i=1; i<=nlines; ++i) (*this)[i].resize(frame[i]);
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
int YoungTableaux<T>::sum() const
{
#ifdef DEBUG
if(is_valid()) laerror("invalid young frame");
#endif
int sum=0;
int nrows=this->size();
for(int i=1; i<=nrows; ++i) sum += (*this)[i].size();
return sum;
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



/***************************************************************************//**
 * forced instantization in the corresponding object file
 ******************************************************************************/
template class NRPerm<int>;
template class CyclePerm<int>;
template class CompressedPartition<int>;
template class Partition<int>;
template class YoungTableaux<int>;

#define INSTANTIZE(T) \
template std::istream & operator>>(std::istream &s, CyclePerm<T> &x); \
template std::ostream & operator<<(std::ostream &s, const CyclePerm<T> &x); \
template std::ostream & operator<<(std::ostream &s, const CompressedPartition<T> &x); \
template std::ostream & operator<<(std::ostream &s, const YoungTableaux<T> &x); \



INSTANTIZE(int)




}//namespace
