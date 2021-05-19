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
std::list<NRVec_from1<T> > cyclelist={};
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
for(auto l=cyclelist.begin(); l!=cyclelist.end(); ++l) (*this)[i++] = *l;
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
Partition<T> CyclePerm<T>::cycles(const T n) const
{
#ifdef DEBUG
if(!this->is_valid()) laerror("operation with an invalid cycleperm");
#endif
Partition<T> r(n); r.clear();
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
for(auto l=cycle.begin(); l!=cycle.end(); ++l) c[++i] = *l;

return closebracket+1;
}

template <typename T>
void CyclePerm<T>::readfrom(const std::string &line)
{
const char *p=line.c_str();
std::list<NRVec<T> > cyclelist={};
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
for(auto l=cyclelist.begin(); l!=cyclelist.end(); ++l) (*this)[++i] = *l;
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


/***************************************************************************//**
 * forced instantization in the corresponding object file
 ******************************************************************************/
template class NRPerm<int>;
template class CyclePerm<int>;
template class Partition<int>;

#define INSTANTIZE(T) \
template std::istream & operator>>(std::istream &s, CyclePerm<T> &x); \
template std::ostream & operator<<(std::ostream &s, const CyclePerm<T> &x); \



INSTANTIZE(int)




}//namespace
