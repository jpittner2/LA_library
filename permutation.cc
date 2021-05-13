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
namespace LA {

template <typename T>
void NRPerm<T>::identity()
{
T n=this->size();
#ifdef DEBUG
        if(n<0) laerror("invalid permutation size");
#endif
if(n==0) return;
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




/***************************************************************************//**
 * forced instantization in the corresponding object file
 ******************************************************************************/
template class NRPerm<int>;
}
