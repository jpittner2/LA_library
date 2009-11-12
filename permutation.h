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
#ifndef _PERMUTATION_H
#define _PERMUTATION_H
namespace LA {
template<typename T>
const NRVec<T> inversepermutation(const NRVec<T> &p, const T offset=0)
{
int n=p.size();
NRVec<T> q(n);
if(!offset) for(int i=0; i<n; ++i) q[p[i]]=i;
else for(int i=0; i<n; ++i) q[p[i]-offset]=i+offset;
return q;
}
}//namespace
#endif
