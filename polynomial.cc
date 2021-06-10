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

#include "polynomial.h"
#include <stdio.h>
#include <string.h>


namespace LA {

template <typename T>
void Polynomial<T>::polydiv(const Polynomial &rhs, Polynomial &q, Polynomial &r) const
{
if(rhs[rhs.degree()]==(T)0) laerror("division by a polynomial with zero leading coefficient - simplify it first");
if(rhs.degree()==0) //scalar division
	{
	q= *this/rhs[0];
	r.resize(0,false);
	r[0]=0;
	return;
	}
int rdegree= rhs.degree();
int qdegree= degree()-rdegree;
if(qdegree<0)
	{
	q.resize(0,false);
	q[0]=0;
	r= *this;
	return;
	}
//general case
q.resize(qdegree,false);
r= *this; r.copyonwrite();
for(int i=degree(); i>=rdegree; --i)
	{
	T tmp= r[i]/rhs[rdegree];
	q[i-rdegree]= tmp;
	r -= rhs.shifted(i-rdegree)*tmp;
	}
r.resize(rhs.degree()-1,true);
}


/***************************************************************************//**
 * forced instantization in the corresponding object file
 ******************************************************************************/
template class Polynomial<int>;
template class Polynomial<double>;
template class Polynomial<std::complex<double> >;

#define INSTANTIZE(T) \



//INSTANTIZE(double)



}//namespace
