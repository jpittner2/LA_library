/*
    LA: linear algebra C++ interface library
    Copyright (C) 2008-2023 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>

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

#include "numbers.h"


namespace LA {

template<typename N>
N primefactor(const N &x, const N &last)
{
N i,t;
   if ( x <= 2 ) return x;
   if ( !(x & 1) ) return 2;
   i = last >3 ? last : 3;
   for (;;) { 
      t = x / i; 
    if ( t < i ) break;
      if ( t * i == x )
         return i;
      i += 2;
      }
   return x;
}


template<typename N>
FACTORIZATION<N>  factorization(const N &x)
{
FACTORIZATION<N> f;
N y=x;
N last=0;
while(y>1)
	{
	N z=primefactor(y,last);
	if(z!=last)
		{	
		std::pair<N,N> p;
		p.first=z;
		p.second=1;
		f.push_back(p);
		}
	else
		++f.back().second;
	last=z;
	y/=z;
	}
return f;
}


template<typename N>
N nextprime(N x)
{
if ( x < 2 ) return 2;
if ( !(x & 1) ) x--;
while ( !isprime(x += 2) );
return x;
}


template <typename N>
std::ostream & operator<<(std::ostream &s, const FACTORIZATION<N>  &x) 
{
for(auto p=x.begin(); p!=x.end(); ++p) s<<"("<<p->first<<","<<p->second<<") ";
return s;
}


template <typename N>
N eulerphi(const FACTORIZATION<N> &x)
{
N e=1;
for(auto p=x.begin(); p!=x.end(); ++p) 
	{
	e *= (p->first-1);
	if(p->second>1) e*= pow(p->first, p->second-1);
	}
return e;
}


template <typename N>
N pow(const N &x, N i)
{
if(i==0) return 1;
if(i==1) return x;
N y,z;  
z=x;            
while(!(i&1))   
        {       
        z = z*z;
        i >>= 1;
        }       
y=z;    
while((i >>= 1)/*!=0*/)
                {
                z = z*z;
                if(i&1) y = y*z;
                }
return y;
}



//force instantization

#define INSTANTIZE(N) \
template N primefactor(const N &x, const N &last); \
template FACTORIZATION<N>  factorization(const N &x); \
template N nextprime(N x); \
template std::ostream & operator<<(std::ostream &s, const FACTORIZATION<N>  &x); \
template N pow(const N &x, N i); \
template N eulerphi(const FACTORIZATION<N> &f); \





INSTANTIZE(uint64_t)

}//namespace
