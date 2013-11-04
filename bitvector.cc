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

#include "bitvector.h"

namespace LA {

//inefficient I/O operators
std::ostream & operator<<(std::ostream &s, const bitvector &x)
{
for(unsigned int i=0; i<x.size(); ++i) s<< x[i];
return s;
}

std::istream & operator>>(std::istream  &s, bitvector &x)
{
bool a;
x.copyonwrite();
for(unsigned int i=0; i<x.size(); ++i) {s >>a; x.assign(i,a);}
return s;
}


//implemented so that vectors of different length are considered different automatically
bool bitvector::operator!=(const bitvector &rhs) const
{
if(nn!=rhs.nn || modulo!=rhs.modulo) return 1;
if(v==rhs.v) return 0;
if(!modulo) return memcmp(v,rhs.v,nn*sizeof(bitvector_block));
if(memcmp(v,rhs.v,(nn-1)*sizeof(bitvector_block))) return 1;
bitvector_block a=v[nn-1];
bitvector_block b=rhs.v[nn-1];
//zero out the irrelevant bits
bitvector_block mask= ~((bitvector_block)0);
mask <<=modulo;
mask = ~mask;
a&=mask; b&=mask;
return a!=b;
}


bool bitvector::operator>(const bitvector &rhs) const
{
if(nn!=rhs.nn || modulo!=rhs.modulo) laerror("at the moment only bitvectors of the same length comparable");
if(v==rhs.v) return 0;
if(!modulo) return memcmp(v,rhs.v,nn*sizeof(bitvector_block)>0);
int r;
if((r=memcmp(v,rhs.v,(nn-1)*sizeof(bitvector_block)))) return r>0;
bitvector_block a=v[nn-1];
bitvector_block b=rhs.v[nn-1];
//zero out the irrelevant bits
bitvector_block mask= ~((bitvector_block)0);
mask <<=modulo;
mask = ~mask;
a&=mask; b&=mask;
return a>b;
}

bool bitvector::operator<(const bitvector &rhs) const
{
if(nn!=rhs.nn || modulo!=rhs.modulo) laerror("at the moment only bitvectors of the same length comparable");
if(v==rhs.v) return 0;
if(!modulo) return memcmp(v,rhs.v,nn*sizeof(bitvector_block)<0);
int r;
if((r=memcmp(v,rhs.v,(nn-1)*sizeof(bitvector_block)))) return r<0;
bitvector_block a=v[nn-1];
bitvector_block b=rhs.v[nn-1];
//zero out the irrelevant bits
bitvector_block mask= ~((bitvector_block)0);
mask <<=modulo;
mask = ~mask;
a&=mask; b&=mask;
return a<b;
}



bitvector bitvector::operator~() const
{
bitvector r((*this).size());
for(int i=0; i<nn; ++i) r.v[i] = ~v[i];
return r;
}

bitvector& bitvector::operator&=(const bitvector &rhs)
{
#ifdef DEBUG
if(nn!=rhs.nn || modulo!=rhs.modulo) laerror("operation on incompatible bitvectors");
#endif
copyonwrite();
for(int i=0; i<nn; ++i) v[i] &= rhs.v[i];
return *this;
}

bitvector& bitvector::operator|=(const bitvector &rhs)
{
#ifdef DEBUG
if(nn!=rhs.nn || modulo!=rhs.modulo) laerror("operation on incompatible bitvectors");
#endif
copyonwrite();
for(int i=0; i<nn; ++i) v[i] |= rhs.v[i];
return *this;
}

bitvector& bitvector::operator^=(const bitvector &rhs)
{
#ifdef DEBUG
if(nn!=rhs.nn || modulo!=rhs.modulo) laerror("operation on incompatible bitvectors");
#endif
copyonwrite();
for(int i=0; i<nn; ++i) v[i] ^= rhs.v[i];
return *this;
}


/*number of ones in a binary number, from "Hacker's delight" book*/
#ifdef LONG_IS_32 
static unsigned int word_popul(unsigned long x) 
{
x -= ((x>>1)&0x55555555);
x = (x&0x33333333) + ((x>>2)&0x33333333);
x=(x + (x>>4))&0x0f0f0f0f;
x+= (x>>8);
x+= (x>>16);
return x&0x3f;
}
#else
//@@@@ use an efficient trick
static unsigned int word_popul(unsigned long x)
{
unsigned int s=0;
for(int i=0; i<64; ++i)
        {
        if(x&1) ++s;
        x>>=1;
        }
return s;
}

#endif


unsigned int bitvector::population(const unsigned int before) const
{
int i;
unsigned int s=0;
for(i=0; i<nn-1; ++i) s+=word_popul(v[i]);
bitvector_block a=v[nn-1];
if(modulo)
	{
	bitvector_block mask= ~((bitvector_block)0);
	mask <<=modulo;
	a &= ~mask;
	}
return s+word_popul(a);
}

unsigned int bitvector::operator%(const bitvector &y) const
{
if(nn!=y.nn) laerror("incompatible size in bitdifference");

unsigned int s=0;
for(int i=0; i<nn-1; ++i) s+=word_popul(v[i]^y.v[i]);
bitvector_block a=v[nn-1]^y.v[nn-1];
if(modulo)
        {
        bitvector_block mask= ~((bitvector_block)0);
        mask <<=modulo;
        a &= ~mask;
        }
return s+word_popul(a);
}


}//namespace
