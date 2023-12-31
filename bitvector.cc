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

#include "bitvector.h"
#include <unistd.h>

namespace LA {

//inefficient I/O operators
std::ostream & operator<<(std::ostream &s, const bitvector &x)
{
for(unsigned int i=0; i<x.size(); ++i) s<< x[i];
return s;
}

std::istream & operator>>(std::istream  &s, bitvector &x)
{
std::string str;
s >> str;
x.resize(str.size());
for(unsigned int i=0; i<x.size(); ++i) {x.assign(i,str[i]!='0');}
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
if(size()<rhs.size()) resize(rhs.size(),true);
copyonwrite();
for(int i=0; i<nn; ++i) v[i] &= (i>=rhs.nn? 0 : rhs.v[i]);
return *this;
}

bitvector& bitvector::operator|=(const bitvector &rhs)
{
if(size()<rhs.size()) resize(rhs.size(),true);
copyonwrite();
for(int i=0; i<nn && i<rhs.nn; ++i) v[i] |= rhs.v[i];
return *this;
}

bitvector& bitvector::operator^=(const bitvector &rhs)
{
if(size()<rhs.size()) resize(rhs.size(),true);
copyonwrite();
for(int i=0; i<nn && i<rhs.nn; ++i) v[i] ^= rhs.v[i];
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
//@@@@ use an efficient trick too
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


bitvector& bitvector::operator>>=(unsigned int i)
{
if(i==0) return *this;
copyonwrite();
unsigned int imod = i%blockbits;
unsigned int ishift = i/blockbits;
for(int dest=0; dest<nn; ++dest)
	{
	int src=dest+ishift;
	if(src>=nn) v[dest]=0;
	else
		{
		v[dest] = v[src]>>imod;
		if(imod && (src+1<nn)) v[dest] |= (v[src+1]&((1ULL<<imod)-1)) <<(blockbits-imod);
		}
	}
return *this;
}

bitvector& bitvector::leftshift(unsigned int i, bool autoresize)
{
if(i==0) return *this;
copyonwrite();
unsigned int imod = i%blockbits;
unsigned int ishift = i/blockbits;
if(autoresize) resize(size()+i,true);
for(int dest=nn-1; dest>=0; --dest)
        {
        int src=dest-ishift;
        if(src<0) v[dest]=0;
        else
                {
                v[dest] = v[src]<<imod;
                if(imod && (src-1>=0)) v[dest] |= (v[src-1]& (((1ULL<<imod)-1) <<(blockbits-imod)))>>(blockbits-imod);
                }
        }

return *this;
}


void  bitvector::randomize()
{
copyonwrite();
for(int i=0; i<nn; ++i) v[i]=RANDINT64();
//zero the excess bits
if(modulo) 
	{
	bitvector_block mask = (1ULL<<modulo)-1;
	v[nn-1] &= mask;
	}
}



unsigned int bitvector::population(const unsigned int before) const
{
if(before) laerror("before parameter in population() not implemented yet");
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

unsigned int bitvector::bitdiff(const bitvector &y) const
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

static unsigned int nlz64(uint64_t x0)
{
int64_t x=x0;
uint64_t y;
unsigned int n;
n=0;
y=x;
L: if ( x<0) return n;
if(y==0) return 64-n;
++n;
x<<=1;
y>>=1;
goto L;
}

static unsigned int ntz64(uint64_t x)
{
unsigned int n;
if(x==0) return 64;
n=1;
if((x&0xffffffff)==0) {n+=32; x>>=32;}
if((x&0xffff)==0) {n+=16; x>>=16;}
if((x&0xff)==0) {n+=8; x>>=8;}
if((x&0xf)==0) {n+=4; x>>=4;}
if((x&0x3)==0) {n+=2; x>>=2;}
return n-(x&1);
}

unsigned int bitvector::nlz() const
{
int leadblock=nn-1;
unsigned int n=0;
while(leadblock>0 && v[leadblock] == 0) 
	{
	--leadblock;
	n+=blockbits;
	}
n+= nlz64(v[leadblock]);
if(modulo) n-= blockbits-modulo;
return n;
}

unsigned int bitvector::ntz() const
{
int tailblock=0;
unsigned int n=0;
if(iszero()) return size();
while(tailblock<nn-1 && v[tailblock] == 0)
        {
        ++tailblock;
        n+=blockbits;
        }
n+= ntz64(v[tailblock]);
return n;
}

//NOTE: naive algorithm, just for testing
//does not perform modulo irreducible polynomial, is NOT GF(2^n) multiplication
bitvector bitvector::multiply(const bitvector &rhs, bool autoresize) const
{
int maxsize=size(); if(rhs.size()>maxsize) maxsize=rhs.size();
bitvector r(autoresize?size()+rhs.size():maxsize);
r.clear();
bitvector tmp(rhs);
if(autoresize) tmp.resize(size()+rhs.size(),true);
for(int i=0; i<=degree(); ++i)
	{
	if((*this)[i]) r+= tmp;
	tmp.leftshift(1,false);
	}
return r;
}


//this is GF(2^n) multiplication
bitvector bitvector::field_mult(const bitvector &rhs, const bitvector &irpolynom) const
{
int d=irpolynom.degree();
if(d>size()||d>rhs.size()) laerror("inconsistent dimensions in field_mult");
bitvector r(size());
r.clear();
bitvector tmp(*this);
tmp.resize(size()+1,true);
int rd=rhs.degree();
for(int i=0; i<=rd; ++i) //avoid making a working copy of rhs and shifting it
        {
        if(rhs[i]) r+= tmp;
        tmp.leftshift(1,false);
  	if(tmp[d]) tmp -= irpolynom;
        }
return r;
}



//this is GF(2^n) multiplicative inverseion
//cf. https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
bitvector bitvector::field_inv(const bitvector &irpolynom) const
{
int d=irpolynom.degree();
if(d>size()) laerror("inconsistent dimensions in field_inv");

bitvector t(size()); t.clear();
bitvector newt(size()); newt.clear(); newt.set(0);
bitvector r(irpolynom); r.copyonwrite();
bitvector newr(*this); if(r.size()>newr.size()) newr.resize(r.size(),true); newr.copyonwrite();
int rs=r.size();


while(!newr.is_zero())
	{
	//std::cout <<"r "<<r<<" newr "<<newr <<" "; std::cout <<"t "<<t<<" newt "<<newt; std::cout <<std::endl;
	bitvector remainder(rs);
	bitvector quotient = r.division(newr,remainder);
	r=newr; newr=remainder;
	remainder= t - quotient.multiply(newt,false); //avoid size growth
	t=newt; newt=remainder;
	}

if(r.degree()>0) laerror("field_inv: polynomial is not irreducible or input is its multiple");
if(!r[0]) laerror("zero in field_inv");

return t;
}



void bitvector::resize(const unsigned int n, bool preserve)
{
int old=size(); 
NRVec<bitvector_block>::resize((n+blockbits-1)/blockbits,preserve); 
modulo=n%blockbits; 
if(preserve)  //clear newly allocated memory
	{
	for(int i=old; i<nn*blockbits; ++i) this->reset(i);
	} 
else clear();
}


bitvector bitvector::division(const bitvector &rhs,  bitvector &remainder) const
{
if(rhs.is_zero()) laerror("division by zero binary polynomial");
if(is_zero() || rhs.is_one()) {remainder.clear(); return *this;}
bitvector r(size());
r.clear();
remainder= *this;
remainder.copyonwrite();

int rhsd = rhs.degree();
int d;
while((d=remainder.degree()) >= rhsd)
	{
	unsigned int pos = d-rhsd;
	r.set(pos);
	remainder -= rhs<<pos;
	}
remainder.resize(rhs.size(),true);
return r;
}

bitvector bitvector::gcd(const bitvector &rhs) const
{
bitvector big,small;

if(degree()>=rhs.degree()) 
	{big= *this; small=rhs;}
else
	{big=rhs; small= *this;}

if(big.is_zero())
	{
	if(small.is_zero()) laerror("two zero arguments in gcd");
	return small;
	}

if(small.is_zero()) return big;
if(small.is_one()) return small;
if(big.is_one()) return big;

do      {
        bitvector help=small;
        small= big%small;
        big=help;
        }
while(! small.is_zero());
return big;



}

void bitvector::read(int fd, bool dimensions, bool transp)
{
if(dimensions) 
	{
	int r = ::read(fd,&modulo,sizeof(modulo));
	if(r!=sizeof(modulo)) laerror("cannot read in bitvector");
	}
NRVec<bitvector_block>::get(fd,dimensions,transp);
}

void bitvector::write(int fd, bool dimensions, bool transp)
{
if(dimensions)
        {
        int r = ::write(fd,&modulo,sizeof(modulo));
        if(r!=sizeof(modulo)) laerror("cannot write in bitvector");
        }
NRVec<bitvector_block>::put(fd,dimensions,transp);
}




}//namespace
