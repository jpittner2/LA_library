#include "bitvector.h"

//inefficient I/O operators
ostream & operator<<(ostream &s, const bitvector &x)
{
for(unsigned int i=0; i<x.size(); ++i) s<< x[i];
return s;
}

istream & operator>>(istream  &s, bitvector &x)
{
bool a;
for(unsigned int i=0; i<x.size(); ++i) {s >>a; x.assign(i,a);}
return s;
}


