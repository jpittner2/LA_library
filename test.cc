#include "bitvector.h"
#include "qsort.h"
#include "la.h"
#include "fourindex.h"
#include "laerror.h"
#include "traceback.h"

void test2(char *msg)
{
laerror(msg);
}

int main(void)
{
sigtraceback(SIGSEGV,1);
sigtraceback(SIGABRT,1);
sigtraceback(SIGBUS,1);
sigtraceback(SIGFPE,1);

bitvector v(100);
v.fill();
bitvector x(50); x=v;
v.copyonwrite();
for(unsigned int i=0; i<100; i+=2) v.reset(i);
x.fill();
x= ~x;
for(unsigned int i=0; i<100; ++i) x.assign(i,i&1);
cout <<v<< endl;
cout <<x<< endl;
cout <<"TEST "<<(x==v)<< " "<<x.population()<<endl;

v.clear(); x.clear();
v.set(31); x.set(32);
cout <<" TEST < "<< (x<v)<<endl;

NRVec<int> t(10);
for(int i=0; i<10; ++i) t[i]=i;
cout <<t;
t.sort(1);
cout <<t;
NRVec<int> u(10);
for(int i=0; i<10;++i) u[i]=i;
ptrqsortup(&t[0],&t[9],&u[0]);
cout<<t <<"U= "<<u;
ptrqsortdown<int,int>(&t[0],&t[9]);
cout<<t;

NRSMat_from1<double> a(5),b(5),c;
c=a+b;

test2("pokus");

fourindex<int,double> f;
fourindex_dense<twoelectronrealmullikan,double,int> ff(f);
}
