#include "vec.h"

int main(void)
{
NRVec<double> *p = new NRVec<double>[1000];
NRVec<double> q(10); q=1.;
p[500]|=q;
}
