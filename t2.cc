
#include <time.h>
#include "la.h"
#include "sparsemat.h"
#include "matexp.h"
#include "fourindex.h"
#include "bitvector.h"

#ifdef USE_TRACEBACK
#include "traceback.h"
#endif

void test(const NRVec<double> &x)
{
NRMat<double> aa(0.,2,2);
NRMat<double> cc(aa);
cc.copyonwrite(); cc[0][0]=1.;

cout << aa << cc <<"\n";
cout <<"test x= "<<x<<"\n";
}

