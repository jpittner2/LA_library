#include "vec.h"
#include "smat.h"
#include "mat.h"
#include "sparsemat.h"
#include "nonclass.h"

//GMRES solution of a linear system

//matrix can be any class which has nrows(), ncols(), diagonalof() and NRVec::gemv() available
//does not even have to be explicitly stored

template<typename T, typename Matrix>
extern void gmres(const Matrix &bigmat, const NRVec<T> &b, NRVec<T> &x, const bool doguess=1, const double eps=1e-7, const int MAXIT=50, const bool verbose=1, bool square=1,const bool precondition=1, int neustart=0);
