#include "vec.h"
#include "smat.h"
#include "mat.h"
#include "sparsemat.h"
#include "nonclass.h"

//conjugate gradient solution of a linear system

//matrix can be any class which has nrows(), ncols(), diagonalof() and NRVec::gemv() available
//does not even have to be explicitly stored


template<typename T, typename Matrix>
extern void conjgrad(const Matrix &bigmat, const NRVec<T> &b, NRVec<T> &x, const bool doguess, const double tol, const int itmax, const bool verbose, bool issquare,const bool precondition);
