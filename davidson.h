#include "vec.h"
#include "smat.h"
#include "mat.h"
#include "sparsemat.h"
#include "nonclass.h"

//Davidson diagonalization of real symmetric matrix

//matrix can be any class which has nrows(), ncols(), diagonalof() and NRVec::gemv() available
//does not even have to be explicitly stored - direct CI
//n<0 highest eigenvalues, n>0 lowest eigenvalues

export template <typename T, typename Matrix>
extern void davidson(const Matrix &bigmat, NRVec<T> *eivecs /*input-output*/, NRVec<T> &eivals, int nroots=1, const double accur=1e-6, int nits=50, const int ndvdmx = 500, const bool incore=1);
