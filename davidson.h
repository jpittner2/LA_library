#include "vec.h"
#include "smat.h"
#include "mat.h"
#include "sparsemat.h"
#include "nonclass.h"

//Davidson diagonalization of real symmetric matrix (modified Lanczos)

//matrix can be any class which has nrows(), ncols(), diagonalof() and NRVec::gemv() available
//does not even have to be explicitly stored - direct CI

export template <typename T, typename Matrix>
extern void davidson(const Matrix &bigmat, NRVec<T> &eivals, void *eivecs, 
		int nroots=1,  const bool verbose=0, const double eps=1e-6,
	 	const bool incore=1, int maxit=100, const int maxkrylov = 500);
