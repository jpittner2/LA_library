
//Davidson diagonalization of real symmetric matrix

//matrix can be any class which provides nrows(), diagonalof() and operator*(const  NRVec<T>) methods
//does not even have to be explicitly stored - direct CI
//n<0 highest eigenvalues, n>0 lowest eigenvalues

template <class T, class matrix>
extern void davidson(const matrix &a, NRVec<T> *vecs /*input-output*/, T *vals, const int n=1, const double eps=1e-10);
