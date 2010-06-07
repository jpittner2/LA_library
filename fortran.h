#ifdef FORTRAN_
#define FORNAME(x) x##_
#else
#define FORNAME(x) x
#endif

#define FORTRAN_INT long
