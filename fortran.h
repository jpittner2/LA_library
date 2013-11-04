#ifdef FORTRAN_
#define FORNAME(x) x##_
#else
#define FORNAME(x) x
#endif

#ifdef LONG_FORTRAN_INT
#define FORINT
#define FINT long
#else
#undef FORINT
#define FINT int
#endif

#define BLAS_FORTRANCASE(x) toupper(x)
#define LAPACK_FORTRANCASE(x) toupper(x)
