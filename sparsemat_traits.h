////////////////////////////////////////////////////////////////////////////
//traits classes

#ifndef _SPARSEMAT_TRAITS_INCL
#define _SPARSEMAT_TRAITS_INCL


template<> struct NRMat_traits<SparseMat<double> > {
typedef double elementtype;
typedef SparseMat<double> producttype;
static double norm (const SparseMat<double> &x) {return x.norm();}
static void axpy (SparseMat<double>&s, const SparseMat<double> &x, const double c) {s.axpy(c,x);}
};

#endif
