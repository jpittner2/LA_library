////////////////////////////////////////////////////////////////////////////
//traits classes

#ifndef _LA_TRAITS_INCL
#define _LA_TRAITS_INCL

//default one, good for numbers
template<class C> struct NRMat_traits {
typedef C elementtype;
typedef C producttype;
static C norm (const  C &x) {return abs(x);}
static void axpy (C &s, const C &x, const C &c) {s+=x*c;}
};

//specializations
template<> struct NRMat_traits<NRMat<double> > {
typedef double elementtype;
typedef NRMat<double> producttype;
static double norm (const NRMat<double> &x) {return x.norm();}
static void axpy (NRMat<double>&s, const NRMat<double> &x, const double c) {s.axpy(c,x);}
};

template<> struct NRMat_traits<NRSMat<double> > {
typedef double elementtype;
typedef NRMat<double> producttype;
static const double norm (const NRSMat<double> &x) {return x.norm(0.);}
static void axpy (NRSMat<double>&s, const NRSMat<double> &x, const double c) {s.axpy(c,x);}
};


template<> struct NRMat_traits<NRMat<complex<double> > > {
typedef complex<double> elementtype;
typedef NRMat<complex<double> > producttype;
static double norm (const NRMat<complex<double> >  &x) {return x.norm();}
static void axpy (NRMat<complex<double> >&s, const NRMat<complex<double> > &x, const complex<double> c) {s.axpy(c,x);}
};



#endif
