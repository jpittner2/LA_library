lib_LTLIBRARIES = libla.la
include_HEADERS = fortran.h cuda_la.h auxstorage.h  davidson.h   laerror.h    mat.h          qsort.h             vec.h bisection.h   diis.h       la.h         noncblas.h     smat.h bitvector.h   fourindex.h  la_traits.h  nonclass.h     sparsemat.h sparsesmat.h csrmat.h conjgrad.h    gmres.h      matexp.h     permutation.h  
libla_la_SOURCES = vec.cc mat.cc smat.cc sparsemat.cc sparsesmat.cc csrmat.cc laerror.cc noncblas.cc  bitvector.cc strassen.cc nonclass.cc cuda_la.cc
check_PROGRAMS = t test
t_SOURCES = t.cc t2.cc 
test_SOURCES = test.cc
LDADD = .libs/libla.a
ACLOCAL_AMFLAGS = -I m4
EXTRA_DIST = LICENSE doxygen.cfg aminclude.am acinclude.m4 footer.html

.cu.o:
	$(NVCC) -o $@ -c $< $(NVCCFLAGS)

#todo: achieve portability of the fortran calls via autoconf?

#use ./configure CXXFLAGS="" LDFLAGS="" to avoid defaults

AM_LDFLAGS=
AM_CXXFLAGS=

AM_CXXFLAGS += -g 
AM_CXXFLAGS += $(OPTIMIZEOPT) $(CUDAOPT) $(FORINTOPT) $(DEBUGOPT) $(MATPTROPT)
AM_CXXFLAGS += -DNO_STRASSEN -DFORTRAN_

AM_CXXFLAGS += $(CBLASOPT) $(CLAPACKOPT)
AM_CXXFLAGS += $(TRACEBACKOPT)
#AM_LDFLAGS += .libs/libla.a
AM_LDFLAGS += $(CBLASLIB)
AM_LDFLAGS += $(BLASLIB)
AM_LDFLAGS += $(ATLASLIB)
AM_LDFLAGS += $(CUDALIBS)
AM_LDFLAGS += $(TRACEBACKLIB)

include $(top_srcdir)/aminclude.am
