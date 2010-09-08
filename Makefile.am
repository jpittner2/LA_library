lib_LTLIBRARIES = libla.la
include_HEADERS = fortran.h cuda_la.h auxstorage.h  davidson.h   laerror.h    mat.h          qsort.h             vec.h bisection.h   diis.h       la.h         noncblas.h     smat.h bitvector.h   fourindex.h  la_traits.h  nonclass.h     sparsemat.h sparsesmat.h conjgrad.h    gmres.h      matexp.h     permutation.h  
libla_la_SOURCES = vec.cc mat.cc smat.cc sparsemat.cc sparsesmat.cc laerror.cc noncblas.cc  bitvector.cc strassen.cc nonclass.cc cuda_la.cc
check_PROGRAMS = t test
t_SOURCES = t.cc t2.cc 
test_SOURCES = test.cc
LDADD = .libs/libla.a
ACLOCAL_AMFLAGS = -I m4
EXTRA_DIST = LICENSE 

.cu.o:
	$(NVCC) -o $@ -c $< $(NVCCFLAGS)

#todo: achieve portability of the fortran calls via autoconf?

#use ./configure CXXFLAGS="" LDFLAGS="" to avoid defaults
CXXFLAGS += -g 
CXXFLAGS += $(OPTIMIZEOPT) $(CUDAOPT) $(FORINTOPT) $(DEBUGOPT) $(MATPTROPT)
CXXFLAGS += -DNO_STRASSEN -DFORTRAN_

CXXFLAGS += $(CBLASOPT) $(CLAPACKOPT)
LDFLAGS += $(CBLASLIB)
CXXFLAGS += $(TRACEBACKOPT)
LDFLAGS += $(CUDALIBS)
LDFLAGS += $(TRACEBACKLIB)

