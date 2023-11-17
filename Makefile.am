lib_LTLIBRARIES = libla.la
include_HEADERS = version.h simple.h vecmat3.h quaternion.h fortran.h cuda_la.h auxstorage.h  davidson.h   laerror.h    mat.h          qsort.h             vec.h bisection.h   diis.h       la.h         noncblas.h     smat.h bitvector.h   fourindex.h  la_traits.h random.h  nonclass.h     sparsemat.h sparsesmat.h csrmat.h conjgrad.h    gmres.h      matexp.h     permutation.h   polynomial.h contfrac.h graph.h
libla_la_SOURCES = simple.cc quaternion.cc vecmat3.cc vec.cc mat.cc smat.cc sparsemat.cc sparsesmat.cc csrmat.cc laerror.cc noncblas.cc  bitvector.cc strassen.cc nonclass.cc cuda_la.cc fourindex.cc permutation.cc polynomial.cc contfrac.cc graph.cc random.cc
nodist_libla_la_SOURCES = version.cc
check_PROGRAMS = t test tX
t_SOURCES = t.cc t2.cc 
#nodist_t_SOURCES = 
test_SOURCES = test.cc
tX_SOURCES = tX.cc
LDADD = .libs/libla.a
ACLOCAL_AMFLAGS = -I m4
EXTRA_DIST = get_git_version LICENSE doxygen.cfg aminclude.am acinclude.m4 footer.html
CLEANFILES = version.cc
#noinst_PROGRAMS = 

.cu.o:
	$(NVCC) -o $@ -c $< $(NVCCFLAGS)

#todo: achieve portability of the fortran calls via autoconf?

#use ./configure CXXFLAGS="" LDFLAGS="" to avoid defaults

AM_LDFLAGS=
AM_CXXFLAGS=

AM_CXXFLAGS += -g 
AM_CXXFLAGS += $(OPTIMIZEOPT) $(CUDAOPT) $(FORINTOPT) $(DEBUGOPT) $(PROFILEOPT) $(MATPTROPT)
AM_CXXFLAGS += -DNO_STRASSEN -DFORTRAN_

AM_CXXFLAGS += $(CBLASOPT) $(CLAPACKOPT) 
AM_CXXFLAGS += $(MKLOPT)
#AM_LDFLAGS += .libs/libla.a
AM_LDFLAGS += $(CBLASLIB)
AM_LDFLAGS += $(BLASLIB)
AM_LDFLAGS += $(ATLASLIB)
AM_LDFLAGS += $(MKLLIB)
AM_LDFLAGS += $(CUDALIBS)

tX_LDFLAGS = $(AM_LDFLAGS) -lX11 -lm 


.PHONY: always
version.cc: always
	./get_git_version >version.cc

# commented out since autoreconf complains about duplicit .PHONY
#include $(top_srcdir)/aminclude.am
