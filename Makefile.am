lib_LIBRARIES = libla.a
include_HEADERS = auxstorage.h  davidson.h   laerror.h    mat.h          qsort.h             vec.h bisection.h   diis.h       la.h         noncblas.h     smat.h bitvector.h   fourindex.h  la_traits.h  nonclass.h     sparsemat.h conjgrad.h    gmres.h      matexp.h     permutation.h  sparsemat_traits.h 
libla_a_SOURCES = vec.cc mat.cc smat.cc sparsemat.cc laerror.cc noncblas.cc  bitvector.cc strassen.cc nonclass.cc
check_PROGRAMS = t test
t_SOURCES = t.cc t2.cc 
test_SOURCES = test.cc
LDADD = libla.a

EXTRA_DIST = LICENSE 

#
CXXFLAGS += -fno-omit-frame-pointer -O3 -g -fPIC -finline-limit=1000
CXXFLAGS += -DNO_STRASSEN -DFORTRAN_

#remove this for production code efficiency
CXXFLAGS += -DDEBUG

CXXFLAGS += $(CBLASOPT) $(CLAPACKOPT)
LDFLAGS += $(CBLASLIB)
CXXFLAGS += $(TRACEBACKOPT)
LDFLAGS += $(TRACEBACKLIB)


