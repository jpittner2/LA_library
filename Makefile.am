lib_LTLIBRARIES = libla.la
include_HEADERS = LA_version.h optimize.h tensor.h miscfunc.h simple.h vecmat3.h quaternion.h fortran.h cuda_la.h auxstorage.h  davidson.h lanczos.h  laerror.h    mat.h          qsort.h             vec.h bisection.h   diis.h       la.h         noncblas.h     smat.h numbers.h bitvector.h   fourindex.h  la_traits.h la_random.h  nonclass.h     sparsemat.h sparsesmat.h csrmat.h conjgrad.h    gmres.h      matexp.h     permutation.h   polynomial.h contfrac.h graph.h reg.h regsurf.h
libla_la_SOURCES = simple.cc optimize.cc tensor.cc miscfunc.cc quaternion.cc vecmat3.cc vec.cc mat.cc smat.cc sparsemat.cc sparsesmat.cc csrmat.cc laerror.cc noncblas.cc  numbers.cc bitvector.cc strassen.cc nonclass.cc cuda_la.cc fourindex.cc permutation.cc polynomial.cc contfrac.cc graph.cc la_random.cc reg.cc regsurf.cc
nodist_libla_la_SOURCES = version.cc
noinst_HEADERS = testoptimize.h
check_PROGRAMS = t test tX test_reg test_regsurf testoptimize
TESTS = testoptimize
t_SOURCES = t.cc t2.cc
#nodist_t_SOURCES = 
test_SOURCES = test.cc
tX_SOURCES = tX.cc
test_reg_SOURCES = test_reg.cc
test_regsurf_SOURCES = test_regsurf.cc
testoptimize_SOURCES = testoptimize.cc
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


.PHONY: always manual 
version.cc: always
	./get_git_version >version.cc

# commented out since autoreconf complains about duplicit .PHONY
#include $(top_srcdir)/aminclude.am

#documentation: make manual (alias doc) builds doc/manual.pdf, make doxygen the doxygen pages in doxygen-doc,
#make all builds both besides the library (each only if its tool was found by configure)
MANUAL_TEX = doc/biblio.tex doc/bitvector.tex doc/building.tex doc/concepts.tex doc/fourindex.tex doc/introduction.tex doc/iterative.tex doc/lamacros.tex doc/manual.tex doc/mat.tex doc/nonclass.tex doc/optimize.tex doc/permutation.tex doc/polynomial.tex doc/quaternion.tex doc/regression.tex doc/smat.tex doc/sparse.tex doc/tensor.tex doc/utilities.tex doc/vec.tex doc/vecmat3.tex \
	doc/bib/iterative.tex doc/bib/optimize.tex doc/bib/permutation.tex doc/bib/polynomial.tex doc/bib/quaternion.tex doc/bib/utilities.tex doc/bib/vecmat3.tex
EXTRA_DIST += doc/manual.pdf $(MANUAL_TEX)

doc/manual.pdf: $(MANUAL_TEX)
	mkdir -p doc/.build
	cd doc && for i in 1 2 3; do $(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=.build manual.tex >/dev/null 2>&1 || { tail -20 .build/manual.log; exit 1; }; done
	cp doc/.build/manual.pdf $@

manual doc: doc/manual.pdf

doxygen-doc/html/index.html: doxygen.cfg $(include_HEADERS) $(libla_la_SOURCES)
	$(DX_ENV) $(DX_DOXYGEN) $(DX_CONFIG) >/dev/null

doxygen: doxygen-doc/html/index.html

ALL_DOCS =
if HAVE_PDFLATEX
ALL_DOCS += doc/manual.pdf
endif
#if DX_COND_doc
#ALL_DOCS += doxygen-doc/html/index.html
#endif
all-local: $(ALL_DOCS)

clean-local:
	rm -rf doc/.build doxygen-doc
