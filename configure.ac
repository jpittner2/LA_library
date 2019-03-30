#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.63])
AC_INIT([libla], [0.8], [jiri@pittnerovi.com])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])
AM_PROG_AR
AM_INIT_AUTOMAKE([-Wall -Werror foreign])
LT_INIT


# Checks for programs.
AC_PROG_CXX
AC_PROG_LIBTOOL
AC_PROG_CC
#AC_PROG_RANLIB obsoleted by libtool
AC_LANG(C++)


# Checks for mandatory libraries.
ATLASLIB=""	
AC_CHECK_LIB([atlas], [ATL_zgemv],ATLASLIB="-lblas -latlas", [
        echo "ATLAS not found, I hope you are using some other (more) efficient BLAS!"
        ])
AC_SUBST([ATLASLIB])

AC_CHECK_LIB([blas], [dgemm_],BLASLIB="-lblas", [
        echo ERROR: BLAS not found! You have to install the BLAS library.
        exit
        ],[$ATLASLIB])
AC_SUBST([BLASLIB])


MATPTROPT=""
AC_ARG_ENABLE([matptr],[  --enable-matptr   switch to double** matrix representation (CUDA incompatible)  [[default=no]]],
    [case "${enableval}" in
     yes) MATPTROPT="-DMATPTR" ;;
     no)  ;;
     *) AC_MSG_ERROR([bad value ${enableval} for --enable-matptr]) ;;
     esac],
 ,)


#cblas and clapack available?
AC_CHECK_LIB([cblas], [cblas_ddot], [CBLASLIB="-lcblas"], [CBLASOPT=-DNONCBLAS],[$ATLASLIB])
AC_CHECK_HEADER([cblas.h],,[CBLASOPT=-DNONCBLAS CBLASLIB=""], AC_INCLUDES_DEFAULT)
AC_SUBST([CBLASLIB])
AC_SUBST([CBLASOPT])


AC_CHECK_LIB([lapack], [dgeev_],, [
        echo ERROR: LAPACK not found! You have to install the LAPACK library
        exit
        ],[$CBLASLIB $BLASLIB $ATLASLIB])


AC_CHECK_LIB([lapack], [clapack_dgesv], , [CLAPACKOPT=-DNONCLAPACK],[$CBLASLIB $BLASLIB $ATLASLIB])
AC_CHECK_HEADER([clapack.h],,[CLAPACKOPT=-DNONCLAPACK], AC_INCLUDES_DEFAULT)
AC_SUBST([CLAPACKOPT])


#CUDA available? link with cublas and avoid cblas and clapack then...
AC_CHECK_LIB([cublas], [cublasInit], [MATPTROPT="" NVCC=nvcc NVCCFLAGS="-O -arch sm_20" CUDALIBS=-lcublas CUDAOPT=-DCUDALA CBLASOPT=-DNONCBLAS CLAPACKOPT=-DNONCLAPACK CBLASLIB=""], [CUDALIB="" CUDAOPT=""])
AC_CHECK_HEADER([cublas.h],,[CUDAOPT="" CUDALIBS=""], AC_INCLUDES_DEFAULT)
AC_SUBST([CUDALIBS])
AC_SUBST([CUDAOPT])
AC_SUBST([NVCC])
AC_SUBST([NVCCFLAGS])
AC_SUBST([MATPTROPT])


#the check for traceback needs bfd to be linked into
AC_CHECK_LIB([bfd], [bfd_fprintf_vma])
AC_CHECK_LIB([traceback], [sigtraceback], [TRACEBACKLIB="-ltraceback -lbfd" TRACEBACKOPT="-DUSE_TRACEBACK -fno-omit-frame-pointer"])
AC_SUBST([TRACEBACKOPT])
AC_SUBST([TRACEBACKLIB])

#process options
FORINTOPT=""
AC_ARG_ENABLE([fotran64int],[  --enable-fotran64int to link with 64-bit-integer-BLAS+LAPACK ],
    [case "${enableval}" in
     yes) FORINTOPT="-DLONG_FORTRAN_INT" 
	CBLASLIB=""
	CBLASOPT=-DNONCBLAS
	CLAPACKOPT=-DNONCLAPACK
	;;
     no) FORINTOPT="" ;;
     *) FORINTOPT=${enableval}
     esac],
 ,)
AC_SUBST([FORINTOPT])


OPTIMIZEOPT="-O3 -finline-limit=1000"
AC_ARG_ENABLE([optimize],[  --enable-optimize   to compile with optimization [[default= -O3 -finline-limit=1000]]],
    [case "${enableval}" in
     yes) ;;
     no) OPTIMIZEOPT="" ;;
     *) OPTIMIZEOPT=${enableval}
     esac],
 ,)
AC_SUBST([OPTIMIZEOPT])

DEBUGOPT="-DDEBUG"
AC_ARG_ENABLE([debug],[  --disable-debug   not to perform some range-checking [[default=yes]]],
    [case "${enableval}" in
     yes) ;;
     no) DEBUGOPT="" ;;
     *) AC_MSG_ERROR([bad value ${enableval} for --enable-debug]) ;;
     esac],
 ,)
AC_SUBST([DEBUGOPT])




# Checks for header files.
AC_CHECK_HEADERS([fcntl.h stdlib.h string.h sys/vfs.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_SIZE_T
AC_TYPE_SSIZE_T
AC_CHECK_MEMBERS([struct stat.st_blksize])

# Checks for library functions.
AC_CHECK_FUNCS([getpagesize memset sqrt])

#Doxygen support
DX_INIT_DOXYGEN($PACKAGE_NAME, doxygen.cfg)

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
echo
echo "**********************************************************************************"
echo "The LA library has now been configured. You may run make; make check; make install"
echo "Please make sure that the generated Makefile employs a proper version of optimized"
echo "BLAS/LAPACK library. If not, re-run configure with CXXFLAGS and LDFLAGS options   "
echo "set to '-I path' and '-L path' for your preferred BLAS/LAPACK library version.    "
echo "In addition, similarly you might set include and link paths for Nvidia CUBLAS.    "
echo "Use  ./configure --disable-optimize CXXFLAGS='' LDFLAGS='' for a fast compile.    "
echo "Use --enable-fotran64int to link with BLAS and LAPACK using 64-bit integers.      "
echo "For documentation, 'make doxygen-doc' and see doxygen-doc/html/index.html         "
echo "For usage examples see file t.cc. Do not forget using copyonwrite() before        "
echo "changing individual matrix/vector elements via l.h.s. operator()                  "
echo "**********************************************************************************"
echo

