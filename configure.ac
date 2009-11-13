#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.63])
AC_INIT([libla], [0.5], [jiri@pittnerovi.com])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])
AM_INIT_AUTOMAKE([-Wall -Werror foreign])
LT_INIT


# Checks for programs.
AC_PROG_CXX
AC_PROG_LIBTOOL
AC_PROG_CC
#AC_PROG_RANLIB obsoleted by libtool
AC_LANG(C++)


# Checks for mandatory libraries.
AC_CHECK_LIB([blas], [dgemm_],, [
        echo ERROR: BLAS not found! You have to install the BLAS library.
        exit
        ])

AC_CHECK_LIB([atlas], [ATL_zgemv],, [
        echo "ATLAS not found, I hope you are using some other (more) efficient BLAS!"
        ])

AC_CHECK_LIB([lapack], [dgeev_],, [
        echo ERROR: LAPACK not found! You have to install the LAPACK library
        exit
        ])


#check for optional libraries
AC_CHECK_LIB([cblas], [cblas_ddot], [CBLASLIB=-lcblas], [CBLASOPT=-DNONCBLAS])
AC_SUBST([CBLASLIB])
AC_SUBST([CBLASOPT])

AC_CHECK_LIB([lapack], [clapack_dgesv], , [CLAPACKOPT=-DNONCLAPACK])
AC_SUBST([CLAPACKOPT])

#the check for traceback needs bfd to be linked into
AC_CHECK_LIB([bfd], [bfd_fprintf_vma])
AC_CHECK_LIB([traceback], [sigtraceback], [TRACEBACKLIB="-ltraceback -lbfd" TRACEBACKOPT="-DUSE_TRACEBACK -fno-omit-frame-pointer"])
AC_SUBST([TRACEBACKOPT])
AC_SUBST([TRACEBACKLIB])

#process options
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

MATPTROPT=""
AC_ARG_ENABLE([matptr],[  --enable-matptr   switch to double** matrix representation  [[default=no]]],
    [case "${enableval}" in
     yes) MATPTROPT="-DMATPTR" ;;
     no)  ;;
     *) AC_MSG_ERROR([bad value ${enableval} for --enable-matptr]) ;;
     esac],
 ,)
AC_SUBST([MATPTROPT])




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

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
echo
echo "**********************************************************************************"
echo "The LA library has now been configured. You may run make; make check; make install"
echo "Please make sure that the generated Makefile employs a proper version of optimized"
echo "BLAS/LAPACK library. If not, re-run configure with CXXFLAGS and LDFLAGS options   "
echo "set to '-I path' and '-L path' for your preferred BLAS/LAPACK library version     "
echo "For usage examples see file t.cc. Do not forget using copyonwrite() before        "
echo "changing individual matrix/vector elements via l.h.s. operator[] or operator()    "
echo "**********************************************************************************"
