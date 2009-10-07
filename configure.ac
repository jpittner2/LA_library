#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.63])
AC_INIT([libla], [0.4], [jiri@pittnerovi.com])
AC_CONFIG_HEADERS([config.h])
AM_INIT_AUTOMAKE([-Wall -Werror foreign])

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC
AC_PROG_RANLIB

# Checks for libraries.
AC_CHECK_LIB([blas], [dgemm_])
AC_CHECK_LIB([atlas], [ATL_zgemv])
AC_CHECK_LIB([lapack], [dgeev_])

#optional
AC_CHECK_LIB([cblas], [cblas_ddot], [CBLASLIB=-lcblas], [CBLASOPT=-DNONCBLAS])
AC_SUBST([CBLASLIB])
AC_SUBST([CBLASOPT])

AC_CHECK_LIB([lapack], [clapack_dgesv], , [CLAPACKOPT=-DNONCLAPACK])
AC_SUBST([CLAPACKOPT])



#the check for traceback needs bfd to be linked into
AC_CHECK_LIB([bfd], [bfd_fprintf_vma])
AC_CHECK_LIB([traceback], [sigtraceback], [TRACEBACKLIB="-ltraceback -lbfd" TRACEBACKOPT=-DUSE_TRACEBACK])
AC_SUBST([TRACEBACKOPT])
AC_SUBST([TRACEBACKLIB])

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
