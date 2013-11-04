/*
    LA: linear algebra C++ interface library
    Copyright (C) 2008 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// LA and general error handler
#include <iostream>
#include "laerror.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdarg.h>

#include "cuda_la.h"

#ifdef USE_TRACEBACK
#include "traceback.h"
#endif

namespace LA {

//enforce GPU initialization by a global class instantization constructor
#ifdef CUDALA
GPU_START gpu_start_instant;
#endif

bool _LA_count_check=true;

extern "C" void _findme(void) {}; //for autoconf test we need a function with C linkage

void laerror2(const char *s1, const char *s2)
{
  std::cerr << "LA:ERROR - ";
  std::cout << "LA:ERROR - ";
  if(s1)
  {
    std::cerr << s2 << ": " << s1 << "\n";
    std::cout << s2 << ": " << s1 << "\n";
  }
#ifdef CUDALA
{
cublasStatus s = cublasGetError();
std::cerr << "CUBLAS status = " << s << std::endl;
std::cout << "CUBLAS status = " << s << std::endl;
}
#endif
  if(errno) perror("system error");

throw LAerror(s1); 
}


//stub for f77 blas called from strassen routine
extern "C" void xerbla_(const char name[6], int *n){
	char msg[1024];
	strcpy(msg,"LAPACK or BLAS error in routine ");
	strncat(msg,name,6);
	sprintf(msg+strlen(msg),": illegal value of parameter #%d",*n);
	laerror(msg);
}


//with atlas-cblas another error routine is necessary
extern "C" void ATL_xerbla(int p, char *rout, char *form, ...){
	char msg0[1024], *msg;
	va_list argptr;
	va_start(argptr, form);
	strcpy(msg0,"ATLAS error\n");
	msg=msg0+strlen(msg0);
	if (p) {sprintf(msg, "Parameter %d to routine %s was incorrect\n", p, rout); msg+=strlen(msg);}
	vsprintf(msg, form, argptr);
	va_end(argptr);
	laerror(msg0);
}

#ifndef NONCBLAS
#include "cblas.h"
#include <stdarg.h>
extern "C" void cblas_xerbla(int p, const char *rout, const char *form, ...)
{
   va_list argptr;

   va_start(argptr, form);
   if (p)
      {
      fprintf(stdout, "Parameter %d to routine %s was incorrect\n", p, rout);
      fprintf(stderr, "Parameter %d to routine %s was incorrect\n", p, rout);
      }
   vfprintf(stdout, form, argptr);
   vfprintf(stderr, form, argptr);
   va_end(argptr);
   laerror("terminating in cblas_xerbla");
}

extern "C" int cblas_errprn(int ierr, int info, char *form, ...) {
	char msg0[1024], *msg;
	va_list argptr;
	va_start(argptr, form);
	sprintf(msg0,"CBLAS error %d %d\n",ierr,info);
	msg=msg0+strlen(msg0);
	vsprintf(msg, form, argptr);
	va_end(argptr);
	laerror(msg0);
	return 0;
}
#endif

}//namespace
