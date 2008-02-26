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
#include <errno.h>
#include <stdarg.h>


#ifdef USE_TRACEBACK
#include "traceback.h"
#endif

void laerror(const char *s1)
{
  std::cerr << "LA:ERROR - ";
  std::cout << "LA:ERROR - ";
  if(s1)
  {
    std::cerr << s1  << "\n";
    std::cout << s1  << "\n";
  }
  if(errno) perror("system error");

#ifdef USE_TRACEBACK
  traceback(1);
  exit(1);
#else
  throw LAerror(s1); //traceback possible via sigtraceback, but BFD library might fail for -O3 code on some machines and source lines would not be available
#endif
}

//stub for f77 blas called from strassen routine
extern "C" void xerbla_(const char name[6], int *n)
{
char msg[1024];
strcpy(msg,"LAPACK or BLAS error in routine ");
strncat(msg,name,6);
sprintf(msg+strlen(msg),": illegal value of parameter #%d",*n);
laerror(msg);
}


//with atlas-cblas another error routine is necessary

extern "C" void ATL_xerbla(int p, char *rout, char *form, ...)
{
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

int cblas_errprn(int ierr, int info, char *form, ...)
{
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
