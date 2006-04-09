// LA and general error handler
#include <iostream>
#include "laerror.h"
#include <stdio.h>
#include <errno.h>

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
char msg[128];
strcpy(msg,"LAPACK or BLAS error in routine ");
strncat(msg,name,6);
sprintf(msg+strlen(msg),": illegal value of parameter #%d",*n);
laerror(msg);
}


