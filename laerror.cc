// LA and general error handler
#include <iostream>
#include "laerror.h"
#include <stdio.h>
#include <errno.h>
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
  throw LAerror(s1);
  exit(1);
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


