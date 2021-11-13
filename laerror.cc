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
#include <signal.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <setjmp.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/prctl.h>
#include <fcntl.h>


#include "cuda_la.h"


namespace LA {

//enforce GPU initialization by a global class instantization constructor
#ifdef CUDALA
GPU_START gpu_start_instant;
#endif

bool _LA_count_check=true; //intentionally not const
const bool _LA_warn_empty_copyonwrite=false;

extern "C" void _findme(void) {}; //for autoconf test we need a function with C linkage

//traceback routines
extern "C" {


//simple traceback by calling gdb on ourselves 
int traceback(int flags)
{
    char pid_buf[256];
    sprintf(pid_buf, "%d", getpid());
    char name_buf[512];
    name_buf[readlink("/proc/self/exe", name_buf, 511)]=0;
    prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);
    int child_pid = fork();
    if (!child_pid) {
        dup2(2,1); // redirect output to stderr - edit: unnecessary?
        execl("/usr/bin/gdb", "gdb", "--batch", "-n", "-ex", "thread", "-ex", "bt", name_buf, pid_buf, NULL);
        abort(); /* If gdb failed to start */
    } else {
        waitpid(child_pid,NULL,0);
    }
return 0;
}

static int tracebackflags;
static int washere=0;

void tracebackhandler(int i, siginfo_t *info, void *x)
{
fflush(stdout);

traceback(tracebackflags);

if(!washere) {washere=1; raise(i);} /*generate core*/
else _exit(1); /*prevent endless loop if something goes very wrong*/
}

void sigtraceback(int sig,int flags)
{
struct sigaction action;
action.sa_sigaction=tracebackhandler;
action.sa_flags=SA_ONESHOT|SA_SIGINFO;
memset(&action.sa_mask,0,sizeof(sigset_t));
tracebackflags=flags;
if(sigaction(sig,&action,NULL)) perror("cannot install signal handler");
}



}//extern C

void laerror2(const char *s1, const char *s2)
{
  std::cout<<"\n";
  std::cerr<<"\n";
  std::cout.flush();
  std::cerr.flush();
  if(s1)
  {
    std::cerr <<"LA:ERROR - "<< s2 << ": " << s1 << "\n";
    std::cout <<"LA:ERROR - "<< s2 << ": " << s1 << "\n";
  }
  else 
	{
	std::cerr <<"LA:ERROR - unspecified\n";
	std::cout <<"LA:ERROR - unspecified\n";
	}
#ifdef CUDALA
{
cublasStatus s = cublasGetError();
std::cerr << "CUBLAS status = " << s << std::endl;
std::cout << "CUBLAS status = " << s << std::endl;
}
#endif
  if(errno) perror("system error");

int f=open("/proc/self/cmdline",O_RDONLY);
if(f)
	{
	std::string line;
	char z;
	while(1==read(f,&z,1))
		{
		line+= z?z:' ';
		}
	close(f);
	std::cout<<"Command LINE: "<<line <<"\n";
  	std::cerr<<"Command LINE: "<<line <<"\n";
  	std::cout.flush();
  	std::cerr.flush();
	}

int r=system("echo INFO:; whoami; hostname; pwd; ls -l -t .;");

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
extern "C" {
#include "cblas.h"
}
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
