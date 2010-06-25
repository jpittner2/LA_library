#include "la_traits.h"
#include "cuda_la.h"

#ifdef CUDALA

namespace LA {

GPUID DEFAULT_LOC = cpu;

void set_default_loc(const GPUID loc) 
{
DEFAULT_LOC = loc;
}

void *gpualloc(size_t size)
{
cublasStatus status;
void *ptr=NULL;
status = cublasAlloc(size,1,&ptr);
if(status != CUBLAS_STATUS_SUCCESS) laerror("Error in cublasAlloc");
return ptr;
}


void gpufree(void *ptr)
{
cublasStatus status = cublasFree(ptr);
if (status != CUBLAS_STATUS_SUCCESS) laerror("Error in cublasFree");
}

void gpuget(size_t n, size_t elsize, const void *from, void *to)
{
cublasStatus status;
status=cublasGetVector(n,elsize,from,1,to,1);
if (status != CUBLAS_STATUS_SUCCESS) laerror("Error in cublasGetVector");
}

void gpuput(size_t n, size_t elsize, const void *from, void *to)
{
cublasStatus status;
status=cublasSetVector(n,elsize,from,1,to,1);
if (status != CUBLAS_STATUS_SUCCESS) laerror("Error in cublasSetVector");
}

double *gpuputdouble(const double &x)
{
cublasStatus status;
void *ptr=NULL;
status = cublasAlloc(1,sizeof(double),&ptr);
if(status != CUBLAS_STATUS_SUCCESS) laerror("Error in cublasAlloc");
status=cublasSetVector(1,sizeof(double),&x,1,ptr,1);
if (status != CUBLAS_STATUS_SUCCESS) laerror("Error in cublasSetVector");
return (double *)ptr;
}

complex<double> *gpuputcomplex(const complex<double> &x)
{
cublasStatus status;
void *ptr=NULL;
status = cublasAlloc(1,sizeof(complex<double>),&ptr);
if(status != CUBLAS_STATUS_SUCCESS) laerror("Error in cublasAlloc");
status=cublasSetVector(1,sizeof(complex<double>),&x,1,ptr,1);
if (status != CUBLAS_STATUS_SUCCESS) laerror("Error in cublasSetVector");
return (complex<double> *)ptr;
}



}
#endif
