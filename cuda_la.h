#ifndef _CUDA_LA_H
#define _CUDA_LA_H

#ifdef CUDALA
#undef MATPTR
#include "cublas.h"
#endif

#include "la_traits.h"

namespace LA {

#ifdef CUDALA
#define CPU_GPU(x,y) {if((x)!=cpu && (y)!=cpu) laerror("one operand must be in CPU memory");}
#define NOT_GPU(x) {if((x).getlocation()!=cpu) laerror("Operation not implemented on GPU (yet). Use .moveto(0) first.");}
#define SAME_LOC(x,y) {if((x).getlocation()!=(y).getlocation()) laerror("Operands have different location. Use .moveto() first.");}
#define SAME_LOC3(x,y,z) {if((x).getlocation()!=(y).getlocation() || (x).getlocation()!=(z).getlocation()) laerror("Operands have different location. Use .moveto() first.");}
#else
#define CPU_GPU(x,y) {}
#define NOT_GPU(x) {}
#define SAME_LOC(x,y) {}
#define SAME_LOC3(x,y,z) {}
#endif

typedef enum {undefined=-1, cpu=0, gpu1=1, gpu2=2, gpu3=3, gpu4=4} GPUID;

#ifdef CUDALA

//global static instantiation of this class will provide automatic init/shutdown of GPU
class GPU_START {
public:
	GPU_START(void)
		{
		cublasStatus status = cublasInit();
		if (status != CUBLAS_STATUS_SUCCESS) laerror("Cannot init GPU for CUBLAS");
		}
	~GPU_START(void)
		{
		cublasStatus status = cublasShutdown();
		if (status != CUBLAS_STATUS_SUCCESS) laerror("Cannot cleanly shutdown GPU");
                }
};

extern void *gpualloc(size_t size);
extern void gpufree(void *ptr);
extern void gpuget(size_t n, size_t elsize, const void *from, void *to);
extern void gpuput(size_t n, size_t elsize, const void *from, void *to);
extern double *gpuputdouble(const double &x);
extern complex<double> *gpuputcomplex(const complex<double> &x);

void set_default_loc(const GPUID loc);

extern GPUID DEFAULT_LOC;


#endif
}
#endif
