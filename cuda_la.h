//------------------------------------------------------------------------------
/* vim: set ts=8 sw=8 sts=8 noexpandtab cindent: */
//------------------------------------------------------------------------------
#ifndef _CUDA_LA_H
#define _CUDA_LA_H

#include <errno.h>
#ifdef CUDALA
#undef MATPTR
#include "cublas.h"
#endif

#include "la_traits.h"

namespace LA {

#ifdef CUDALA
#define CPU_GPU(x,y) {if((x)!=cpu && (y)!=cpu) laerror("one operand must be in CPU memory");}
#define NOT_GPU(x) {if((x).getlocation()!=cpu) laerror("Operation not implemented on GPU (yet). Use .moveto(0) first.");}
#define NOT_CPU(x) {if((x).getlocation()==cpu) laerror("Operation not implemented on CPU (yet). Use .moveto(>0) first.");}
#define SAME_LOC(x,y) {if((x).getlocation()!=(y).getlocation()) laerror("Operands have different location. Use .moveto() first.");}
#define SAME_LOC3(x,y,z) {if((x).getlocation()!=(y).getlocation() || (x).getlocation()!=(z).getlocation()) laerror("Operands have different location. Use .moveto() first.");}
#else
#define CPU_GPU(x,y) {}
#define NOT_GPU(x) {}
#define SAME_LOC(x,y) {}
#define SAME_LOC3(x,y,z) {}
#endif

#ifdef DEBUG
	#ifdef __GNUG__
		#define TEST_CUBLAS(X) { if(cublasGetError() != CUBLAS_STATUS_SUCCESS){ laerror2(#X, __PRETTY_FUNCTION__); } }
	#else
		#define TEST_CUBLAS(X) { if(cublasGetError() != CUBLAS_STATUS_SUCCESS){ laerror2(#X, __func__); } }
	#endif
#else
	#define TEST_CUBLAS(X) {}
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
		errno = 0;
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

template <typename T>
void smart_gpu_set(size_t n, const T& val, void *gpu_to, size_t _step = 1){
	void *ptr(NULL);
	if(sizeof(T)%sizeof(float) != 0){ laerror("memory alignment error"); }

	cublasAlloc(1, sizeof(T), &ptr);
	TEST_CUBLAS("cublasAlloc");

	cublasSetVector(1, sizeof(T), &val, 1, ptr, 1);
	TEST_CUBLAS("cublasSetVector");

	if(sizeof(T) == sizeof(float)){
		cublasScopy(n, (float*)ptr, 0, ((float*)gpu_to), _step);
		TEST_CUBLAS("cublasScopy");

	}else if(sizeof(T) == sizeof(double)){
		cublasDcopy(n, (double*)ptr, 0, ((double*)gpu_to), _step);
		TEST_CUBLAS("cublasDcopy");

	}else if(sizeof(T) == sizeof(complex<double>)){
		cublasZcopy(n, (cuDoubleComplex*)ptr, 0, (cuDoubleComplex*)gpu_to, _step);
		TEST_CUBLAS("cublasZcopy");

	}else{
		for(register int i=0; i<sizeof(T)/sizeof(float); i++){
			cublasScopy(n, (float*)ptr + i, 0, ((float*)gpu_to) + i, sizeof(T)/sizeof(float)*_step);
			TEST_CUBLAS("cublasScopy");
		}
	}

	cublasFree(ptr);
	TEST_CUBLAS("cublasFree");
}

extern GPUID DEFAULT_LOC;

#endif
}
#endif
