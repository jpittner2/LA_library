/* vim: set ts=8 sw=8 sts=8 noexpandtab cindent: */
/*******************************************************************************
*******************************************************************************/
#include "la_traits.h"
#include "cuda_la.h"

#ifdef CUDALA

namespace LA {

GPUID DEFAULT_LOC = cpu;

void set_default_loc(const GPUID loc){
	DEFAULT_LOC = loc;
}

void *gpualloc(size_t size){
	void *ptr = NULL;
	cublasAlloc(size, 1, &ptr);
	TEST_CUBLAS("cublasAlloc");
	return ptr;
}


void gpufree(void *ptr){
	cublasFree(ptr);
	TEST_CUBLAS("cublasFree");
}

void gpuget(size_t n, size_t elsize, const void *from, void *to){
	cublasGetVector(n, elsize, from, 1, to, 1);
	TEST_CUBLAS("cublasGetVector");
}

void gpuput(size_t n, size_t elsize, const void *from, void *to){
	cublasSetVector(n, elsize, from, 1, to, 1);
	TEST_CUBLAS("cublasSetVector");
}

double *gpuputdouble(const double &x){
	void *ptr = NULL;
	cublasAlloc(1, sizeof(double), &ptr);
	TEST_CUBLAS("cublasAlloc");

	cublasSetVector(1, sizeof(double), &x, 1, ptr, 1);
	TEST_CUBLAS("cublasSetVector");

	return (double *)ptr;
}

complex<double> *gpuputcomplex(const complex<double> &x){
	void *ptr = NULL;
	cublasAlloc(1, sizeof(complex<double>), &ptr);
	TEST_CUBLAS("cublasAlloc");

	cublasSetVector(1, sizeof(complex<double>), &x, 1, ptr, 1);
	TEST_CUBLAS("cublasSetVector");

	return (complex<double> *)ptr;
}


}
#endif
