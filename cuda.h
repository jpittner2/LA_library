#ifndef _CUDA_LA_H
#define _CUDA_LA_H

#ifdef CUDA
#define GPU_NOT_IMPLEMENTED {laerror("Operation not implemented on GPU. Use .moveto(0) first.");}
#else
#define GPU_NOT_IMPLEMENTED {}
#endif

#endif
