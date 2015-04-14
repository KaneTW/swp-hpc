/*****************************************************
 * CG Solver (HPC Software Lab)
 *
 * Parallel Programming Models for Applications in the 
 * Area of High-Performance Computation
 *====================================================
 * IT Center (ITC)
 * RWTH Aachen University, Germany
 * Author: Tim Cramer (cramer@itc.rwth-aachen.de)
 * Date: 2010 - 2015
 *****************************************************/


#ifndef __SOLVER_H__
#define __SOLVER_H__

#include "def.h"


#ifdef __cplusplus
	extern "C" {
#endif
	void vectorDot(const floatType* a, const floatType* b, const int n, floatType* ab);
        void vectorSquare(const floatType* x, const int n, floatType* aa);
	__global__ void axpy(const floatType a, const floatType* __restrict__ x, const int n, floatType* __restrict__ y);
	__global__ void xpay(const floatType* __restrict__ x, const floatType a, const int n, floatType* __restrict__ y);
	__global__ void matvec(const int n, const int nnz, const int maxNNZ, const floatType* __restrict__ data, const int* __restrict__ indices, const int* __restrict__ length, const floatType* __restrict__ x, floatType* __restrict__ y);
	void nrm2(const floatType* x, const int n, floatType* nrm);
	void cg(const int n, const int nnz, const int maxNNZ, const floatType* data, const int* indices, const int* length, const floatType* b, floatType* x, struct SolverConfig* sc);
#ifdef __cplusplus
	}
#endif


#endif
