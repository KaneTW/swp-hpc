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
	void vectorDot(const floatType* restrict a, const floatType* restrict b, const int n, floatType* restrict ab);
	void vectorSquare(const floatType* restrict a, const int n, floatType* restrict aa);
	void axpy(const floatType a, const floatType* restrict x, const int n, floatType* restrict y);
	void xpay(const floatType* restrict x, const floatType a, const int n, floatType* restrict y);
	void matvec(const int n, const int nnz, const int maxNNZ, const floatType* restrict data, const int* restrict indices, const int* restrict length, const floatType* restrict x, floatType* restrict y);
	void nrm2(const floatType* restrict x, const int n, floatType* restrict nrm);
	void parallelMemcpy(const floatType* restrict x, const int n, floatType* restrict y);
	void getDiag(const int n, const int nnz, const int maxNNZ, const floatType* restrict data, const int* restrict indices, const int* restrict length, floatType* restrict diag);
	void diagMult(const floatType* restrict diag, const floatType* restrict x, const int n, floatType* restrict out, floatType* restrict dot);
	void cg(const int n, const int nnz, const int maxNNZ, const floatType* restrict data, const int* restrict indices, const int* restrict length, const floatType* restrict b, floatType* restrict x, struct SolverConfig* sc);
#ifdef __cplusplus
	}
#endif


#endif
