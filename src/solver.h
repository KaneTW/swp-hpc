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
	void cg(const int n, const int nnz, const int maxNNZ, const floatType* data, const int* indices, const int* length, const floatType* b, floatType* x, struct SolverConfig* sc);
#ifdef __cplusplus
	}
#endif


#endif
