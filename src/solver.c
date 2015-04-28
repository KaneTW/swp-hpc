/*****************************************************
 * CG Solver (HPC Software Lab)
 *
 * Parallel Programming Models for Applications in the 
 * Area of High-Performance Computation
 *====================================================
 * IT Center (ITC)
 * RWTH Aachen University, Germany
 * Author: Tim Cramer (cramer@itc.rwth-aachen.de)
 * 	   Fabian Schneider (f.schneider@itc.rwth-aachen.de)
 * Date: 2010 - 2015
 *****************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENACC
# include <openacc.h>
#endif

#ifdef CUDA
# include <cuda.h>
#endif

#include "solver.h"
#include "output.h"


/* y <- ax + y */
void axpy(const floatType a, const floatType* __restrict__ const x, const int n, floatType* __restrict__ const y){
	int i;

	#pragma acc kernels loop independent
	for (i = 0; i < n; i++) {
		y[i]=a*x[i]+y[i];
	}
}

/* y <- x + ay */
void xpay(const floatType* __restrict__ const x, const floatType a, const int n, floatType* __restrict__ const y){
	int i;

	#pragma acc kernels loop independent
	for (i = 0; i < n; i++) {
		y[i]=x[i]+a*y[i];
	}
}


/* y <- A*x
 * Remember that A is stored in the ELLPACK-R format (data, indices, length, n, nnz, maxNNZ). */
void matvec(const int n, const int nnz, const int maxNNZ, const floatType* __restrict__ const data, const int* __restrict__ const indices, const int* __restrict__ const length, const floatType* __restrict__ const x, floatType* __restrict__ const y){
	int row;
	#pragma acc kernels loop independent
	for (row = 0; row < n; row++) {
		floatType temp = 0;	
		int col;
		#pragma acc loop vector reduction(+:temp)
		for (col = 0; col < length[row]; col++) {
			int k = col * n + row;
			temp += data[k] * x[indices[k]];
		}
		y[row] = temp;
	}
}

void vectorDot(const floatType* __restrict__ const a, const floatType* __restrict__ const b, const int n, floatType* __restrict__ const ab) {
	int i;
	floatType temp = 0;

	#pragma acc kernels loop reduction(+:temp)
	for (i=0; i<n; i++){
		temp += a[i]*b[i];
	}
	*ab = temp;
}

void vectorSquare(const floatType* __restrict__ const x, const int n, floatType* __restrict__ const ab) {
	int i;
	floatType temp = 0;

	#pragma acc kernels loop reduction(+:temp)
	for (i=0; i<n; i++){
		temp += x[i]*x[i];
	}
	*ab = temp;
}

void nrm2(const floatType* __restrict__ const x, const int n, floatType* __restrict__ const nrm) {
	floatType temp;
	vectorSquare(x, n, &temp);
	*nrm = rsqrt(temp);
}

void diagMult(const floatType* __restrict__ const diag, const floatType* __restrict__ const x, const int n, floatType* __restrict__ const out) {
	int i;

	#pragma acc kernels loop independent
	for (i=0; i<n; i++){
		out[i] = x[i]/diag[i];
	}
}

void getDiag(const int n, const int nnz, const int maxNNZ, const floatType* __restrict__ const data, const int* __restrict__ const indices, const int* __restrict__ const length, floatType* __restrict__ const diag) {
	int i;

	#pragma acc kernels loop independent
	for (i=0; i<n; i++) {
		int j;
		for (j = 0; j < length[i]; j++) {
			int idx = j*n + i;
			int realcol = indices[idx];
			if (i == realcol) {
				diag[i] = data[idx];
			}
		}
	}
}

/***************************************
 *         Conjugate Gradient          *
 *   This function will do the CG      *
 *  algorithm without preconditioning. *
 *    For optimiziation you must not   *
 *        change the algorithm.        *
 ***************************************
 r(0)    = b - Ax(0)
 p(0)    = r(0)
 rho(0)    =  <r(0),r(0)>                
 ***************************************
 for k=0,1,2,...,n-1
   q(k)      = A * p(k)                 
   dot_pq    = <p(k),q(k)>             
   alpha     = rho(k) / dot_pq
   x(k+1)    = x(k) + alpha*p(k)      
   r(k+1)    = r(k) - alpha*q(k)     
   check convergence ||r(k+1)||_2 < eps  
	 rho(k+1)  = <r(k+1), r(k+1)>         
   beta      = rho(k+1) / rho(k)
   p(k+1)    = r(k+1) + beta*p(k)      
***************************************/
void cg(const int n, const int nnz, const int maxNNZ, const floatType* __restrict__ const data, const int* __restrict__ const indices, const int* __restrict__ const length, const floatType* __restrict__ const b, floatType* __restrict__ const x, struct SolverConfig* sc){
	floatType *r, *p, *q, *z, *diag;
	floatType alpha, beta, rho, rho_old, dot_pq, bnrm2, check;
	int iter;

	/* allocate memory */
	const size_t fvecSize = n * sizeof(floatType);
	const size_t ivecSize = n * sizeof(int);
	const size_t matCount = n * maxNNZ;
	const size_t fmatSize = matCount * sizeof(floatType);
	const size_t imatSize = matCount * sizeof(int);


	#pragma acc enter data create(r[0:n], q[0:n])
	#pragma acc enter data create(z[0:n], diag[0:n]) copyin(b[0:n])
	#pragma acc enter data copyin(data[0:matCount], indices[0:matCount], length[0:n])
	#pragma acc enter data copyin(x[0:n])

	getDiag(n, nnz, maxNNZ, data, indices, length, diag);


	matvec(n, nnz, maxNNZ, data, indices, length, x, r);
	
	xpay(b, -1.0, n, r);
	diagMult(diag, r, n, z)
	memcpy(p, z, fvecSize);
	#pragma acc enter data copyin(p[0:n])
	
	/* Calculate initial residuum */
	nrm2(r, n, &bnrm2);

	/* check(0)    =  <r(0),r(0)> */
	/* rho(0)    =  <r(0),z(0)> */
	vectorDot(r, z, n, &rho);
	vectorSquare(r, n, &check);
	printf("rho_0=%e/%e\n", rho, check);
	for(iter = 0; iter < sc->maxIter; iter++){
		DBGMSG("=============== Iteration %d ======================\n", iter);
	
		/* q(k)      = A * p(k) */

		matvec(n, nnz, maxNNZ, data, indices, length, q);

		/* dot_pq    = <p(k),q(k)> */
		vectorDot(p, q, n, &dot_pq);

		/* alpha     = rho(k) / dot_pq */
		alpha = rho / dot_pq;

		/* x(k+1)    = x(k) + alpha*p(k) */
		axpy(alpha, p, n, x);

		/* r(k+1)    = r(k) - alpha*q(k) */
		axpy(-alpha, q, n, r);

		rho_old = rho;

		/* rho(k+1)  = <r(k+1), z(k+1)> */
		diagMult(diag, r, n, z);

		vectorDot(r, z, n, &rho);
		vectorSquare(r, n, &check);

		/* Normalize the residual with initial one */
		sc->residual = sqrt(check) * bnrm2;
   	
		/* Check convergence ||r(k+1)||_2 < eps
		 * If the residual is smaller than the CG
		 * tolerance specified in the CG_TOLERANCE
		 * environment variable our solution vector
		 * is good enough and we can stop the 
		 * algorithm. */
		printf("res_%d=%e\n", iter+1, sc->residual);
		
		if(sc->residual < sc->tolerance) {
			break;
		}
		

		/* beta      = rho(k+1) / rho(k) */
		beta = rho / rho_old;

		/* p(k+1)    = r(k+1) + beta*p(k) */
		xpay(z, beta, n, p);
	}
	
	#pragma openacc exit data copyout(x[0:n])
	printf("res_%d=%e\n", iter+1, sc->residual);


	/* Store the number of iterations and the 
	 * time for the sparse matrix vector
	 * product which is the most expensive 
	 * function in the whole CG algorithm. */
	sc->iter = iter;
	sc->timeMatvec = timeMatvec;

	/* Clean up */
	// todo do we really need to?2
}
