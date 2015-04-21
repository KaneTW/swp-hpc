
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

#include "globals.h"
#include "solver.h"
#include "output.h"

/* ab <- a' * b */
void vectorDot(const floatType* restrict a, const floatType* restrict b, const int n, floatType* restrict ab){
	int i;
	floatType temp = 0;

	#pragma omp parallel for simd aligned(a,b:CG_ALIGN) reduction(+:temp) schedule(static) private(i) default(none)
	for(i=0; i<n; i++){
		temp += a[i]*b[i];
	}
	*ab = temp;
}

/* aa <- <a,a>*/
void vectorSquare(const floatType* restrict a, const int n, floatType* restrict aa){
	int i;
	floatType temp = 0;

	#pragma omp parallel for simd aligned(a:CG_ALIGN) reduction(+:temp) schedule(static) private(i) default(none)
	for(i=0; i<n; i++){
		temp += a[i]*a[i];
	}
	*aa = temp;
}

/* y <- ax + y */
void axpy(const floatType a, const floatType* restrict x, const int n, floatType* restrict y){
	int i;
	#pragma omp parallel for simd aligned(x,y:CG_ALIGN) default(none) private(i) schedule(static) shared(a)
	for(i=0; i<n; i++){
		y[i]=a*x[i]+y[i];
	}
}

/* y <- x + ay */
void xpay(const floatType* restrict x, const floatType a, const int n, floatType* restrict y){
	int i;
	#pragma omp parallel for simd aligned(x,y:CG_ALIGN) default(none) private(i) schedule(static)  shared(a)
	for(i=0; i<n; i++){
		y[i]=x[i]+a*y[i];
	}
}

/* y <- A*x
 * Remember that A is stored in the ELLPACK-R format (data, indices, length, n, nnz, maxNNZ). */
void matvec(const int n, const int nnz, const int maxNNZ, const floatType* restrict data, const int* restrict indices, const int* restrict length, const floatType* restrict x, floatType* restrict y){
	int row;
	__assume_aligned(data, CG_ALIGN);
	__assume_aligned(indices, CG_ALIGN);
	__assume_aligned(length, CG_ALIGN);
	__assume_aligned(x, CG_ALIGN);
	__assume_aligned(y, CG_ALIGN);
	#pragma omp parallel for default(none) private(row) schedule(static) shared(x,y,data,indices,length,maxNNZ)
	for (row = 0; row < n; row++) {
		int col;
		int off = row*maxNNZ;
		__assume(length[row] % 2 == 0);
		__assume(off % 2 == 0);
		y[row] = 0;
		#pragma omp simd aligned(data,indices,length,x,y) linear(col) simdlen(4)
		for (col = 0; col < length[row]; col++) {
			int idx = col + off;
			y[row] += data[idx] * x[indices[idx]];
		}
	}
}

/* nrm <- ||x||_2 */
void nrm2(const floatType* restrict x, const int n, floatType* restrict nrm){
	int i;
	floatType temp = 0;

	#pragma omp parallel for simd aligned(x:CG_ALIGN) reduction(+:temp) default(none) private(i)  schedule(static) 
	for(i = 0; i<n; i++){
		temp+=(x[i]*x[i]);
	}
	*nrm=sqrt(temp);
}

void diagMult(const floatType* restrict diag, const floatType* restrict x, const int n, floatType* restrict out, floatType* restrict dot) {
	int i;
	floatType sum = 0;

	#pragma omp parallel for simd reduction(+:sum) aligned(x,diag,out:CG_ALIGN) default(none) private(i) schedule(static) 
	for (i = 0; i < n; i++) {
		floatType temp = diag[i] * x[i];
		out[i] = temp;
		sum += temp*x[i];
	}
	*dot = sum;
}

void getDiag(const int n, const int nnz, const int maxNNZ, const floatType* restrict data, const int* restrict indices, const int* restrict length, floatType* restrict diag) {
	int i, j;
	__assume_aligned(data, CG_ALIGN);
	__assume_aligned(diag, CG_ALIGN);
	__assume_aligned(indices, CG_ALIGN);

	#pragma omp parallel for default(none) schedule(static) private(i,j) shared(diag,data,indices,maxNNZ,length)
	for (i = 0; i < n; i++) {
		for (j = 0; j < length[i]; j++) {
			int idx = j+i*maxNNZ;
			int realcol = indices[idx];
			if (i == realcol) {
				diag[i] = 1.0/data[idx];
			}
		}
	}
}

void parallelMemcpy(const floatType* restrict x, const int n, floatType* restrict y) {
	int i;
	#pragma omp parallel for simd aligned(x,y:CG_ALIGN) default(none) schedule(static) private(i)
	for (i=0; i < n; i++) {
		y[i] = x[i];
	}
}

/*******************************************
 *           Conjugate Gradient            *
 *      This function will do the CG       *
 *  algorithm with Jacobi preconditioning. *
 *      For optimiziation you must not     *
 *          change the algorithm.          *
 *******************************************
 r(0)    = b - Ax(0)
 p(0)    = r(0)
 rho(0)    =  <r(0),r(0)>                
 *******************************************
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
#pragma optimization_level 0 // kill me
void cg(const int n, const int nnz, const int maxNNZ, const floatType* data, const int* indices, const int* length, const floatType* b, floatType* x, struct SolverConfig* sc){
	floatType* r, *p, *q, *diag, *z;
	floatType alpha, beta, rho, rho_old, check, dot_pq, bnrm2;
	int iter;
 	double timeMatvec_s;
 	double timeMatvec=0;

	/* allocate memory */
	r = (floatType*)_mm_malloc (n * sizeof(floatType), CG_ALIGN);
	p = (floatType*)_mm_malloc (n * sizeof(floatType), CG_ALIGN);
	q = (floatType*)_mm_malloc (n * sizeof(floatType), CG_ALIGN);
	z = (floatType*)_mm_malloc (n * sizeof(floatType), CG_ALIGN);
	diag = (floatType*)_mm_malloc (n * sizeof(floatType), CG_ALIGN);

	getDiag(n, nnz, maxNNZ, data, indices, length, diag);

	DBGMAT("Start matrix A = ", n, nnz, maxNNZ, data, indices, length)
	DBGVEC("diag = ", diag, n);
	DBGVEC("b = ", b, n);
	DBGVEC("x = ", x, n);

	/* r(0)    = b - Ax(0) */
	timeMatvec_s = getWTime();
	matvec(n, nnz, maxNNZ, data, indices, length, x, r);
	timeMatvec += getWTime() - timeMatvec_s;
	xpay(b, -1.0, n, r);
	DBGVEC("r = b - Ax = ", r, n);

	/* z0 = M^-1r0 */
	diagMult(diag, r, n, z, &rho);

	/* Calculate initial residuum */
	nrm2(r, n, &bnrm2);
	bnrm2 = 1.0/bnrm2;
	printf("bnrm2: %e\n", bnrm2);
	
	parallelMemcpy(z, n, p);
	DBGVEC("p = z = ", p, n);

	/* rho(0)    =  <r(0),z(0)>, check(0) = <r(0),r(0)> */
	vectorSquare(r, n, &check);
	printf("rho_0=%e/%e\n", rho, check);
	int maxIter = sc->maxIter;
	for(iter = 0; iter < maxIter; iter++){
		DBGMSG("=============== Iteration %d ======================\n", iter);
	
		/* q(k)      = A * p(k) */
		timeMatvec_s = getWTime();
		matvec(n, nnz, maxNNZ, data, indices, length, p, q);
		timeMatvec += getWTime() - timeMatvec_s;
		DBGVEC("q = A * p= ", q, n);

		/* dot_pq    = <p(k),q(k)> */
		vectorDot(p, q, n, &dot_pq);
		DBGSCA("dot_pq = <p, q> = ", dot_pq);

		/* alpha     = rho(k) / dot_pq */
		alpha = rho / dot_pq;
		DBGSCA("alpha = rho / dot_pq = ", alpha);

		/* x(k+1)    = x(k) + alpha*p(k) */
		axpy(alpha, p, n, x);
		DBGVEC("x = x + alpha * p= ", x, n);

		/* r(k+1)    = r(k) - alpha*q(k) */
		axpy(-alpha, q, n, r);
		DBGVEC("r = r - alpha * q= ", r, n);

		rho_old = rho;
		DBGSCA("rho_old = rho = ", rho_old);

		/* z(k+1) = M^-1r(k+1), rho = <r,z> */
		diagMult(diag, r, n, z, &rho);

		vectorSquare(r, n, &check);

		/* Normalize the residual with initial one */
		sc->residual = sqrt(check) * bnrm2;

		/* Check convergence ||r(k+1)||_2 < eps
		 * If the residual is smaller than the CG
		 * tolerance specified in the CG_TOLERANCE
		 * environment variable our solution vector
		 * is good enough and we can stop the 
		 * algorithm. */
		//printf("res_%d=%e\n", iter+1, sc->residual);
		if(sc->residual <= sc->tolerance)
			break;

		/* beta      = rho(k+1) / rho(k) */
		beta = rho / rho_old;
		DBGSCA("beta = rho / rho_old= ", beta);

		/* p(k+1)    = z(k+1) + beta*p(k) */
		xpay(z, beta, n, p);
		DBGVEC("p = z + beta * p> = ", p, n);

	}
	printf("res_%d=%e\n", iter+1, sc->residual);

	/* Store the number of iterations and the 
	 * time for the sparse matrix vector
	 * product which is the most expensive 
	 * function in the whole CG algorithm. */
	sc->iter = iter;
	sc->timeMatvec = timeMatvec;

	/* Clean up */
	_mm_free(r);
	_mm_free(p);
	_mm_free(q);
	_mm_free(diag);
	_mm_free(z);
}
