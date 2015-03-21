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
	floatType temp;
	temp = 0;
	#pragma omp parallel for simd aligned(a:CG_ALIGN,b:CG_ALIGN) reduction(+:temp) schedule(static) private(i) shared(n) default(none)
	for(i=0; i<n; i++){
		temp += a[i]*b[i];
	}
	*ab = temp;
}

/* y <- ax + y */
void axpy(const floatType a, const floatType* restrict x, const int n, floatType* restrict y){
	int i;
	#pragma omp parallel for simd aligned(x:CG_ALIGN,y:CG_ALIGN) default(none) private(i) shared(a,n) schedule(static) 
	for(i=0; i<n; i++){
		y[i]=a*x[i]+y[i];
	}
}

/* y <- x + ay */
void xpay(const floatType* restrict x, const floatType a, const int n, floatType* restrict y){
	int i;
	#pragma omp parallel for simd aligned(x:CG_ALIGN,y:CG_ALIGN) default(none) private(i) shared(n,a) schedule(static) 
	for(i=0; i<n; i++){
		y[i]=x[i]+a*y[i];
	}
}

/* y <- A*x
 * Remember that A is stored in the ELLPACK-R format (data, indices, length, n, nnz, maxNNZ). */
void matvec(const int n, const int nnz, const int maxNNZ, const floatType* restrict data, const int* restrict indices, const int* restrict length, const floatType* restrict x, floatType* restrict y){
	int row, col, idx;
	floatType sum;

	#pragma omp parallel for default(none) private(row, col, idx, sum) shared(n, x, y, data, indices, length) schedule(static) 
	for (row = 0; row < n; row++) {
		sum = 0;
		#pragma omp simd aligned(length:CG_ALIGN, data:CG_ALIGN, x: CG_ALIGN, indices:CG_ALIGN)  private(col, idx) 
		for (col = 0; col < length[row]; col++) {
			idx = col + row*maxNNZ;
			sum += data[idx] * x[indices[idx]];
		}
		y[row] = sum;
	}
}

/* nrm <- ||x||_2 */
void nrm2(const floatType* restrict x, const int n, floatType* restrict nrm){
	int i;
	floatType temp;
	temp = 0;

	#pragma omp parallel for simd aligned(x:CG_ALIGN) reduction(+:temp) default(none) private(i) shared(n) schedule(static) 
	for(i = 0; i<n; i++){
		temp+=(x[i]*x[i]);
	}
	*nrm=sqrt(temp);
}

void diagMult(const floatType* restrict diag, const floatType* restrict x, const int n, floatType* restrict out) {
	int i;
	#pragma omp parallel for simd default(none) aligned(out:CG_ALIGN, x:CG_ALIGN, diag:CG_ALIGN) private(i) shared(n) schedule(static) 
	for (i = 0; i < n; i++) {
		out[i] = x[i]*diag[i];
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
void cg(const int n, const int nnz, const int maxNNZ, const floatType* data, const int* indices, const int* length, const floatType* b, floatType* x, struct SolverConfig* sc){
	floatType* r, *p, *q, *diag, *z;
	floatType alpha, beta, rho, rho_old, check, dot_pq, bnrm2, rz;
	int iter, i, j;
 	double timeMatvec_s;
 	double timeMatvec=0;

	/* allocate memory */
	r = (floatType*)_mm_malloc (n * sizeof(floatType), CG_ALIGN);
	p = (floatType*)_mm_malloc (n * sizeof(floatType), CG_ALIGN);
	q = (floatType*)_mm_malloc (n * sizeof(floatType), CG_ALIGN);
	z = (floatType*)_mm_malloc (n * sizeof(floatType), CG_ALIGN);
	diag = (floatType*)_mm_malloc (n * sizeof(floatType), CG_ALIGN);
	printf("hi");

	#pragma omp parallel for default(none) schedule(static) private(i,j) shared(n,diag,data,indices,length)
	for (i = 0; i < n; i++) {
		diag[i] = 1;
		for (j = 0; j < length[i]; j++) {
			int idx = j+i*maxNNZ;
			int realcol = indices[idx];
			if (i == realcol) {
				diag[i] = 1.0/data[idx];
			}
		}
	}

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
	diagMult(diag, r, n, z);

	/* Calculate initial residuum */
	nrm2(r, n, &bnrm2);
	bnrm2 = 1.0/bnrm2;

	/* p(0)    = z(0) */
	#pragma omp parallel for simd default(none) schedule(static) private(i) shared(n,p,z) 
	for (i=0; i < n; i++) {
		p[i] = z[i];
	}

	DBGVEC("p = r = ", p, n);

	/* rho(0)    =  <r(0),z(0)>, check(0) = <r(0),r(0)> */
	vectorDot(r, z, n, &rho);
	vectorDot(r, r, n, &check);
	printf("rho_0=%e/%e\n", rho, check);

	for(iter = 0; iter < sc->maxIter; iter++){
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

		/* z(k+1) = M^-1r(k+1) */
		diagMult(diag, r, n, z);

		vectorDot(r, r, n, &check);
		/* Normalize the residual with initial one */
		sc->residual = sqrt(check) * bnrm2;

		/* Check convergence ||r(k+1)||_2 < eps
		 * If the residual is smaller than the CG
		 * tolerance specified in the CG_TOLERANCE
		 * environment variable our solution vector
		 * is good enough and we can stop the 
		 * algorithm. */
		printf("res_%d=%e\n", iter+1, sc->residual);
		if(sc->residual <= sc->tolerance)
			break;

		rho_old = rho;
		DBGSCA("rho_old = rho = ", rho_old);

		/* rho(k+1)  = <r(k+1), z(k+1)> */
		vectorDot(r, z, n, &rho);
		DBGSCA("rho = <r, z> = ", rho);

		/* beta      = rho(k+1) / rho(k) */
		beta = rho / rho_old;
		DBGSCA("beta = rho / rho_old= ", beta);

		/* p(k+1)    = z(k+1) + beta*p(k) */
		xpay(z, beta, n, p);
		DBGVEC("p = z + beta * p> = ", p, n);

	}

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
