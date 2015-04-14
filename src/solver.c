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


/* ab <- a' * b */

template<unsigned int blockSize> 
 __global__ inline void unrolledReduction(const int tid, floatType* sdata) {
 	#define UNROLLED_ADD_SYNC(n) { \
		if (blockSize >= n) { \
			if (tid < n/2) { \
				sdata[tid] += sdata[tid + n/2]; \
			} __syncthreads(); \
		} \
	}

	UNROLLED_ADD_SYNC(512)
	UNROLLED_ADD_SYNC(256)
	UNROLLED_ADD_SYNC(128)
	#undef UNROLLED_ADD_SYNC

	#define UNROLLED_ADD(n) { \
		if (blockSize >= n) { \
			sdata[tid] += sdata[tid + n/2];
		} \
	}

	if (tid < 32) {
		UNROLLED_ADD(64)
		UNROLLED_ADD(32)
		UNROLLED_ADD(16)
		UNROLLED_ADD(8)
		UNROLLED_ADD(4)
		UNROLLED_ADD(2)
	}
	#undef UNROLLED_ADD
 }

 template<unsigned int blockSize>
__global__ void devVectorDot(const floatType* __restrict__ a, const floatType* __restrict__ b, const int n, floatType* __restrict__ ab){
	extern __shared__ floatType sdata[];
	int tid = threadIdx.x;
	int i = blockIdx.x * (blockSize*2) + tid;
	int gridSize = blockSize*2 * gridDim.x;
	sdata[tid] = 0;

	while (i < n) {
		sdata[tid] += a[i]*b[i] + a[i + blockSize]*b[i+blockSize];
		i += gridSize;
	}
	__syncthreads();

	unrolledReduction<blockSize>(tid, sdata);
	
	if (tid == 0) {
		ab[blockIdx.x] = sdata[0];
	}
}

/* y <- ax + y */
void axpy(const floatType a, const floatType* __restrict__ x, const int n, floatType* __restrict__ y){
	int i;
	for(i=0; i<n; i++){
		y[i]=a*x[i]+y[i];
	}
}

/* y <- x + ay */
void xpay(const floatType* __restrict__ x, const floatType a, const int n, floatType* __restrict__ y){
	int i;
	for(i=0; i<n; i++){
		y[i]=x[i]+a*y[i];
	}
}

/* y <- A*x
 * Remember that A is stored in the ELLPACK-R format (data, indices, length, n, nnz, maxNNZ). */
void matvec(const int n, const int nnz, const int maxNNZ, const floatType* __restrict__ data, const int* __restrict__ indices, const int* __restrict__ length, const floatType* __restrict__ x, floatType* __restrict__ y){
	int i, j, k;
	for (i = 0; i < n; i++) {
		y[i] = 0;
		for (j = 0; j < length[i]; j++) {
			k = j * n + i;
			y[i] += data[k] * x[indices[k]];
		}
	}
}

/* a <- <x,x> */
template<unsigned int blockSize>
__global__ void devVectorSquare(const floatType* __restrict__ x, const int n, floatType* __restrict__ a){
	extern __shared__ floatType sdata[];
	int tid = threadIdx.x;
	int i = blockIdx.x * (blockSize*2) + tid;
	int gridSize = blockSize*2 * gridDim.x;
	sdata[tid] = 0;

	while (i < n) {
		sdata[tid] += x[i]*x[i] + x[i + blockSize]*x[i+blockSize];
		i += gridSize;
	}
	__syncthreads();

	unrolledReduction<blockSize>(tid, sdata);
	
	if (tid == 0) {
		a[blockIdx.x] = sdata[0];
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
void cg(const int n, const int nnz, const int maxNNZ, const floatType* data, const int* indices, const int* length, const floatType* b, floatType* x, struct SolverConfig* sc){
	floatType* r, *p, *q;
	floatType alpha, beta, rho, rho_old, dot_pq, bnrm2;
	int iter;
 	double timeMatvec_s;
 	double timeMatvec=0;
	
	/* allocate memory */
	r = (floatType*)malloc(n * sizeof(floatType));
	p = (floatType*)malloc(n * sizeof(floatType));
	q = (floatType*)malloc(n * sizeof(floatType));
	
	DBGMAT("Start matrix A = ", n, nnz, maxNNZ, data, indices, length)
	DBGVEC("b = ", b, n);
	DBGVEC("x = ", x, n);

	/* r(0)    = b - Ax(0) */
	timeMatvec_s = getWTime();
	matvec(n, nnz, maxNNZ, data, indices, length, x, r);
	timeMatvec += getWTime() - timeMatvec_s;
	xpay(b, -1.0, n, r);
	DBGVEC("r = b - Ax = ", r, n);
	

	/* Calculate initial residuum */
	nrm2(r, n, &bnrm2);
	bnrm2 = 1.0 /bnrm2;

	/* p(0)    = r(0) */
	memcpy(p, r, n*sizeof(floatType));
	DBGVEC("p = r = ", p, n);

	/* rho(0)    =  <r(0),r(0)> */
	vectorDot(r, r, n, &rho);
	printf("rho_0=%e\n", rho);

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


		rho_old = rho;
		DBGSCA("rho_old = rho = ", rho_old);


		/* rho(k+1)  = <r(k+1), r(k+1)> */
		vectorDot(r, r, n, &rho);
		DBGSCA("rho = <r, r> = ", rho);

		/* Normalize the residual with initial one */
		sc->residual= sqrt(rho) * bnrm2;


   	
		/* Check convergence ||r(k+1)||_2 < eps
		 * If the residual is smaller than the CG
		 * tolerance specified in the CG_TOLERANCE
		 * environment variable our solution vector
		 * is good enough and we can stop the 
		 * algorithm. */
		printf("res_%d=%e\n", iter+1, sc->residual);
		if(sc->residual <= sc->tolerance)
			break;


		/* beta      = rho(k+1) / rho(k) */
		beta = rho / rho_old;
		DBGSCA("beta = rho / rho_old= ", beta);

		/* p(k+1)    = r(k+1) + beta*p(k) */
		xpay(r, beta, n, p);
		DBGVEC("p = r + beta * p> = ", p, n);

	}

	/* Store the number of iterations and the 
	 * time for the sparse matrix vector
	 * product which is the most expensive 
	 * function in the whole CG algorithm. */
	sc->iter = iter;
	sc->timeMatvec = timeMatvec;

	/* Clean up */
	free(r);
	free(p);
	free(q);
}
