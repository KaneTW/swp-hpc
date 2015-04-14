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

#define BLOCK_SIZE 1024

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

/* y <- ax + y */
__global__ void axpy(const floatType a, const floatType* __restrict__ x, const int n, floatType* __restrict__ y){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) 
		y[i]=a*x[i]+y[i];
	}
}

/* y <- x + ay */
__global__ void xpay(const floatType* __restrict__ x, const floatType a, const int n, floatType* __restrict__ y){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		y[i]=x[i]+a*y[i];
	}
}


/* y <- A*x
 * Remember that A is stored in the ELLPACK-R format (data, indices, length, n, nnz, maxNNZ). */
__global__ void matvec(const int n, const int nnz, const int maxNNZ, const floatType* __restrict__ data, const int* __restrict__ indices, const int* __restrict__ length, const floatType* __restrict__ x, floatType* __restrict__ y){
	int row = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < n) {
		float temp = 0;
		for (col = 0; col < maxNNZ; j++) {
			int k = col * n + row;
			temp += data[k] * x[indices[k]];
		}
		y[row] = temp;
	}
}

void vectorSquare(const floatType* __restrict__ x, const int n, floatType* __restrict__ a) {
	const int threadsPerBlock = 1024;
	const int numBlocks = n/threadsPerBlock + 1;
	const size_t size = threadsPerBlock*sizeof(floatType);
	floatType* out = (floatType*)malloc(size);
	floatType* devOut;
	floatType temp = 0;
	
	cudaMalloc(&devOut, size);

	devVectorSquare<threadsPerBlock><<<numBlocks, threadsPerBlock>>>(x, n, devOut);

	cudaMemcpy(out, devOut, size, cudaMemcpyDeviceToHost);
	cudaFree(devOut);

	for (int i = 0; i < threadsPerBlock; i++) {
		temp += out[i];
	}
	*a = temp;
}

void vectorDot(const floatType* __restrict__ a, const floatType* __restrict__ b, const int n, floatType* __restrict__ ab) {
	const int threadsPerBlock = 1024;
	const int numBlocks = n/threadsPerBlock + 1;
	const size_t size = threadsPerBlock*sizeof(floatType);
	floatType* out = (floatType*)malloc(size);
	floatType* devOut;
	floatType temp = 0;
	
	cudaMalloc(&devOut, size);

	devVectorDot<threadsPerBlock><<<numBlocks, threadsPerBlock>>>(a, b, n, devOut);

	cudaMemcpy(out, devOut, size, cudaMemcpyDeviceToHost);
	cudaFree(devOut);

	for (int i = 0; i < threadsPerBlock; i++) {
		temp += out[i];
	}
	*ab = temp;
}

void nrm2(const floatType* __restrict__ x, const int n, floatType* __restrict__ nrm) {
	floatType temp;
	vectorSquare(x, n, &temp);
	*nrm = sqrt(temp);
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
void cg(const int n, const int nnz, const int maxNNZ, const floatType* __restrict__ data, const int* __restrict__ indices, const int* __restrict__ length, const floatType* __restrict__ b, floatType* __restrict__ x, struct SolverConfig* sc){
	floatType *r, *p, *q, *devR, *devP, *devQ;
	floatType alpha, beta, rho, rho_old, dot_pq, bnrm2;
	int iter;
 	double timeMatvec_s;
 	double timeMatvec=0;
	
	/* allocate memory */
	const size_t fvecSize = n * sizeof(floatType);
	const size_t ivecSize = n * sizeof(int);
	const size_t matCount = n * maxNNZ;
	const size_t fmatSize = matCount * sizeof(floatType);
	const size_t imatSize = matCount * sizeof(int);
	r = (floatType*)malloc(fvecSize);
	cudaMalloc(&devR, fvecSize);
	p = (floatType*)malloc(fvecSize);
	cudaMalloc(&devP, fvecSize);
	q = (floatType*)malloc(fvecSize);
	cudaMalloc(&devQ, fvecSize);
	// cuda memory for arguments
	// constant memory
	cudaChannelFormatDesc iChan = cudaCreateChannelDesc<int>();
	cudaChannelFormatDesc fChan = cudaCreateChannelDesc<floatType>();
	floatType *devData, *devB;
	floatType *devIndices, *devLength;
	cudaMalloc(&devB, fvecSize);
	cudaMemcpy(devB, b, fvecSize, cudaMemcpyHostToDevice);

	cudaMalloc(&devData, fmatSize);
	cudaMemcpy(devData, data, fmatSize, cudaMemcpyHostToDevice);

	cudaMalloc(&devIndices, imatSize);
	cudaMemcpy(devIndices, indices, imatSize, cudaMemcpyHostToDevice);

	cudaMalloc(&devLength, ivecSize);
	cudaMemcpy(devLength, length, ivecSize, cudaMemcpyHostToDevice);

	// nonconstant
	floatType *devX;
	cudaMalloc(&devX, fvecSize);
	cudaMemcpy(devX, x, fvecSize, cudaMemcpyHostToDevice);
	
	const int threadsPerBlock = 1024;
	const int numBlocks = n / threadsPerBlock + 1; 

	/* r(0)    = b - Ax(0) */
	timeMatvec_s = getWTime();
	matvec<<<numBlocks, threadsPerBlock>>>(n, nnz, maxNNZ, devData, devIndices, devLength, devX, devR);
	timeMatvec += getWTime() - timeMatvec_s;
	xpay<<<numBlocks, threadsPerBlock>>>(devB, -1.0, n, devR);
	

	/* Calculate initial residuum */
	nrm2(devR, n, &bnrm2);
	bnrm2 = 1.0 /bnrm2;

	/* p(0)    = r(0) */
	cudaMemcpy(devP, devR, fvecSize);

	/* rho(0)    =  <r(0),r(0)> */
	vectorSquare(devR, n, &rho);
	printf("rho_0=%e\n", rho);

	for(iter = 0; iter < sc->maxIter; iter++){
		DBGMSG("=============== Iteration %d ======================\n", iter);
	
		/* q(k)      = A * p(k) */
		timeMatvec_s = getWTime();
		matvec<<<numBlocks, threadsPerBlock>>>(n, nnz, maxNNZ, devData, devIndices, devLength, devP, devQ);
		timeMatvec += getWTime() - timeMatvec_s;

		/* dot_pq    = <p(k),q(k)> */
		vectorDot(devP, devQ, n, &dot_pq);

		/* alpha     = rho(k) / dot_pq */
		alpha = rho / dot_pq;

		/* x(k+1)    = x(k) + alpha*p(k) */
		axpy<<<numBlocks, threadsPerBlock>>>(alpha, devP, n, devX);

		/* r(k+1)    = r(k) - alpha*q(k) */
		axpy<<<numBlocks, threadsPerBlock>>>(-alpha, devQ, n, devR);


		rho_old = rho;

		/* rho(k+1)  = <r(k+1), r(k+1)> */
		vectorDot(devR, devR, n, &rho);

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

		/* p(k+1)    = r(k+1) + beta*p(k) */
		xpay<<<numBlocks, threadsPerBlock>>>(devR, beta, n, devP);

	}
	// copy x back
	cudaMemcpy(x, devX, fvecSize, cudaMemcpyDeviceToHost );

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
