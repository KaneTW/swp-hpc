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
#include "reduction.h"

#ifndef NO_ERROR_CHECKS
#define CHECK_CUDA_ERROR(expr) { if ((expr) != cudaSuccess) { printf("Error when executing cuda function"); } } 
#else
#define CHECK_CUDA_ERROR(expr) { expr; }
#endif

void printError() {
#ifndef NO_ERROR_CHECKS
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(err));
	}
#endif
}

#define UNROLLED_ADD_SYNC(n) { \
if (blockSize >= n) { \
	if (tid < n/2) { \
		sdata[tid] = localSum = localSum + sdata[tid + n/2]; \
		} \
	__syncthreads(); \
	} \
}

#define UNROLLED_ADD(n) { \
	if (blockSize >= n) { \
		sdata[tid] = localSum = localSum + sdata[tid + n/2]; \
	} __syncthreads(); \
}


 template<unsigned int blockSize>
__global__ void devVectorDot(const floatType* __restrict__ const a, const floatType* __restrict__ const b, const int n, floatType* __restrict__ const ab){
	extern __shared__ floatType sdata[];
	const int tid = threadIdx.x;
	int i = blockIdx.x * (blockSize*2) + tid;
	const int gridSize = blockSize*2 * gridDim.x;

	floatType localSum = 0;
	#pragma unroll 2
	while (i < n) {
		localSum += a[i]*b[i];
		#ifdef DEBUG
		printf("vecdot: %i %f %f %f\n", i, a[i], b[i], localSum);
		#endif
		if (i + blockSize < n)
	            localSum += a[i+blockSize]*b[i+blockSize];
	    	#ifdef DEBUG
    		printf("vecdot: %f\n",  localSum);
    		#endif
		i += gridSize;
	}
	sdata[tid] = localSum;
	__syncthreads();

	UNROLLED_ADD_SYNC(1024)
	UNROLLED_ADD_SYNC(512)
	UNROLLED_ADD_SYNC(256)
	UNROLLED_ADD_SYNC(128)	

	if (tid < 32) {
		UNROLLED_ADD(64)
		UNROLLED_ADD(32)
		UNROLLED_ADD(16)
		UNROLLED_ADD(8)
		UNROLLED_ADD(4)
		UNROLLED_ADD(2)
	}

	
	if (tid == 0) {
		ab[blockIdx.x] = localSum;
	}

}

/* a <- <x,x> */
template<unsigned int blockSize>
__global__ void devVectorSquare(const floatType* __restrict__ const x, const int n, floatType* __restrict__ const a){
	extern __shared__ floatType sdata[];
	const int tid = threadIdx.x;
	int i = blockIdx.x * (blockSize*2) + tid;
	const int gridSize = blockSize*2 * gridDim.x;

	floatType localSum = 0;

	#pragma unroll 2
	while (i < n) {
		localSum += x[i]*x[i];
		if (i + blockSize < n)
	            localSum += x[i+blockSize]*x[i+blockSize];
		i += gridSize;
	}
	sdata[tid] = localSum;
	__syncthreads();

	UNROLLED_ADD_SYNC(1024)
	UNROLLED_ADD_SYNC(512)
	UNROLLED_ADD_SYNC(256)
	UNROLLED_ADD_SYNC(128)

	if (tid < 32) {
		UNROLLED_ADD(64)
		UNROLLED_ADD(32)
		UNROLLED_ADD(16)
		UNROLLED_ADD(8)
		UNROLLED_ADD(4)
		UNROLLED_ADD(2)
	}

	if (tid == 0) {
		a[blockIdx.x] = localSum;
	}
}


/* y <- ax + y */
__global__ void axpy(const floatType a, const floatType* __restrict__ const x, const int n, floatType* __restrict__ const y){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		y[i]=a*x[i]+y[i];
	}
}

/* y <- x + ay */
__global__ void xpay(const floatType* __restrict__ const x, const floatType a, const int n, floatType* __restrict__ const y){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		y[i]=x[i]+a*y[i];
	}
}


/* y <- A*x
 * Remember that A is stored in the ELLPACK-R format (data, indices, length, n, nnz, maxNNZ). */
__global__ void matvec(const int n, const int nnz, const int maxNNZ, const floatType* __restrict__ const data, const int* __restrict__ const indices, const int* __restrict__ const length, const floatType* __restrict__ const x, floatType* __restrict__ const y){
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col;
	if (row < n) {
		floatType temp = 0;	
		#pragma unroll 2
		for (col = 0; col < length[row]; col++) {
			int k = col * n + row;
			temp += data[k] * x[indices[k]];
		}
		y[row] = temp;
	}
}

// texref version
// define a global texref for matvec. has to be int2 for double
texture<int2, cudaTextureType1D, cudaReadModeElementType> devPRef;
__global__ void matvecRef(const int n, const int nnz, const int maxNNZ, const floatType* __restrict__ const data, const int* __restrict__ const indices, const int* __restrict__ const length, floatType* __restrict__ const y){
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col;
	if (row < n) {
		floatType temp = 0;	
		#pragma unroll 2
		for (col = 0; col < length[row]; col++) {
			int k = col * n + row;
			int2 pv = tex1Dfetch(devPRef, indices[k]);
			floatType val = __hiloint2double(pv.y, pv.x);
			temp += data[k] * val;
		}
		y[row] = temp;
	}
}


// optimize a bit for G3
#define REDUCTION_BLOCK_SIZE 128
#define REDUCTION_BLOCK_COUNT(n) 64

void vectorSquare(const floatType* __restrict__ const x, const int n, floatType* __restrict__ a) {
	const int threadsPerBlock = REDUCTION_BLOCK_SIZE;
	const int numBlocks = REDUCTION_BLOCK_COUNT(n);
	const size_t size = numBlocks*sizeof(floatType);

	// hacky as fuck, but works
	static floatType* out = NULL;
	static floatType* devOut = NULL;

	if (devOut == NULL) {
		CHECK_CUDA_ERROR(cudaHostAlloc(&out, size, cudaHostAllocMapped));
		CHECK_CUDA_ERROR(cudaHostGetDevicePointer(&devOut, out, 0));
	}
	
	devVectorSquare<threadsPerBlock><<<numBlocks, threadsPerBlock, threadsPerBlock*sizeof(floatType)>>>(x, n, devOut);
	printError();
	cudaDeviceSynchronize();

	*a = reduce(out, numBlocks);
}

void vectorDot(const floatType* __restrict__ const a, const floatType* __restrict__ const b, const int n, floatType* __restrict__ const ab) {
	const int threadsPerBlock = REDUCTION_BLOCK_SIZE;
	const int numBlocks = REDUCTION_BLOCK_COUNT(n);
	const size_t size = numBlocks*sizeof(floatType);

	// hacky as fuck, but works
	static floatType* out = NULL;
	static floatType* devOut = NULL;

	if (devOut == NULL) {
		CHECK_CUDA_ERROR(cudaHostAlloc(&out, size, cudaHostAllocMapped));
		CHECK_CUDA_ERROR(cudaHostGetDevicePointer(&devOut, out, 0));
	}

	devVectorDot<threadsPerBlock><<<numBlocks, threadsPerBlock, threadsPerBlock*sizeof(floatType)>>>(a, b, n, devOut);
	printError();
	cudaDeviceSynchronize();

	*ab = reduce(out, numBlocks);
}

void nrm2(const floatType* __restrict__ const x, const int n, floatType* __restrict__ const nrm) {
	floatType temp;
	vectorSquare(x, n, &temp);
	*nrm = rsqrt(temp);
}

__global__ void diagMult(const floatType* __restrict__ const diag, const floatType* __restrict__ const x, const int n, floatType* __restrict__ const out) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		out[i] = x[i]/diag[i];
	}
}

__global__ void getDiag(const int n, const int nnz, const int maxNNZ, const floatType* __restrict__ const data, const int* __restrict__ const indices, const int* __restrict__ const length, floatType* __restrict__ const diag) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		int j;
		#pragma unroll 2
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
	floatType *devR, *devP, *devQ;
	floatType alpha, beta, rho, rho_old, dot_pq, bnrm2, check;
	int iter;
 	float timeMatvec_s;
 	float timeMatvec=0;

 	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	/* allocate memory */
	const size_t fvecSize = n * sizeof(floatType);
	const size_t ivecSize = n * sizeof(int);
	const size_t matCount = n * maxNNZ;
	const size_t fmatSize = matCount * sizeof(floatType);
	const size_t imatSize = matCount * sizeof(int);

	// varying block sizes 
	#define BLOCK_SIZE(func, bs)  \
		int func##BlockSize = bs; \
		int func##GridSize = (n + func##BlockSize - 1)/func##BlockSize; 
	
	#define LAUNCH(kernel) kernel<<<kernel##GridSize, kernel##BlockSize>>>

	BLOCK_SIZE(matvec, 128);
	BLOCK_SIZE(matvecRef, 128);
	BLOCK_SIZE(xpay, 128);
	BLOCK_SIZE(axpy, 128);
	BLOCK_SIZE(getDiag, 128);
	BLOCK_SIZE(diagMult, 128);



	CHECK_CUDA_ERROR(cudaMalloc(&devP, fvecSize));


	CHECK_CUDA_ERROR(cudaMalloc(&devR, fvecSize));

	CHECK_CUDA_ERROR(cudaMalloc(&devQ, fvecSize));
	// cuda memory for arguments
	//cudaChannelFormatDesc iChan = cudaCreateChannelDesc<int>();
	
	floatType *devData, *devB, *devDiag, *devZ, *devX;
	int *devIndices, *devLength;

	CHECK_CUDA_ERROR(cudaMalloc(&devB, fvecSize));
	CHECK_CUDA_ERROR(cudaMemcpy(devB, b, fvecSize, cudaMemcpyHostToDevice));

	CHECK_CUDA_ERROR(cudaMalloc(&devData, fmatSize));
	CHECK_CUDA_ERROR(cudaMemcpy(devData, data, fmatSize, cudaMemcpyHostToDevice));

	CHECK_CUDA_ERROR(cudaMalloc(&devIndices, imatSize));
	CHECK_CUDA_ERROR(cudaMemcpy(devIndices, indices, imatSize, cudaMemcpyHostToDevice));

	CHECK_CUDA_ERROR(cudaMalloc(&devLength, ivecSize));
	CHECK_CUDA_ERROR(cudaMemcpy(devLength, length, ivecSize, cudaMemcpyHostToDevice));
	
	CHECK_CUDA_ERROR(cudaMalloc(&devDiag, fvecSize));
	LAUNCH(getDiag)(n, nnz, maxNNZ, devData, devIndices, devLength, devDiag);
	printError();

	CHECK_CUDA_ERROR(cudaMalloc(&devZ, fvecSize));

	CHECK_CUDA_ERROR(cudaMalloc(&devX, fvecSize));
	CHECK_CUDA_ERROR(cudaMemcpy(devX, x, fvecSize, cudaMemcpyHostToDevice));

	// texture reference
	size_t offsetP;
	cudaBindTexture(&offsetP, devPRef, devP, fvecSize);

	
	/* r(0)    = b - Ax(0) */
	cudaEventRecord(start);
	LAUNCH(matvec)(n, nnz, maxNNZ, devData, devIndices, devLength, devX, devR);
	printError();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timeMatvec_s, start, stop);
	timeMatvec += timeMatvec_s/1000;

	LAUNCH(xpay)(devB, -1.0, n, devR);
	printError();

	
	LAUNCH(diagMult)(devDiag, devR, n, devZ);
	printError();
	vectorDot(devR, devZ, n, &rho);

	/* Calculate initial residuum */
	nrm2(devR, n, &bnrm2);
	printf("bnrm2: %e\n", bnrm2);

	/* p(0)    = z(0) */
	CHECK_CUDA_ERROR(cudaMemcpy(devP, devZ, fvecSize, cudaMemcpyDeviceToDevice));

	/* check(0)    =  <r(0),r(0)> */
	/* rho(0)    =  <r(0),z(0)> */
	vectorSquare(devR, n, &check);
	printf("rho_0=%e/%e\n", rho, check);
	for(iter = 0; iter < sc->maxIter; iter++){
		DBGMSG("=============== Iteration %d ======================\n", iter);
	
		/* q(k)      = A * p(k) */
		cudaEventRecord(start);
		LAUNCH(matvecRef)(n, nnz, maxNNZ, devData, devIndices, devLength, devQ);
		printError();	
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timeMatvec_s, start, stop);
		timeMatvec += timeMatvec_s/1000;

		/* dot_pq    = <p(k),q(k)> */
		vectorDot(devP, devQ, n, &dot_pq);

		/* alpha     = rho(k) / dot_pq */
		alpha = rho / dot_pq;

		/* x(k+1)    = x(k) + alpha*p(k) */
		LAUNCH(axpy)(alpha, devP, n, devX);
		printError();

		/* r(k+1)    = r(k) - alpha*q(k) */
		LAUNCH(axpy)(-alpha, devQ, n, devR);
		printError();


		rho_old = rho;

		/* rho(k+1)  = <r(k+1), z(k+1)> */
		LAUNCH(diagMult)(devDiag, devR, n, devZ);
		printError();

		vectorDot(devR, devZ, n, &rho);
		vectorSquare(devR, n, &check);

		/* Normalize the residual with initial one */
		sc->residual = sqrt(check) * bnrm2;
   	
		/* Check convergence ||r(k+1)||_2 < eps
		 * If the residual is smaller than the CG
		 * tolerance specified in the CG_TOLERANCE
		 * environment variable our solution vector
		 * is good enough and we can stop the 
		 * algorithm. */
		#ifdef DEBUG
		#define RESIDUAL_DEBUG
		#endif
		#ifdef RESIDUAL_DEBUG
		printf("res_%d=%e\n", iter+1, sc->residual);
		printf("rhores_%d=%e\n", iter+1, sqrt(rho)*bnrm2);
		printf("rhores_%d=%e\n", iter+1, rho);
		printf("check_%d=%e\n", iter+1, check);
		#endif
		if(sc->residual < sc->tolerance) {
			break;
		}
		

		/* beta      = rho(k+1) / rho(k) */
		beta = rho / rho_old;

		/* p(k+1)    = r(k+1) + beta*p(k) */
		LAUNCH(xpay)(devZ, beta, n, devP);
		printError();

	}
	cudaDeviceSynchronize();
	printf("res_%d=%e\n", iter+1, sc->residual);

	// copy x back
	CHECK_CUDA_ERROR(cudaMemcpy(x, devX, fvecSize, cudaMemcpyDeviceToHost ));

	/* Store the number of iterations and the 
	 * time for the sparse matrix vector
	 * product which is the most expensive 
	 * function in the whole CG algorithm. */
	sc->iter = iter;
	sc->timeMatvec = timeMatvec;

	/* Clean up */
	cudaFree(devR);
	cudaFree(devP);
	cudaFree(devQ);
	cudaFree(devB);
	cudaFree(devX);
	cudaFree(devZ);
	cudaFree(devDiag);
	cudaFree(devIndices);
	cudaFree(devData);
	cudaFree(devLength);
}
