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


#define NUM_GANGS 64
#define VECTOR_LENGTH 192
#define RED_NUM_GANGS 64
#define RED_VECTOR_LENGTH 192

/* y <- ax + y */
void axpy(const floatType a, const floatType* restrict const x, const int n, floatType* restrict const y){
	int i;

	#pragma acc parallel num_gangs(NUM_GANGS) vector_length(VECTOR_LENGTH) present(x[0:n], y[0:n])
	#pragma acc loop independent gang, vector  
	for (i = 0; i < n; i++) {
		y[i]=a*x[i]+y[i];
	}
}

/* y <- x + ay */
void xpay(const floatType* restrict const x, const floatType a, const int n, floatType* restrict const y){
	int i;

	#pragma acc parallel num_gangs(NUM_GANGS) vector_length(VECTOR_LENGTH) present(x[0:n],y[0:n])
	#pragma acc loop independent gang, vector 
	for (i = 0; i < n; i++) {
		y[i]=x[i]+a*y[i];
	}
}


/* y <- A*x
 * Remember that A is stored in the ELLPACK-R format (data, indices, length, n, nnz, maxNNZ). */
void matvec(const int n, const int nnz, const int maxNNZ, const floatType* restrict const data, const int* restrict const indices, const int* restrict const length, const floatType* restrict const x, floatType* restrict const y){
	int row;

	#pragma acc parallel vector_length(128) present(data[0:n*maxNNZ], indices[0:n*maxNNZ], length[0:n], x[0:n], y[0:n])
        #pragma acc loop independent gang, vector private(row) 
	for (row = 0; row < n; row++) {
		floatType temp = 0;	
		int col;
		int len = length[row];
		#pragma acc loop seq private(col) 
		for (col = 0; col < len; col++) {
			int k = col * n + row;
			temp += data[k] * x[indices[k]];
		}

		y[row] = temp;
	}
}

void vectorDot(const floatType* restrict const a, const floatType* restrict const b, const int n, floatType* restrict const ab) {
	int i;
	floatType temp = 0;
	#pragma acc parallel num_gangs(RED_NUM_GANGS) vector_length(RED_VECTOR_LENGTH) present(a[0:n],b[0:n])
	#pragma acc loop reduction(+:temp) private(i) 
	for (i=0; i<n; i++){
		temp += a[i]*b[i];
	}
	*ab = temp;
}

void vectorSquare(const floatType* restrict const x, const int n, floatType* restrict const ab) {
	int i;
	floatType temp = 0;

	#pragma acc parallel num_gangs(RED_NUM_GANGS) vector_length(RED_VECTOR_LENGTH) present(x[0:n])
	#pragma acc loop reduction(+:temp)
	for (i=0; i<n; i++){
		temp += x[i]*x[i];
	}

	*ab = temp;
}

void nrm2(const floatType* restrict const x, const int n, floatType* restrict const nrm) {
	floatType temp;
	vectorSquare(x, n, &temp);
	*nrm = 1/sqrt(temp);
}

void diagMult(const floatType* restrict const diag, const floatType* restrict const x, const int n, floatType* restrict const out) {
	int i;

	#pragma acc parallel num_gangs(NUM_GANGS) vector_length(VECTOR_LENGTH) present(x[0:n],diag[0:n], out[0:n])
	#pragma acc loop independent gang, vector
	for (i=0; i<n; i++){
		out[i] = x[i]*diag[i];
	}
}

void getDiag(const int n, const int nnz, const int maxNNZ, const floatType* restrict const data, const int* restrict const indices, const int* restrict const length, floatType* restrict const diag) {
	int i;

	#pragma acc parallel num_gangs(NUM_GANGS) vector_length(VECTOR_LENGTH) present(data[0:n*maxNNZ], indices[0:n*maxNNZ], length[0:n], diag[0:n])
	#pragma acc loop independent gang, vector
	for (i=0; i<n; i++) {
		int j;
		for (j = 0; j < length[i]; j++) {
			int idx = j*n + i;
			int realcol = indices[idx];
			if (i == realcol) {
				diag[i] = 1/data[idx];
			}
		}
	}
}

void cg(const int n, const int nnz, const int maxNNZ, const floatType* restrict const data, const int* restrict const indices, const int* restrict const length, const floatType* restrict const b, floatType* restrict const x, struct SolverConfig* sc){
	floatType *r, *p, *q, *z, *diag;
	floatType alpha, beta, rho, rho_old, dot_pq, bnrm2, check;
	int iter;

	/* allocate memory */
	const size_t fvecSize = n * sizeof(floatType);
	const size_t ivecSize = n * sizeof(int);
	const size_t matCount = n * maxNNZ;
	const size_t fmatSize = matCount * sizeof(floatType);
	const size_t imatSize = matCount * sizeof(int);
	r = malloc(fvecSize);
	p = malloc(fvecSize);
	q = malloc(fvecSize);
	z = malloc(fvecSize);
	diag = malloc(fvecSize);

	#pragma acc enter data create(r[0:n], q[0:n], diag[0:n], z[0:n])
	#pragma acc enter data copyin(data[0:matCount])
	#pragma acc enter data copyin(indices[0:matCount], length[0:n])
	#pragma acc enter data copyin(b[0:n])
	#pragma acc data copy(x[0:n])
	{

	getDiag(n, nnz, maxNNZ, data, indices, length, diag);

	matvec(n, nnz, maxNNZ, data, indices, length, x, r);
	
	xpay(b, -1.0, n, r);
	//todo ugh
	diagMult(diag, r, n, z);
	#pragma acc update host(z[0:n])
	memcpy(p, z, fvecSize);
	#pragma acc enter data copyin(p[0:n])
	
	/* Calculate initial residuum */
	nrm2(r, n, &bnrm2);

	/* check(0)  = <r(0),r(0)> */
	/* rho(0)  = <r(0),z(0)> */
	vectorDot(r, z, n, &rho);
	vectorSquare(r, n, &check);
	printf("rho_0=%e/%e\n", rho, check);
	for(iter = 0; iter < sc->maxIter; iter++){
		DBGMSG("=============== Iteration %d ======================\n", iter);
	
		/* q(k)   = A * p(k) */
		matvec(n, nnz, maxNNZ, data, indices, length, p, q);

		/* dot_pq  = <p(k),q(k)> */
		vectorDot(p, q, n, &dot_pq);

		/* alpha   = rho(k) / dot_pq */
		alpha = rho / dot_pq;

		/* x(k+1)  = x(k) + alpha*p(k) */
		axpy(alpha, p, n, x);

		/* r(k+1)  = r(k) - alpha*q(k) */
		axpy(-alpha, q, n, r);

		rho_old = rho;

		/* rho(k+1) = <r(k+1), z(k+1)> */
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
		

		/* beta   = rho(k+1) / rho(k) */
		beta = rho / rho_old;

		/* p(k+1)  = r(k+1) + beta*p(k) */
		xpay(z, beta, n, p);
	}
	}
	printf("res_%d=%e\n", iter+1, sc->residual);


	/* Store the number of iterations and the 
	 * time for the sparse matrix vector
	 * product which is the most expensive 
	 * function in the whole CG algorithm. */
	sc->iter = iter;
	sc->timeMatvec = 0;

	/* Clean up */
	// todo do we really need to?2
}
