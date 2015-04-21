#include "reduction.h"
#include <assert.h>
#include <stdint.h>
#include <emmintrin.h>
#include <pmmintrin.h>
floatType reduce(floatType* __restrict__ const array, int n) {
	//array is aligned due to cuda, and n is 64 in current code
	assert(n == 64);
	assert(((uintptr_t)array & 63) == 0);
	int i;
	floatType temp1 = 0;
	floatType temp2 = 0;
	floatType temp3 = 0;
	floatType temp4 = 0;
	for (i = 0; i < n; i += 4) {
		temp1 += array[i];
		temp2 += array[i+1];
		temp3 += array[i+2];
		temp4 += array[i+3];
	}
	return temp1+temp2+temp3+temp4;
}
