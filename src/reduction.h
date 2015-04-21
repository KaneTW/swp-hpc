#ifndef __REDUCTION_H
#define __REDUCTION_H
#include "def.h"

#define REDUCTION_GRID_SIZE 64
#ifdef __cplusplus
extern "C" {
#endif

floatType reduce(floatType* __restrict__ array, int n);

#ifdef __cplusplus
}
#endif

#endif
