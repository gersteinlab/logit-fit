#include "f2c.h"

__device__ int daxpy_(int *n, doublereal *da, doublereal *dx, 
	int *incx, doublereal *dy, int *incy);