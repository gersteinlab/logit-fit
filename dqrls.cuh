#include "f2c.h"

__device__ int dqrls_(doublereal *x, int *n, int *p, doublereal 
	*y, int *ny, doublereal *tol, doublereal *b, doublereal *rsd, 
	doublereal *qty, int *k, int *jpvt, doublereal *qraux, 
	doublereal *work);
	