#include "f2c.h"

__device__ int dqrls_(doublereal *x, integer *n, integer *p, doublereal 
	*y, integer *ny, doublereal *tol, doublereal *b, doublereal *rsd, 
	doublereal *qty, integer *k, integer *jpvt, doublereal *qraux, 
	doublereal *work);
	