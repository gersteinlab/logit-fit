extern "C" {
	__device__ void dqrls(double *x, int *n, int *p, double *y, int *ny, double *tol, double *b, 
					 double *rsd, double *qty, int *k, int *jpvt, double *qraux, double *work);
}
