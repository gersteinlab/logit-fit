// #include "glm.logit.cuh"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <limits.h>
// #include <limits>
// #include <utility>
// #include <iostream>
#include <sys/stat.h>
#include <errno.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

using namespace std;

#include "f2c.h"
#include "daxpy.cuh"
#include "dcopy.cuh"
#include "dscal.cuh"
#include "dswap.cuh"
#include "ddot.cuh"
#include "dnrm2.cuh"
#include "dqrdc.cuh"
#include "dqrdc2.cuh"
#include "dqrsl.cuh"
#include "dqrls.cuh"
#include "lmfit.cu"
#include "fit.cpp"

#define STRSIZE 1024
// #define STRSIZE 256
#define NUM_BLOCKS 24
#define THREADS_PER_BLOCK 128

/* This code is a CUDA implementation of a logistic regression fitting function */

inline void GPUassert(cudaError_t code, const char * file, int line, bool Abort=true)
{
    if (code != 0) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),file,line);
        if (Abort) exit(code);
    }       
}

#define GPUerrchk(ans) { GPUassert((ans), __FILE__, __LINE__); }

/* Base helper functions and macros */
#ifndef M_LN_SQRT_2PI
#define M_LN_SQRT_2PI	0.918938533204672741780329736406	/* log(sqrt(2*pi)) */
#endif

#ifndef M_LN_2PI
#define M_LN_2PI	1.837877066409345483560659472811	/* log(2*pi) */
#endif

#ifndef M_LOG10_2
#define M_LOG10_2	0.301029995663981195213738894724	/* log10(2) */
#endif

#define n_max (100)

static const double DOUBLE_EPS = 1e-8;
static const double THRESH = 30.;
static const double MTHRESH = -30.;
static const double INVEPS = 1/DOUBLE_EPS;

__device__ double x_d_omx(double x) {
  if (x < 0 || x > 1) {
		printf("Error: Value %g out of range (0, 1)", x);
		asm("trap;");
	}
  return x/(1 - x);
}

__device__ double x_d_opx(double x) {return x/(1 + x);}

int imin2(int x, int y)
{
    return (x < y) ? x : y;
}

__device__ double R_D_exp (double x, bool log_p) {
	return (log_p	? (x)	: exp(x));
}

// Returns the value ln[gamma(xx)] for xx > 0.
__device__ double gammln(double xx) {
	double x,y,tmp,ser;
	double cof[6]={76.18009172947146,-86.50532032941677,
												24.01409824083091,-1.231739572450155,
												0.1208650973866179e-2,-0.5395239384953e-5};
	int j;
	
	y=x=xx;
	tmp=x+5.5;
	tmp -= (x+0.5)*log(tmp);
	ser=1.000000000190015;
	for (j=0;j<=5;j++) ser += cof[j]/++y;
	return -tmp+log(2.5066282746310005*ser/x);
}

__device__ double stirlerr(double n) {

#define S0 0.083333333333333333333       /* 1/12 */
#define S1 0.00277777777777777777778     /* 1/360 */
#define S2 0.00079365079365079365079365  /* 1/1260 */
#define S3 0.000595238095238095238095238 /* 1/1680 */
#define S4 0.0008417508417508417508417508/* 1/1188 */

/*
  error for 0, 0.5, 1.0, 1.5, ..., 14.5, 15.0.
*/
    const double sferr_halves[31] = {
	0.0, /* n=0 - wrong, place holder only */
	0.1534264097200273452913848,  /* 0.5 */
	0.0810614667953272582196702,  /* 1.0 */
	0.0548141210519176538961390,  /* 1.5 */
	0.0413406959554092940938221,  /* 2.0 */
	0.03316287351993628748511048, /* 2.5 */
	0.02767792568499833914878929, /* 3.0 */
	0.02374616365629749597132920, /* 3.5 */
	0.02079067210376509311152277, /* 4.0 */
	0.01848845053267318523077934, /* 4.5 */
	0.01664469118982119216319487, /* 5.0 */
	0.01513497322191737887351255, /* 5.5 */
	0.01387612882307074799874573, /* 6.0 */
	0.01281046524292022692424986, /* 6.5 */
	0.01189670994589177009505572, /* 7.0 */
	0.01110455975820691732662991, /* 7.5 */
	0.010411265261972096497478567, /* 8.0 */
	0.009799416126158803298389475, /* 8.5 */
	0.009255462182712732917728637, /* 9.0 */
	0.008768700134139385462952823, /* 9.5 */
	0.008330563433362871256469318, /* 10.0 */
	0.007934114564314020547248100, /* 10.5 */
	0.007573675487951840794972024, /* 11.0 */
	0.007244554301320383179543912, /* 11.5 */
	0.006942840107209529865664152, /* 12.0 */
	0.006665247032707682442354394, /* 12.5 */
	0.006408994188004207068439631, /* 13.0 */
	0.006171712263039457647532867, /* 13.5 */
	0.005951370112758847735624416, /* 14.0 */
	0.005746216513010115682023589, /* 14.5 */
	0.005554733551962801371038690  /* 15.0 */
    };
    double nn;

    if (n <= 15.0) {
	nn = n + n;
	if (nn == (int)nn) return(sferr_halves[(int)nn]);
	return(gammln(n + 1.) - (n + 0.5)*log(n) + n - M_LN_SQRT_2PI);
    }

    nn = n*n;
    if (n>500) return((S0-S1/nn)/n);
    if (n> 80) return((S0-(S1-S2/nn)/nn)/n);
    if (n> 35) return((S0-(S1-(S2-S3/nn)/nn)/nn)/n);
    /* 15 < n <= 35 : */
    return((S0-(S1-(S2-(S3-S4/nn)/nn)/nn)/nn)/n);
}

/* From R, currently only used for kode = 1, m = 1, n in {0,1,2,3} : */
void dpsifn(double x, int n, int kode, int m, double *ans, int *nz, int *ierr)
{
    const static double bvalues[] = {	/* Bernoulli Numbers */
	 1.00000000000000000e+00,
	-5.00000000000000000e-01,
	 1.66666666666666667e-01,
	-3.33333333333333333e-02,
	 2.38095238095238095e-02,
	-3.33333333333333333e-02,
	 7.57575757575757576e-02,
	-2.53113553113553114e-01,
	 1.16666666666666667e+00,
	-7.09215686274509804e+00,
	 5.49711779448621554e+01,
	-5.29124242424242424e+02,
	 6.19212318840579710e+03,
	-8.65802531135531136e+04,
	 1.42551716666666667e+06,
	-2.72982310678160920e+07,
	 6.01580873900642368e+08,
	-1.51163157670921569e+10,
	 4.29614643061166667e+11,
	-1.37116552050883328e+13,
	 4.88332318973593167e+14,
	-1.92965793419400681e+16
    };

    int i, j, k, mm, mx, nn, np, nx, fn;
    double arg, den, elim, eps, fln, fx, rln, rxsq,
	r1m4, r1m5, s, slope, t, ta, tk, tol, tols, tss, tst,
	tt, t1, t2, wdtol, xdmln, xdmy, xinc, xln = 0.0 /* -Wall */,
	xm, xmin, xq, yint;
    double trm[23], trmr[n_max + 1];

    *ierr = 0;
    if (n < 0 || kode < 1 || kode > 2 || m < 1) {
	*ierr = 1;
	return;
    }
    if (x <= 0.) {
	/* use	Abramowitz & Stegun 6.4.7 "Reflection Formula"
	 *	psi(k, x) = (-1)^k psi(k, 1-x)	-  pi^{n+1} (d/dx)^n cot(x)
	 */
	if (x == round(x)) {
	    /* non-positive integer : +Inf or NaN depends on n */
	    for(j=0; j < m; j++) /* k = j + n : */
		ans[j] = ((j+n) % 2) ? INFINITY : nan("");
	    return;
	}
	/* This could cancel badly */
	dpsifn(1. - x, n, /*kode = */ 1, m, ans, nz, ierr);
	/* ans[j] == (-1)^(k+1) / gamma(k+1) * psi(k, 1 - x)
	 *	     for j = 0:(m-1) ,	k = n + j
	 */

	/* Cheat for now: only work for	 m = 1, n in {0,1,2,3} : */
	if(m > 1 || n > 3) {/* doesn't happen for digamma() .. pentagamma() */
	    /* not yet implemented */
	    *ierr = 4; return;
	}
	x *= M_PI; /* pi * x */
	if (n == 0)
	    tt = cos(x)/sin(x);
	else if (n == 1)
	    tt = -1/pow(sin(x), 2.0);
	else if (n == 2)
	    tt = 2*cos(x)/pow(sin(x), 3.0);
	else if (n == 3)
	    tt = -2*(2*pow(cos(x), 2.0) + 1.)/pow(sin(x), 4.0);
	else /* can not happen! */
	    tt = nan("");
	/* end cheat */

	s = (n % 2) ? -1. : 1.;/* s = (-1)^n */
	/* t := pi^(n+1) * d_n(x) / gamma(n+1)	, where
	 *		   d_n(x) := (d/dx)^n cot(x)*/
	t1 = t2 = s = 1.;
	for(k=0, j=k-n; j < m; k++, j++, s = -s) {
	    /* k == n+j , s = (-1)^k */
	    t1 *= M_PI;/* t1 == pi^(k+1) */
	    if(k >= 2)
		t2 *= k;/* t2 == k! == gamma(k+1) */
	    if(j >= 0) /* by cheat above,  tt === d_k(x) */
		ans[j] = s*(ans[j] + t1/t2 * tt);
	}
	if (n == 0 && kode == 2) /* unused from R, but "wrong": xln === 0 :*/
	    ans[0] += xln;
	return;
    } /* x <= 0 */

    /* else :  x > 0 */
    *nz = 0;
    xln = log(x);
    if(kode == 1 && m == 1) {/* the R case  ---  for very large x: */
	double lrg = 1/(2. * DBL_EPSILON);
	if(n == 0 && x * xln > lrg) {
	    ans[0] = -xln;
	    return;
	}
	else if(n >= 1 && x > n * lrg) {
	    ans[0] = exp(-n * xln)/n; /* == x^-n / n  ==  1/(n * x^n) */
	    return;
	}
    }
    mm = m;
    nx = imin2((int)-DBL_MIN_EXP, (int)DBL_MAX_EXP);/* = 1021 */
    r1m5 = M_LOG10_2;
    r1m4 = DBL_EPSILON * 0.5;
    wdtol = fmax(r1m4, 0.5e-18); /* 1.11e-16 */

    /* elim = approximate exponential over and underflow limit */
    elim = 2.302 * (nx * r1m5 - 3.0);/* = 700.6174... */
    for(;;) {
	nn = n + mm - 1;
	fn = nn;
	t = (fn + 1) * xln;

	/* overflow and underflow test for small and large x */

	if (fabs(t) > elim) {
	    if (t <= 0.0) {
		*nz = 0;
		*ierr = 2;
		return;
	    }
	}
	else {
	    if (x < wdtol) {
		ans[0] = pow(x, (double)-n-1);
		if (mm != 1) {
		    for(k = 1; k < mm ; k++)
			ans[k] = ans[k-1] / x;
		}
		if (n == 0 && kode == 2)
		    ans[0] += xln;
		return;
	    }

	    /* compute xmin and the number of terms of the series,  fln+1 */

	    rln = r1m5 * DBL_MANT_DIG;
	    rln = fmin(rln, 18.06);
	    fln = fmax(rln, 3.0) - 3.0;
	    yint = 3.50 + 0.40 * fln;
	    slope = 0.21 + fln * (0.0006038 * fln + 0.008677);
	    xm = yint + slope * fn;
	    mx = (int)xm + 1;
	    xmin = mx;
	    if (n != 0) {
		xm = -2.302 * rln - fmin(0.0, xln);
		arg = xm / n;
		arg = fmin(0.0, arg);
		eps = exp(arg);
		xm = 1.0 - eps;
		if (fabs(arg) < 1.0e-3)
		    xm = -arg;
		fln = x * xm / eps;
		xm = xmin - x;
		if (xm > 7.0 && fln < 15.0)
		    break;
	    }
	    xdmy = x;
	    xdmln = xln;
	    xinc = 0.0;
	    if (x < xmin) {
		nx = (int)x;
		xinc = xmin - nx;
		xdmy = x + xinc;
		xdmln = log(xdmy);
	    }

	    /* generate w(n+mm-1, x) by the asymptotic expansion */

	    t = fn * xdmln;
	    t1 = xdmln + xdmln;
	    t2 = t + xdmln;
	    tk = fmax(fabs(t), fmax(fabs(t1), fabs(t2)));
	    if (tk <= elim) /* for all but large x */
		goto L10;
	}
	nz++; /* underflow */
	mm--;
	ans[mm] = 0.;
	if (mm == 0)
	    return;
    } /* end{for()} */
    nn = (int)fln + 1;
    np = n + 1;
    t1 = (n + 1) * xln;
    t = exp(-t1);
    s = t;
    den = x;
    for(i=1; i <= nn; i++) {
	den += 1.;
	trm[i] = pow(den, (double)-np);
	s += trm[i];
    }
    ans[0] = s;
    if (n == 0 && kode == 2)
	ans[0] = s + xln;

    if (mm != 1) { /* generate higher derivatives, j > n */

	tol = wdtol / 5.0;
	for(j = 1; j < mm; j++) {
	    t /= x;
	    s = t;
	    tols = t * tol;
	    den = x;
	    for(i=1; i <= nn; i++) {
		den += 1.;
		trm[i] /= den;
		s += trm[i];
		if (trm[i] < tols)
		    break;
	    }
	    ans[j] = s;
	}
    }
    return;

  L10:
    tss = exp(-t);
    tt = 0.5 / xdmy;
    t1 = tt;
    tst = wdtol * tt;
    if (nn != 0)
	t1 = tt + 1.0 / fn;
    rxsq = 1.0 / (xdmy * xdmy);
    ta = 0.5 * rxsq;
    t = (fn + 1) * ta;
    s = t * bvalues[2];
    if (fabs(s) >= tst) {
	tk = 2.0;
	for(k = 4; k <= 22; k++) {
	    t = t * ((tk + fn + 1)/(tk + 1.0))*((tk + fn)/(tk + 2.0)) * rxsq;
	    trm[k] = t * bvalues[k-1];
	    if (fabs(trm[k]) < tst)
		break;
	    s += trm[k];
	    tk += 2.;
	}
    }
    s = (s + t1) * tss;
    if (xinc != 0.0) {

	/* backward recur from xdmy to x */

	nx = (int)xinc;
	np = nn + 1;
	if (nx > n_max) {
	    *nz = 0;
	    *ierr = 3;
	    return;
	}
	else {
	    if (nn==0)
		goto L20;
	    xm = xinc - 1.0;
	    fx = x + xm;

	    /* this loop should not be changed. fx is accurate when x is small */
	    for(i = 1; i <= nx; i++) {
		trmr[i] = pow(fx, (double)-np);
		s += trmr[i];
		xm -= 1.;
		fx = x + xm;
	    }
	}
    }
    ans[mm-1] = s;
    if (fn == 0)
	goto L30;

    /* generate lower derivatives,  j < n+mm-1 */

    for(j = 2; j <= mm; j++) {
	fn--;
	tss *= xdmy;
	t1 = tt;
	if (fn!=0)
	    t1 = tt + 1.0 / fn;
	t = (fn + 1) * ta;
	s = t * bvalues[2];
	if (fabs(s) >= tst) {
	    tk = 4 + fn;
	    for(k=4; k <= 22; k++) {
		trm[k] = trm[k] * (fn + 1) / tk;
		if (fabs(trm[k]) < tst)
		    break;
		s += trm[k];
		tk += 2.;
	    }
	}
	s = (s + t1) * tss;
	if (xinc != 0.0) {
	    if (fn == 0)
		goto L20;
	    xm = xinc - 1.0;
	    fx = x + xm;
	    for(i=1 ; i<=nx ; i++) {
		trmr[i] = trmr[i] * fx;
		s += trmr[i];
		xm -= 1.;
		fx = x + xm;
	    }
	}
	ans[mm - j] = s;
	if (fn == 0)
	    goto L30;
    }
    return;

  L20:
    for(i = 1; i <= nx; i++)
	s += 1. / (x + (nx - i)); /* avoid disastrous cancellation, PR#13714 */

  L30:
    if (kode != 2) /* always */
	ans[0] = s - xdmln;
    else if (xdmy != x) {
	xq = xdmy / x;
	ans[0] = s - log(xq);
    }
    return;
} /* dpsifn() */

double digamma(double x)
{
    double ans;
    int nz, ierr;
    if(isnan(x)) return x;
    dpsifn(x, 0, 1, 1, &ans, &nz, &ierr);
    // ML_TREAT_psigam(ierr);
    if (ierr != 0) {
    	return nan("");
    }
    return -ans;
}

double trigamma(double x)
{
    double ans;
    int nz, ierr;
    if(isnan(x)) return x;
    dpsifn(x, 1, 1, 1, &ans, &nz, &ierr);
    // ML_TREAT_psigam(ierr);
    if (ierr != 0) {
    	return nan("");
    }
    return ans;
}

__device__ double bd0(double x, double np) {
    double ej, s, s1, v;
    int j;

    if (isnan(x) || isnan(np) || np == 0.0) {
    	printf("Error: bd0 given an argument that is not a number.\n");
			asm("trap;");
		}

    if (fabs(x-np) < 0.1*(x+np)) {
	v = (x-np)/(x+np);  // might underflow to 0
	s = (x-np)*v;/* s using v -- change by MM */
	if(fabs(s) < DBL_MIN) return s;
	ej = 2*x*v;
	v = v*v;
	for (j = 1; j < 1000; j++) { /* Taylor series; 1000: no infinite loop
					as |v| < .1,  v^2000 is "zero" */
	    ej *= v;// = v^(2j+1)
	    s1 = s+ej/((j<<1)+1);
	    if (s1 == s) /* last term was effectively 0 */
		return s1 ;
	    s = s1;
	}
    }
    /* else:  | x - np |  is not too small */
    return(x*log(x/np)+np-x);
}

/* A log-likelihood helper function */
// __device__ double loglik (int n, double th, 
// 							 const vector<double> &mu, const vector<double> &y) {
// 	
// 	double sum = 0;
// 	unsigned int n1 = (unsigned int)n;
// 	for (unsigned int i = 0; i < n1; i++) {
// 		sum += (gammln(th+y[i]) - gammln(th) - gammln(y[i]+1) + th * 
// 					 log(th) + y[i] * log(mu[i] + (y[i] == 0 ? 1 : 0)) - (th + y[i]) * 
// 					 log(th + mu[i]));
// 	}
// 	return sum;
// }

/* Link functions */
double linkfun (double mu) {
	return log(mu);
}

__device__ double linkinv (double eta) {
	double temp = (eta < MTHRESH) ? DOUBLE_EPS : ((eta > THRESH) ? INVEPS : exp(eta));
	return x_d_opx(temp);
	// return (max(exp(eta), DBL_MIN));
}

__device__ void mu_eta (double* eta, double* mu_eta_val, int size) {
	// vector<double> result;
	for (unsigned int i = 0; i < size; i++) {
		double opexp = 1 + exp(eta[i]);
		mu_eta_val[i] = ((eta[i] > THRESH || eta[i] < MTHRESH) ? DOUBLE_EPS : exp(eta[i])/(opexp * opexp));
		// result.push_back(max(exp(eta[i]), DBL_MIN));
	}
	// return result;
}

// Our own dbinom function
__device__ double* dbinom (double* x, double* n, double* p, bool log_p, int size) {
	
	// Error checking
	for (unsigned int i = 0; i < size; i++) {
		if ((p[i] < 0) || (p[i] > 1)) {
			printf("Error in dbinom(): p must be between 0 and 1\n");
			asm("trap;");
		}
	}
	for (unsigned int i = 0; i < size; i++) {
		if (x[i] < 0) {
			printf("Error in dbinom(): x must be >=0\n");
			asm("trap;");
		}
	}
	for (unsigned int i = 0; i < size; i++) {
		if (n[i] < x[i]) {
			printf("Error in dbinom(); x must be <= than the binomial denominator\n");
			asm("trap;");
		}
	}
	double* q = (double *)malloc(size*sizeof(double));
	for (unsigned int i = 0; i < size; i++) {
		q[i] = 1 - p[i];
	}
	
	double* result = (double *)malloc(size*sizeof(double));
	
	for (unsigned int i = 0; i < size; i++) {
	
		double lf, lc;
	
		if (p[i] == 0) {
			result[i] = (((x[i] == 0) ? (log_p ? 0. : 1.) : (log_p ? -DBL_MAX : 0.)));
			continue;
		}
		if (q[i] == 0) {
			result[i] = (((x[i] == n[i]) ? (log_p ? 0. : 1.) : (log_p ? -DBL_MAX : 0.)));
			continue;
		}
	
		if (x[i] == 0) {
			if (n[i] == 0) {
				result[i] = ((log_p ? 0. : 1.));
				continue;
			}
			lc = (p[i] < 0.1) ? -bd0(n[i],n[i]*q[i]) - n[i]*p[i] : n[i]*log(q[i]);
			result[i] = (( (log_p	?  (lc)	 : exp(lc)) ));
			continue;
		}
		if (x[i] == n[i]) {
			lc = (q[i] < 0.1) ? -bd0(n[i],n[i]*p[i]) - n[i]*q[i] : n[i]*log(p[i]);
			result[i] = (( (log_p	?  (lc)	 : exp(lc)) ));
			continue;
		}
		if (x[i] < 0 || x[i] > n[i]) {
			result[i] = (( (log_p ? -DBL_MAX : 0.) ));
			continue;
		}
	
		/* n*p or n*q can underflow to zero if n and p or q are small.  This
			 used to occur in dbeta, and gives NaN as from R 2.3.0.  */
		lc = stirlerr(n[i]) - stirlerr(x[i]) - stirlerr(n[i]-x[i]) - bd0(x[i],n[i]*p[i]) - bd0(n[i]-x[i],n[i]*q[i]);

		/* f = (M_2PI*x*(n-x))/n; could overflow or underflow */
		/* Upto R 2.7.1:
		 * lf = log(M_2PI) + log(x) + log(n-x) - log(n);
		 * -- following is much better for  x << n : */
		lf = M_LN_2PI + log(x[i]) + log1p(- x[i]/n[i]);

		result[i] = (R_D_exp((lc - 0.5*lf), log_p));
	}
	return result;
}

/* Logistic binomial helper functions */

// Calculate variance from mu
__device__ void logit_variance (double* mu, double* varmu, int size) {
	// vector<double> result;
	for (unsigned int i = 0; i < size; i++) {
		varmu[i] = mu[i]*(1-mu[i]);
	}
	// return result;
}

// Returns TRUE if mu vector is within acceptable variance
__device__ bool logit_validmu (double* mu, int size) {
	for (unsigned int i = 0; i < size; i++) {
		if (isinf(mu[i]) || mu[i] <= 0.0 || mu[i] >= 1.0) {
			return false;
		}
	}
	return true;
}

__device__ double y_log_y (double y, double mu) {
	return (y != 0.) ? (y * log(y/mu)) : 0;
}

// Return deviance residuals
__device__ void logit_dev_residuals (double* y, double* mu, double* dev, int size) {
	// vector<double> result;
	
	// int i;
	unsigned int n = size;
	unsigned int lmu = size;
	// int nprot = 1;
	
	for (unsigned int i = 0; i < n; i++) {
		dev[i] = y[i];
	}
	
	if (lmu != n && lmu != 1) {
		printf("Error: Argument mu must be a numeric vector of length 1 or length %d\n", n);
		asm("trap;");
	}
	
	for (int i = 0; i < n; i++) {
		dev[i] = 2 * (y_log_y(y[i], mu[i]) + y_log_y(1 - y[i], 1 - mu[i]));
	}
	// return result;
}

// Calculate the AIC
__device__ double logit_aic (double* y, double* mu, int size) {
	// vector<double> m;
// 	bool is_n = false;
// 	for (unsigned int i = 0; i < n.size(); i++) {
// 		if (n[i] > 1) {
// 			is_n = true;
// 			break;
// 		}
// 	}
	
// 	if (is_n) {
// 		m = n;
// 	} else {
	double* m = (double *)malloc(size*sizeof(double));
	for (unsigned int i = 0; i < size; i++) {
		m[i] = 1.0;
	}
// 	}
	
// 	double sum_m = 0;
// 	double sum = 0;
// 	for (unsigned int i = 0; i < n.size(); i++) {
// 		sum_m += m[i];
// 	}
// 	if (sum_m > 0) {
// 		for (unsigned int i = 0; i < n.size(); i++) {
// 			sum += 1/m[i];
// 		}
// 	}
	
	double* m_prod_y = (double *)malloc(size*sizeof(double));;
// 	vector<double> m_rounded;
	
	for (unsigned int i = 0; i < size; i++) {
		m_prod_y[i] = round(y[i]);
	}
	
// 	for (unsigned int i = 0; i < y.size(); i++) {
// 		m_rounded.push_back(round(m[i]));
// 	}
	
	// DEBUG
	// printf("Breakpoint Alpha\n");
// 	for (unsigned int i = 0; i < y.size(); i++) {
// 		printf("%d: %f\n", i, m_prod_y[i]);
// 	}
// 	for (unsigned int i = 0; i < y.size(); i++) {
// 		printf("%d: %f\n", i, m[i]);
// 	}
// 	for (unsigned int i = 0; i < y.size(); i++) {
// 		printf("%d: %f\n", i, mu[i]);
// 	}
	
	double* db = dbinom(m_prod_y, m, mu, true, size);
	
	// DEBUG
	// printf("Breakpoint Beta\n");
	
	double sum = 0;
	for (unsigned int i = 0; i < size; i++) {
// 		if (m[i] > 0) {
		sum += db[i];
// 		}
	}
	
	return -2*sum;
}

// [ret] dqrdc2(vector<vector<double> > x, int n, int n, int p, double tol, int *k,
// 						 double *qraux, vector<int> jpvt, double *work) {
// 						 
// 	
// 
// [ret] dqrls(vector<vector<double> > x, int n, int p, vector<double> y, int ny, double tol,
// 						vector<vector<double> > b, vector<double> rsd, vector<double> qty,
// 						int *k, vector<int> jpvt, double *qraux, double *work) {
// 	
// 	// Internal variables
// 	int info, j, jj, kk;
// 	
// 	// Reduce x
// 	dqrdc2(x,n,n,p,tol,k,qraux,jpvt,work);

__device__ lmfit Cdqrls(double** x, double* y, double tol, bool chk, int y_size, int x_size) {
	
	// double ans;
	double *qr;
	double *y_for;
	double *coefficients;
	double *residuals;
	double *effects;
	int *pivot;
	double *qraux;
	
	int n, ny = 1, p, rank, pivoted = 0;
	// int nprotect = 4;
	
	double rtol = tol;
	double *work;
	
	n = y_size; // x[0].size
	p = x_size; // x.size
	
	// Sanity checks
// 	if (n != (int)y.size()) {
// 		printf("Error: Dimensions of \'x\' (%d,%d) and \'y\' (%d) do not match", n, p, (int)y.size());
// 		exit(1);
// 	}
	
	for (int i = 0; i < p; i++) {
		for (int j = 0; j < n; j++) {
			if (isinf(x[i][j])) {
				printf("Error: NA/NaN/Inf in \'x\' matrix: (%d,%d)\n", i, j);
				asm("trap;");
			}
		}
	}
	
	for (unsigned int i = 0; i < p; i++) {
		if (isinf(y[i])) {
			printf("Error: NA/NaN/Inf in \'y\' vector: element %d\n", i);
			asm("trap;");
		}
	}
	
	// Encoding in C types
	// qr = x;
	// Use column-major order for Fortran
	int flat_size = n*p;
	qr = (double *)malloc(flat_size*sizeof(double));
	for (int i = 0; i < p; i++) {
		for (int j = 0; j < n; j++) {
			qr[i*n+j] = x[i][j];
		}
	}
	
	// Turn y into a matrix for use in Fortran
	// y_for = y;
	y_for = (double *)malloc(n*sizeof(double));
	for (unsigned int i = 0; i < y_size; i++) {
		y_for[i] = y[i];
	}
	
	// Coefficients mapping
	coefficients = (double *)malloc(p*sizeof(double));
	for (int i = 0; i < p; i++) {
		coefficients[i] = 0.0;
	}

	// residuals = y;
	residuals = (double *)malloc((int)y_size*sizeof(double));
	for (unsigned int i = 0; i < y_size; i++) {
		residuals[i] = y[i];
	}
	
	// effects = y;
	effects = (double *)malloc((int)y_size*sizeof(double));
	for (unsigned int i = 0; i < y_size; i++) {
		effects[i] = y[i];
	}
	
	pivot = (int *)malloc(p*sizeof(int));
	for (int i = 0; i < p; i++) {
		pivot[i] = i+1;
	}
	
	qraux = (double *)malloc(p*sizeof(double));
	work = (double *)malloc(2*p*sizeof(double));
	
	// DEBUG
	printf("Breakpoint Yocto\n");
	// exit(0);
	
	// Call dqrls
	dqrls_(qr, (integer *)&n, (integer *)&p, y_for, (integer *)&ny, &rtol, coefficients, residuals, effects, (integer *)&rank, 
				(integer *)pivot, qraux, work);
				
	for	(int i = 0; i < p; i++) {
		if (pivot[i] != i+1) {
			pivoted = 1;
			break;
		}
	}
	
	// DEBUG
 	printf("Breakpoint Zeta\n");
// 	for (int i = 0; i < (int)x.size(); i++) {
// 		for (int j = 9; j < 10; j++) {
// 			if (qr[i][j]) {
// 				printf("True\n");
// 			} else {
// 				printf("False\n");
// 			}
// 		}
// 	}
// 	exit(0);

	// Re-encode as a double** matrix
	double** qr_new = (double **)malloc(p*sizeof(double *));
	for (int i = 0; i < p; i++) {
		qr_new[i] = (double *)malloc(n*sizeof(double));
		for (int j = 0; j < n; j++) {
			qr_new[i][j] = qr[i*n+j];
		}
	}
	
	// Re-encode in C++ vectors
// 	vector<vector<double> > qr_vec;
// 	for (int i = 0; i < (int)x.size(); i++) {
// 		vector<double> temp;
// 		for (int j = 0; j < (int)x[i].size(); j++) {
// 			temp.push_back(qr[i*n+j]);
// 		}
// 		qr_vec.push_back(temp);
// 	}
	
	// DEBUG
	// printf("Breakpoint Molto\n");
	// exit(0);
	
// 	vector<double> coefficients_vec;
// 	for (int i = 0; i < p; i++) {
// 		coefficients_vec.push_back(coefficients[i]);
// 	}
// 	
// 	vector<double> residuals_vec;
// 	for (int i = 0; i < (int)y.size(); i++) {
// 		residuals_vec.push_back(residuals[i]);
// 	}
// 	
// 	vector<double> effects_vec;
// 	for (int i = 0; i < (int)y.size(); i++) {
// 		effects_vec.push_back(effects[i]);
// 	}
// 	
// 	vector<int> pivot_vec;
// 	for (int i = 0; i < p; i++) {
// 		pivot_vec.push_back(pivot[i]);
// 	}
// 	
// 	vector<double> qraux_vec;
// 	for (int i = 0; i < p; i++) {
// 		qraux_vec.push_back(qraux[i]);
// 	}
	
	lmfit lm1;
	lm1.qr = qr_new;
	lm1.coefficients = coefficients;
	lm1.residuals = residuals;
	lm1.effects = effects;
	lm1.rank = rank;
	lm1.pivot = pivot;
	lm1.qraux = qraux;
	lm1.tol = tol;
	lm1.pivoted = pivoted;
	
	// DEBUG
	// printf("Breakpoint Eta\n");
	// exit(0);
	
	return lm1;
}

/*
 * Theta_ml's score function
 */
// double theta_score (double n, double th, vector<double> &mu, vector<double> &y) {
// 	double sum = 0.0;
// 	for (unsigned int i = 0; i < y.size(); i++) {
// 		sum += digamma(th + y[i]) - digamma(th) + log(th) + 1 - log(th + mu[i]) - 
// 					 (y[i] + th)/(mu[i] + th);
// 	}
// 	return sum;
// }

/*
 * Theta_ml's info function
 */
// double theta_info (double n, double th, vector<double> &mu, vector<double> &y) {
// 	double sum = 0.0;
// 	for (unsigned int i = 0; i < y.size(); i++) {
// 		sum += -trigamma(th + y[i]) + trigamma(th) - 1/th + 2/(mu[i] + th) - 
// 					 (y[i] + th)/pow((mu[i] + th),2.0);
// 	}
// 	return sum;
// }

/*
 * This is the theta ML estimator code
 * Assume equal weights (=1) for all predictors
 */
// pair <double,double> theta_ml (vector<double> &y, vector<double> &mu, double n, int limit) {
// 	
// 	// Control variables
// 	double epsilon = 1e-8;
// 	
// 	double denom = 0.0;
// 	for (unsigned int i = 0; i < y.size(); i++) {
// 		denom += pow((y[i]/mu[i] - 1.0), 2.0);
// 	}
// 	double t0 = n/denom;
// 	int it = 0;
// 	double del = 1.0;
// 	double i;
// 	for (; (it < limit) && fabs(del) > epsilon; it++) {
// 		t0 = fabs(t0);
// 		i = theta_info(n, t0, mu, y);
// 		del = theta_score(n, t0, mu, y)/i;
// 		t0 += del;
// 	}
// 	if (t0 < 0.0) {
// 		t0 = 0.0;
// 		printf("Warning: theta estimate truncated at zero\n");
// 	}
// 	if (it == limit) {
// 		printf("Warning: theta estimator iteration limit reached\n");
// 	}
// 	double se = sqrt(1/i);
// 	pair <double,double> retval (t0,se);
// 	return retval;
// 	// return t0;
// }
	
/* 
 * This is really a pared down glm_fit that only implements the portion needed
 * for the logistic regression fit
 */
__device__ void glm_fit (double* y, double** x, int* y_size, int* x_size, int* lm_pivot, bool* good, double* mu, double* eta, double* devold_vec, double* coef, double* coefold, double* w, double* varmu, double* mu_eta_val, double* z, double* prefit_y, double** prefit_x, double* lm_coefficients, double* start, double* residuals, double** Rmat, double* wt, double* wtdmu_vec, double* weights) {

	// DEBUG
	// printf("Breakpoint 1\n");

	// Control variables
	double epsilon = 1e-8;
	int maxit = 25;
	
	// Setup variables
	bool conv = false;
	int nobs = *y_size;
	int nvars = *x_size;
	
// 	vector<int> lm_pivot;
// 	vector<bool> good;
	
	// Eta always valid
	
	// Initialize the mu's
	// vector<double> mu;
	// vector<double> n;
	for (unsigned int i = 0; i < nobs; i++) {
		if (y[i] < 0) {
			printf("Error: negative values not allowed for the negative binomial model. Exiting.\n");
			asm("trap;");
		}
		// n.push_back(1.0);
		mu[i] = ((y[i] + 0.5)/2);
	}
	
	// Initialize the eta's
	// vector<double> eta;
// 	if (etastart.size() == 0) {
	for (unsigned int i = 0; i < nobs; i++) {
		eta[i] = log(x_d_omx(mu[i]));
	}
// 	} else {
// 		eta = etastart;
// 	}
	
	for (unsigned int i = 0; i < nobs; i++) {
		double tmp = (eta[i] < MTHRESH) ? DOUBLE_EPS : ((eta[i] > THRESH) ? INVEPS : exp(eta[i]));
		mu[i] = x_d_opx(tmp);
	}
		
	if (!logit_validmu(mu, nobs)) {
		printf("Invalid starting mu detected. Exiting.\n");
		asm("trap;");
	}
	
	// calculate initial deviance and coefficient
	logit_dev_residuals(y, mu, devold_vec, nobs);
	double devold = 0.0;
	for (unsigned int i = 0; i < nobs; i++) {
		devold += devold_vec[i];
	}
	
	bool boundary = false;
	
	// Coefficient vector declarations
// 	vector<double> coef;
// 	vector<double> coefold;
	coefold[0] = -INFINITY;
	
	lmfit lm;
// 	vector<double> w;
	double dev;
	
	// DEBUG
	// printf("Breakpoint Upsilon\n");
	// exit(0);
	
	// DEBUG
	// printf("Breakpoint 2\n");
	
	// The iteratively reweighting L.S. iteration
	int iter = 0;
	for (; iter < maxit; iter++) {
		
		// DEBUG
		// printf("Iter: %d\n", iter);
		
		logit_variance(mu, varmu, nobs);
		for (unsigned int j = 0; j < nobs; j++) {
			if (isnan(varmu[j])) {
				printf("NaNs in the mu variance: cannot proceed. Exiting.\n");
				asm("trap;");
			} else if (varmu[j] == 0) {
				printf("Zeroes in the mu variance: cannot proceed. Exiting.\n");
				asm("trap;");
			}
		}
		mu_eta(eta, mu_eta_val, nobs);
		for (unsigned int j = 0; j < nobs; j++) {
			if (isnan(mu_eta_val[j])) {
				printf("NaNs in the d(mu)/d(eta)\n");
				asm("trap;");
			}
		}
		
		// Save the good ones (i.e. the ones where mu_eta_val is nonzero)
		// good.clear();
		unsigned int num_false = 0;
		for (unsigned int i = 0; i < nobs; i++) {
			if (mu_eta_val[i] != 0) {
				good[i] = true;
			} else {
				good[i] = false;
				num_false++;
			}
		}
		
		if (num_false == nobs) { // All not good
			conv = false;
			printf("Error: No observations informative at iteration %d\n", iter);
			break;
		}
		
		// vector<double> z;
		for (unsigned int j = 0; j < nobs; j++) {
			if (good[j] == true) {
				double this_val = eta[j] + (y[j] - mu[j])/mu_eta_val[j];
				z[j] = this_val;
			}
		}
		// vector<double> w;
		for (unsigned int j = 0; j < nobs; j++) {
			if (good[j] == true) {
				double this_val = sqrt(pow(mu_eta_val[j],2.0))/varmu[j];
				w[j] = this_val;
			}
		}
		
		// Set up dot products for Cdqrls
		// vector<vector<double> > prefit_x;
		// vector<double> prefit_y;
		for (unsigned int j = 0; j < nvars; j++) {
			// vector<double> temp;
			for (unsigned int k = 0; k < nobs; k++) {
				if (good[k] == true) {
					prefit_x[j][k] = x[j][k] * w[k];
				}
			}
			// prefit_x.push_back(temp);
		}
		
		for (unsigned int j = 0; j < nobs; j++) {
			prefit_y[j] = z[j] * w[j];
		}
		
		// if (iter > 0) {
		// delete &lm;
		// }
		
		// DEBUG
		// printf("Breakpoint Upsilon 2\n");
		// exit(0);
		
		lm = Cdqrls(prefit_x, prefit_y, min(1e-7, epsilon/1000), false, nobs, nvars);
		
		// DEBUG
		// printf("Breakpoint Tau 2\n");
		// exit(0);
		
		lm_coefficients = lm.coefficients;
		for (int j = 0; j < nvars; j++) {
			if (isinf(lm_coefficients[j])) {
				conv = false;
				printf("Warning: Non-finite coefficients at iteration %d\n", iter);
				break;
			}
		}
		
		// Stop if not enough parameters
		if (nobs < lm.rank) {
			printf("Error: X matrix has rank %d, but only %d observation(s).\n", lm.rank, nobs);
			asm("trap;");
		}
		
		// calculate updated values of eta and mu with the new coef
		lm_pivot = lm.pivot;
		// vector<double> start; // (nvars,0.0);
		for (int j = 0; j < nvars; j++) {
			// This code not necessary: if (lm_pivot[j]) {
				start[j] = lm_coefficients[lm_pivot[j]-1];
			// }
		}
		
		// New eta needs a matrix calculation
		// vector<double> new_eta;
		for (int j = 0; j < nobs; j++) {
			double new_eta_one = 0.0;
			for (int k = 0; k < nvars; k++) {
				new_eta_one += x[k][j] * start[k];
			}
			eta[j] = new_eta_one;
		}
		// eta = new_eta;
		
		// vector<double> new_mu;
		for (unsigned int j = 0; j < nobs; j++) {
			mu[j] = linkinv(eta[j]);
		}
		// mu = new_mu;
		
		logit_dev_residuals(y, mu, devold_vec, nobs);
		dev = 0.0;
		for (unsigned int j = 0; j < nobs; j++) {
			dev += devold_vec[j];
		}
		
		// check for divergence
		boundary = false;
		
		if (isinf(dev)) {
			if (coefold[0] == -INFINITY) {
				printf("Error: divergence in function fitting. No valid set of ");
				printf("coefficients has been found. Exiting.\n");
				asm("trap;");
			}
			int ii = 1;
			while (isinf(dev)) {
				if (ii > maxit) {
					printf("Error: Inner loop 1; cannot correct step size\n");
					asm("trap;");
				}
				ii++;
				// Element-wise addition
				for (unsigned int j = 0; j < nvars; j++) {
					start[j] = (start[j] + coefold[j])/2;
				}
					
				// New eta needs a matrix calculation
				// new_eta.clear();
				for (int j = 0; j < nobs; j++) {
					double new_eta_one = 0.0;
					for (int k = 0; k < nvars; k++) {
						new_eta_one += x[k][j] * start[k];
					}
					eta[j] = new_eta_one;
				}
				// eta = new_eta;
				
				// new_mu.clear();
				for (unsigned int j = 0; j < nobs; j++) {
					mu[j] = linkinv(eta[j]);
				}
				// mu = new_mu;
				
				logit_dev_residuals(y, mu, devold_vec, nobs);
				dev = 0.0;
				for (unsigned int j = 0; j < nobs; j++) {
					dev += devold_vec[j];
				}
			}
			boundary = true;
		}
		
		// check for fitted values outside the domain
		if (!logit_validmu(mu, nobs)) {
			if (coefold[0] == -INFINITY) {
				printf("Error: Fitted mu is outside the valid domain. No valid set of ");
				printf("coefficients has been found. Exiting.\n");
				asm("trap;");
			}
			int ii = 1;
			while (!logit_validmu(mu, nobs)) {
				if (ii > maxit) {
					printf("Error: Inner loop 2; cannot correct step size\n");
					asm("trap;");
				}
				ii++;
				// Element-wise addition
				for (unsigned int j = 0; j < nvars; j++) {
					start[j] = (start[j] + coefold[j])/2;
				}
				
				// New eta needs a matrix calculation
				// new_eta.clear();
				for (int j = 0; j < nobs; j++) {
					double new_eta_one = 0.0;
					for (int k = 0; k < nvars; k++) {
						new_eta_one += x[k][j] * start[k];
					}
					eta[j] = new_eta_one;
				}
				// eta = new_eta;
				
				// new_mu.clear();
				for (unsigned int j = 0; j < nobs; j++) {
					mu[j] = linkinv(eta[j]);
				}
				// mu = new_mu;
			}
			boundary = true;
			logit_dev_residuals(y, mu, devold_vec, nobs);
			dev = 0.0;
			for (unsigned int j = 0; j < nobs; j++) {
				dev += devold_vec[j];
			}
		}
		
		// check for convergence
		if (abs(dev - devold)/(0.1 + abs(dev)) < epsilon) {
			conv = true;
			coef = start;
			break;
		} else {
			devold = dev;
			coefold = start;
			coef = coefold;
		}
	}
	// end IRLS iteration
	
	// DEBUG
	// printf("Breakpoint Tau\n");
	
	// DEBUG
	// printf("Breakpoint 3\n");
	
	if (!conv) {
		printf("Warning: fitting algorithm did not converge\n");
	}
	if (boundary) {
		printf("Warning: fitting algorithm stopped at boundary value\n");
	}
	// Binomial family check
	double eps = 10*epsilon;
	for (unsigned int i = 0; i < nobs; i++) {
		if ((mu[i] > 1-eps) || (mu[i] < eps)) {
			printf("Warning: Fitted probabilities numerically 0 or 1 occurred\n");
			break;
		}
	}
	
	// DEBUG
	// printf("Breakpoint 3a\n");
	
	// If X matrix was not full rank then columns were pivoted,
  // hence we need to re-label the names ...
  lm_pivot = lm.pivot;
  if (lm.rank < nvars) {
  	for (unsigned int i = lm.rank; i < (unsigned int)nvars; i++) {
  		coef[lm_pivot[i]-1] = 0;
  	}
  }
  
  // DEBUG
	// printf("Breakpoint 3b\n");
  
  // update by accurate calculation, including 0-weight cases.
  mu_eta(eta, mu_eta_val, nobs);
  // vector<double> residuals;
  for (unsigned int i = 0; i < nobs; i++) {
  	double temp = (y[i] - mu[i])/mu_eta_val[i];
  	residuals[i] = temp;
  }
  double** lm_qr = lm.qr;
  
  // DEBUG
	// printf("Breakpoint 3c\n");
  
  int sum_good = 0;
  for (unsigned int i = 0; i < nobs; i++) {
  	if (good[i] == true) {
  		sum_good++;
  	}
  }
  int nr = min(sum_good, nvars);
  // vector<vector<double> > Rmat;
  if (nr < nvars) {
  	for (unsigned int i = 0; i < (unsigned int)nvars; i++) {
  		// vector<double> temp;
  		if (i < (unsigned int)nr) {
  			for (unsigned int j = 0; j < (unsigned int)nvars; j++) {
  				if (i > j) {
						Rmat[i][j] = 0.0;
					} else {
						Rmat[i][j] = lm_qr[i][j];
					}
				}
  		} else {
				for (unsigned int j = 0; j < (unsigned int)nvars; j++) {
					if (i == j) {
						Rmat[i][j] = 1.0;
					} else {
						Rmat[i][j] = 0.0;
					}
				}
			}
			// Rmat.push_back(temp);
		}
  	
  } else {
  
		for (unsigned int i = 0; i < (unsigned int)nvars; i++) {
			// vector<double> temp;
			for (unsigned int j = 0; j < (unsigned int)nvars; j++) {
				if (i > j) {
					Rmat[i][j] = 0.0;
				} else {
					Rmat[i][j] = lm_qr[i][j];
				}
			}
			// Rmat.push_back(temp);
		}
	}
	
	// DEBUG
	// printf("Breakpoint 3d\n");
	
  // vector<double> wt;
  for (unsigned int i = 0; i < nobs; i++) {
  	if (good[i]) {
  		wt[i] = pow(w[i],2.0);
  	} else {
  		wt[i] = 0.0;
  	}
  }
  // calculate null deviance -- corrected in glm() if offset and intercept
  double wtdmu = 0.0;
  for (unsigned int i = 0; i < nobs; i++) {
  	wtdmu += y[i];
  }
  wtdmu = wtdmu/(double)nobs;
  
  // DEBUG
	// printf("Breakpoint 3e\n");
  
  // Produce a vector version of wtdmu
  // vector<double> wtdmu_vec;
  for (unsigned int i = 0; i < nobs; i++) {
  	wtdmu_vec[i] = wtdmu;
  }
  
  // DEBUG
	// printf("Breakpoint 3f\n");
  
  logit_dev_residuals(y, wtdmu_vec, devold_vec, nobs);
  double nulldev = 0.0;
  for (unsigned int i = 0; i < nobs; i++) {
  	nulldev += devold_vec[i];
  }
  
  // calculate df
  int n_ok = nobs;
  int nulldf = n_ok - 1;
  int rank = lm.rank;
  int resdf = n_ok - rank;
  
  // DEBUG
	// printf("Breakpoint 3g\n");
  
  // calculate AIC
  double aic_model = logit_aic(y, mu, nobs) + 2*rank;
  
  // DEBUG
	// printf("Breakpoint 3h\n");
  
  double* effects = lm.effects;
  
  // vector<double> weights;
  for (int i = 0; i < nobs; i++) {
  	weights[i] = 1.0;
  }
  
  double** qr = lm.qr;
	double* qraux = lm.qraux;
	int* pivot = lm.pivot;
	
	// DEBUG
	// printf("Breakpoint 4\n");
  
//   fit this_fit(coef, residuals, mu, effects, Rmat, rank, qr, qraux, pivot, lm.getTol(),
//   						 eta, dev, aic_model, nulldev, iter, wt, weights, resdf, nulldf, y, 
//   						 conv, boundary);
//   // delete &lm;
//   *outfit = this_fit;
	
	// Output the values of "outfit"
	// double* coefficients = *outfit.getCoefficients();
	printf("<-- Coefficients -->\n");
	for (unsigned int i = 0; i < nvars; i++) {
		printf("%f", coef[i]);
		if (i != nvars-1) {
			printf("\t");
		} else {
			printf("\n\n");
		}
	}
	
	// double* residuals = outfit.getResiduals();
	printf("<-- Residuals -->\n");
	for (unsigned int i = 0; i < nobs; i++) {
		printf("%f", residuals[i]);
		if (i != nobs-1) {
			printf("\t");
		} else {
			printf("\n\n");
		}
	}
	
	// double* fitted_values = outfit.getFittedValues();
	printf("<-- Fitted Values -->\n");
	for (unsigned int i = 0; i < nobs; i++) {
		printf("%f", mu[i]);
		if (i != nobs-1) {
			printf("\t");
		} else {
			printf("\n\n");
		}
	}
	
	// double* effects = outfit.getEffects();
	printf("<-- Effects -->\n");
	for (unsigned int i = 0; i < nobs; i++) {
		printf("%f", effects[i]);
		if (i != nobs-1) {
			printf("\t");
		} else {
			printf("\n\n");
		}
	}
	
	// double** R = outfit.getR();
	printf("<-- R -->\n");
	for (unsigned int i = 0; i < nvars; i++) {
		for (unsigned int j = 0; j < nvars; j++) {
			printf("%f", Rmat[i][j]);
			if (j != nvars-1) {
				printf("\t");
			} else {
				printf("\n");
			}
		}
	}
	
	printf("\n");
	
	printf("<-- Rank -->\n");
	printf("%d\n\n", rank);
	
	// double** qr = outfit.getQr();
	printf("<-- QR matrix -->\n");
	for (int i = 0; i < nvars; i++) {
		for (int j = 0; j < nobs; i++) {
			printf("%f", qr[i][j]);
			if (j != nobs-1) {
				printf("\t");
			} else {
				printf("\n");
			}
		}
	}
	
	printf("\n");
	
	// vector<double> qraux = outfit.getQraux();
	printf("<-- Qraux -->\n");
	for (int i = 0; i < nvars; i++) {
		printf("%f", qraux[i]);
		if (i != nvars-1) {
			printf("\t");
		} else {
			printf("\n\n");
		}
	}
	
	// vector<int> pivot = outfit.getPivot();
	printf("<-- Pivot vector -->\n");
	for (int i = 0; i < nvars; i++) {
		printf("%d", pivot[i]);
		if (i != nvars-1) {
			printf("\t");
		} else {
			printf("\n\n");
		}
	}
	
	printf("<-- Tol -->\n");
	printf("%f\n\n", lm.tol);
	
	// double* linear_predictors = outfit.getLinearPredictors();
	printf("<-- Linear Predictors -->\n");
	for (unsigned int i = 0; i < nobs; i++) {
		printf("%f", eta[i]);
		if (i != nobs-1) {
			printf("\t");
		} else {
			printf("\n\n");
		}
	}
	
	printf("<-- Deviance -->\n");
	printf("%f\n\n", dev);
	
	printf("<-- AIC -->\n");
	printf("%f\n\n", aic_model);
	
	printf("<-- Null deviance -->\n");
	printf("%f\n\n", nulldev);
	
	printf("<-- Number of iterations -->\n");
	printf("%d\n\n", iter);
	
	// double* weights = outfit.getWeights();
	printf("<-- Weights -->\n");
	for (unsigned int i = 0; i < nobs; i++) {
		printf("%f", wt[i]);
		if (i != nobs-1) {
			printf("\t");
		} else {
			printf("\n\n");
		}
	}
	
	// double* prior_weights = outfit.getPriorWeights();
	printf("<-- Prior Weights -->\n");
	for (unsigned int i = 0; i < nobs; i++) {
		printf("%f", weights[i]);
		if (i != nobs-1) {
			printf("\t");
		} else {
			printf("\n\n");
		}
	}
	
	printf("<-- Degrees of freedom residual -->\n");
	printf("%d\n\n", resdf);
	
	printf("<-- Degrees of freedom null -->n");
	printf("%d\n\n", nulldf);
	
	printf("<-- Converged -->\n");
	const char* bool_out = (conv) ? "true" : "false";
	printf("%s\n\n", bool_out);
	
	printf("<-- Boundary -->\n");
	bool_out = (boundary) ? "true" : "false";
	printf("%s\n\n", bool_out);
}

__global__ void apportionWork(double* y_gpu, double** x_gpu, int* y_size, int* x_size, int* lm_pivot_gpu, bool* good_gpu, double* mu_gpu, double* eta_gpu, double* devold_vec_gpu, double* coef_gpu, double* coefold_gpu, double* w_gpu, double* varmu_gpu, double* mu_eta_gpu, double* z_gpu, double* prefit_y_gpu, double** prefit_x_gpu, double* lm_coefficients_gpu, double* start_gpu, double* residuals_gpu, double** Rmat_gpu, double* wt_gpu, double* wtdmu_vec_gpu, double* weights_gpu) {
	
	// Which thread am I?
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	// For now, we're just going to have thread 0 run a logistic regression
	if (tid == 0) {
		glm_fit(y_gpu, x_gpu, y_size, x_size, lm_pivot_gpu, good_gpu, mu_gpu, eta_gpu, devold_vec_gpu, coef_gpu, coefold_gpu, w_gpu, varmu_gpu, mu_eta_gpu, z_gpu, prefit_y_gpu, prefit_x_gpu, lm_coefficients_gpu, start_gpu, residuals_gpu, Rmat_gpu, wt_gpu, wtdmu_vec_gpu, weights_gpu);
	}
}

/* Main function */
int main (int argc, char* argv[]) {

	// The response vector (y) file
	// Each value is listed one per row
	// const char y_file[STRSIZE];
	
 	// The predictor matrix (x) file
 	// Rows are observations, columns are predictor variables
 	// Assumes tab-delimited values
 	// const char x_file[STRSIZE];
 	
 	// The initial theta to use in fitting
 	// double init_theta;
 	
 	// DEBUG
 	// printf("Breakpoint Pre-Sigma\n");
 	
 	// Argument checking
 	if (argc != 3) {
 		printf("Incorrect number of arguments: found %d but expected 2. Exiting.\n", argc-1);
 		printf("Usage: glm.logit [response file] [predictor file]\n");
 		return 1;
 	}
	const char* y_file = argv[1];
	const char* x_file = argv[2];
 		// init_theta = atof(argv[3]);
 	
 	// DEBUG
 	printf("Breakpoint Sigma\n");
 	
 	// Data structures for imported data
 	double y[1024];
 	int ysize = 0;
	double x[1024][1024];
	// int xsize = 0;
	// int xisize = 0;
	
	// Verify files, and import data to memory
	struct stat ybuf;
	if (stat(y_file, &ybuf)) { // Report the error and exit
		printf("Error trying to stat %s: %s\n", y_file, strerror(errno));
		return 1;
	}
	// Check that the file is not empty
	if (ybuf.st_size == 0) {
		printf("Error: Response file cannot be empty. Exiting.\n");
		return 1;
	}
	
	struct stat xbuf;
	if (stat(x_file, &xbuf)) { // Report the error and exit
		printf("Error trying to stat %s: %s\n", x_file, strerror(errno));
		return 1;
	}
	// Check that the file is not empty
	if (xbuf.st_size == 0) {
		printf("Error: Predictor file cannot be empty. Exiting.\n");
		return 1;
	}
	
	// Bring response file data into memory
	char linebuf[STRSIZE];
	FILE *yfile_ptr = fopen(y_file, "r");
	while (fgets(linebuf, STRSIZE, yfile_ptr) != NULL) {
		// string line = string(linebuf);
		linebuf[strlen(linebuf)-1] = '\0';
		
// 		size_t ws_index = line.find_last_of("\n");
// 		string in = line.substr(0, ws_index);
		
		y[ysize] = atof(linebuf);
		ysize++;
	}
	// Check feof of file
	if (feof(yfile_ptr)) { // We're good
		fclose(yfile_ptr);
	} else { // It's an error
		char errstring[STRSIZE];
		sprintf(errstring, "Error reading from %s", y_file);
		perror(errstring);
		return 1;
	}
	
	// Initial version of x matrix
	double x_tr[1024][1024];
	
	// DEBUG
	printf("Breakpoint Delta\n");
	
	// Bring predictor file data into memory
	FILE *xfile_ptr = fopen(x_file, "r");
	int row = 0;
	int col = 0;
	while (fgets(linebuf, STRSIZE, xfile_ptr) != NULL) {
		// string line = string(linebuf);
		
		// vector<double> vec;
		
		// DEBUG
		// printf("Breakpoint Upsilon\n");
		
		// DEBUG
		// for (int i = 0; i < 21; i++) {
		char *line = linebuf;
		
		col = 0;
		while (strcmp(line, "") != 0) {
			int i;
			for (i = 0; i < strlen(line); i++) {
				if (line[i] == '\t' || line[i] == '\n') {
					break;
				}
			}
			
			char temp[STRSIZE];
			strncpy(temp, line, i*sizeof(char));
			temp[i] = '\0';
			x_tr[row][col] = atof(temp);
			col++;
			
			if (line[i] == '\n') {
				break;
			} else {
				char temp2[STRSIZE];
				strcpy(temp2, line+i+1);
				line = temp2;
			}
		}
		row++;
			// printf("%s\n", line.c_str()); // DEBUG
// 			size_t ws_index = line.find_first_of("\t\n");
// 			string in = line.substr(0, ws_index);
// 			vec.push_back(atof(in.c_str()));
// 			
// 			// Check if we've reached the end-of-line
// // 			if (ws_index+1 >= line.length()) {
// // 				break;
// // 			} else {
// 				line = line.substr(ws_index+1);
// // 			}
// // 		}
// 		}
		
		// DEBUG
		// printf("Breakpoint Tau\n");
		// exit(0);
		
		// x_tr.push_back(vec);
	}
	// Check feof of file
	if (feof(xfile_ptr)) { // We're good
		fclose(xfile_ptr);
	} else { // It's an error
		char errstring[STRSIZE];
		sprintf(errstring, "Error reading from %s", x_file);
		perror(errstring);
		return 1;
	}
	
	// Need to transpose the x matrix
	for (unsigned int i = 0; i < col; i++) {
		// vector<double> vec;
		for (unsigned int j = 0; j < row; j++) {
			x[i][j] = x_tr[j][i];
		}
		// x.push_back(vec);
	}
	
	// x_tr.clear();
	int xsize = col;
	int xisize = row;
	
	// DEBUG
	printf("Breakpoint Gamma\n");
	printf("ysize: %d\n", ysize);
	// exit(0);
	
	// CUDA stuff
	double *y_cpu = (double *)malloc(ysize*sizeof(double));
	for (unsigned int i = 0; i < ysize; i++) {
		y_cpu[i] = y[i];
	}
	
	// DEBUG
	printf("Breakpoint G1\n");
	
	double *y_gpu;
	cudaMalloc((void**)&y_gpu, ysize*sizeof(double));
	cudaMemcpy(y_gpu, y_cpu, ysize*sizeof(double), cudaMemcpyHostToDevice);
	
	double **x_gpu, **x_gpu_b;
	cudaMalloc((void**)&x_gpu, xsize*sizeof(double *));
	x_gpu_b = (double **)malloc(xsize*sizeof(double *));
	for (int i = 0; i < (int)xsize; i++) {
		cudaMalloc((void**)&x_gpu_b[i], xisize*sizeof(double));
		double *xi = (double *)malloc(xisize*sizeof(double));
		for (int j = 0; j < (int)xisize; j++) {
			xi[j] = x[i][j];
		}
		cudaMemcpy(x_gpu_b[i], xi, xisize*sizeof(double), cudaMemcpyHostToDevice);
	}
	cudaMemcpy(x_gpu, x_gpu_b, xsize*sizeof(double *), cudaMemcpyHostToDevice);
	
	// DEBUG
	printf("Breakpoint G2\n");
	
	int y_size_cpu = (int)ysize;
	int *y_size;
	cudaMalloc((void**)&y_size, sizeof(int));
	cudaMemcpy(y_size, &y_size_cpu, sizeof(int), cudaMemcpyHostToDevice);
	int x_size_cpu = (int)xsize;
	int *x_size;
	cudaMalloc((void**)&x_size, sizeof(int));
	cudaMemcpy(x_size, &x_size_cpu, sizeof(int), cudaMemcpyHostToDevice);
	
	// DEBUG
	printf("Breakpoint G3\n");
	
	int *lm_pivot_gpu;
	cudaMalloc((void**)&lm_pivot_gpu, ysize*sizeof(int));
	bool *good_gpu;
	cudaMalloc((void**)&good_gpu, ysize*sizeof(bool));
	double *mu_gpu;
	cudaMalloc((void**)&mu_gpu, ysize*sizeof(double));
	double *eta_gpu;
	cudaMalloc((void**)&eta_gpu, ysize*sizeof(double));
	double *devold_vec_gpu;
	cudaMalloc((void**)&devold_vec_gpu, ysize*sizeof(double));
	double *coef_gpu;
	cudaMalloc((void**)&coef_gpu, xsize*sizeof(double));
	double *coefold_gpu;
	cudaMalloc((void**)&coefold_gpu, xsize*sizeof(double));
	double *w_gpu;
	cudaMalloc((void**)&w_gpu, ysize*sizeof(double));
	double *varmu_gpu;
	cudaMalloc((void**)&varmu_gpu, ysize*sizeof(double));
	double *mu_eta_gpu;
	cudaMalloc((void**)&mu_eta_gpu, ysize*sizeof(double));
	double *z_gpu;
	cudaMalloc((void**)&z_gpu, ysize*sizeof(double));
	
	double *prefit_y_gpu;
	cudaMalloc((void**)&prefit_y_gpu, ysize*sizeof(double));
	
	// DEBUG
	printf("Breakpoint G4\n");
	
	double **prefit_x_gpu, **prefit_x_gpu_b;
	cudaMalloc((void**)&prefit_x_gpu, xisize*sizeof(double *));
	prefit_x_gpu_b = (double **)malloc(xisize*sizeof(double *));
	for (int i = 0; i < (int)xsize; i++) {
		cudaMalloc((void**)&prefit_x_gpu_b[i], xsize*sizeof(double));
		// cudaMemcpy(prefit_x_gpu_b[i], x[i], x[i].size()*sizeof(double), cudaMemcpyHostToDevice);
	}
	cudaMemcpy(prefit_x_gpu, prefit_x_gpu_b, xisize*sizeof(double *), cudaMemcpyHostToDevice);
	
	double* lm_coefficients_gpu;
	cudaMalloc((void**)&lm_coefficients_gpu, xsize*sizeof(double));
	double* start_gpu;
	cudaMalloc((void**)&start_gpu, xsize*sizeof(double));
	double* residuals_gpu;
	cudaMalloc((void**)&residuals_gpu, ysize*sizeof(double));
	
	double **Rmat_gpu, **Rmat_gpu_b;
	cudaMalloc((void**)&Rmat_gpu, xsize*sizeof(double *));
	Rmat_gpu_b = (double **)malloc(xsize*sizeof(double *));
	for (int i = 0; i < (int)xsize; i++) {
		cudaMalloc((void**)&Rmat_gpu_b[i], xsize*sizeof(double));
	}
	cudaMemcpy(Rmat_gpu, Rmat_gpu_b, xsize*sizeof(double *), cudaMemcpyHostToDevice);
	
	double* wt_gpu;
	cudaMalloc((void**)&wt_gpu, ysize*sizeof(double));
	double* wtdmu_vec_gpu;
	cudaMalloc((void**)&wtdmu_vec_gpu, ysize*sizeof(double));
	double* weights_gpu;
	cudaMalloc((void**)&weights_gpu, ysize*sizeof(double));
	fit* outfit;
	cudaMalloc((void**)&outfit, sizeof(fit));
	
	// DEBUG
	printf("Breakpoint Malta\n");
	
	// Do the actual glm_logit fitting
	// Launch CUDA kernels
	apportionWork<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(y_gpu, x_gpu, y_size, x_size, lm_pivot_gpu, good_gpu, mu_gpu, eta_gpu, devold_vec_gpu, coef_gpu, coefold_gpu, w_gpu, varmu_gpu, mu_eta_gpu, z_gpu, prefit_y_gpu, prefit_x_gpu, lm_coefficients_gpu, start_gpu, residuals_gpu, Rmat_gpu, wt_gpu, wtdmu_vec_gpu, weights_gpu);
	GPUerrchk(cudaPeekAtLastError());
	cudaDeviceSynchronize();
	GPUerrchk(cudaPeekAtLastError());
	// fit outfit = glm_fit(y, x);
	
	// Output the values of "outfit"
// 	double* coefficients = *outfit.getCoefficients();
// 	printf("<-- Coefficients -->\n");
// 	for (unsigned int i = 0; i < x.size(); i++) {
// 		printf("%f", coefficients[i]);
// 		if (i != x.size()-1) {
// 			printf("\t");
// 		} else {
// 			printf("\n\n");
// 		}
// 	}
// 	
// 	double* residuals = outfit.getResiduals();
// 	printf("<-- Residuals -->\n");
// 	for (unsigned int i = 0; i < y.size(); i++) {
// 		printf("%f", residuals[i]);
// 		if (i != y.size()-1) {
// 			printf("\t");
// 		} else {
// 			printf("\n\n");
// 		}
// 	}
// 	
// 	double* fitted_values = outfit.getFittedValues();
// 	printf("<-- Fitted Values -->\n");
// 	for (unsigned int i = 0; i < y.size(); i++) {
// 		printf("%f", fitted_values[i]);
// 		if (i != y.size()-1) {
// 			printf("\t");
// 		} else {
// 			printf("\n\n");
// 		}
// 	}
// 	
// 	double* effects = outfit.getEffects();
// 	printf("<-- Effects -->\n");
// 	for (unsigned int i = 0; i < y.size(); i++) {
// 		printf("%f", effects[i]);
// 		if (i != y.size()-1) {
// 			printf("\t");
// 		} else {
// 			printf("\n\n");
// 		}
// 	}
// 	
// 	double** R = outfit.getR();
// 	printf("<-- R -->\n");
// 	for (unsigned int i = 0; i < x.size(); i++) {
// 		for (unsigned int j = 0; j < x.size(); j++) {
// 			printf("%f", R[i][j]);
// 			if (j != x.size()-1) {
// 				printf("\t");
// 			} else {
// 				printf("\n");
// 			}
// 		}
// 	}
// 	
// 	printf("\n");
// 	
// 	printf("<-- Rank -->\n");
// 	printf("%d\n\n", outfit.getRank());
// 	
// 	double** qr = outfit.getQr();
// 	printf("<-- QR matrix -->\n");
// 	for (int i = 0; i < x.size(); i++) {
// 		for (int j = 0; j < y.size(); i++) {
// 			printf("%f", qr[i][j]);
// 			if (j != y.size()-1) {
// 				printf("\t");
// 			} else {
// 				printf("\n");
// 			}
// 		}
// 	}
// 	
// 	printf("\n");
// 	
// 	vector<double> qraux = outfit.getQraux();
// 	printf("<-- Qraux -->\n");
// 	for (int i = 0; i < x.size(); i++) {
// 		printf("%f", qraux[i]);
// 		if (i != x.size()-1) {
// 			printf("\t");
// 		} else {
// 			printf("\n\n");
// 		}
// 	}
// 	
// 	vector<int> pivot = outfit.getPivot();
// 	printf("<-- Pivot vector -->\n");
// 	for (int i = 0; i < x.size(); i++) {
// 		printf("%d", pivot[i]);
// 		if (i != x.size()-1) {
// 			printf("\t");
// 		} else {
// 			printf("\n\n");
// 		}
// 	}
// 	
// 	printf("<-- Tol -->\n");
// 	printf("%f\n\n", outfit.getTol());
// 	
// 	double* linear_predictors = outfit.getLinearPredictors();
// 	printf("<-- Linear Predictors -->\n");
// 	for (unsigned int i = 0; i < y.size(); i++) {
// 		printf("%f", linear_predictors[i]);
// 		if (i != y.size()-1) {
// 			printf("\t");
// 		} else {
// 			printf("\n\n");
// 		}
// 	}
// 	
// 	printf("<-- Deviance -->\n");
// 	printf("%f\n\n", outfit.getDeviance());
// 	
// 	printf("<-- AIC -->\n");
// 	printf("%f\n\n", outfit.getAIC());
// 	
// 	printf("<-- Null deviance -->\n");
// 	printf("%f\n\n", outfit.getNullDeviance());
// 	
// 	printf("<-- Number of iterations -->\n");
// 	printf("%d\n\n", outfit.getIter());
// 	
// 	double* weights = outfit.getWeights();
// 	printf("<-- Weights -->\n");
// 	for (unsigned int i = 0; i < y.size(); i++) {
// 		printf("%f", weights[i]);
// 		if (i != y.size()-1) {
// 			printf("\t");
// 		} else {
// 			printf("\n\n");
// 		}
// 	}
// 	
// 	double* prior_weights = outfit.getPriorWeights();
// 	printf("<-- Prior Weights -->\n");
// 	for (unsigned int i = 0; i < y.size(); i++) {
// 		printf("%f", prior_weights[i]);
// 		if (i != y.size()-1) {
// 			printf("\t");
// 		} else {
// 			printf("\n\n");
// 		}
// 	}
// 	
// 	printf("<-- Degrees of freedom residual -->\n");
// 	printf("%d\n\n", outfit.getDFResidual());
// 	
// 	printf("<-- Degrees of freedom null -->n");
// 	printf("%d\n\n", outfit.getDFNull());
// 	
// 	printf("<-- Converged -->\n");
// 	string bool_out = (outfit.getConverged()) ? "true" : "false";
// 	printf("%s\n\n", bool_out.c_str());
// 	
// 	printf("<-- Boundary -->\n");
// 	bool_out = (outfit.getBoundary()) ? "true" : "false";
// 	printf("%s\n\n", bool_out.c_str());
	
// 	printf("<-- Theta -->\n");
// 	printf("%f\n\n", outfit.getTheta());
// 	
// 	printf("<-- SE Theta -->\n");
// 	printf("%f\n\n", outfit.getSETheta());
// 	
// 	printf("<-- Two Log Likelihood -->\n");
// 	printf("%f\n\n", outfit.getTwoLogLik());
	
	// delete &outfit;
	
	return 0;
}
