#include "glm.logit.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <limits>
#include <utility>
#include <iostream>
#include <sys/stat.h>
#include <errno.h>

using namespace std;

#include "lmfit.cpp"
#include "fit.cpp"

#define STRSIZE 1024
// #define STRSIZE 256

/* This code is a C++ implementation of a negative binomial fitting function */

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

double x_d_omx(double x) {
  if (x < 0 || x > 1) {
		fprintf(stderr, "Value %g out of range (0, 1)", x);
		exit(1);
	}
  return x/(1 - x);
}

double x_d_opx(double x) {return x/(1 + x);}

int imin2(int x, int y)
{
    return (x < y) ? x : y;
}

double R_D_exp (double x, bool log_p) {
	return (log_p	? (x)	: exp(x));
}

// Returns the value ln[gamma(xx)] for xx > 0.
double gammln(double xx) {
	double x,y,tmp,ser;
	static double cof[6]={76.18009172947146,-86.50532032941677,
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

double stirlerr(double n) {

#define S0 0.083333333333333333333       /* 1/12 */
#define S1 0.00277777777777777777778     /* 1/360 */
#define S2 0.00079365079365079365079365  /* 1/1260 */
#define S3 0.000595238095238095238095238 /* 1/1680 */
#define S4 0.0008417508417508417508417508/* 1/1188 */

/*
  error for 0, 0.5, 1.0, 1.5, ..., 14.5, 15.0.
*/
    const static double sferr_halves[31] = {
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
		ans[j] = ((j+n) % 2) ? std::numeric_limits<double>::infinity() : nan("");
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

double bd0(double x, double np) {
    double ej, s, s1, v;
    int j;

    if (isnan(x) || isnan(np) || np == 0.0) {
    	perror("Error: bd0 given an argument that is not a number.\n");
			exit(1);
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
double loglik (int n, double th, 
							 const vector<double> &mu, const vector<double> &y) {
	
	double sum = 0;
	unsigned int n1 = (unsigned int)n;
	for (unsigned int i = 0; i < n1; i++) {
		sum += (gammln(th+y[i]) - gammln(th) - gammln(y[i]+1) + th * 
					 log(th) + y[i] * log(mu[i] + (y[i] == 0 ? 1 : 0)) - (th + y[i]) * 
					 log(th + mu[i]));
	}
	return sum;
}

/* Link functions */
double linkfun (double mu) {
	return log(mu);
}

double linkinv (double eta) {
	double temp = (eta < MTHRESH) ? DOUBLE_EPS : ((eta > THRESH) ? INVEPS : exp(eta));
	return x_d_opx(temp);
	// return (max(exp(eta), DBL_MIN));
}

vector<double> mu_eta (vector<double> &eta) {
	vector<double> result;
	for (unsigned int i = 0; i < eta.size(); i++) {
		double opexp = 1 + exp(eta[i]);
		result.push_back((eta[i] > THRESH || eta[i] < MTHRESH) ? DOUBLE_EPS : exp(eta[i])/(opexp * opexp));
		// result.push_back(max(exp(eta[i]), DBL_MIN));
	}
	return result;
}

// Our own dbinom function
vector<double> dbinom (vector<double> x, vector<double> n, vector<double> p, bool log_p) {
	
	// Error checking
	for (unsigned int i = 0; i < p.size(); i++) {
		if ((p[i] < 0) || (p[i] > 1)) {
			perror("p must be between 0 and 1\n");
			exit(1);
		}
	}
	for (unsigned int i = 0; i < x.size(); i++) {
		if (x[i] < 0) {
			perror("x must be >=0\n");
			exit(1);
		}
	}
	for (unsigned int i = 0; i < x.size(); i++) {
		if (n[i] < x[i]) {
			perror("x must be <= than the binomial denominator\n");
			exit(1);
		}
	}
	vector<double> q;
	for (unsigned int i = 0; i < p.size(); i++) {
		q[i] = 1 - p[i];
	}
	
	vector<double> result;
	
	for (unsigned int i = 0; i < x.size(); i++) {
	
		double lf, lc;
	
		if (p[i] == 0) {
			result.push_back(((x[i] == 0) ? (log_p ? 0. : 1.) : (log_p ? -DBL_MAX : 0.)));
			continue;
		}
		if (q[i] == 0) {
			result.push_back(((x[i] == n[i]) ? (log_p ? 0. : 1.) : (log_p ? -DBL_MAX : 0.)));
			continue;
		}
	
		if (x[i] == 0) {
			if (n[i] == 0) {
				result.push_back((log_p ? 0. : 1.));
				continue;
			}
			lc = (p[i] < 0.1) ? -bd0(n[i],n[i]*q[i]) - n[i]*p[i] : n[i]*log(q[i]);
			result.push_back(( (log_p	?  (lc)	 : exp(lc)) ));
			continue;
		}
		if (x[i] == n[i]) {
			lc = (q[i] < 0.1) ? -bd0(n[i],n[i]*p[i]) - n[i]*q[i] : n[i]*log(p[i]);
			result.push_back(( (log_p	?  (lc)	 : exp(lc)) ));
			continue;
		}
		if (x[i] < 0 || x[i] > n[i]) {
			result.push_back(( (log_p ? -DBL_MAX : 0.) ));
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

		result.push_back(R_D_exp((lc - 0.5*lf), log_p));
	}
	return result;
}

/* Logistic binomial helper functions */

// Calculate variance from mu
vector<double> logit_variance (vector<double> &mu) {
	vector<double> result;
	for (unsigned int i = 0; i < mu.size(); i++) {
		result.push_back(mu[i]*(1-mu[i]));
	}
	return result;
}

// Returns TRUE if mu vector is within acceptable variance
bool logit_validmu (vector<double> &mu) {
	for (unsigned int i = 0; i < mu.size(); i++) {
		if (isinf(mu[i]) || mu[i] <= 0.0 || mu[i] >= 1.0) {
			return false;
		}
	}
	return true;
}

double y_log_y (double y, double mu) {
	return (y != 0.) ? (y * log(y/mu)) : 0;
}

// Return deviance residuals
vector<double> logit_dev_residuals (vector<double> &y, vector<double> &mu) {
	vector<double> result;
	
	// int i;
	unsigned int n = y.size();
	unsigned int lmu = mu.size();
	// int nprot = 1;
	
	for (unsigned int i = 0; i < n; i++) {
		result.push_back(y[i]);
	}
	
	if (lmu != n && lmu != 1) {
		fprintf(stderr, "Error: Argument mu must be a numeric vector of length 1 or length %d\n", n);
		exit(1);
	}
	
	for (int i = 0; i < (int)n; i++) {
		result[i] = 2 * (y_log_y(y[i], mu[i]) + y_log_y(1 - y[i], 1 - mu[i]));
	}
	return result;
}

// Calculate the AIC
double logit_aic (vector<double> &y, vector<double> &mu) {
	vector<double> m;
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
	for (unsigned int i = 0; i < y.size(); i++) {
		m.push_back(1);
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
	
	vector<double> m_prod_y;
	vector<double> m_rounded;
	
	for (unsigned int i = 0; i < y.size(); i++) {
		m_prod_y.push_back(round(m[i]*y[i]));
	}
	
	for (unsigned int i = 0; i < y.size(); i++) {
		m_rounded.push_back(round(m[i]));
	}
	
	vector<double> db = dbinom(m_prod_y, m_rounded, mu, true);
	double sum = 0;
	for (unsigned int i = 0; i < n.size(); i++) {
		if (m[i] > 0) {
			sum += 1/m[i]*db[i];
		}
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

lmfit Cdqrls(vector<vector<double> > &x, vector<double> &y, double tol, bool chk) {
	
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
	
	n = (int)x[0].size();
	p = (int)x.size();
	
	// Sanity checks
	if (n != (int)y.size()) {
		printf("Error: Dimensions of \'x\' (%d,%d) and \'y\' (%d) do not match", n, p, (int)y.size());
		exit(1);
	}
	
	for (unsigned int i = 0; i < x.size(); i++) {
		for (unsigned int j = 0; j < y.size(); j++) {
			if (isinf(x[i][j])) {
				printf("Error: NA/NaN/Inf in \'x\' matrix: (%d,%d)\n", i, j);
				exit(1);
			}
		}
	}
	
	for (unsigned int i = 0; i < x.size(); i++) {
		if (isinf(y[i])) {
			printf("Error: NA/NaN/Inf in \'y\' vector: element %d\n", i);
			exit(1);
		}
	}
	
	// Encoding in C types
	// qr = x;
	// Use column-major order for Fortran
	int flat_size = n*p;
	qr = (double *)malloc(flat_size*sizeof(double));
	for (int i = 0; i < (int)x.size(); i++) {
		for (int j = 0; j < (int)x[i].size(); j++) {
			qr[i*n+j] = x[i][j];
		}
	}
	
	// Turn y into a matrix for use in Fortran
	// y_for = y;
	y_for = (double *)malloc(n*sizeof(double));
	for (unsigned int i = 0; i < y.size(); i++) {
		y_for[i] = y[i];
	}
	
	// Coefficients mapping
	coefficients = (double *)malloc(p*sizeof(double));
	for (int i = 0; i < p; i++) {
		coefficients[i] = 0.0;
	}

	// residuals = y;
	residuals = (double *)malloc((int)y.size()*sizeof(double));
	for (unsigned int i = 0; i < y.size(); i++) {
		residuals[i] = y[i];
	}
	
	// effects = y;
	effects = (double *)malloc((int)y.size()*sizeof(double));
	for (unsigned int i = 0; i < y.size(); i++) {
		effects[i] = y[i];
	}
	
	pivot = (int *)malloc(p*sizeof(int));
	for (int i = 0; i < p; i++) {
		pivot[i] = i+1;
	}
	
	qraux = (double *)malloc(p*sizeof(double));
	work = (double *)malloc(2*p*sizeof(double));
	
	// DEBUG
	// printf("Breakpoint Yocto\n");
	// exit(0);
	
	// Call dqrls
	dqrls_(qr, &n, &p, y_for, &ny, &rtol, coefficients, residuals, effects, &rank, 
				pivot, qraux, work);
				
	for	(int i = 0; i < p; i++) {
		if (pivot[i] != i+1) {
			pivoted = 1;
			break;
		}
	}
	
	// DEBUG
// 	printf("Breakpoint Zeta\n");
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
	
	// Re-encode in C++ vectors
	vector<vector<double> > qr_vec;
	for (int i = 0; i < (int)x.size(); i++) {
		vector<double> temp;
		for (int j = 0; j < (int)x[i].size(); j++) {
			temp.push_back(qr[i*n+j]);
		}
		qr_vec.push_back(temp);
	}
	
	// DEBUG
	// printf("Breakpoint Molto\n");
	// exit(0);
	
	vector<double> coefficients_vec;
	for (int i = 0; i < p; i++) {
		coefficients_vec.push_back(coefficients[i]);
	}
	
	vector<double> residuals_vec;
	for (int i = 0; i < (int)y.size(); i++) {
		residuals_vec.push_back(residuals[i]);
	}
	
	vector<double> effects_vec;
	for (int i = 0; i < (int)y.size(); i++) {
		effects_vec.push_back(effects[i]);
	}
	
	vector<int> pivot_vec;
	for (int i = 0; i < p; i++) {
		pivot_vec.push_back(pivot[i]);
	}
	
	vector<double> qraux_vec;
	for (int i = 0; i < p; i++) {
		qraux_vec.push_back(qraux[i]);
	}
	
	lmfit lm1 (qr_vec, coefficients_vec, residuals_vec, effects_vec, rank, pivot_vec, qraux_vec, tol, pivoted);
	
	// DEBUG
	// printf("Breakpoint Eta\n");
	// exit(0);
	
	return lm1;
}

/*
 * Theta_ml's score function
 */
double theta_score (double n, double th, vector<double> &mu, vector<double> &y) {
	double sum = 0.0;
	for (unsigned int i = 0; i < y.size(); i++) {
		sum += digamma(th + y[i]) - digamma(th) + log(th) + 1 - log(th + mu[i]) - 
					 (y[i] + th)/(mu[i] + th);
	}
	return sum;
}

/*
 * Theta_ml's info function
 */
double theta_info (double n, double th, vector<double> &mu, vector<double> &y) {
	double sum = 0.0;
	for (unsigned int i = 0; i < y.size(); i++) {
		sum += -trigamma(th + y[i]) + trigamma(th) - 1/th + 2/(mu[i] + th) - 
					 (y[i] + th)/pow((mu[i] + th),2.0);
	}
	return sum;
}

/*
 * This is the theta ML estimator code
 * Assume equal weights (=1) for all predictors
 */
pair <double,double> theta_ml (vector<double> &y, vector<double> &mu, double n, int limit) {
	
	// Control variables
	double epsilon = 1e-8;
	
	double denom = 0.0;
	for (unsigned int i = 0; i < y.size(); i++) {
		denom += pow((y[i]/mu[i] - 1.0), 2.0);
	}
	double t0 = n/denom;
	int it = 0;
	double del = 1.0;
	double i;
	for (; (it < limit) && fabs(del) > epsilon; it++) {
		t0 = fabs(t0);
		i = theta_info(n, t0, mu, y);
		del = theta_score(n, t0, mu, y)/i;
		t0 += del;
	}
	if (t0 < 0.0) {
		t0 = 0.0;
		printf("Warning: theta estimate truncated at zero\n");
	}
	if (it == limit) {
		printf("Warning: theta estimator iteration limit reached\n");
	}
	double se = sqrt(1/i);
	pair <double,double> retval (t0,se);
	return retval;
	// return t0;
}
	
/* 
 * This is really a pared down glm_fit that only implements the portion needed
 * for the logistic regression fit
 */
fit glm_fit (vector<double> &y, vector<vector<double> > &x) {

	// DEBUG
	printf("Breakpoint 1\n");

	// Control variables
	double epsilon = 1e-8;
	int maxit = 25;
	
	// Setup variables
	bool conv = false;
	int nobs = (int)y.size();
	int nvars = (int)x.size();
	
	vector<int> lm_pivot;
	vector<bool> good;
	
	// Eta always valid
	
	// Initialize the mu's
	vector<double> mu;
	vector<double> n;
	for (unsigned int i = 0; i < y.size(); i++) {
		if (y[i] < 0) {
			printf("Error: negative values not allowed for the negative binomial model. Exiting.\n");
			exit(1);
		}
		n.push_back(1.0);
		mu.push_back((y[i] + 0.5)/2);
	}
	
	// Initialize the eta's
	vector<double> eta;
// 	if (etastart.size() == 0) {
	for (unsigned int i = 0; i < mu.size(); i++) {
		eta.push_back(log(x_d_omx(mu[i])));
	}
// 	} else {
// 		eta = etastart;
// 	}
	
	for (unsigned int i = 0; i < eta.size(); i++) {
		double tmp = (eta[i] < MTHRESH) ? DOUBLE_EPS : ((eta[i] > THRESH) ? INVEPS : exp(eta[i]));
		mu[i] = x_d_opx(tmp);
	}
		
	if (!logit_validmu(mu)) {
		printf("Invalid starting mu detected. Exiting.\n");
		exit(1);
	}
	
	// calculate initial deviance and coefficient
	vector<double> devold_vec = logit_dev_residuals(y, mu);
	double devold = 0.0;
	for (unsigned int i = 0; i < devold_vec.size(); i++) {
		devold += devold_vec[i];
	}
	
	bool boundary = false;
	
	// Coefficient vector declarations
	vector<double> coef;
	vector<double> coefold;
	
	lmfit lm;
	vector<double> w;
	double dev;
	
	// DEBUG
	// printf("Breakpoint Upsilon\n");
	// exit(0);
	
	// DEBUG
	printf("Breakpoint 2\n");
	
	// The iteratively reweighting L.S. iteration
	int iter = 0;
	for (; iter < maxit; iter++) {
		
		// DEBUG
		// printf("Iter: %d\n", iter);
		
		vector<double> varmu = logit_variance(mu);
		for (unsigned int j = 0; j < varmu.size(); j++) {
			if (isnan(varmu[j])) {
				printf("NaNs in the mu variance: cannot proceed. Exiting.\n");
				exit(1);
			} else if (varmu[j] == 0) {
				printf("Zeroes in the mu variance: cannot proceed. Exiting.\n");
				exit(1);
			}
		}
		vector<double> mu_eta_val = mu_eta(eta);
		for (unsigned int j = 0; j < mu_eta_val.size(); j++) {
			if (isnan(mu_eta_val[j])) {
				printf("NaNs in the d(mu)/d(eta)\n");
				exit(1);
			}
		}
		
		// Save the good ones (i.e. the ones where mu_eta_val is nonzero)
		good.clear();
		unsigned int num_false = 0;
		for (unsigned int i = 0; i < mu_eta_val.size(); i++) {
			if (mu_eta_val[i] != 0) {
				good.push_back(true);
			} else {
				good.push_back(false);
				num_false++;
			}
		}
		
		if (num_false == good.size()) { // All not good
			conv = false;
			fprintf(stderr, "No observations informative at iteration %d\n", iter);
			break;
		}
		
		vector<double> z;
		for (unsigned int j = 0; j < y.size(); j++) {
			if (good[j] == true) {
				double this_val = eta[j] + (y[j] - mu[j])/mu_eta_val[j];
				z.push_back(this_val);
			}
		}
		// vector<double> w;
		for (unsigned int j = 0; j < y.size(); j++) {
			if (good[j] == true) {
				double this_val = sqrt(pow(mu_eta_val[j],2.0))/varmu[j];
				w.push_back(this_val);
			}
		}
		
		// Set up dot products for Cdqrls
		vector<vector<double> > prefit_x;
		vector<double> prefit_y;
		for (unsigned int j = 0; j < x.size(); j++) {
			vector<double> temp;
			for (unsigned int k = 0; k < y.size(); k++) {
				if (good[k] == true) {
					temp.push_back(x[j][k] * w[k]);
				}
			}
			prefit_x.push_back(temp);
		}
		
		for (unsigned int j = 0; j < z.size(); j++) {
			prefit_y.push_back(z[j] * w[j]);
		}
		
		// if (iter > 0) {
		// delete &lm;
		// }
		
		// DEBUG
		// printf("Breakpoint Upsilon 2\n");
		// exit(0);
		
		lm = Cdqrls(prefit_x, prefit_y, min(1e-7, epsilon/1000), false);
		
		// DEBUG
		// printf("Breakpoint Tau 2\n");
		// exit(0);
		
		vector<double> lm_coefficients = lm.getCoefficients();
		for (int j = 0; j < nvars; j++) {
			if (isinf(lm_coefficients[j])) {
				conv = false;
				printf("Warning: Non-finite coefficients at iteration %d\n", iter);
				break;
			}
		}
		
		// Stop if not enough parameters
		if (nobs < lm.getRank()) {
			printf("Error: X matrix has rank %d, but only %d observation(s).\n", lm.getRank(), nobs);
			exit(1);
		}
		
		// calculate updated values of eta and mu with the new coef
		lm_pivot = lm.getPivot();
		vector<double> start; // (nvars,0.0);
		for (int j = 0; j < nvars; j++) {
			// This code not necessary: if (lm_pivot[j]) {
				start.push_back(lm_coefficients[lm_pivot[j]-1]);
			// }
		}
		
		// New eta needs a matrix calculation
		vector<double> new_eta;
		for (int j = 0; j < nobs; j++) {
			double new_eta_one = 0.0;
			for (int k = 0; k < nvars; k++) {
				new_eta_one += x[k][j] * start[k];
			}
			new_eta.push_back(new_eta_one);
		}
		eta = new_eta;
		
		vector<double> new_mu;
		for (unsigned int j = 0; j < eta.size(); j++) {
			new_mu.push_back(linkinv(eta[j]));
		}
		mu = new_mu;
		
		vector<double> dev_vec = logit_dev_residuals(y, mu);
		dev = 0.0;
		for (unsigned int j = 0; j < dev_vec.size(); j++) {
			dev += dev_vec[j];
		}
		
		// check for divergence
		boundary = false;
		
		if (isinf(dev)) {
			if (coefold.size() == 0) {
				printf("Error: divergence in function fitting. No valid set of ");
				printf("coefficients has been found. Exiting.\n");
				exit(1);
			}
			int ii = 1;
			while (isinf(dev)) {
				if (ii > maxit) {
					printf("Error: Inner loop 1; cannot correct step size\n");
					exit(1);
				}
				ii++;
				// Element-wise addition
				for (unsigned int j = 0; j < start.size(); j++) {
					start[j] = (start[j] + coefold[j])/2;
				}
					
				// New eta needs a matrix calculation
				new_eta.clear();
				for (int j = 0; j < nobs; j++) {
					double new_eta_one = 0.0;
					for (int k = 0; k < nvars; k++) {
						new_eta_one += x[k][j] * start[k];
					}
					new_eta.push_back(new_eta_one);
				}
				eta = new_eta;
				
				new_mu.clear();
				for (unsigned int j = 0; j < eta.size(); j++) {
					new_mu.push_back(linkinv(eta[j]));
				}
				mu = new_mu;
				
				vector<double> dev_vec = logit_dev_residuals(y, mu);
				dev = 0.0;
				for (unsigned int j = 0; j < dev_vec.size(); j++) {
					dev += dev_vec[j];
				}
			}
			boundary = true;
		}
		
		// check for fitted values outside the domain
		if (!logit_validmu(mu)) {
			if (coefold.size() == 0) {
				printf("Error: Fitted mu is outside the valid domain. No valid set of ");
				printf("coefficients has been found. Exiting.\n");
				exit(1);
			}
			int ii = 1;
			while (!logit_validmu(mu)) {
				if (ii > maxit) {
					printf("Error: Inner loop 2; cannot correct step size\n");
					exit(1);
				}
				ii++;
				// Element-wise addition
				for (unsigned int j = 0; j < start.size(); j++) {
					start[j] = (start[j] + coefold[j])/2;
				}
				
				// New eta needs a matrix calculation
				new_eta.clear();
				for (int j = 0; j < nobs; j++) {
					double new_eta_one = 0.0;
					for (int k = 0; k < nvars; k++) {
						new_eta_one += x[k][j] * start[k];
					}
					new_eta.push_back(new_eta_one);
				}
				eta = new_eta;
				
				new_mu.clear();
				for (unsigned int j = 0; j < eta.size(); j++) {
					new_mu.push_back(linkinv(eta[j]));
				}
				mu = new_mu;
			}
			boundary = true;
			vector<double> dev_vec = logit_dev_residuals(y, mu);
			dev = 0.0;
			for (unsigned int j = 0; j < dev_vec.size(); j++) {
				dev += dev_vec[j];
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
	printf("Breakpoint 3\n");
	
	if (!conv) {
		fprintf(stderr, "Warning: fitting algorithm did not converge\n");
	}
	if (boundary) {
		fprintf(stderr, "Warning: fitting algorithm stopped at boundary value\n");
	}
	// Binomial family check
	double eps = 10*epsilon;
	for (unsigned int i = 0; i < mu.size(); i++) {
		if ((mu[i] > 1-eps) || (mu[i] < eps)) {
			fprintf(stderr, "Warning: Fitted probabilities numerically 0 or 1 occurred\n");
			break;
		}
	}
	
	// DEBUG
	printf("Breakpoint 3a\n");
	
	// If X matrix was not full rank then columns were pivoted,
  // hence we need to re-label the names ...
  lm_pivot = lm.getPivot();
  if (lm.getRank() < nvars) {
  	for (unsigned int i = lm.getRank(); i < (unsigned int)nvars; i++) {
  		coef[lm_pivot[i]-1] = 0;
  	}
  }
  
  // DEBUG
	printf("Breakpoint 3b\n");
  
  // update by accurate calculation, including 0-weight cases.
  vector<double> mu_eta_vec = mu_eta(eta);
  vector<double> residuals;
  for (unsigned int i = 0; i < y.size(); i++) {
  	double temp = (y[i] - mu[i])/mu_eta_vec[i];
  	residuals.push_back(temp);
  }
  vector<vector<double> > lm_qr = lm.getQr();
  
  // DEBUG
	printf("Breakpoint 3c\n");
  
  int sum_good = 0;
  for (unsigned int i = 0; i < good.size(); i++) {
  	if (good[i] == true) {
  		sum_good++;
  	}
  }
  int nr = min(sum_good, nvars);
  vector<vector<double> > Rmat;
  if (nr < nvars) {
  	for (unsigned int i = 0; i < (unsigned int)nvars; i++) {
  		vector<double> temp;
  		if (i < (unsigned int)nr) {
  			for (unsigned int j = 0; j < (unsigned int)nvars; j++) {
  				if (i > j) {
						temp.push_back(0.0);
					} else {
						temp.push_back(lm_qr[i][j]);
					}
				}
  		} else {
				for (unsigned int j = 0; j < (unsigned int)nvars; j++) {
					if (i == j) {
						temp.push_back(1.0);
					} else {
						temp.push_back(0.0);
					}
				}
			}
			Rmat.push_back(temp);
		}
  	
  } else {
  
		for (unsigned int i = 0; i < (unsigned int)nvars; i++) {
			vector<double> temp;
			for (unsigned int j = 0; j < (unsigned int)nvars; j++) {
				if (i > j) {
					temp.push_back(0.0);
				} else {
					temp.push_back(lm_qr[i][j]);
				}
			}
			Rmat.push_back(temp);
		}
	}
	
	// DEBUG
	printf("Breakpoint 3d\n");
	
  vector<double> wt;
  for (unsigned int i = 0; i < y.size(); i++) {
  	if (good[i]) {
  		wt.push_back(pow(w[i],2.0));
  	} else {
  		wt.push_back(0.0);
  	}
  }
  // calculate null deviance -- corrected in glm() if offset and intercept
  double wtdmu = 0.0;
  for (unsigned int i = 0; i < y.size(); i++) {
  	wtdmu += y[i];
  }
  wtdmu = wtdmu/(double)nobs;
  
  // DEBUG
	printf("Breakpoint 3e\n");
  
  // Produce a vector version of wtdmu
  vector<double> wtdmu_vec;
  for (unsigned int i = 0; i < y.size(); i++) {
  	wtdmu_vec.push_back(wtdmu);
  }
  
  // DEBUG
	printf("Breakpoint 3f\n");
  
  vector<double> nulldev_vec = logit_dev_residuals(y, wtdmu_vec);
  double nulldev = 0.0;
  for (unsigned int i = 0; i < nulldev_vec.size(); i++) {
  	nulldev += nulldev_vec[i];
  }
  
  // calculate df
  int n_ok = nobs;
  int nulldf = n_ok - 1;
  int rank = lm.getRank();
  int resdf = n_ok - rank;
  
  // DEBUG
	printf("Breakpoint 3g\n");
  
  // calculate AIC
  double aic_model = logit_aic(y, mu) + 2*rank;
  
  // DEBUG
	printf("Breakpoint 3h\n");
  
  vector<double> effects = lm.getEffects();
  
  vector<double> weights;
  for (int i = 0; i < (int)y.size(); i++) {
  	weights.push_back(1.0);
  }
  
  vector<vector<double> > qr = lm.getQr();
	vector<double> qraux = lm.getQraux();
	vector<int> pivot = lm.getPivot();
	
	// DEBUG
	printf("Breakpoint 4\n");
  
  fit this_fit(coef, residuals, mu, effects, Rmat, rank, qr, qraux, pivot, lm.getTol(),
  						 eta, dev, aic_model, nulldev, iter, wt, weights, resdf, nulldf, y, 
  						 conv, boundary);
  // delete &lm;
  return this_fit;
}

/* The actual function. Can be called from main, or from other packages */
// fit glm_logit (vector<double> &y, vector<vector<double> > &x) {
// 	// Assume link function is log
// 	// Exclude model frame building since we will be conducting those operations manually
// 	// negbin_family(y, x, mu, init_theta);
// 	
// 	// Control variables
// 	double epsilon = 1e-8;
// 	int maxit = 25;
// 	
// 	// Initial fit
// 	vector<double> vec0;
// 	
// 	// DEBUG
// 	// printf("Breakpoint Upsilon\n");
// 	// exit(0);
// 	
// 	fit this_fit = glm_fit(y, x, init_theta, vec0);
// 	
// 	// DEBUG
// 	// printf("Breakpoint Tau\n");
// 	// exit(0);
// 	
// 	vector<double> mu = this_fit.getFittedValues();
// 	pair <double,double> theta_ret = theta_ml(y, mu, (int)y.size(), 25);
// 	double th = theta_ret.first;
// 	double theta = th;
// 	int n = (int)y.size();
// 	
// 	int iter = 0;
// 	double d1 = sqrt(2 * max(1, this_fit.getDFResidual()));
// 	double del = 1.0;
// 	double d2 = 1.0;
// 	// g refers to the link function
// 	double Lm = loglik(n, th, mu, y);
// 	double Lm0 = Lm + 2 * d1;
// 	fit cur_fit;
// 	
// 	// DEBUG
// 	// printf("Breakpoint Alpha\n");
// 	// exit(0);
// 	
// 	while ((iter < maxit) && (abs(Lm0 - Lm)/d1 + abs(del)/d2) > epsilon) {
// 		vector<double> eta;
// 		for (unsigned int i = 0; i < mu.size(); i++) {
// 			eta.push_back(linkfun(mu[i]));
// 		}
// 		cur_fit = glm_fit(y, x, theta, eta);
// 		double t0 = th;
// 		theta_ret = theta_ml(y, mu, (int)y.size(), 25);
// 		th = theta_ret.first;
// 		theta = th;
// 		mu = this_fit.getFittedValues();
// 		del = t0 - th;
// 		Lm0 = Lm;
// 		Lm = loglik(n, th, mu, y);
// 		
// 		iter++;
// 	}
// 	
// 	// DEBUG
// 	// printf("Breakpoint Beta\n");
// 	
// 	if (iter > maxit) {
// 		printf("Warning: alternation limit reached\n");
// 	}
// 	cur_fit.setTheta(th);
// 	cur_fit.setSETheta(theta_ret.second);
// 	cur_fit.setTwoLogLik(2 * Lm);
// 	cur_fit.setAIC((-1)*cur_fit.getTwoLogLik() + 2 * (double)cur_fit.getRank() + 2);
// 	return cur_fit;
// }

/* Main function */
int main (int argc, char* argv[]) {

	// The response vector (y) file
	// Each value is listed one per row
	string y_file;
	
 	// The predictor matrix (x) file
 	// Rows are observations, columns are predictor variables
 	// Assumes tab-delimited values
 	string x_file;
 	
 	// The initial theta to use in fitting
 	// double init_theta;
 	
 	// DEBUG
 	// printf("Breakpoint Pre-Sigma\n");
 	
 	// Argument checking
 	if (argc != 3) {
 		printf("Incorrect number of arguments: found %d but expected 2. Exiting.\n", argc-1);
 		printf("Usage: glm.logit [response file] [predictor file]\n");
 		return 1;
 	} else {
 		y_file = string(argv[1]);
 		x_file = string(argv[2]);
 		// init_theta = atof(argv[3]);
 	}
 	
 	// DEBUG
 	// printf("Breakpoint Sigma\n");
 	
 	// Data structures for imported data
 	vector<double> y;
	vector<vector<double> > x;
	
	// Verify files, and import data to memory
	struct stat ybuf;
	if (stat(y_file.c_str(), &ybuf)) { // Report the error and exit
		printf("Error trying to stat %s: %s\n", y_file.c_str(), strerror(errno));
		return 1;
	}
	// Check that the file is not empty
	if (ybuf.st_size == 0) {
		printf("Error: Response file cannot be empty. Exiting.\n");
		return 1;
	}
	
	struct stat xbuf;
	if (stat(x_file.c_str(), &xbuf)) { // Report the error and exit
		printf("Error trying to stat %s: %s\n", x_file.c_str(), strerror(errno));
		return 1;
	}
	// Check that the file is not empty
	if (xbuf.st_size == 0) {
		printf("Error: Predictor file cannot be empty. Exiting.\n");
		return 1;
	}
	
	// Bring response file data into memory
	char linebuf[STRSIZE];
	FILE *yfile_ptr = fopen(y_file.c_str(), "r");
	while (fgets(linebuf, STRSIZE, yfile_ptr) != NULL) {
		string line = string(linebuf);
		
		size_t ws_index = line.find_last_of("\n");
		string in = line.substr(0, ws_index);
		
		y.push_back(atof(in.c_str()));
	}
	// Check feof of file
	if (feof(yfile_ptr)) { // We're good
		fclose(yfile_ptr);
	} else { // It's an error
		char errstring[STRSIZE];
		sprintf(errstring, "Error reading from %s", y_file.c_str());
		perror(errstring);
		return 1;
	}
	
	// Initial version of x matrix
	vector<vector<double> > x_tr;
	
	// DEBUG
	// printf("Breakpoint Delta\n");
	
	// Bring predictor file data into memory
	FILE *xfile_ptr = fopen(x_file.c_str(), "r");
	while (fgets(linebuf, STRSIZE, xfile_ptr) != NULL) {
		string line = string(linebuf);
		
		vector<double> vec;
		
		// DEBUG
		// printf("Breakpoint Upsilon\n");
		
		// DEBUG
		// for (int i = 0; i < 21; i++) {
		
		while (line != "") {
			// printf("%s\n", line.c_str()); // DEBUG
			size_t ws_index = line.find_first_of("\t\n");
			string in = line.substr(0, ws_index);
			vec.push_back(atof(in.c_str()));
			
			// Check if we've reached the end-of-line
// 			if (ws_index+1 >= line.length()) {
				break;
// 			} else {
				line = line.substr(ws_index+1);
// 			}
// 		}
		}
		
		// DEBUG
		// printf("Breakpoint Tau\n");
		// exit(0);
		
		x_tr.push_back(vec);
	}
	// Check feof of file
	if (feof(xfile_ptr)) { // We're good
		fclose(xfile_ptr);
	} else { // It's an error
		char errstring[STRSIZE];
		sprintf(errstring, "Error reading from %s", x_file.c_str());
		perror(errstring);
		return 1;
	}
	
	// Need to transpose the x matrix
	for (unsigned int i = 0; i < x_tr[0].size(); i++) {
		vector<double> vec;
		for (unsigned int j = 0; j < x_tr.size(); j++) {
			vec.push_back(x_tr[j][i]);
		}
		x.push_back(vec);
	}
	
	x_tr.clear();
	
	// DEBUG
	printf("Breakpoint Gamma\n");
	// exit(0);
	
	// Do the actual glm_logit fitting
	fit outfit = glm_fit(y, x);
	
	// Output the values of "outfit"
	vector<double> coefficients = outfit.getCoefficients();
	printf("<-- Coefficients -->\n");
	for (unsigned int i = 0; i < coefficients.size(); i++) {
		printf("%f", coefficients[i]);
		if (i != coefficients.size()-1) {
			printf("\t");
		} else {
			printf("\n\n");
		}
	}
	
	vector<double> residuals = outfit.getResiduals();
	printf("<-- Residuals -->\n");
	for (unsigned int i = 0; i < residuals.size(); i++) {
		printf("%f", residuals[i]);
		if (i != residuals.size()-1) {
			printf("\t");
		} else {
			printf("\n\n");
		}
	}
	
	vector<double> fitted_values = outfit.getFittedValues();
	printf("<-- Fitted Values -->\n");
	for (unsigned int i = 0; i < fitted_values.size(); i++) {
		printf("%f", fitted_values[i]);
		if (i != fitted_values.size()-1) {
			printf("\t");
		} else {
			printf("\n\n");
		}
	}
	
	vector<double> effects = outfit.getEffects();
	printf("<-- Effects -->\n");
	for (unsigned int i = 0; i < effects.size(); i++) {
		printf("%f", effects[i]);
		if (i != effects.size()-1) {
			printf("\t");
		} else {
			printf("\n\n");
		}
	}
	
	vector<vector<double> > R = outfit.getR();
	printf("<-- R -->\n");
	for (unsigned int i = 0; i < R.size(); i++) {
		for (unsigned int j = 0; j < R[i].size(); j++) {
			printf("%f", R[i][j]);
			if (j != R[i].size()-1) {
				printf("\t");
			} else {
				printf("\n");
			}
		}
	}
	
	printf("\n");
	
	printf("<-- Rank -->\n");
	printf("%d\n\n", outfit.getRank());
	
	vector<vector<double> > qr = outfit.getQr();
	printf("<-- QR matrix -->\n");
	for (int i = 0; i < (int)x.size(); i++) {
		for (int j = 0; j < (int)x[i].size(); i++) {
			printf("%f", qr[i][j]);
			if (j != (int)x[i].size()-1) {
				printf("\t");
			} else {
				printf("\n");
			}
		}
	}
	
	printf("\n");
	
	vector<double> qraux = outfit.getQraux();
	printf("<-- Qraux -->\n");
	for (int i = 0; i < (int)x.size(); i++) {
		printf("%f", qraux[i]);
		if (i != (int)x.size()-1) {
			printf("\t");
		} else {
			printf("\n\n");
		}
	}
	
	vector<int> pivot = outfit.getPivot();
	printf("<-- Pivot vector -->\n");
	for (int i = 0; i < (int)x.size(); i++) {
		printf("%d", pivot[i]);
		if (i != (int)x.size()-1) {
			printf("\t");
		} else {
			printf("\n\n");
		}
	}
	
	printf("<-- Tol -->\n");
	printf("%f\n\n", outfit.getTol());
	
	vector<double> linear_predictors = outfit.getLinearPredictors();
	printf("<-- Linear Predictors -->\n");
	for (unsigned int i = 0; i < linear_predictors.size(); i++) {
		printf("%f", linear_predictors[i]);
		if (i != linear_predictors.size()-1) {
			printf("\t");
		} else {
			printf("\n\n");
		}
	}
	
	printf("<-- Deviance -->\n");
	printf("%f\n\n", outfit.getDeviance());
	
	printf("<-- AIC -->\n");
	printf("%f\n\n", outfit.getAIC());
	
	printf("<-- Null deviance -->\n");
	printf("%f\n\n", outfit.getNullDeviance());
	
	printf("<-- Number of iterations -->\n");
	printf("%d\n\n", outfit.getIter());
	
	vector<double> weights = outfit.getWeights();
	printf("<-- Weights -->\n");
	for (unsigned int i = 0; i < weights.size(); i++) {
		printf("%f", weights[i]);
		if (i != weights.size()-1) {
			printf("\t");
		} else {
			printf("\n\n");
		}
	}
	
	vector<double> prior_weights = outfit.getPriorWeights();
	printf("<-- Prior Weights -->\n");
	for (unsigned int i = 0; i < prior_weights.size(); i++) {
		printf("%f", prior_weights[i]);
		if (i != prior_weights.size()-1) {
			printf("\t");
		} else {
			printf("\n\n");
		}
	}
	
	printf("<-- Degrees of freedom residual -->\n");
	printf("%d\n\n", outfit.getDFResidual());
	
	printf("<-- Degrees of freedom null -->n");
	printf("%d\n\n", outfit.getDFNull());
	
	printf("<-- Converged -->\n");
	string bool_out = (outfit.getConverged()) ? "true" : "false";
	printf("%s\n\n", bool_out.c_str());
	
	printf("<-- Boundary -->\n");
	bool_out = (outfit.getBoundary()) ? "true" : "false";
	printf("%s\n\n", bool_out.c_str());
	
	printf("<-- Theta -->\n");
	printf("%f\n\n", outfit.getTheta());
	
	printf("<-- SE Theta -->\n");
	printf("%f\n\n", outfit.getSETheta());
	
	printf("<-- Two Log Likelihood -->\n");
	printf("%f\n\n", outfit.getTwoLogLik());
	
	// delete &outfit;
	
	return 0;
}
