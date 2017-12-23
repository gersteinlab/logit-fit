/* /home/fas/gerstein/ll426/scratch/code/logit-fit/dnrm2.f -- translated by f2c (version 20160102).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#include <stdio.h>
#include "f2c.h"

/* > \brief \b DNRM2 */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/*  Definition: */
/*  =========== */

/*       DOUBLE PRECISION FUNCTION DNRM2(N,X,INCX) */

/*       .. Scalar Arguments .. */
/*       INTEGER INCX,N */
/*       .. */
/*       .. Array Arguments .. */
/*       DOUBLE PRECISION X(*) */
/*       .. */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > DNRM2 returns the euclidean norm of a vector via the function */
/* > name, so that */
/* > */
/* >    DNRM2 := sqrt( x'*x ) */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date November 2011 */

/* > \ingroup double_blas_level1 */

/* > \par Further Details: */
/*  ===================== */
/* > */
/* > \verbatim */
/* > */
/* >  -- This version written on 25-October-1982. */
/* >     Modified on 14-October-1993 to inline the call to DLASSQ. */
/* >     Sven Hammarling, Nag Ltd. */
/* > \endverbatim */
/* > */
/*  ===================================================================== */
__device__ doublereal dnrm2_(integer *n, doublereal *x, integer *incx)
{
    /* System generated locals */
    integer i__1, i__2;
    doublereal ret_val, d__1;

    /* Builtin functions */
    // double sqrt(doublereal);

    /* Local variables */
    integer ix;
    doublereal ssq, norm, scale, absxi;


/*  -- Reference BLAS level1 routine (version 3.4.0) -- */
/*  -- Reference BLAS is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     November 2011 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
    /* Parameter adjustments */
    // --x;
    
    // DEBUG
    printf("Breakpoint A1\n");
    printf("n: %d\n", *n);
    printf("incx: %d\n", *incx);

    /* Function Body */
    if (*n < 1 || *incx < 1) {
	norm = 0.;
    } else if (*n == 1) {
	norm = abs(x[0]);
    } else {
	scale = 0.;
	ssq = 1.;
/*        The following loop is equivalent to this call to the LAPACK */
/*        auxiliary routine: */
/*        CALL DLASSQ( N, X, INCX, SCALE, SSQ ) */

	// DEBUG
  printf("Breakpoint A2\n");

	i__1 = (*n - 1) * *incx + 1;
	i__2 = *incx;
	for (ix = 0; ix < i__1; ix += i__2) {
	
		// DEBUG
    printf("Breakpoint A3: %d\n", ix);
	
	    if (x[ix] != 0.) {
		absxi = (d__1 = x[ix], abs(d__1));
		if (scale < absxi) {
/* Computing 2nd power */
		    d__1 = scale / absxi;
		    ssq = ssq * (d__1 * d__1) + 1.;
		    scale = absxi;
		} else {
/* Computing 2nd power */
		    d__1 = absxi / scale;
		    ssq += d__1 * d__1;
		}
	    }
/* L10: */
	}
	norm = scale * sqrt(ssq);
    }
    
  // DEBUG
  printf("Breakpoint A4\n");

    ret_val = norm;
    return ret_val;

/*     End of DNRM2. */

} /* dnrm2_ */

