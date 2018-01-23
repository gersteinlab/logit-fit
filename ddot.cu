/* /home/fas/gerstein/ll426/scratch/code/logit-fit/ddot.f -- translated by f2c (version 20160102).
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

/* > \brief \b DDOT */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/*  Definition: */
/*  =========== */

/*       DOUBLE PRECISION FUNCTION DDOT(N,DX,INCX,DY,INCY) */

/*       .. Scalar Arguments .. */
/*       int INCX,INCY,N */
/*       .. */
/*       .. Array Arguments .. */
/*       DOUBLE PRECISION DX(*),DY(*) */
/*       .. */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* >    DDOT forms the dot product of two vectors. */
/* >    uses unrolled loops for increments equal to one. */
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
/* >     jack dongarra, linpack, 3/11/78. */
/* >     modified 12/3/93, array(1) declarations changed to array(*) */
/* > \endverbatim */
/* > */
/*  ===================================================================== */
__device__ doublereal ddot_(int *n, doublereal *dx, int *incx, doublereal *dy, 
	int *incy)
{
    /* System generated locals */
    int i__1;
    doublereal ret_val;

    /* Local variables */
    int i__, m, ix, iy, mp1;
    doublereal dtemp;


/*  -- Reference BLAS level1 routine (version 3.4.0) -- */
/*  -- Reference BLAS is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     November 2011 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
    /* Parameter adjustments */
    // --dy;
    // --dx;

    /* Function Body */
    ret_val = 0.;
    dtemp = 0.;
    if (*n <= 0) {
	return ret_val;
    }
    if (*incx == 1 && *incy == 1) {
    
    	printf("Breakpoint C1\n"); // DEBUG

/*        code for both increments equal to 1 */


/*        clean-up loop */

	m = *n % 5;
	if (m != 0) {
	    i__1 = m;
	    for (i__ = 0; i__ < i__1; ++i__) {
		dtemp += dx[i__] * dy[i__];
	    }
	    if (*n < 5) {
		ret_val = dtemp;
		return ret_val;
	    }
	}
	printf("Breakpoint C2\n"); // DEBUG
	mp1 = m + 1;
	i__1 = *n;
	for (i__ = mp1-1; i__ < i__1; i__ += 5) {
		// DEBUG
		printf("Breakpoint C3: %d\n", i__);
		printf("dx[i]: %f\n", dx[i__]);
		printf("dy[i]: %f\n", dy[i__]);
		printf("dx[i+1]: %f\n", dx[i__ + 1]);
		printf("dy[i+1]: %f\n", dy[i__ + 1]);
		printf("dx[i+2]: %f\n", dx[i__ + 2]);
		printf("dy[i+2]: %f\n", dy[i__ + 2]);
		printf("dx[i+3]: %f\n", dx[i__ + 3]);
		printf("dy[i+3]: %f\n", dy[i__ + 3]);
		printf("dx[i+4]: %f\n", dx[i__ + 4]);
		printf("dy[i+4]: %f\n", dy[i__ + 4]);
		
	    dtemp = dtemp + dx[i__] * dy[i__] + dx[i__ + 1] * dy[i__ + 1] + 
		    dx[i__ + 2] * dy[i__ + 2] + dx[i__ + 3] * dy[i__ + 3] + 
		    dx[i__ + 4] * dy[i__ + 4];
	}
	printf("Breakpoint C4\n"); // DEBUG
    } else {

/*        code for unequal increments or equal increments */
/*          not equal to 1 */

	ix = 1;
	iy = 1;
	if (*incx < 0) {
	    ix = (-(*n) + 1) * *incx + 1;
	}
	if (*incy < 0) {
	    iy = (-(*n) + 1) * *incy + 1;
	}
	i__1 = *n;
	for (i__ = 0; i__ < i__1; ++i__) {
	    dtemp += dx[ix] * dy[iy];
	    ix += *incx;
	    iy += *incy;
	}
    }
    ret_val = dtemp;
    return ret_val;
} /* ddot_ */

