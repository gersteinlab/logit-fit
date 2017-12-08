*> \brief \b DSCAL
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at 
*            http://www.netlib.org/lapack/explore-html/ 
*
*  Definition:
*  ===========
*
*       SUBROUTINE DSCAL(N,DA,DX,INCX)
* 
*       .. Scalar Arguments ..
*       DOUBLE PRECISION DA
*       INTEGER INCX,N
*       ..
*       .. Array Arguments ..
*       DOUBLE PRECISION DX(*)
*       ..
*  
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*>    DSCAL scales a vector by a constant.
*>    uses unrolled loops for increment equal to one.
*> \endverbatim
*
*  Authors:
*  ========
*
*> \author Univ. of Tennessee 
*> \author Univ. of California Berkeley 
*> \author Univ. of Colorado Denver 
*> \author NAG Ltd. 
*
*> \date November 2011
*
*> \ingroup double_blas_level1
*
*> \par Further Details:
*  =====================
*>
*> \verbatim
*>
*>     jack dongarra, linpack, 3/11/78.
*>     modified 3/93 to return if incx .le. 0.
*>     modified 12/3/93, array(1) declarations changed to array(*)
*> \endverbatim
*>
*  =====================================================================
      SUBROUTINE dscal(N,DA,DX,INCX)
*
*  -- Reference BLAS level1 routine (version 3.4.0) --
*  -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     November 2011
*
*     .. Scalar Arguments ..
      DOUBLE PRECISION DA
      INTEGER INCX,N
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION DX(*)
*     ..
*
*  =====================================================================
*
*     .. Local Scalars ..
      INTEGER I,M,MP1,NINCX
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC mod
*     ..
      IF (n.LE.0 .OR. incx.LE.0) RETURN
      IF (incx.EQ.1) THEN
*
*        code for increment equal to 1
*
*
*        clean-up loop
*
         m = mod(n,5)
         IF (m.NE.0) THEN
            DO i = 1,m
               dx(i) = da*dx(i)
            END DO
            IF (n.LT.5) RETURN
         END IF
         mp1 = m + 1
         DO i = mp1,n,5
            dx(i) = da*dx(i)
            dx(i+1) = da*dx(i+1)
            dx(i+2) = da*dx(i+2)
            dx(i+3) = da*dx(i+3)
            dx(i+4) = da*dx(i+4)
         END DO
      ELSE
*
*        code for increment not equal to 1
*
         nincx = n*incx
         DO i = 1,nincx,incx
            dx(i) = da*dx(i)
         END DO
      END IF
      RETURN
      END
