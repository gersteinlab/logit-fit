*> \brief \b DCOPY
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at 
*            http://www.netlib.org/lapack/explore-html/ 
*
*  Definition:
*  ===========
*
*       SUBROUTINE DCOPY(N,DX,INCX,DY,INCY)
* 
*       .. Scalar Arguments ..
*       INTEGER INCX,INCY,N
*       ..
*       .. Array Arguments ..
*       DOUBLE PRECISION DX(*),DY(*)
*       ..
*  
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*>    DCOPY copies a vector, x, to a vector, y.
*>    uses unrolled loops for increments equal to one.
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
*>     modified 12/3/93, array(1) declarations changed to array(*)
*> \endverbatim
*>
*  =====================================================================
      SUBROUTINE dcopy(N,DX,INCX,DY,INCY)
*
*  -- Reference BLAS level1 routine (version 3.4.0) --
*  -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     November 2011
*
*     .. Scalar Arguments ..
      INTEGER INCX,INCY,N
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION DX(*),DY(*)
*     ..
*
*  =====================================================================
*
*     .. Local Scalars ..
      INTEGER I,IX,IY,M,MP1
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC mod
*     ..
      IF (n.LE.0) RETURN
      IF (incx.EQ.1 .AND. incy.EQ.1) THEN
*
*        code for both increments equal to 1
*
*
*        clean-up loop
*
         m = mod(n,7)
         IF (m.NE.0) THEN
            DO i = 1,m
               dy(i) = dx(i)
            END DO
            IF (n.LT.7) RETURN
         END IF   
         mp1 = m + 1
         DO i = mp1,n,7
            dy(i) = dx(i)
            dy(i+1) = dx(i+1)
            dy(i+2) = dx(i+2)
            dy(i+3) = dx(i+3)
            dy(i+4) = dx(i+4)
            dy(i+5) = dx(i+5)
            dy(i+6) = dx(i+6)
         END DO
      ELSE      
*
*        code for unequal increments or equal increments
*          not equal to 1
*
         ix = 1
         iy = 1
         IF (incx.LT.0) ix = (-n+1)*incx + 1
         IF (incy.LT.0) iy = (-n+1)*incy + 1
         DO i = 1,n
            dy(iy) = dx(ix)
            ix = ix + incx
            iy = iy + incy
         END DO
      END IF
      RETURN
      END
