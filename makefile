all: glm.logit

glm.nb: glm.logit.o daxpy.o dcopy.o ddot.o dnrm2.o dqrdc.o dqrdc2.o dqrls.o dqrsl.o dscal.o dswap.o
	g++ -Wall -o glm.logit glm.nb.o daxpy.o dcopy.o ddot.o dnrm2.o dqrdc.o dqrdc2.o dqrls.o dqrsl.o dscal.o dswap.o

glm.logit.o: glm.logit.cpp lmfit.cpp fit.cpp
	g++ -Wall -c glm.logit.cpp
	
daxpy.o: daxpy.f
	gfortran -c daxpy.f
	
dcopy.o: dcopy.f
	gfortran -c dcopy.f

ddot.o: ddot.f
	gfortran -c ddot.f
	
dnrm2.o: dnrm2.f
	gfortran -c dnrm2.f
	
dqrdc.o: dqrdc.f
	gfortran -c dqrdc.f
	
dqrdc2.o: dqrdc2.f
	gfortran -c dqrdc2.f
	
dqrls.o: dqrls.f
	gfortran -c dqrls.f
	
dqrsl.o: dqrsl.f
	gfortran -c dqrsl.f
	
dscal.o: dscal.f
	gfortran -c dscal.f

dswap.o: dswap.f
	gfortran -c dswap.f
