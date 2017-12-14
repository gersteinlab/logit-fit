all: glm.logit

glm.logit: glm.logit.o daxpy.o dcopy.o ddot.o dnrm2.o dqrdc.o dqrdc2.o dqrls.o dqrsl.o dscal.o dswap.o
	gfortran -o glm.logit glm.logit.o daxpy.o dcopy.o ddot.o dnrm2.o dqrdc.o dqrdc2.o dqrls.o dqrsl.o dscal.o dswap.o -lgcc -L/ysm-gpfs/apps/software/CUDA/9.0.176/lib64/ -lcuda -lcudart

glm.logit.o: glm.logit.cu lmfit.cu
	nvcc -lineinfo -dc glm.logit.cu lmfit.cu
	
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
