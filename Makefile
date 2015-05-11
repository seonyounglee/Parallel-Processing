
cudart :  cudart.cu
	#nvcc -O3 -lm -o cudart cudart.cu
	nvcc -lm -g -o cudart cudart.cu -arch sm_20

