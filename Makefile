simulator: simulator.cu simulator.h kernel.cu
	nvcc -lcurand -ccbin g++ -std=c++11 -o simulator simulator.cu

clean:
	rm simulator 
