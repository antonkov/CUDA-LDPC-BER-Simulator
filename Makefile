simulator: input.o kernel.o
	nvcc -lcurand -ccbin g++ -std=c++11 -o simulator input.o kernel.o

kernel.o: simulator.cu simulator.h kernel.cu
	nvcc -ccbin g++ -std=c++11 -o kernel.o -c simulator.cu

input.o: input.cpp input.h
	nvcc -ccbin g++ -std=c++11 -c input.cpp

clean:
	rm simulator kernel.o input.o
