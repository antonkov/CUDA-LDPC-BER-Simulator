simulator: bin/input.o bin/kernel.o bin/algebra.o bin/kernelCPU.o bin/filesystem.o
	nvcc -lcurand -lboost_system -lboost_filesystem -ccbin g++ -std=c++11 -o simulator bin/input.o bin/kernel.o bin/algebra.o bin/kernelCPU.o bin/filesystem.o

bin/kernel.o: simulator.cu simulator.h kernel.cu
	nvcc -ccbin g++ -std=c++11 -o bin/kernel.o -c simulator.cu

bin/kernelCPU.o: simulator.h kernelCPU.h kernelCPU.cpp
	nvcc -ccbin g++ -std=c++11 -o bin/kernelCPU.o -c kernelCPU.cpp

bin/input.o: input.cpp input.h
	nvcc -ccbin g++ -std=c++11 -o bin/input.o -c input.cpp

bin/algebra.o: algebra.cpp algebra.h
	nvcc -ccbin g++ -std=c++11 -o bin/algebra.o -c algebra.cpp

bin/filesystem.o: filesystem.cpp filesystem.h
	nvcc -ccbin g++ -std=c++11 -o bin/filesystem.o -c filesystem.cpp

clean:
	rm simulator bin/*
