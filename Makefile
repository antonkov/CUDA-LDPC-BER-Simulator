main: codeSnr simulator

codeSnr: bin/filesystem.o bin/libsimulator.a codeSnr.cpp
	nvcc -O3 -Lbin -lcurand -lboost_system -lboost_filesystem -lsimulator -ccbin g++ -std=c++11 -o codeSnr bin/filesystem.o codeSnr.cpp

simulator: bin/filesystem.o bin/libsimulator.a simulator_frontend.cpp 
	nvcc -O3 -Lbin -lcurand -lboost_system -lboost_filesystem -lsimulator -ccbin g++ -std=c++11 -o simulator bin/filesystem.o simulator_frontend.cpp

bin/libsimulator.a: bin/input.o bin/kernel.o bin/algebra.o bin/kernelCPU.o
	ar rcs bin/libsimulator.a bin/input.o bin/kernel.o bin/algebra.o bin/kernelCPU.o

bin/kernel.o: simulator.cu simulator.h kernel.cu
	nvcc -ccbin g++ -std=c++11 -o bin/kernel.o -c simulator.cu

bin/kernelCPU.o: simulator.h kernelCPU.h kernelCPU.cpp
	nvcc -O3 -ccbin g++ -std=c++11 -o bin/kernelCPU.o -c kernelCPU.cpp

bin/input.o: input.cpp input.h
	nvcc -O3 -ccbin g++ -std=c++11 -o bin/input.o -c input.cpp

bin/algebra.o: algebra.cpp algebra.h
	nvcc -O3 -ccbin g++ -std=c++11 -o bin/algebra.o -c algebra.cpp

bin/filesystem.o: filesystem.cpp filesystem.h
	nvcc -O3 -ccbin g++ -std=c++11 -o bin/filesystem.o -c filesystem.cpp

findTop: bin/filesystem.o bin/libsimulator.a findTop.cpp
	nvcc -O3 -Lbin -lcurand -lboost_system -lboost_filesystem -lsimulator -ccbin g++ -std=c++11 -o findTop bin/filesystem.o findTop.cpp

clean:
	rm findTop codeSnr simulator bin/*
