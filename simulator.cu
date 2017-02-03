#include "simulator.h"
#include "input.h"
#include "algebra.h"
#include "kernel.cu"
#include "kernelCPU.h"

#include <cuda.h>
#include <curand.h>
#include <string>
#include <vector>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <cassert>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    exit(EXIT_FAILURE);}} while(0)

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    exit(EXIT_FAILURE);}} while(0)

#define MALLOC(x, y) CUDA_CALL(cudaMallocManaged(x, y))
#define FREE(x) CUDA_CALL(cudaFree(x))

const int block_size = 512;
const int decoders = 100;
const int blocks = decoders;
const float SNR = 4;
const int MAX_ITERATIONS = 50;
const int NUMBER_OF_CODEWORDS = 10 * 1000;
const bool callGPU = true;

void fillInput(std::string, CodeInfo**, Edge**, Edge**, Matrix &);
SimulationReport simulate(std::string);

int main(int argc, char* argv[])
{
    std::vector<std::string> inputFilenames;
    for (int i = 1; i < argc; i++)
    {
        inputFilenames.push_back(argv[i]);
    }

    std::cout << "Results" << std::endl;
    std::cout << "Filename Time(ms) BER% FER%" << std::endl;

    // Create time measure structures
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    for (auto filename : inputFilenames)
    {
        cudaEventRecord(start);

        // Calling main simulation
        SimulationReport report = simulate(filename);

        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        std::cout << filename << " ";
        std::cout << milliseconds << " ";
        std::cout << report.BER << " ";
        std::cout << report.FER << " ";
        std::cout << std::endl;
    }
}

void writeRandomCodeword(float * a, Matrix const & Gt)
{
    std::fill(a, a + Gt.rows.size(), 0);
    for (int j = 0; j < Gt.cols.size(); j++)
        if (rand() % 2) // take this codeword
            for (auto p : Gt.cols[j])
                a[p.first]  = 1 - a[p.first];
}

SimulationReport simulate(std::string filename)
{
    CodeInfo* codeInfo;
    Edge* edgesFromVariable;
    Edge* edgesFromCheck;
    Matrix Gt;
    fillInput(filename, &codeInfo, &edgesFromVariable, &edgesFromCheck, Gt);
    float sigma2 = pow(10.0, -SNR / 10.0);

    float* probP;
    float* probQ;
    float* probR;
    MALLOC(&probP, decoders * codeInfo->varNodes * sizeof(float));
    MALLOC(&probQ, decoders * codeInfo->totalEdges * sizeof(float));
    MALLOC(&probR, decoders * codeInfo->totalEdges * sizeof(float));

    float* noisedVector;
    int noisedVectorSize = decoders * codeInfo->varNodes;
    if (noisedVectorSize % 2 == 1)
        noisedVectorSize++; // curandGenerateNormal works only with even
    MALLOC(&noisedVector, noisedVectorSize * sizeof(float));
    curandGenerator_t gen;
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 19ULL));
    srand(19);
    float* codewords;
    MALLOC(&codewords, noisedVectorSize * sizeof(float));
    for (int i = 0; i < decoders; i++)
        writeRandomCodeword(codewords + codeInfo->varNodes * i, Gt);
    float* estimation;
    MALLOC(&estimation, noisedVectorSize * sizeof(float));
    ErrorInfo* errorInfo;
    MALLOC(&errorInfo, sizeof(ErrorInfo));
    errorInfo->bitErrors = 0;
    errorInfo->frameErrors = 0;

    int numberKernelRuns = NUMBER_OF_CODEWORDS / decoders;
    float cntFrames = decoders * numberKernelRuns;
    float cntBits = cntFrames * codeInfo->varNodes;
    for (int run = 0; run < numberKernelRuns; run++)
    {
        // Generate n float on device with 
        // normal distribution mean = 0, stddev = sqrt(sigma2)
        CURAND_CALL(curandGenerateNormal(gen, noisedVector, 
                    noisedVectorSize, 0.0, sqrt(sigma2)));
        // Kernel execution
        if (callGPU)
        {
        decodeAWGN<<<blocks, block_size>>>(
                codeInfo,
                edgesFromVariable,
                edgesFromCheck,
                probP,
                probQ,
                probR,
                sigma2,
                estimation,
                codewords,
                noisedVector,
                MAX_ITERATIONS,
                errorInfo);
        } else {
            CUDA_CALL(cudaDeviceSynchronize());
            decodeAWGN_CPU(codeInfo,edgesFromVariable,edgesFromCheck,probP,probQ,probR,sigma2,estimation,codewords,noisedVector,MAX_ITERATIONS,errorInfo,blocks,block_size);
        }
    }
    CUDA_CALL(cudaDeviceSynchronize());
    float BER = errorInfo->bitErrors * 100.0 / cntBits;
    float FER = errorInfo->frameErrors * 100.0 / cntFrames;

    // Freeing resources
    FREE(errorInfo);
    FREE(estimation);
    FREE(codewords);
    FREE(noisedVector);
    FREE(probR);
    FREE(probQ);
    FREE(probP);
    FREE(edgesFromCheck);
    FREE(edgesFromVariable);
    FREE(codeInfo);

    SimulationReport report;
    report.BER = BER;
    report.FER = FER;
    return report;
}

void fillInput(
        std::string filename,
        CodeInfo** codeInfo,
        Edge** edgesFromVariable,
        Edge** edgesFromCheck,
        Matrix & Gt)
{
    Matrix H;
    std::ifstream matrixStream(filename);
    readMatrix(matrixStream, &H);
    Gt = codingMatrix(H);
    assert(isZero(multiply(H, Gt)));

    MALLOC(codeInfo, sizeof(CodeInfo));
    (*codeInfo)->checkNodes = H.k;
    (*codeInfo)->varNodes = H.n;
    (*codeInfo)->totalEdges = H.totalCells;

    MALLOC(edgesFromVariable, (*codeInfo)->totalEdges * sizeof(Edge));
    MALLOC(edgesFromCheck, (*codeInfo)->totalEdges * sizeof(Edge));

    int currentEdge = 0;
    for (int j = 0; j < H.n; j++)
    {
        int connectedToNode = 0;
        int absoluteStartIndex = currentEdge;
        for (auto iIdPair : H.cols[j])
        {
            int i = iIdPair.first;
            int id = iIdPair.second;
            Edge& edge = (*edgesFromVariable)[currentEdge++];
            edge.index = id;
            edge.vn = j;
            edge.cn = i;
            edge.absoluteStartIndex = absoluteStartIndex;
            edge.relativeIndexFromNode = connectedToNode;
            connectedToNode++;
        }
        for (int id = 0; id < connectedToNode; id++)
        {
            Edge& edge = (*edgesFromVariable)[absoluteStartIndex + id];
            edge.edgesConnectedToNode = connectedToNode;
            //std::cout << edge.index << " " << edge.vn << " " << edge.cn
            //    << " " << edge.edgesConnectedToNode
            //    << " " << edge.absoluteStartIndex
            //    << " " << edge.relativeIndexFromNode << std::endl;
        }
    }

    //std::cout << "Table II" << std::endl;
    currentEdge = 0;
    for (int i = 0; i < H.k; i++)
    {
        int connectedToNode = 0;
        int absoluteStartIndex = currentEdge;
        for (auto jIdPair : H.rows[i])
        {
            int j = jIdPair.first;
            int id = jIdPair.second;
            Edge& edge = (*edgesFromCheck)[currentEdge++];
            edge.index = id;
            edge.vn = j;
            edge.cn = i;
            edge.absoluteStartIndex = absoluteStartIndex;
            edge.relativeIndexFromNode = connectedToNode;
            connectedToNode++;
        }
        for (int id = 0; id < connectedToNode; id++)
        {
            Edge& edge = (*edgesFromCheck)[absoluteStartIndex + id];
            edge.edgesConnectedToNode = connectedToNode;
            //std::cout << edge.index << " " << edge.vn << " " << edge.cn
            //    << " " << edge.edgesConnectedToNode
            //    << " " << edge.absoluteStartIndex
            //    << " " << edge.relativeIndexFromNode << std::endl;
        }
    }
}
