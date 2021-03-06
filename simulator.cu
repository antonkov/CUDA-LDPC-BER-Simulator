#include "simulator.h"
#include "input.h"
#include "algebra.h"
#include "kernel.cu"
#include "kernelCPU.h"

#include <cuda.h>
#include <curand.h>
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

const int BLOCK_SIZE = 512;
const int DECODERS = 100;
const int BLOCKS = DECODERS;
const int MAX_ITERATIONS = 50;
const bool CALL_GPU = true;

SimulationReport simulateImpl(Data const &, simulation_params_t const &);

Data::Data(std::string const & filename)
{
    Matrix H;
    std::ifstream matrixStream(filename);
    readMatrix(matrixStream, &H);
    Gt = codingMatrix(H);
    assert(isZero(multiply(H, Gt)));

    MALLOC(&codeInfo, sizeof(CodeInfo));
    codeInfo->checkNodes = H.k;
    codeInfo->varNodes = H.n;
    codeInfo->totalEdges = H.totalCells;

    MALLOC(&edgesFromVariable, codeInfo->totalEdges * sizeof(Edge));
    MALLOC(&edgesFromCheck, codeInfo->totalEdges * sizeof(Edge));

    int currentEdge = 0;
    for (int j = 0; j < H.n; j++)
    {
        int connectedToNode = 0;
        int absoluteStartIndex = currentEdge;
        for (auto iIdPair : H.cols[j])
        {
            int i = iIdPair.first;
            int id = iIdPair.second;
            Edge& edge = edgesFromVariable[currentEdge++];
            edge.index = id;
            edge.vn = j;
            edge.cn = i;
            edge.absoluteStartIndex = absoluteStartIndex;
            edge.relativeIndexFromNode = connectedToNode;
            connectedToNode++;
        }
        for (int id = 0; id < connectedToNode; id++)
        {
            Edge& edge = edgesFromVariable[absoluteStartIndex + id];
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
            Edge& edge = edgesFromCheck[currentEdge++];
            edge.index = id;
            edge.vn = j;
            edge.cn = i;
            edge.absoluteStartIndex = absoluteStartIndex;
            edge.relativeIndexFromNode = connectedToNode;
            connectedToNode++;
        }
        for (int id = 0; id < connectedToNode; id++)
        {
            Edge& edge = edgesFromCheck[absoluteStartIndex + id];
            edge.edgesConnectedToNode = connectedToNode;
            //std::cout << edge.index << " " << edge.vn << " " << edge.cn
            //    << " " << edge.edgesConnectedToNode
            //    << " " << edge.absoluteStartIndex
            //    << " " << edge.relativeIndexFromNode << std::endl;
        }
    }

    MALLOC(&probP, DECODERS * codeInfo->varNodes * sizeof(float));
    MALLOC(&probQ, DECODERS * codeInfo->totalEdges * sizeof(float));
    MALLOC(&probR, DECODERS * codeInfo->totalEdges * sizeof(float));
    noisedVectorSize = DECODERS * codeInfo->varNodes;
    if (noisedVectorSize % 2 == 1)
        noisedVectorSize++; // curandGenerateNormal works only with even
    MALLOC(&noisedVector, noisedVectorSize * sizeof(float));
    MALLOC(&codewords, noisedVectorSize * sizeof(float));
    MALLOC(&estimation, noisedVectorSize * sizeof(float));
    MALLOC(&errorInfo, sizeof(ErrorInfo));
}

Data::~Data()
{
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
}

SimulationReport simulate(Data const & data, simulation_params_t const & params)
{
    // Create time measure structures
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    cudaEventRecord(start);

    SimulationReport report = simulateImpl(data, params);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    report.timeMs = milliseconds;

    return report;
}

void writeRandomCodeword(float * a, Matrix const & Gt)
{
    std::fill(a, a + Gt.rows.size(), 0);
    for (int j = 0; j < Gt.cols.size(); j++)
        if (rand() % 2) // take this codeword
            for (auto p : Gt.cols[j])
                a[p.first]  = 1 - a[p.first];
}


SimulationReport simulateImpl(Data const & data, simulation_params_t const & params)
{
    float sigma2 = pow(10.0, -params.snr / 10.0);

    curandGenerator_t gen;
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 19ULL));
    srand(19);
    for (int i = 0; i < DECODERS; i++)
        writeRandomCodeword(data.codewords + data.codeInfo->varNodes * i, data.Gt);
    data.errorInfo->bitErrors = 0;
    data.errorInfo->frameErrors = 0;

    float cntFrames = 0;
    float cntBits = 0;
    while (true)
    {
        cntFrames += DECODERS;
        cntBits += DECODERS * data.codeInfo->varNodes;
        // Generate n float on device with 
        // normal distribution mean = 0, stddev = sqrt(sigma2)
        CURAND_CALL(curandGenerateNormal(gen, data.noisedVector, 
                    data.noisedVectorSize, 0.0, sqrt(sigma2)));
        // Kernel execution
        if (CALL_GPU)
        {
            decodeAWGN<<<BLOCKS, BLOCK_SIZE>>>(
                    data.codeInfo,
                    data.edgesFromVariable,
                    data.edgesFromCheck,
                    data.probP,
                    data.probQ,
                    data.probR,
                    sigma2,
                    data.estimation,
                    data.codewords,
                    data.noisedVector,
                    MAX_ITERATIONS,
                    data.errorInfo);
        } else {
            CUDA_CALL(cudaDeviceSynchronize());
            decodeAWGN_CPU(data.codeInfo,data.edgesFromVariable,data.edgesFromCheck,data.probP,data.probQ,data.probR,sigma2,data.estimation,data.codewords,data.noisedVector,MAX_ITERATIONS,data.errorInfo,BLOCKS,BLOCK_SIZE);
        }
        CUDA_CALL(cudaDeviceSynchronize());

        if (cntFrames >= params.numberOfCodewords ||
            data.errorInfo->frameErrors >= params.numberOfFrameErrors)
            break;
    }
    float BER = data.errorInfo->bitErrors / cntBits;
    float FER = data.errorInfo->frameErrors / cntFrames;

    SimulationReport report;
    report.BER = BER;
    report.FER = FER;
    return report;
}
