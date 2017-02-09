#include "simulator.h"
#include "input.h"
#include "algebra.h"
#include "kernel.cu"
#include "kernelCPU.h"
#include "filesystem.h"

#include <cuda.h>
#include <curand.h>
#include <string>
#include <vector>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <cassert>
#include <unistd.h>
#include <queue>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    exit(EXIT_FAILURE);}} while(0)

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    exit(EXIT_FAILURE);}} while(0)

#define MALLOC(x, y) CUDA_CALL(cudaMallocManaged(x, y))
#define FREE(x) CUDA_CALL(cudaFree(x))

void fillInput(std::string, CodeInfo**, Edge**, Edge**, Matrix &);
SimulationReport simulate(std::string, float);

const int BLOCK_SIZE = 512;
const int DECODERS = 100;
const int BLOCKS = DECODERS;
const int MAX_ITERATIONS = 50;
const int DEFAULT_NUMBER_OF_CODEWORDS = 10 * 1000;
const int MAX_NUMBER_OF_CODEWORDS = 1000 * 1000 * 1000;
const int DEFAULT_NUMBER_OF_MIN_FER = 100; 
const float DEFAULT_SNR = 3;
const bool CALL_GPU = true;

struct settings_t
{
    enum NumberOfRuns { MIN_FER, CODEWORDS } runsType = CODEWORDS;
    float snrFrom = DEFAULT_SNR;
    float snrTo = DEFAULT_SNR + 0.1;
    float snrStep = 1;
    int numberOfCodewords = DEFAULT_NUMBER_OF_CODEWORDS;
    int numberOfMinFER = DEFAULT_NUMBER_OF_MIN_FER;
    bool runsTypeSet = false;
    float ferThreshold = 0.01;
    // if FER lower than this value, don't calc further 
    // and print this value for rest snrs because
    // it can take too long

    void checkRunsTypeAndSet(NumberOfRuns type)
    {
        if (runsTypeSet)
        {
            std::cerr << "-n and -f should not be set at the same time" << std::endl;;
            exit(1);
        }
        runsType = type;
    }
} settings;

int main(int argc, char* argv[])
{
    int opt;
    while ((opt = getopt(argc, argv, "n:f:s:")) != -1) {
        switch (opt) {
            case 's':
                sscanf(optarg, "%f:%f:%f", &settings.snrFrom,
                        &settings.snrTo,
                        &settings.snrStep);
                break;
            case 'n':
                settings.checkRunsTypeAndSet(settings_t::CODEWORDS);
                settings.numberOfCodewords = atoi(optarg);
                break;
            case 'f':
                settings.checkRunsTypeAndSet(settings_t::MIN_FER);
                settings.numberOfMinFER = atoi(optarg);
                break;
            case 't':
                settings.ferThreshold = atof(optarg);
                break;
            default:
                printf("Usage: %s [-s snrFrom:snrTo:snrStep] files*\n",
                        argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    std::queue<std::string> inputFilenames;
    for (int i = optind; i < argc; i++)
    {
        inputFilenames.push(argv[i]);
    }

    std::cout << "Results" << std::endl;
    std::cout << "Filename SNR Time(ms) BER% FER%" << std::endl;

    // Create time measure structures
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    while (!inputFilenames.empty())
    {
        std::string filename = inputFilenames.front();
        inputFilenames.pop();

        if (isDirectory(filename))
        {
            auto files = filesInDirectory(filename);
            for (auto file : files)
                inputFilenames.push(file);
            continue;
        }

        SimulationReport report;
        report.FER = 100;
        for (float snr = settings.snrFrom;
                snr < settings.snrTo;
                snr += settings.snrStep)
        {
            cudaEventRecord(start);

            // Check if FER is still big enough
            if (report.FER >= settings.ferThreshold)
            {
                // Calling main simulation
                report = simulate(filename, snr);
            }

            cudaEventRecord(stop);

            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);

            std::cout << filename << " ";
            std::cout << snr << " ";
            std::cout << milliseconds << " ";
            std::cout << report.BER << " ";
            std::cout << report.FER << " ";
            std::cout << std::endl;
        }
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

SimulationReport simulate(std::string filename, float SNR)
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
    MALLOC(&probP, DECODERS * codeInfo->varNodes * sizeof(float));
    MALLOC(&probQ, DECODERS * codeInfo->totalEdges * sizeof(float));
    MALLOC(&probR, DECODERS * codeInfo->totalEdges * sizeof(float));

    float* noisedVector;
    int noisedVectorSize = DECODERS * codeInfo->varNodes;
    if (noisedVectorSize % 2 == 1)
        noisedVectorSize++; // curandGenerateNormal works only with even
    MALLOC(&noisedVector, noisedVectorSize * sizeof(float));
    curandGenerator_t gen;
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 19ULL));
    srand(19);
    float* codewords;
    MALLOC(&codewords, noisedVectorSize * sizeof(float));
    for (int i = 0; i < DECODERS; i++)
        writeRandomCodeword(codewords + codeInfo->varNodes * i, Gt);
    float* estimation;
    MALLOC(&estimation, noisedVectorSize * sizeof(float));
    ErrorInfo* errorInfo;
    MALLOC(&errorInfo, sizeof(ErrorInfo));
    errorInfo->bitErrors = 0;
    errorInfo->frameErrors = 0;

    float cntFrames = 0;
    float cntBits = 0;
    while (true)
    {
        cntFrames += DECODERS;
        cntBits += DECODERS * codeInfo->varNodes;
        // Generate n float on device with 
        // normal distribution mean = 0, stddev = sqrt(sigma2)
        CURAND_CALL(curandGenerateNormal(gen, noisedVector, 
                    noisedVectorSize, 0.0, sqrt(sigma2)));
        // Kernel execution
        if (CALL_GPU)
        {
            decodeAWGN<<<BLOCKS, BLOCK_SIZE>>>(
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
            decodeAWGN_CPU(codeInfo,edgesFromVariable,edgesFromCheck,probP,probQ,probR,sigma2,estimation,codewords,noisedVector,MAX_ITERATIONS,errorInfo,BLOCKS,BLOCK_SIZE);
        }
        CUDA_CALL(cudaDeviceSynchronize());

        if (settings.runsType == settings_t::CODEWORDS &&
                cntFrames >= settings.numberOfCodewords)
            break;
        if (settings.runsType == settings_t::MIN_FER &&
                errorInfo->frameErrors >= settings.numberOfMinFER)
            break;
        if (cntFrames >= MAX_NUMBER_OF_CODEWORDS)
            break;
    }
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
