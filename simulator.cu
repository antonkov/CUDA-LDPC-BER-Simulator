#include "simulator.h"
#include "kernel.cu"
#include <vector>
#include <cstdio>
#include <iostream>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    exit(EXIT_FAILURE);}} while(0)

#define MALLOC(x, y) CUDA_CALL(cudaMallocManaged(x, y))

const int block_size = 512;
const int decoders = 100;
const int blocks = decoders;
const float SNR = 2;

void fillInput(CodeInfo**, Edge**, Edge**);

int main()
{
    CodeInfo* codeInfo;
    Edge* edgesFromVariable;
    Edge* edgesFromCheck;
    fillInput(&codeInfo, &edgesFromVariable, &edgesFromCheck);
    float sigma2 = pow(10.0, -SNR / 10);

    float* probP;
    float* probQ;
    float* probR;
    MALLOC(&probP, decoders * codeInfo->varNodes * sizeof(float));
    MALLOC(&probQ, decoders * codeInfo->totalEdges * sizeof(float));
    MALLOC(&probR, decoders * codeInfo->totalEdges * sizeof(float));

    float* noisedVector;
    int noisedVectorSize = decoders * codeInfo->varNodes;
    MALLOC(&noisedVector, noisedVectorSize);
    // adding noise
    for (int i = 0; i < noisedVectorSize; i++)
    {
        noisedVector[i] = -1;
    }

    // Kernel execution
    decodeAWGN<<<blocks, block_size>>>(
            codeInfo,
            edgesFromVariable,
            edgesFromCheck,
            probP,
            probQ,
            probR,
            sigma2,
            noisedVector);
    //cudaMemcpy(berOut, berOut_obj, sizeof(float)
    CUDA_CALL(cudaDeviceSynchronize());
}

void fillInput(
        CodeInfo** codeInfo, Edge** edgesFromVariable, Edge** edgesFromCheck)
{
    MALLOC(codeInfo, sizeof(CodeInfo));
    freopen("matrix.txt", "r", stdin);
    int n, k;
    std::cin >> k >> n;
    (*codeInfo)->checkNodes = k;
    (*codeInfo)->varNodes = n;
    std::vector<std::vector<int>> h(k, std::vector<int>(n));
    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::cin >> h[i][j];
            if (h[i][j])
                (*codeInfo)->totalEdges++;
        }
    }

    MALLOC(edgesFromVariable, (*codeInfo)->totalEdges * sizeof(Edge));
    MALLOC(edgesFromCheck, (*codeInfo)->totalEdges * sizeof(Edge));

    int currentEdge = 0;
    std::vector<std::vector<int>> e(k, std::vector<int>(n));
    for (int j = 0; j < n; j++)
    {
        int connectedToNode = 0;
        int absoluteStartIndex = currentEdge;
        for (int i = k - 1; i >= 0; i--)
        {
            if (h[i][j])
            {
                e[i][j] = currentEdge++;
                Edge& edge = (*edgesFromVariable)[e[i][j]];
                edge.index = e[i][j];
                edge.vn = j;
                edge.cn = i;
                edge.absoluteStartIndex = absoluteStartIndex;
                edge.relativeIndexFromNode = connectedToNode;
                connectedToNode++;
            }
        }
        for (int i = k - 1; i >= 0; i--)
        {
            if (h[i][j])
            {
                Edge& edge = (*edgesFromVariable)[e[i][j]];
                edge.edgesConnectedToNode = connectedToNode;
                /*std::cout << edge.index << " " << edge.vn << " " << edge.cn
                    << " " << edge.edgesConnectedToNode
                    << " " << edge.absoluteStartIndex
                    << " " << edge.relativeIndexFromNode << std::endl;*/
            }
        }
    }

    //std::cout << "Table II" << std::endl;
    currentEdge = 0;
    for (int i = 0; i < k; i++)
    {
        int connectedToNode = 0;
        int absoluteStartIndex = currentEdge;
        for (int j = 0; j < n; j++)
        {
            if (h[i][j])
            {
                currentEdge++;
                Edge& edge = (*edgesFromCheck)[e[i][j]];
                edge.index = e[i][j];
                edge.vn = j;
                edge.cn = i;
                edge.absoluteStartIndex = absoluteStartIndex;
                edge.relativeIndexFromNode = connectedToNode;
                connectedToNode++;
            }
        }
        for (int j = 0; j < n; j++)
        {
            if (h[i][j])
            {
                Edge& edge = (*edgesFromCheck)[e[i][j]];
                edge.edgesConnectedToNode = connectedToNode;
                /*std::cout << edge.index << " " << edge.vn << " " << edge.cn
                    << " " << edge.edgesConnectedToNode
                    << " " << edge.absoluteStartIndex
                    << " " << edge.relativeIndexFromNode << std::endl;*/
            }
        }
    }
}
