#include "simulator.h"

__device__ void iterateToVariables(
        CodeInfo* codeInfo,
        Edge* edges,
        float* probQ,
        float* probR)
{
    int eSt = blockIdx.x * codeInfo->totalEdges;
    for (int p = 0; p < codeInfo->totalEdges; p += blockDim.x)
    {
        Edge& e = edges[p + threadIdx.x];
        float r0 = 1;
        float r1 = 1;
        for (int id = 0; id < e.edgesConnectedToNode; id++)
        {
            if (id == e.relativeIndexFromNode)
                continue;
            int i = e.absoluteStartIndex + id;
            r0 *= (1 - 2 * probQ[eSt + i]);
        }
        r0 = (1 + r0) / 2;
        r1 = 1 - r0;
        probR[eSt + p] = r1;
    }
}

__device__ void iterateToCheck(
        CodeInfo* codeInfo,
        Edge* edges,
        float* probP,
        float* probR,
        float* probQ)
{
    int vSt = blockIdx.x * codeInfo->varNodes;
    int eSt = blockIdx.x * codeInfo->totalEdges;
    for (int p = 0; p < codeInfo->totalEdges; p += blockDim.x)
    {
        Edge& e = edges[p + threadIdx.x];
        float q1 = probP[vSt + e.vn];
        for (int id = 0; id < e.edgesConnectedToNode; id++)
        {
            if (id == e.relativeIndexFromNode)
                continue;
            int i = e.absoluteStartIndex + id;
            q1 *= probR[eSt + i];
        }
        probQ[eSt + p] = q1;
    }
}

__global__ void decodeAWGN(
        CodeInfo* codeInfo,
        Edge* edgesFromVariable,
        Edge* edgesFromCheck,
        float* probP,
        float* probR,
        float* probQ,
        float sigma2,
        float* noisedVector)
{
    int totalEdges = codeInfo->totalEdges;
    // start for current decoder in data arrays indexed by edges
    int eSt = blockIdx.x * codeInfo->totalEdges;
    // start for current decoder in data arrays indexed by vars
    int vSt = blockIdx.x * codeInfo->varNodes;

    // initial messages to check nodes
    for (int p = threadIdx.x; p < totalEdges; p += blockDim.x)
    {
        // initProbCalcAWGN
        int j = edgesFromVariable[p].vn;
        float y = noisedVector[vSt + j];
        probP[vSt + j] = 1.0 / (1.0 + exp(-2 * y / sigma2));
        probQ[eSt + p] = probP[vSt + j];
    }
    __syncthreads();
    // iteration back to variable nodes
    for (int p = threadIdx.x; p < totalEdges; p += blockDim.x)
    {
        iterateToVariables(codeInfo, edgesFromCheck, probQ, probR);
    }
    __syncthreads();
    // calculate the estimation
    for (int p = threadIdx.x; p < totalEdges; p += blockDim.x)
    {
    }
    __syncthreads();
    // calculate the syndrome
    for (int p = threadIdx.x; p < totalEdges; p += blockDim.x)
    {
    }
    __syncthreads();
}

/*__global__ void berSimulate(
        CodeInfo* codeInfo,
        Edge* edgesFromVariable,
        Edge* edgesFromCheck,
        EdgeData* edgeDataInitToCheck,
        EdgeData* edgeDataToVariable,
        EdgeData* edgeDataToCheck,
        float sigma2,
        float* noisedVector)
{
    decodeAWGN(
            codeInfo,
            edgesFromVariable,
            edgesFromCheck,
            edgeDataInitToCheck,
            edgeDataToVariable,
            edgeDataToCheck,
            sigma2,
            noisedVector);
}

*/
