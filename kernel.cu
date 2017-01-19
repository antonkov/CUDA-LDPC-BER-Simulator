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
        probR[eSt + p + threadIdx.x] = r1;
    }
}

__device__ void iterateToChecks(
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
        probQ[eSt + p + threadIdx.x] = q1;
    }
}

__device__ void estimationCalc(
        CodeInfo* codeInfo, 
        Edge* edges,
        float* probR,
        float* estimation)
{
    int eSt = blockIdx.x * codeInfo->totalEdges;
    for (int p = 0; p < codeInfo->totalEdges; p += blockDim.x)
    {
        Edge& e = edges[p + threadIdx.x];
        float q1 = probR[eSt + p + threadIdx.x];
        float q0 = 1 - q1;
        for (int id = 0; id < e.edgesConnectedToNode; id++)
        {
            int i = e.absoluteStartIndex + id;
            q1 *= probR[eSt + p + i];
            q0 *= (1 - probR[eSt + p + i]);
        }
        int index = e.vn;
        if (q1 > q0)
            estimation[index] = 1;
        else
            estimation[index] = 0;
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
        float* estimation,
        float* noisedVector,
        int MAX_ITERATIONS,
        ErrorInfo* errorInfo)
{
    int totalEdges = codeInfo->totalEdges;
    // start for current decoder in data arrays indexed by edges
    int eSt = blockIdx.x * codeInfo->totalEdges;
    // start for current decoder in data arrays indexed by vars
    int vSt = blockIdx.x * codeInfo->varNodes;

    for (int iter = 0; iter < MAX_ITERATIONS; iter++)
    {
        if (iter == 0) {
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
        } else {
            // iteration to check nodes
            for (int p = threadIdx.x; p < totalEdges; p += blockDim.x)
                iterateToChecks(codeInfo, edgesFromVariable, probP, probR, probQ);
            __syncthreads();
        }
        // iteration back to variable nodes
        for (int p = threadIdx.x; p < totalEdges; p += blockDim.x)
            iterateToVariables(codeInfo, edgesFromCheck, probQ, probR);
        __syncthreads();
        // calculate the estimation
        for (int p = threadIdx.x; p < totalEdges; p += blockDim.x)
            estimationCalc(codeInfo, edgesFromVariable, probR, estimation);
        __syncthreads();
        // check that zero
        __shared__ bool allZero;
        allZero = true;
        __syncthreads();
        for (int p = threadIdx.x; p < totalEdges; p += blockDim.x)
        {
            int j = edgesFromVariable[p].vn;
            if (estimation[vSt + j])
                allZero = false;
        }
        __syncthreads();
        if (allZero) {
            return;
        }
    }
    // calculating error
    if (threadIdx.x == 0)
    { // to count only once
        errorInfo->frameErrors++;
        for (int i = 0; i < codeInfo->varNodes; i++)
            if (estimation[vSt + i])
                errorInfo->bitErrors++;
    }
}
