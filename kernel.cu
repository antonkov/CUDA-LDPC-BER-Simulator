#include "simulator.h"

__device__ void iterateToVariables(
        CodeInfo* codeInfo,
        Edge* edges,
        float* probQ,
        float* probR)
{
    for (int p = threadIdx.x; p < codeInfo->totalEdges; p += blockDim.x)
    {
        Edge& e = edges[p];
        float r0 = 1;
        float r1 = 1;
        for (int id = 0; id < e.edgesConnectedToNode; id++)
        {
            if (id == e.relativeIndexFromNode)
                continue;
            int i = e.absoluteStartIndex + id;
            r0 *= (1 - 2 * probQ[i]);
        }
        r0 = (1 + r0) / 2;
        r1 = 1 - r0;
        probR[p] = r1;
    }
}

__device__ void iterateToChecks(
        CodeInfo* codeInfo,
        Edge* edges,
        float* probP,
        float* probR,
        float* probQ)
{
    for (int p = threadIdx.x; p < codeInfo->totalEdges; p += blockDim.x)
    {
        Edge& e = edges[p];
        float q1 = probP[e.vn];
        for (int id = 0; id < e.edgesConnectedToNode; id++)
        {
            if (id == e.relativeIndexFromNode)
                continue;
            int i = e.absoluteStartIndex + id;
            q1 *= probR[i];
        }
        probQ[p] = q1;
    }
}

__device__ void estimationCalc(
        CodeInfo* codeInfo, 
        Edge* edges,
        float* probR,
        float* estimation)
{
    for (int p = threadIdx.x; p < codeInfo->totalEdges; p += blockDim.x)
    {
        Edge& e = edges[p];
        float q1 = probR[p];
        float q0 = 1 - q1;
        for (int id = 0; id < e.edgesConnectedToNode; id++)
        {
            int i = e.absoluteStartIndex + id;
            q1 *= probR[i];
            q0 *= (1 - probR[i]);
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
        float* probQ,
        float* probR,
        float sigma2,
        float* estimation,
        float* noisedVector,
        int MAX_ITERATIONS,
        ErrorInfo* errorInfo)
{
    int totalEdges = codeInfo->totalEdges;
    {
        // start for current decoder in data arrays indexed by edges
        int eSt = blockIdx.x * codeInfo->totalEdges;
        probQ += eSt;
        probR += eSt;
        // start for current decoder in data arrays indexed by vars
        int vSt = blockIdx.x * codeInfo->varNodes;
        probP += vSt;
        estimation += vSt;
        noisedVector += vSt;
    }

    __shared__ int notZeros;
    for (int iter = 0; iter < MAX_ITERATIONS; iter++)
    {
        if (iter == 0) {
            // initial messages to check nodes
            for (int p = threadIdx.x; p < totalEdges; p += blockDim.x)
            {
                // initProbCalcAWGN
                int j = edgesFromVariable[p].vn;
                float y = noisedVector[j];
                probP[j] = 1.0 / (1.0 + exp(-2 * y / sigma2));
                probQ[p] = probP[j];
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
        if (threadIdx.x == 0)
            notZeros = 0;
        __syncthreads();
        for (int p = threadIdx.x; p < totalEdges; p += blockDim.x)
        {
            Edge& e = edgesFromVariable[p];
            if (e.relativeIndexFromNode == 0 && estimation[e.vn])
                atomicAdd(&notZeros, 1);
        }
        __syncthreads();
        if (notZeros == 0) {
            return;
        }
    }
    // calculation error
    if (threadIdx.x == 0)
    { // to count only once
        atomicAdd(&errorInfo->frameErrors, 1);
        atomicAdd(&errorInfo->bitErrors, notZeros);
    }
}
