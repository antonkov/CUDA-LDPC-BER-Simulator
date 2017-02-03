#include "simulator.h"

__device__ float logtanh(float x)
{
    // if x is positive then y is positive
    float t = exp(x); // should be just exp(x) if in pseudocode and fabs in matlab
    float y = log((t + 1.0) / (t - 1.0));
    return y;
}

__device__ void iterateToZ(
        CodeInfo* codeInfo,
        Edge* edges,
        float* L,
        float* Z)
{
    for (int p = threadIdx.x; p < codeInfo->totalEdges; p += blockDim.x)
    {
        Edge& e = edges[p];
        float alphaProd = 1;
        float fSum = 0;
        for (int id = 0; id < e.edgesConnectedToNode; id++)
        {
            if (id == e.relativeIndexFromNode)
                continue;
            int i = e.absoluteStartIndex + id;
            Edge& eAdj = edges[i];
            float alpha = (L[eAdj.index] < 0) ? -1 : 1;
            alphaProd *= alpha;
            fSum += logtanh(fabs(L[eAdj.index]));
        }
        float val = alphaProd * logtanh(fSum); // fSum is positive
        val = min(val, 19.07);
        val = max(val, -19.07);
        Z[e.index] = val;
    }
    __syncthreads();
}

__device__ void iterateToL(
        CodeInfo* codeInfo,
        Edge* edges,
        float* y,
        float* Z,
        float* L)
{
    for (int p = threadIdx.x; p < codeInfo->totalEdges; p += blockDim.x)
    {
        Edge& e = edges[p];
        float val = y[e.vn];
        for (int id = 0; id < e.edgesConnectedToNode; id++)
        {
            if (id == e.relativeIndexFromNode)
                continue;
            int i = e.absoluteStartIndex + id;
            Edge& eAdj = edges[i];
            val += Z[eAdj.index];
        }
        L[e.index] = val;
    }
    __syncthreads();
}

__device__ void calcZeros(CodeInfo* codeInfo, int* notZeros, float* estimation, float* codewords)
{
    // check that zero
    for (int p = threadIdx.x; p < codeInfo->varNodes; p += blockDim.x)
        if (estimation[p] != codewords[p])
            atomicAdd(notZeros, 1);
    __syncthreads();
}

__device__ void estimationCalc(
        CodeInfo* codeInfo, 
        Edge* edges,
        float* y,
        float* Z,
        float* estimation,
        float* codewords,
        int* notZeros)
{
    for (int p = threadIdx.x; p < codeInfo->totalEdges; p += blockDim.x)
    {
        Edge& e = edges[p];
        float sumZ = y[e.vn];
        for (int id = 0; id < e.edgesConnectedToNode; id++)
        {
            int i = e.absoluteStartIndex + id;
            Edge& eAdj = edges[i];
            sumZ += Z[eAdj.index];
        }
        int index = e.vn;
        if (sumZ < 0)
            estimation[index] = 1;
        else
            estimation[index] = 0;
    }
    if (threadIdx.x == 0)
        notZeros = 0;
    __syncthreads();
    calcZeros(codeInfo, notZeros, estimation, codewords);
}

__global__ void decodeAWGN(
        CodeInfo* codeInfo,
        Edge* edgesFromVariable,
        Edge* edgesFromCheck,
        float* y,
        float* L,
        float* Z,
        float sigma2,
        float* estimation,
        float* codewords,
        float* noisedVector,
        int MAX_ITERATIONS,
        ErrorInfo* errorInfo)
{
    {
        // start for current decoder in data arrays indexed by edges
        int eSt = blockIdx.x * codeInfo->totalEdges;
        L += eSt;
        Z += eSt;
        // start for current decoder in data arrays indexed by vars
        int vSt = blockIdx.x * codeInfo->varNodes;
        y += vSt;
        estimation += vSt;
        codewords += vSt;
        noisedVector += vSt;
    }
    // initial messages to check nodes
    for (int p = threadIdx.x; p < codeInfo->varNodes; p += blockDim.x)
    {
        float val = codewords[p] * 2 - 1 + noisedVector[p];
        y[p] = -2 * val / sigma2;
    }
    for (int p = threadIdx.x; p < codeInfo->totalEdges; p += blockDim.x)
        Z[p] = L[p] = 0;
    __syncthreads();

    __shared__ int notZeros;
    estimationCalc(codeInfo, edgesFromVariable, y, Z, estimation, codewords, &notZeros);
    if (notZeros == 0)
        return;
    for (int iter = 0; iter < MAX_ITERATIONS; iter++)
    {
        iterateToL(codeInfo, edgesFromVariable, y, Z, L);
        iterateToZ(codeInfo, edgesFromCheck, L, Z);
        estimationCalc(codeInfo, edgesFromVariable, y, Z, estimation, codewords, &notZeros);
        if (notZeros == 0)
            return;
    }
    // calculation error
    if (threadIdx.x == 0)
    { // to count only once
        atomicAdd(&errorInfo->frameErrors, 1);
        atomicAdd(&errorInfo->bitErrors, notZeros);
    }
}
