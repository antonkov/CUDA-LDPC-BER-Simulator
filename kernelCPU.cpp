#include "kernelCPU.h"

#include <math.h>
#include <algorithm>
#include <stdlib.h>

namespace
{
    float logtanh(float x)
    {
        float t = exp(abs(x)); // should be just exp(x) if in pseudocode and abs in matlab
        float y = log((t + 1.0) / (t - 1.0));
        return y;
    }

    void iterateToZ(
            CodeInfo* codeInfo,
            Edge* edges,
            float* L,
            float* Z)
    {
        for (int p = 0; p < codeInfo->totalEdges; p++)
        {
            Edge& e = edges[p];
            float alphaProd = 1;
            float fSum = 0;
            for (int id = 0; id < e.edgesConnectedToNode; id++)
            {
                if (id == e.relativeIndexFromNode)
                    continue;
                int i = e.absoluteStartIndex + id;
                float alpha = (L[i] < 0) ? -1 : 1;
                alphaProd *= alpha;
                fSum += logtanh(abs(L[i]));
            }
            Z[p] = alphaProd * logtanh(fSum);
            Z[p] = std::min(Z[p], 19.07f);
            Z[p] = std::max(Z[p], -19.07f);
        }
    }

    void iterateToL(
            CodeInfo* codeInfo,
            Edge* edges,
            float* y,
            float* Z,
            float* L)
    {
        for (int p = 0; p < codeInfo->totalEdges; p++)
        {
            Edge& e = edges[p];
            float val = y[e.vn];
            for (int id = 0; id < e.edgesConnectedToNode; id++)
            {
                if (id == e.relativeIndexFromNode)
                    continue;
                int i = e.absoluteStartIndex + id;
                val += Z[i];
            }
            L[p] = val;
        }
    }

    void estimationCalc(
            CodeInfo* codeInfo, 
            Edge* edges,
            float* y,
            float* Z,
            float* estimation)
    {
        for (int p = 0; p < codeInfo->totalEdges; p++)
        {
            Edge& e = edges[p];
            float sumZ = y[e.vn];
            for (int id = 0; id < e.edgesConnectedToNode; id++)
            {
                int i = e.absoluteStartIndex + id;
                sumZ += Z[i];
            }
            int index = e.vn;
            if (sumZ < 0)
                estimation[index] = 1;
            else
                estimation[index] = 0;
        }
    }
}

void decodeAWGN_CPU(
        CodeInfo* codeInfo,
        Edge* edgesFromVariable,
        Edge* edgesFromCheck,
        float* yInp,
        float* LInp,
        float* ZInp,
        float sigma2,
        float* estimationInp,
        float* codewordsInp,
        float* noisedVectorInp,
        int MAX_ITERATIONS,
        ErrorInfo* errorInfo,
        int cntBlocks,
        int cntThreads)
{
    for (int blockIdx = 0; blockIdx < cntBlocks; blockIdx++)
    {
        // start for current decoder in data arrays indexed by edges
        int eSt = blockIdx * codeInfo->totalEdges;
        float* L = LInp + eSt;
        float* Z = ZInp + eSt;
        // start for current decoder in data arrays indexed by vars
        int vSt = blockIdx * codeInfo->varNodes;
        float* y = yInp + vSt;
        float* estimation = estimationInp + vSt;
        float* codewords = codewordsInp + vSt;
        float* noisedVector = noisedVectorInp + vSt;

        // initial messages to check nodes
        for (int p = 0; p < codeInfo->varNodes; p++)
        {
            float val = codewords[p] * 2 - 1 + noisedVector[p];
            y[p] = -2 * val / sigma2;
        }
        for (int p = 0; p < codeInfo->totalEdges; p++)
            Z[p] = L[p] = 0;
        //__syncthreads();

        /*__shared__*/ int notZeros;
        for (int iter = 0; iter < MAX_ITERATIONS; iter++)
        {
            iterateToL(codeInfo, edgesFromVariable, y, Z, L);
            //__syncthreads();
            iterateToZ(codeInfo, edgesFromCheck, L, Z);
            //__syncthreads();
            // calculate the estimation
            estimationCalc(codeInfo, edgesFromVariable, y, Z, estimation);
            //__syncthreads();
            // check that zero
            notZeros = 0;
            //__syncthreads();
            for (int p = 0; p < codeInfo->varNodes; p++)
                if (estimation[p] != codewords[p])
                    notZeros += 1;
            //__syncthreads();
            if (notZeros == 0) {
                return;
            }
        }
        // calculation error
        errorInfo->frameErrors++;
        errorInfo->bitErrors += notZeros;
    }
}
