#pragma once

#include "simulator.h"

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
        int cntThreads);
