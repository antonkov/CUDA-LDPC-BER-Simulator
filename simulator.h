#pragma once

#include "input.h"

#include <string>

const int DEFAULT_NUMBER_OF_FRAME_ERRORS = 100; 
const int MAX_NUMBER_OF_CODEWORDS = 1000 * 1000 * 1000;

struct simulation_params_t
{
    int numberOfCodewords = MAX_NUMBER_OF_CODEWORDS;
    int numberOfFrameErrors = DEFAULT_NUMBER_OF_FRAME_ERRORS;
    float snr;
};

struct Edge {
    int index; // e array
    int vn;    // v array
    int cn;    // c array
    int edgesConnectedToNode;  // t array
    int absoluteStartIndex;    // s array
    int relativeIndexFromNode; // u array
};

struct CodeInfo {
    int totalEdges; // number of edges
    int varNodes;   // number of variable nodes
    int checkNodes; // number of check nodes
};

struct ErrorInfo {
    unsigned long long bitErrors;
    unsigned long long frameErrors;
};

struct Data
{
    CodeInfo* codeInfo;
    Edge* edgesFromVariable;
    Edge* edgesFromCheck;
    Matrix Gt;
    float* probP;
    float* probQ;
    float* probR;
    float* noisedVector;
    int noisedVectorSize;
    float* codewords;
    float* estimation;
    ErrorInfo* errorInfo;

    Data(std::string const & filename);
    ~Data();
};

struct SimulationReport {
    float FER = 0;
    float BER = 0;
    float timeMs;
};

SimulationReport simulate(Data const & data, simulation_params_t const &);
