#pragma once

#include <string>

const int DEFAULT_NUMBER_OF_CODEWORDS = 10 * 1000;
const int DEFAULT_NUMBER_OF_MIN_FER = 100; 
enum NumberOfRuns { MIN_FER, CODEWORDS };

struct simulation_params_t
{
    NumberOfRuns runsType = MIN_FER;
    int numberOfCodewords = DEFAULT_NUMBER_OF_CODEWORDS;
    int numberOfMinFER = DEFAULT_NUMBER_OF_MIN_FER;
    std::string filename;
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

struct SimulationReport {
    float FER = 0;
    float BER = 0;
    float timeMs;
};

SimulationReport simulate(simulation_params_t const &);
