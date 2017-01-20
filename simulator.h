#pragma once

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
