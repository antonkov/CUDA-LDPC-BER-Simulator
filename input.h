#pragma once

#include <vector>
#include <istream>

typedef std::vector<std::vector<std::pair<int, int> > > cells_t;

struct Matrix
{
    cells_t rows;
    cells_t cols;
    int n, k;
    int totalCells;
};

void readMatrix(std::istream& in, Matrix* m);
