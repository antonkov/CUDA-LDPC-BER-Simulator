#include "input.h"

#include <algorithm>
#include <iostream>

void readMatrix(std::istream& in, Matrix* mPtr)
{
    Matrix& m = *mPtr;
    std::string inputType;
    in >> inputType;
    in >> m.k >> m.n;
    m.rows = cells_t(m.k, std::vector<std::pair<int, int>>());
    m.cols = cells_t(m.n, std::vector<std::pair<int, int>>());
    if (inputType == "matrix")
    {
        m.totalCells = 0;
        std::vector<std::vector<int>> h(m.k, std::vector<int>(m.n));
        for (int i = 0; i < m.k; i++)
        {
            for (int j = 0; j < m.n; j++)
            {
                in >> h[i][j];
                if (h[i][j])
                {
                    int id = m.totalCells;
                    m.rows[i].push_back(std::make_pair(j, id));
                    m.cols[j].push_back(std::make_pair(i, id));
                    m.totalCells++;
                }
            }
        }
    } else if (inputType == "coo_matrix") {
        in >> m.totalCells;
        for (int i = 0; i < m.totalCells; i++)
        {
            int row, col;
            in >> row >> col;
            m.rows[row].push_back(std::make_pair(col, i));
            m.cols[col].push_back(std::make_pair(row, i));
        }
        for (int i = 0; i < m.k; i++)
            std::sort(m.rows[i].begin(), m.rows[i].end());
        for (int j = 0; j < m.n; j++)
            std::sort(m.cols[j].begin(), m.cols[j].end());
    } else if (inputType == "proto_matrix") {
        std::cout << "Not supported format" << std::endl;
        exit(1);
    }
}
