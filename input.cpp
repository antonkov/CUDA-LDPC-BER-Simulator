#include "input.h"

#include <algorithm>
#include <iostream>

cells_t vecOfVecs(int size)
{
    return cells_t(size, std::vector<std::pair<int, int>>());
}

void readMatrix(std::istream& in, Matrix* mPtr)
{
    Matrix& m = *mPtr;
    std::string inputType;
    in >> inputType;
    if (inputType == "matrix")
    {
        in >> m.k >> m.n;
        m.rows = vecOfVecs(m.k);
        m.cols = vecOfVecs(m.n);

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
        in >> m.k >> m.n;
        m.rows = vecOfVecs(m.k);
        m.cols = vecOfVecs(m.n);

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
        int n, k;
        in >> k >> n;
        int M;
        in >> M;
        m.n = n * M;
        m.k = k * M;

        m.rows = vecOfVecs(m.k);
        m.cols = vecOfVecs(m.n);

        m.totalCells = 0;
        std::vector<std::vector<int>> hd(k, std::vector<int>(n));
        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < n; j++)
            {
                in >> hd[i][j];
                if (hd[i][j] >= 0)
                {
                    for (int idx = 0; idx < M; idx++)
                    {
                        int row = i * M + idx;
                        int col = j * M + (idx + hd[i][j]) % M;
                        m.rows[row].push_back(
                                std::make_pair(col, m.totalCells));
                        m.cols[col].push_back(
                                std::make_pair(row, m.totalCells));
                        m.totalCells++;
                    }
                }
            }
        }
        for (int i = 0; i < m.k; i++)
            std::sort(m.rows[i].begin(), m.rows[i].end());
        for (int j = 0; j < m.n; j++)
            std::sort(m.cols[j].begin(), m.cols[j].end());
    }
}
