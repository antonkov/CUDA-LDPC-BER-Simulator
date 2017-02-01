#include "algebra.h"
#include <iostream>

bool isZero(Matrix const & m)
{
    return m.totalCells == 0;
}

typedef std::vector<int> vi;

std::vector<vi> asDense(Matrix const & m)
{
    std::vector<vi> res(m.k, vi(m.n));
    for (int i = 0; i < m.k; i++) 
    {
        for (auto & p : m.rows[i])
        {
            res[i][p.first] = 1;
        }
    }
    return res;
}

Matrix fromDense(std::vector<vi> const & v)
{
    Matrix res;
    res.totalCells = 0;
    res.k = v.size();
    res.n = v[0].size();
    res.rows = cells_t(res.k);
    res.cols = cells_t(res.n);
    for (int i = 0; i < res.k; i++)
    {
        for (int j = 0; j < res.n; j++)
        {
            if (v[i][j])
            {
                int num = res.totalCells++;
                res.rows[i].push_back(std::make_pair(j, num));
                res.cols[j].push_back(std::make_pair(i, num));
            }
        }
    }
    return res;
}

std::vector<vi> transpose(std::vector<vi> const & m)
{
    std::vector<vi> res(m[0].size(), vi(m.size()));
    for (int i = 0; i < m.size(); i++) 
    {
        for (int j = 0; j < m[i].size(); j++)
        {
            res[j][i] = m[i][j];
        }
    }
    return res;
}

std::vector<vi> gauss(std::vector<vi> const & inpA, std::vector<vi> & P)
{
    std::vector<vi> A(inpA);
    int n = A.size(), m = A[0].size();
    P = std::vector<vi>(n, vi(n));
    for (int i = 0; i < n; i++)
        P[i][i] = 1;
    int ci = 0;
    for (int j = 0; j < m; j++)
    {
        int maxI = ci;
        for (int i2 = ci; i2 < n; i2++)
        {
            if (A[i2][j] > A[maxI][j])
                maxI = i2;
        }
        if (A[maxI][j])
        {
            std::swap(A[maxI], A[ci]);
            std::swap(P[maxI], P[ci]);
            for (int i2 = 0; i2 < n; i2++)
            {
                if (i2 != ci && A[i2][j])
                {
                    for (int j2 = 0; j2 < m; j2++)
                        A[i2][j2] ^= A[ci][j2];
                    for (int j2 = 0; j2 < n; j2++)
                        P[i2][j2] ^= P[ci][j2];
                }
            }
            ci++;
        }
        if (ci == A.size())
            break;
    }
    return A;
}

void print(std::vector<vi> const & v)
{
    std::cout << "-----matrix-------" << std::endl;
    for (auto & vv : v)
    {
        for (int x : vv)
            std::cout << x << " ";
        std::cout << std::endl;
    }
    std::cout << "------------------" << std::endl;
}

Matrix codingMatrix(Matrix const & H)
{
    std::vector<vi> Ht = transpose(asDense(H));
    int n = Ht.size(), k = Ht[0].size();
    std::vector<vi> Qt;
    std::vector<vi> H2t = gauss(Ht, Qt); // H2t == Qt . Ht --> H . Q == H2
    std::vector<vi> P;
    std::vector<vi> J = gauss(transpose(H2t), P); // J == P . H2 --> J == P . H . Q
    // H . Gt == 0 --> invP . J . invQ . Gt == 0 --> J . invQ . Gt == 0 
    // Y == invQ . Gt == vstack[0 I_n-k] --> Q . Y == Gt
    std::vector<vi> Q(transpose(Qt));
    std::vector<vi> Gt(n, vi(n - k));
    for (int i = 0; i < n; i++)
    {
        for (int j = k; j < n; j++)
        {
            Gt[i][j - k] = Q[i][j];
        }
    }
    return fromDense(Gt);
}

Matrix multiply(Matrix const & a, Matrix const & b)
{
    if (a.n != b.k)
        exit(1);

    Matrix res;
    res.totalCells = 0;
    res.k = a.k;
    res.n = b.n;
    res.rows = cells_t(res.k);
    res.cols = cells_t(res.n);
    for (int i = 0; i < res.k; i++)
    {
        for (int j = 0; j < res.n; j++)
        {
            int ri = 0, ci = 0;
            int cur = 0;
            while (ri < a.rows[i].size() && ci < b.cols[j].size())
            {
                int rval = a.rows[i][ri].first;
                int cval = b.cols[j][ci].first;
                if (rval < cval)
                    ri++;
                else if (cval < rval)
                    ci++;
                else
                {
                    cur ^= 1;
                    ri++;
                    ci++;
                }
            }
            if (cur)
            {
                int num = res.totalCells++;
                res.rows[i].push_back(std::make_pair(j, num));
                res.cols[j].push_back(std::make_pair(i, num));
            }
        }
    }
    return res;
}
