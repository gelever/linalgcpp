/*! @file*/

#include <stdio.h>
#include <assert.h>

#include "sparsesolve.hpp"
#include "linalgcpp.hpp"

using namespace linalgcpp;

template <typename T = double>
void EliminateRowCol(SparseMatrix<T>& mat, int row_col);

template <typename T = double>
SparseMatrix<T> MakeMatrix(size_t size, size_t neighbors);

int main(int argc, char** argv)
{
    size_t size = 20;
    size_t neighbors = 4;

    SparseMatrix<double> A = MakeMatrix(size, neighbors);

    A.PrintDense("Sparse:");

    EliminateRowCol(A, 0);
    A.PrintDense("Sparse Elim:");

    SparseSolver sp_solver(A);

    Vector<double> rhs(A.Rows());
    rhs.Randomize();

    auto sol = sp_solver.Mult(rhs);

    auto diff = A.Mult(sol);
    diff -= rhs;
    diff /= rhs.L2Norm();

    std::cout.precision(16);
    std::cout << "rhs: " << rhs;
    std::cout << "sol: " << sol;
    std::cout << "diff: " << diff;

    std::cout << "Error: " << diff.L2Norm() << "\n";
}

template <typename T>
SparseMatrix<T> MakeMatrix(size_t size, size_t neighbors)
{
    assert(size > 2 * neighbors);

    CooMatrix<T> coo;

    for (size_t i = 0; i < size; ++i)
    {
        coo.AddSym(i, i, 2 * neighbors);

        for (size_t j = i + 1; j < i + neighbors + 1; ++j)
        {
            coo.AddSym(i, j % size, -1);
        }
    }

    return coo.ToSparse();
}

template <typename T>
void EliminateRowCol(SparseMatrix<T>& mat, int row_col)
{
    std::vector<T> ones(mat.Rows(), (T) 1);
    std::vector<T> zeros(mat.Rows(), (T) 0);

    ones[row_col] = (T) 0;
    zeros[row_col] = (T) 1;

    mat.ScaleRows(ones);
    mat.ScaleCols(ones);
    mat.AddDiag(zeros);
}
