#include <iostream>

#include "../include/ex_utilities.hpp"

double PowerIterate(const Operator& A, int max_iter, double tol, bool verbose)
{
    Vector x(A.Rows());
    Vector y(A.Rows());

    x.Randomize(-1.0, 1.0);
    x.Normalize();

    double rayleigh = 0.0;
    double old_rayleigh = 0.0;

    for (int i = 0; i < max_iter; ++i)
    {
        A.Mult(x, y);

        rayleigh = (y * x) / (x * x);

        x = y;
        x /= x.L2Norm();

        if (verbose)
        {
            std::cout << std::scientific;
            std::cout << " i: " << i << " ray: " << rayleigh;
            std::cout << " rate: " << (std::fabs(rayleigh - old_rayleigh) / rayleigh) << "\n";
        }

        if (std::fabs(rayleigh - old_rayleigh) / std::fabs(rayleigh) < tol)
        {
            break;
        }

        old_rayleigh = rayleigh;
    }

    return rayleigh;
}

SparseMatrix AdjToLaplacian(const SparseMatrix& A)
{
    SparseMatrix lap_A(A);

    int n = A.Rows();

    auto& indptr = lap_A.GetIndptr();
    auto& indices = lap_A.GetIndices();
    auto& data = lap_A.GetData();

    for (int row = 0; row < n; ++row)
    {
        int diag_index = -1;
        double val = 0.0;

        for (int j = indptr[row]; j < indptr[row + 1]; ++j)
        {
            int col = indices[j];

            if (col == row)
            {
                diag_index = j;
            }
            else
            {
                val++;
                data[j] = -1.0;
            }
        }

        assert(diag_index >= 0);

        data[diag_index] = val;
    }

    return lap_A;
}

int main()
{
    SparseMatrix vertex_edge = linalgcpp::ReadCooList("data/internet.coo");
    SparseMatrix edge_vertex = vertex_edge.Transpose();
    SparseMatrix vertex_vertex = vertex_edge.Mult(edge_vertex);

    SparseMatrix A = AdjToLaplacian(vertex_vertex);

    double eval = PowerIterate(A, 1000, 1e-7, true);

    std::cout << "Max Eval: " << eval << "\n";

    std::cout << "A:" << A.Rows() << "\n";
}
