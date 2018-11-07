#ifndef EX_UTILITIES_HPP
#define EX_UTILITIES_HPP

#include <iostream>

#include "linalgcpp.hpp"

using Vector = linalgcpp::Vector<double>;
using VectorView = linalgcpp::VectorView<double>;
using DenseMatrix = linalgcpp::DenseMatrix;
using SparseMatrix = linalgcpp::SparseMatrix<double>;
using CooMatrix = linalgcpp::CooMatrix<double>;
using Timer = linalgcpp::Timer;
using Operator = linalgcpp::Operator;
using linalgcpp::linalgcpp_assert;
using linalgcpp::linalgcpp_verify;


inline
DenseMatrix parse_dense(const std::string& filename)
{
    return linalgcpp::ReadCooList(filename).ToDense();
}

inline
DenseMatrix random_mat(int n, int m)
{
    std::vector<double> data(n * m);

    std::random_device r;
    std::default_random_engine dev(r());
    std::uniform_real_distribution<double> gen(-1.0, 1.0);

    for (double& val : data)
    {
        val = gen(dev);
    }

    return DenseMatrix(n, m, std::move(data));
}

inline
DenseMatrix symmetrize(DenseMatrix A)
{
    for (int i = 0; i < A.Rows(); ++i)
    {
        for (int j = i; j < A.Cols(); ++j)
        {
            A(i, j) = A(j, i);
        }
    }

    return A;
}

#endif // EX_UTILITIES_HPP
