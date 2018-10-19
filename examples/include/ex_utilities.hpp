#ifndef EX_UTILITIES_HPP
#define EX_UTILITIES_HPP

#include <iostream>

#include "parser.hpp"
#include "vector.hpp"
#include "densematrix.hpp"
#include "sparsematrix.hpp"
#include "eigensolver.hpp"
#include "timer.hpp"

using Vector = linalgcpp::Vector<double>;
using VectorView = linalgcpp::VectorView<double>;
using DenseMatrix = linalgcpp::DenseMatrix;
using SparseMatrix = linalgcpp::SparseMatrix<double>;
using CooMatrix = linalgcpp::CooMatrix<double>;
using Timer = linalgcpp::Timer;


#endif // EX_UTILITIES_HPP
