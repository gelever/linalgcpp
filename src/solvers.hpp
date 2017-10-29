/*! @file */

#ifndef SOLVERS_HPP__
#define SOLVERS_HPP__

#include "operator.hpp"
#include "vector.hpp"

namespace linalgcpp
{

Vector<double> CG(const Operator& A, const Vector<double>& b,
                  int max_iter = 1000, double tol = 1e-16, bool verbose = false);

void CG(const Operator& A, const Vector<double>& b, Vector<double>& x,
        int max_iter = 1000, double tol = 1e-16, bool verbose = false);

Vector<double> PCG(const Operator& A, const Operator& M, const Vector<double>& b,
                   int max_iter = 1000, double tol = 1e-16, bool verbose = false);

void PCG(const Operator& A, const Operator& M, const Vector<double>& b, Vector<double>& x,
         int max_iter = 1000, double tol = 1e-16, bool verbose = false);

} //namespace linalgcpp

#endif // SOLVERS_HPP__
