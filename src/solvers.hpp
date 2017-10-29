/*! @file */

#ifndef SOLVERS_HPP__
#define SOLVERS_HPP__

#include "operator.hpp"
#include "vector.hpp"

namespace linalgcpp
{

/*! @brief Conjugate Gradient.  Solves Ax = b
    @param A operator to apply the action of A
    @param b right hand side vector
    @param max_iter maxiumum number of iterations to perform
    @param tol relative tolerance for stopping criteria
    @param verbose display additional iteration information
    @retval Vector x the computed solution

    @note Uses random initial guess for x
*/
Vector<double> CG(const Operator& A, const Vector<double>& b,
                  int max_iter = 1000, double tol = 1e-16, bool verbose = false);

/*! @brief Conjugate Gradient.  Solves Ax = b
    @param A operator to apply the action of A
    @param b right hand side vector
    @param[in,out] x intial guess on input and solution on output
    @param max_iter maxiumum number of iterations to perform
    @param tol relative tolerance for stopping criteria
    @param verbose display additional iteration information
*/
void CG(const Operator& A, const Vector<double>& b, Vector<double>& x,
        int max_iter = 1000, double tol = 1e-16, bool verbose = false);

/*! @brief Preconditioned Conjugate Gradient.  Solves Ax = b
           where M is preconditioner for A
    @param A operator to apply the action of A
    @param M operator to apply the preconditioner
    @param b right hand side vector
    @param max_iter maxiumum number of iterations to perform
    @param tol relative tolerance for stopping criteria
    @param verbose display additional iteration information
    @retval Vector x the computed solution

    @note Uses random initial guess for x
*/
Vector<double> PCG(const Operator& A, const Operator& M, const Vector<double>& b,
                   int max_iter = 1000, double tol = 1e-16, bool verbose = false);

/*! @brief Preconditioned Conjugate Gradient.  Solves Ax = b
           where M is preconditioner for A
    @param A operator to apply the action of A
    @param M operator to apply the preconditioner
    @param b right hand side vector
    @param[in,out] x intial guess on input and solution on output
    @param max_iter maxiumum number of iterations to perform
    @param tol relative tolerance for stopping criteria
    @param verbose display additional iteration information
*/
void PCG(const Operator& A, const Operator& M, const Vector<double>& b, Vector<double>& x,
         int max_iter = 1000, double tol = 1e-16, bool verbose = false);

} //namespace linalgcpp

#endif // SOLVERS_HPP__
